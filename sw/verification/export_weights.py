#!/usr/bin/env python3
"""Export explicit int8 weights / scalar shifts into FIACCEL RTL configuration.

JSON input: {"depths":[8,16,32,64,32,16,8], "layers": {
  "enc.0": {"shape":[8,3,3,3], "values":[...], "shift":8}, ...}}
Normal layers use OIHW; dec.0/2/4 use PyTorch IOHW. Values are flattened
in that layout and must already be signed integers in [-128,127].

Alternatively --checkpoint FILE --quantization FILE reads PyTorch tensors.
Quantization JSON maps layer names to {"weight_scale":positive_float,"shift":N}.
This deliberately requires explicit scales: Brevitas scales are not guessed.
"""
import argparse
import json
import math
from pathlib import Path

NAMES = ["enc.0","enc.2","enc.4","aspp1.0","aspp2.0","aspp3.0","aspp4.0",
         "shrink.0","dec.0","dec.2","dec.4","output.0"]


def shapes(depths,output_channels=3):
    if len(depths)!=7 or any(type(x) is not int or x<1 for x in depths):
        raise ValueError("depths must contain seven positive integers")
    a,b,c,d,e,f,g=depths
    return [(a,3,3,3),(b,a,3,3),(c,b,3,3)]+[(d,2*c,3,3)]*4+[
        (d,4*d,1,1),(d,e,3,3),(e,f,3,3),(f,g,3,3),(output_channels,g,3,3)]


def from_checkpoint(args):
    import torch
    quant=json.loads(args.quantization.read_text())
    checkpoint=torch.load(args.checkpoint,map_location="cpu",weights_only=True)
    state=checkpoint.get("state_dict",checkpoint)
    layers={}
    for name in NAMES:
        candidates=[k for k in state if k==name+".weight" or k.endswith("."+name+".weight")]
        if len(candidates)!=1:
            raise ValueError(f"Expected exactly one tensor for {name}.weight, found {candidates}")
        tensor=state[candidates[0]].detach().cpu()
        scale=quant[name]["weight_scale"]
        if not isinstance(scale,(int,float)) or not math.isfinite(scale) or scale<=0:
            raise ValueError(f"{name}: weight_scale must be finite and positive")
        data=(tensor/scale).round()
        if not bool(torch.isfinite(data).all()) or bool(((data< -128)|(data>127)).any()):
            raise ValueError(f"{name}: weights overflow int8; choose a larger weight_scale")
        layers[name]={"shape":list(tensor.shape),"values":data.to(torch.int8).flatten().tolist(),
                      "shift":quant[name]["shift"]}
    return {"depths":args.depths,"layers":layers,"output_channels":args.output_channels}


def normalize(document):
    depths=document.get("depths",[8,16,32,64,32,16,8])
    output_channels=document.get("output_channels",3)
    if output_channels != 3:raise ValueError("FIACCEL requires three output channels")
    result=[]
    for idx,(name,shape) in enumerate(zip(NAMES,shapes(depths,output_channels))):
        layer=document["layers"][name]
        values=layer["values"]
        shift=layer["shift"]
        if tuple(layer["shape"])!=shape:
            raise ValueError(f"{name}: expected shape {shape}, got {layer['shape']}")
        if len(values)!=math.prod(shape) or any(type(v) is not int or v< -128 or v>127 for v in values):
            raise ValueError(f"{name}: expected {math.prod(shape)} signed int8 values")
        if type(shift) is not int or not 0<=shift<=63:
            raise ValueError(f"{name}: shift must be an integer in [0,63]")
        if 8<=idx<=10:
            ci,co,kh,kw=shape
            values=[values[((i*co+o)*kh+y)*kw+x]
                    for o in range(co) for i in range(ci) for y in range(kh) for x in range(kw)]
        result.append((name,values,shift))
    return depths,result


def export(document,destination):
    # Validate everything before creating output files.
    depths,layers=normalize(document)
    destination.mkdir(parents=True,exist_ok=True)
    records=[]
    manifest={"depths":depths,
              "input":"unsigned uint8 RGB, user-supplied input scale",
              "output_activation":"HardTanh",
              "output":f"{document.get('output_channels',3)} signed int16 channels, Q1.14",
              "configuration":"48-bit: [47] shift, [46:43] layer, [42:11] address, [7:0] data",
              "layers":[]}
    for idx,(name,values,shift) in enumerate(layers):
        filename=f"{idx:02d}_{name.replace('.','_')}.hex"
        (destination/filename).write_text("".join(f"{v&255:02x}\n" for v in values))
        records.append((1<<47)|(idx<<43)|shift)
        records.extend((idx<<43)|(address<<11)|(v&255) for address,v in enumerate(values))
        manifest["layers"].append({"id":idx,"name":name,"count":len(values),"shift":shift,"file":filename})
    (destination/"config.hex").write_text("".join(f"{v:012x}\n" for v in records))
    (destination/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return len(records)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input",type=Path)
    source.add_argument("--checkpoint",type=Path)
    parser.add_argument("--quantization",type=Path)
    parser.add_argument("--depths",nargs=7,type=int,default=[8,16,32,64,32,16,8])
    parser.add_argument("--output-channels",type=int,choices=[3],default=3)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.checkpoint and not args.quantization:
        parser.error("--checkpoint requires --quantization with explicit scales and shifts")
    try:
        document=json.loads(args.input.read_text()) if args.input else from_checkpoint(args)
        count=export(document,args.output)
    except (ValueError,KeyError,ImportError) as error:
        parser.error(str(error))
    print(f"Exported {count} configuration writes to {args.output}")


if __name__=="__main__":
    main()
