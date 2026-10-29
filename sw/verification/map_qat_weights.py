#!/usr/bin/env python3
"""Map a trusted FIACCEL Brevitas checkpoint to the current shift-only RTL.

This is a scale-folded deployment approximation, not a bit-exact QAT export.
The existing RTL centers RGB at 128; the training model consumes RGB/255.
Neither that missing input offset nor round-to-nearest can be represented by
the RTL's bias-free convolutions and arithmetic right shifts.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sw.verification.export_weights import NAMES, export, normalize


def fixed_coefficients(coefficients):
    """Choose the most precise nonnegative shift that does not clip INT8."""
    import torch
    coefficients = coefficients.detach().cpu().double()
    if not torch.isfinite(coefficients).all():
        raise ValueError("Nonfinite effective coefficients")
    positive = coefficients.max().clamp_min(0).item()
    negative = (-coefficients.min()).clamp_min(0).item()
    limits = ([127 / positive] if positive else []) + ([128 / negative] if negative else [])
    shift = min(63, math.floor(math.log2(min(limits)))) if limits else 0
    if shift < 0:
        raise ValueError("Effective coefficient exceeds shift-only INT8 range")
    values = (coefficients * (2 ** shift)).round()
    if values.min() < -128 or values.max() > 127:
        raise ValueError("Mapped coefficient overflow")
    error = values / (2 ** shift) - coefficients
    relative = error.square().sum() / coefficients.square().sum().clamp_min(1e-30)
    return values.to(torch.int8), shift, relative.sqrt().item()


def map_model(model):
    import torch
    activations = {}
    native = {}
    for name in NAMES:
        quantized = model.get_submodule(name).quant_weight()
        if quantized.scale.numel() != 1 or quantized.bit_width.item() != 8:
            raise ValueError(f"{name}: requires per-tensor INT8 weights")
        if quantized.zero_point.any():
            raise ValueError(f"{name}: nonzero weight zero-point")
        native[name] = dict(shape=list(quantized.value.shape),
                            values=quantized.int(float_datatype=False).flatten().tolist(),
                            weight_scale=quantized.scale.item())
        if name != "output.0":
            parent, index = name.rsplit(".", 1)
            act = model.get_submodule(f"{parent}.{int(index)+1}").act_quant
            if act.scale().numel() != 1 or act.bit_width().item() != 8:
                raise ValueError(f"{name}: requires scalar UINT8 activation scale")
            scale = act.scale().item()
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError(f"{name}: invalid activation scale")
            activations[name] = scale
    activations["output.0"] = 1 / 16384
    input_scales = {
        "enc.0": [1 / 255] * 3,
        "enc.2": [activations["enc.0"]] * 8,
        "enc.4": [activations["enc.2"]] * 16,
        **{f"aspp{i}.0": [activations["enc.4"]] * 64 for i in range(1, 5)},
        "shrink.0": sum(([activations[f"aspp{i}.0"]] * 64 for i in range(1, 5)), []),
        "dec.0": [activations["shrink.0"]] * 64,
        "dec.2": [activations["dec.0"]] * 32,
        "dec.4": [activations["dec.2"]] * 16,
        "output.0": [activations["dec.4"]] * 8,
    }
    document = dict(depths=[8, 16, 32, 64, 32, 16, 8], output_channels=3, layers={})
    report = {}
    for name in NAMES:
        weight = model.get_submodule(name).quant_weight().value.detach().cpu().double()
        transpose = name.startswith("dec.")
        scales = torch.tensor(input_scales[name], dtype=torch.float64)
        scales = scales.reshape(-1, 1, 1, 1) if transpose else scales.reshape(1, -1, 1, 1)
        effective = weight * scales / activations[name]
        values, shift, error = fixed_coefficients(effective)
        document["layers"][name] = dict(shape=list(values.shape),
            values=values.flatten().tolist(), shift=shift)
        report[name] = dict(input_scales=input_scales[name], output_scale=activations[name],
            native_weight_scale=native[name]["weight_scale"], shift=shift,
            coefficient_relative_l2_error=error, minimum=int(values.min()), maximum=int(values.max()))
    normalize(document)
    return document, native, report


class IntegerNetwork:
    """Exact integer-valued convolutions using FP64, with RTL shift/clamp rules."""
    def __init__(self, document, device="cpu"):
        import torch
        self.torch = torch
        self.weights = {name: torch.tensor(layer["values"], device=device, dtype=torch.float64)
                        .reshape(layer["shape"]) for name, layer in document["layers"].items()}
        self.shifts = {name: layer["shift"] for name, layer in document["layers"].items()}

    def layer(self, x, name):
        from torch.nn import functional as F
        w = self.weights[name]
        if name.startswith("dec."):
            out = F.conv_transpose2d(x, w, stride=2, padding=1, output_padding=1)
        else:
            dilation = int(name[4]) if name.startswith("aspp") else 1
            padding = (0, 0) if name == "shrink.0" else (1, dilation)
            out = F.conv2d(x, w, stride=2 if name.startswith("enc.") else 1,
                           padding=padding, dilation=(1, dilation))
        out = (out / (2 ** self.shifts[name])).floor()
        return out.clamp(-16384, 16384) if name == "output.0" else out.clamp(0, 255)

    def encode(self, rgb):
        x = rgb.double() - 128
        for name in NAMES[:3]:
            x = self.layer(x, name)
        return x

    def flow(self, previous, current):
        torch = self.torch
        joined = torch.cat((previous, current), 1)
        x = torch.cat([self.layer(joined, name) for name in NAMES[3:7]], 1)
        for name in NAMES[7:]:
            x = self.layer(x, name)
        return x.to(torch.int64)


def synthesize(previous, current, flow):
    """NCHW RGB integers; matches fixtures.synthesize, including Q8 coordinates."""
    import torch
    n, c, h, w = previous.shape
    x = torch.arange(w, device=flow.device).reshape(1, 1, w) * 256
    y = torch.arange(h, device=flow.device).reshape(1, h, 1) * 256
    sx, sy = (flow[:, 0] * (w-1)) >> 8, (flow[:, 1] * (h-1)) >> 8

    def sample(rgb, xq, yq):
        xq, yq = xq.clamp(0, (w-1)*256), yq.clamp(0, (h-1)*256)
        x0, y0 = xq >> 8, yq >> 8
        x1, y1 = (x0+1).clamp(max=w-1), (y0+1).clamp(max=h-1)
        xf, yf = xq & 255, yq & 255
        flat = rgb.to(torch.int64).reshape(n, c, h*w)

        def gather(xx, yy):
            indices = (yy*w+xx).reshape(n, 1, h*w).expand(n, c, h*w)
            return flat.gather(2, indices).reshape(n, c, h, w)
        return (gather(x0, y0)*((256-xf)*(256-yf)).unsqueeze(1) +
                gather(x1, y0)*(xf*(256-yf)).unsqueeze(1) +
                gather(x0, y1)*((256-xf)*yf).unsqueeze(1) +
                gather(x1, y1)*(xf*yf).unsqueeze(1))

    a, b = sample(previous, x-sx, y-sy), sample(current, x+sx, y+sy)
    alpha = (16384 + flow[:, 2]).unsqueeze(1)
    return ((a*alpha+b*(32768-alpha)+(1 << 30)) >> 31).clamp(0, 255)


def evaluate(model, document, data_root, count, device, output):
    import numpy as np
    from PIL import Image
    import torch
    import model as model_package
    if (Path(model_package.__file__).parent.parent / "training/data").is_dir():
        from training.data.Vimeo90kTriplet import Vimeo90kTriplet
        from model.pipeline import FIACCELPipeline
    else:
        from datamodules.Vimeo90kTriplet import Vimeo90kTriplet
        from model_pipeline import FIACCELPipeline
    from torchvision.transforms import CenterCrop

    data = Vimeo90kTriplet(data_root, split="test", tensor_transform=CenterCrop((256, 256)))
    pipeline = FIACCELPipeline(model).to(device).eval()
    network = IntegerNetwork(document, device)
    sums = {"qat_psnr_db": 0., "mapped_rtl_psnr_db": 0., "average_frames_psnr_db": 0.}
    rows = []
    for index in range(min(count, len(data))):
        images = data[index].unsqueeze(0).to(device)
        rgb = (images * 255).round().to(torch.int64)
        a, gt, b = rgb[:, 0], images[:, 1], rgb[:, 2]
        qat = pipeline.interpolate(images[:, 0], images[:, 2])
        flow = network.flow(network.encode(a), network.encode(b))
        mapped = synthesize(a, b, flow).float() / 255
        row = {"index": index, "clip": data.folder_list[index]}
        for key, value in (("qat_psnr_db", qat), ("mapped_rtl_psnr_db", mapped),
                           ("average_frames_psnr_db", (images[:, 0]+images[:, 2])/2)):
            psnr = -10 * (value-gt).square().mean().log10().item()
            row[key] = psnr
            sums[key] += psnr
        rows.append(row)
        if index == 0:
            for name, image in (("previous", images[:, 0]), ("target", gt),
                                ("current", images[:, 2]), ("qat", qat), ("mapped_rtl", mapped)):
                array = (image[0].permute(1, 2, 0)*255).round().clamp(0, 255).byte().cpu().numpy()
                Image.fromarray(np.asarray(array)).save(output/f"sample_{name}.png")
        if (index+1) % 8 == 0:
            print(f"evaluated {index+1}/{count}", flush=True)
    return dict(samples=len(rows), split="Vimeo test, first samples, center crop 256x256",
        inference="Independent pairs, cache warmed/reset per triplet",
        arithmetic="FP64 integer-valued convolution; RTL floor, UINT8 clamp, Q1.14 head, Q8 warp",
        scope="Integer reference only, not a board measurement",
        **{key: total/len(rows) for key, total in sums.items()}, per_sample=rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model-root", type=Path, default=ROOT)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--data-root", type=Path)
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    if args.samples < 1:
        p.error("--samples must be positive")
    import torch
    torch.set_num_threads(1)
    sys.path.append(str(args.model_root.resolve()))
    from model.flow_estimation import flow_estimation_int
    # Full Lightning checkpoints contain the saved OmegaConf configuration.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    prefix = "model.flow_estimation."
    model = flow_estimation_int().eval()
    model.load_state_dict({k.removeprefix(prefix): value for k, value in state.items() if k.startswith(prefix)})
    with torch.no_grad():
        document, native, scales = map_model(model)
        records = export(document, args.output)
        (args.output/"quantized_model.json").write_text(json.dumps(document)+"\n")
        (args.output/"native_qat.json").write_text(json.dumps(native)+"\n")
        (args.output/"quantization.json").write_text(json.dumps(scales, indent=2)+"\n")
        manifest_path = args.output/"manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest.update(training="Brevitas INT8 QAT, 10 epochs", checkpoint=str(args.checkpoint.resolve()),
            checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            checkpoint_epoch=checkpoint.get("epoch"), checkpoint_step=checkpoint.get("global_step"),
            input="raw RGB bytes centered as signed int8 RGB-128 by RTL",
            mapping="effective_weight = QAT_dequant_weight * input_scale / output_scale; INT8 / 2**shift",
            exact_qat_equivalence=False,
            limitations=["QAT uses RGB/255 with signed input quantizer; RTL uses RGB-128 without bias compensation",
                         "QAT rounds to nearest; RTL arithmetic right shifts floor the accumulator",
                         "Arbitrary learned scale ratios are folded and requantized into INT8 weights"],
            input_qat_scale=model.quant_input.act_quant.scale().item(),
            exporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            configuration_records=records,
            sha256={f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                    for f in sorted(args.output.glob("*.hex"))})
        manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
        print(f"Exported {records} writes ({records-12} weights, 12 shifts)", flush=True)
        if args.data_root:
            metrics = evaluate(model, document, args.data_root, args.samples, args.device, args.output)
            (args.output/"quality.json").write_text(json.dumps(metrics, indent=2)+"\n")
            print(json.dumps({k: v for k, v in metrics.items() if k != "per_sample"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
