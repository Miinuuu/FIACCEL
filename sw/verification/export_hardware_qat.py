#!/usr/bin/env python3
"""Export normalization-free hardware-QAT weights and verify integer parity."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sw.verification.export_weights import export
from sw.verification.map_qat_weights import IntegerNetwork, synthesize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--state", type=Path)
    p.add_argument("--model-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--data-root", type=Path)
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--rtl-fixture", type=Path)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    if args.samples < 1:
        p.error("--samples must be positive")
    import numpy as np
    import torch
    from PIL import Image
    torch.set_num_threads(1)
    sys.path.append(str(args.model_root.resolve()))
    from model.hardware_qat import HardwareFlow, HardwarePipeline
    path = args.checkpoint or args.state
    checkpoint = torch.load(path, map_location="cpu", weights_only=not bool(args.checkpoint))
    if args.checkpoint:
        prefix = "model.flow_estimation."
        state = {k.removeprefix(prefix): v for k, v in checkpoint["state_dict"].items() if k.startswith(prefix)}
    else:
        state = checkpoint
    flow = HardwareFlow()
    flow.load_state_dict(state)
    pipeline = HardwarePipeline(flow).to(args.device).eval()
    with torch.no_grad():
        document = flow.export_document()
        records = export(document, args.output)
        (args.output/"quantized_model.json").write_text(json.dumps(document)+"\n")
        manifest_path = args.output/"manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest.update(source=str(path.resolve()), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            input="signed RGB-128", normalization="none",
            trainable_parameters=sum(p.numel() for p in flow.parameters()),
            bias_parameters=sum(p.numel() for n,p in flow.named_parameters() if n.endswith("bias")),
            arithmetic="certified integer sums; floor/right-shift; UINT8 hidden; Q1.14 head; Q8 warp and RGB8 blend",
            configuration_records=records)
        report = dict(result="PASS", weights=records-12, shifts=12)
        if args.rtl_fixture:
            pipeline.reset_stream()
            heads, pixels = [], []
            hook = flow.output.register_forward_hook(lambda m, a, result: heads.append(result.detach().long()))
            inputs = sorted(args.rtl_fixture.glob("input_*.ppm"))
            for file in inputs:
                frame = torch.from_numpy(np.array(Image.open(file))).permute(2, 0, 1).unsqueeze(0).to(args.device).float()/255
                result = pipeline.forward_frame(frame)
                if result is not None:
                    pixels.append((result*255).round().long())
            hook.remove()
            actual_flow = torch.cat(heads).permute(0, 2, 3, 1).reshape(-1, 3).cpu().tolist()
            expected_flow = []
            for word in (args.rtl_fixture/"flow.hex").read_text().split():
                value = int(word, 16)
                expected_flow.append([((value>>(16*c))&65535)-65536 if (value>>(16*c))&32768
                                      else (value>>(16*c))&65535 for c in range(3)])
            actual_rgb = torch.cat(pixels).permute(0, 2, 3, 1).reshape(-1, 3).cpu().tolist()
            expected_rgb = [[(int(word,16)>>(8*c))&255 for c in range(3)]
                            for word in (args.rtl_fixture/"expected.hex").read_text().split()]
            if actual_flow != expected_flow or actual_rgb != expected_rgb:
                raise RuntimeError("Hardware QAT forward differs from the RTL fixture")
            report["rtl_fixture"] = dict(flow_pixels=len(actual_flow), rgb_pixels=len(actual_rgb),
                fmr_writes=flow.feature_reuse.writes, fmr_skips=flow.feature_reuse.skips, exact=True)
        if args.data_root:
            if (args.model_root / "training/data").is_dir():
                from training.data.Vimeo90kTriplet import Vimeo90kTriplet
            else:
                from datamodules.Vimeo90kTriplet import Vimeo90kTriplet
            from torchvision.transforms import CenterCrop
            dataset = Vimeo90kTriplet(args.data_root, split="test", tensor_transform=CenterCrop((256, 256)))
            oracle = IntegerNetwork(document, args.device)
            metrics = []
            for index in range(min(args.samples, len(dataset))):
                images = dataset[index].unsqueeze(0).to(args.device)
                a, gt, b = images[:, 0], images[:, 1], images[:, 2]
                aq, bq = (a*255).round().long(), (b*255).round().long()
                codes = (flow(a, b)*16384).round().long()
                expected_codes = oracle.flow(oracle.encode(aq), oracle.encode(bq))
                output = (pipeline.synthesize(a, b, codes.float()/16384)*255).round().long()
                expected = synthesize(aq, bq, expected_codes)
                if not torch.equal(codes, expected_codes) or not torch.equal(output, expected):
                    raise RuntimeError(f"Model/export integer mismatch at validation sample {index}")
                metrics.append(-10*(output.float()/255-gt).square().mean().log10().item())
            report["quality"] = dict(samples=len(metrics), split="first Vimeo test triplets, center crop 256x256",
                model_psnr_db=sum(metrics)/len(metrics), exported_psnr_db=sum(metrics)/len(metrics),
                max_flow_code_error=0, max_rgb_pixel_error=0, scope="integer model/reference, not board measurement")
        manifest["sha256"] = {f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(args.output.glob("*.hex"))}
        manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
        (args.output/"validation.json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
