#!/usr/bin/env python3
"""Generate an exact RGB24 board test stream from exported integer weights."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sw.verification.export_weights import normalize


def sha256(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Export directory")
    parser.add_argument("--input", type=Path, required=True, help="Consecutive RGB24 frames")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.width < 8 or args.height < 8 or args.width % 8 or args.height % 8 or args.threads < 1:
        parser.error("Positive multiples of eight and at least one thread are required")
    frame_bytes = args.width * args.height * 3
    count, remainder = divmod(args.input.stat().st_size, frame_bytes)
    if remainder or count < 2:
        parser.error("Input must contain at least two complete RGB24 frames")
    if args.output.exists():
        parser.error("Choose a new output directory")
    import torch
    from sw.verification.map_qat_weights import IntegerNetwork, synthesize
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.enabled = False
    torch.backends.mkldnn.enabled = False
    torch.backends.cuda.matmul.allow_tf32 = False
    document = json.loads((args.model / "quantized_model.json").read_text())
    normalize(document)
    network = IntegerNetwork(document, args.device)
    args.output.mkdir(parents=True)

    def tensor(data):
        return torch.frombuffer(bytearray(data), dtype=torch.uint8).reshape(
            args.height, args.width, 3).permute(2, 0, 1).unsqueeze(0).to(args.device).long()

    with torch.inference_mode(), args.input.open("rb") as source, \
            (args.output / "input.rgb").open("wb") as copied, \
            (args.output / "expected.rgb").open("wb") as expected:
        data = source.read(frame_bytes)
        copied.write(data)
        expected.write(data)
        previous = tensor(data)
        encoded = network.encode(previous)
        for index in range(1, count):
            data = source.read(frame_bytes)
            copied.write(data)
            current = tensor(data)
            next_encoded = network.encode(current)
            flow = network.flow(encoded, next_encoded)
            midpoint = synthesize(previous, current, flow)
            expected.write(midpoint[0].permute(1, 2, 0).byte().cpu().contiguous().numpy().tobytes())
            expected.write(data)
            # FMR compares all channels' upper three bits per spatial vector.
            # A skipped write retains the old representative for later frames,
            # while this frame's flow still consumes the current exact vector.
            skip = (encoded.long().bitwise_right_shift(5) ==
                    next_encoded.long().bitwise_right_shift(5)).all(dim=1, keepdim=True)
            encoded = torch.where(skip, encoded, next_encoded)
            previous = current
            print(f"Reference midpoint {index}/{count - 1}", flush=True)
        expected.write(data)
    report = dict(format="fiaccel-rgb24-fixture-v1", width=args.width, height=args.height,
        input_frames=count, output_frames=2 * count, interpolated_frames=count - 1,
        model_sha256=sha256(args.model / "quantized_model.json"),
        config_sha256=sha256(args.model / "config.hex"),
        input_source=str(args.input.resolve()),
        reference="FP64 integer convolution, floor/clamp, Q1.14 head, Q8 warp, rounded RGB8",
        selective_transfer="Upper-three-bit comparison across all channels; skipped vectors retain their stored representative",
        scope="Expected stream only; physical board has not been exercised",
        sha256={name: sha256(args.output / name) for name in ("input.rgb", "expected.rgb")})
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
