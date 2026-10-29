#!/usr/bin/env python3
"""Initialize normalization-free hardware QAT from mapped INT8 coefficients."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from model.hardware_qat import HardwareFlow


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mapping", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    model = HardwareFlow(initial_mapping=args.mapping).eval()
    document = model.export_document()
    source = json.loads(args.mapping.read_text())
    if document["layers"] != source["layers"]:
        raise RuntimeError("Initialization changed the supplied integer weights or shifts")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output/"initial_state.pt")
    (args.output/"initial_mapping.json").write_text(json.dumps(document)+"\n")
    report = dict(result="PASS",initialization="scale-folded 10-epoch QAT checkpoint mapping",
        source_mapping=str(args.mapping.resolve()),
        source_sha256=hashlib.sha256(args.mapping.read_bytes()).hexdigest(),
        normalization="none", trainable_parameters=sum(p.numel() for p in model.parameters()),
        identical_integer_weights_and_shifts=True, bias_parameters=0)
    (args.output/"initialization.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
