# Weight export and mapping

Run commands from the repository root. Checkpoint tools require the
[training environment](../../docs/training.md). Load only trusted checkpoints.

## Hardware-compatible QAT export

Export weights and fixed per-layer shifts without scale folding:

```bash
python3 sw/verification/export_hardware_qat.py \
  --checkpoint /path/to/last.ckpt --model-root . \
  --output build/weights
```

Use `--state` instead of `--checkpoint` for a bare `HardwareFlow.state_dict()`.
The output includes `config.hex`, `quantized_model.json` and a manifest.
Optional `--data-root /path/to/vimeo_triplet --samples 32` compares model and
exported integer outputs. The supplied checkpoint must match the current model.

## Legacy Brevitas checkpoint mapping

```bash
python3 sw/verification/map_qat_weights.py \
  --checkpoint /path/to/brevitas.ckpt --model-root . \
  --output build/mapped_weights
```

This folds activation scales into INT8 weights and shift-only scaling. It is an
approximation, rather than a bit-exact export of the original Brevitas graph.
Hardware-compatible QAT uses the exporter above.

## Integer document to hardware records

```bash
python3 sw/verification/export_weights.py \
  --input weights/u200_int8/quantized_model.json --output build/weights
```

This conversion uses Python's standard library. Checkpoint conversion through
this generic exporter requires an explicit `--quantization` file; it does not
infer missing scales.

`make_board_fixture.py --help` describes how to generate Full-HD RGB24 input
and expected-output fixtures from an exported model for the
[deployment bundle](../deploy/README.md). It uses the persistent selective-transfer
representatives when generating expected frames.
