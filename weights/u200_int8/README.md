# Trained U200 INT8 model

Full production channels: **8 → 16 → 32 → 64 → 32 → 16 → 8**,
three flow/mask outputs, **194,224 signed INT8 weights** and 12 shift scales.
The hardware-compatible QAT checkpoint completed 10 epochs / 16,040 steps.

| File | Use |
|---|---|
| `config.hex` | Packed coefficients and shift records consumed by the XRT host |
| `quantized_model.json` | Integer tensors, shapes and shifts |
| `manifest.json` | Layer layout, model provenance and checksums |
| `training.json` | Training and integer-export validation summary |

The board demo uses these coefficients. See the [U200 guide](../../hardware/u200/README.md)
for video inference and the [training guide](../../docs/training.md) for the model.
The published coefficient stream has SHA-256
`84aed863ea2b4c4bc883544547bd1aedacc6780fb10c6a22c88896507148644a`.
