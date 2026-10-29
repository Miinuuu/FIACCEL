# Training

Run from the repository root with the [training environment](../docs/training.md):

```bash
python -m training.train --config-name train_config_int
# Hardware-compatible fine-tuning:
FIACCEL_TRAIN_CONFIG=train_config_h200_hwqat bash training/scripts/train_h200_qat.sh
```

| Path | Role |
|---|---|
| `train.py`, `lightning.py` | Hydra entry point and Lightning training loop |
| `config/` | Model, dataset, optimizer and experiment settings |
| `data/` | Vimeo datasets and datamodules |
| `utils/` | Schedulers, configuration and visualization helpers |
| `scripts/` | QAT initialization and reproducible launch snapshots |
| `environment.yaml` | Research Conda environment |

The network and image synthesis live in [`model/`](../model/).
Hydra targets use `model.pipeline`, `training.data` and `training.utils`.
Older saved Hydra configurations using `model_pipeline`, `datamodules` or
`utils` must use these new package paths when loaded with this layout.
Model parameter names and exported integer coefficients are unchanged.
