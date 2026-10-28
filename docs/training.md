# Training and streaming inference

[← Project overview](../README.md) · [RTL evaluation](../hardware/README.md)

## Environment

The Python model and the protected RTL demo have separate dependencies.
For training, review the PyTorch/CUDA versions in `fiaccel.yaml` for your machine:

```bash
conda env create -f fiaccel.yaml
conda activate fiaccel
```

Run commands from the repository root. Training uses PyTorch, Lightning,
Hydra and Brevitas. The environment is based on the research training setup;
it is not required for the XSim demo.

## Dataset

Use Vimeo-90K triplets and set `data_dir` in
[`config/datamodule/vimeo.yaml`](../config/datamodule/vimeo.yaml).
Update checkpoint directories, Hydra output directories and logging settings
for your environment before starting a run.

```bash
make train_float
make train_int
```

Hydra overrides can be supplied directly:

```bash
python model_train.py --config-name train_config_int ngpu=1 training_loop.train_batch_size=8
```

## Current model interface

The head produces `[flow_x, flow_y, mask]` with a HardTanh activation.
The synthesis pipeline in `model_pipeline.py` converts those channels to an
intermediate image at `t=0.5`. A five-channel checkpoint must be converted or
retrained for this model interface; silently truncating the head is unsupported.

Streaming inference uses `model.eval()`, `reset_stream()` and `forward_frame()`.
The first frame initializes encoder history and returns no interpolated image.
Each following frame is paired with the retained previous frame.
Selective transfer compares encoded feature representatives and retains the
previous representative when the transfer rule permits reuse.

Float inference requires a calibrated feature scale. Quantized inference uses
the encoder's fixed evaluation scale. These model paths do not establish
bit-exact RTL arithmetic or the published PSNR without matching trained weights
and quantization settings.

## Results and figures

<img src="../fig/val_psnr.png" alt="Research training validation PSNR curve" width="720">

The curve is a research training artifact. The public RTL demo uses synthetic
weights and verifies arithmetic/transport against a supplied small fixture;
it does not measure learned interpolation quality.
