# FIACCEL

[![FIACCEL RTL architecture](fig/fiaccel_rtl_top.svg)](docs/architecture.md)

**Memory Efficient Frame Interpolation Accelerator for Full-HD Video**<br>
Min Wu Jeong · Chae Eun Rhee | Inha University | IEEE TCAS-II, 2024

[Paper](https://doi.org/10.1109/TCSII.2023.3329966) · [Architecture](docs/architecture.md) · [Training](docs/training.md) · [RTL guide](hardware/README.md) · [U200 host](hardware/u200/README.md)

INT8 frame interpolation with feature reuse, selective DRAM transfer,
row-wise fusion and kernel decomposition. RTL is provided as **30 encrypted
modules** in [`rtl_vp/`](rtl_vp/), with XSim simulation-only access.

<details>
<summary>Vivado top schematic</summary>

<a href="fig/fiaccel_vivado_top.svg"><img src="fig/fiaccel_vivado_top.svg" width="900" alt="Vivado RTL schematic of the FIACCEL top-level wrapper"></a>

[Core integration schematic](fig/fiaccel_vivado_core.svg) · [View details](docs/architecture.md#vivado-rtl-schematics)

</details>

## Run

Requires **Vivado 2024.2 (Linux)** on `PATH` and **Python 3**.

```bash
make sim        # 32-bit AXI
make sim-wide   # 256-bit AXI
make sim-waves  # selected internal signals / WDB
```

Both check 256 RGB pixels using a synthetic 16×8, three-frame fixture.
Results: `build/public_xsim/validation.json` and `build/public_xsim_256/validation.json`.
This demo checks functionality; see the [RTL guide](hardware/README.md) for scope.

## Contents

| Path | Contents |
|---|---|
| `rtl_vp/` | Encrypted RTL and checksums |
| `hardware/` | AXI testbench, fixtures and simulator runner |
| `sw/` | U200 XRT host and video I/O |
| `reference_papers/` | Reference papers |
| `model/`, `model_pipeline.py` | Network, selective transfer and image synthesis |
| `config/`, `datamodules/`, `model_train.py` | Training |

<details>
<summary>Citation</summary>

```bibtex
@article{jeong2024fiaccel,
  title   = {FIACCEL: Memory Efficient Frame Interpolation Accelerator for Full-HD Video},
  author  = {Jeong, Min Wu and Rhee, Chae Eun},
  journal = {IEEE Transactions on Circuits and Systems II: Express Briefs},
  volume  = {71},
  number  = {4},
  pages   = {2289--2293},
  year    = {2024},
  doi     = {10.1109/TCSII.2023.3329966}
}
```

</details>
