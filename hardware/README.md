# Protected RTL evaluation

Evaluate FIACCEL through its AXI interfaces without distributing the RTL source.
The 29 implementation modules are distributed as IEEE 1735-2014 V2 encrypted
Verilog, with **simulation-only** access for AMD Vivado XSim.

## Run the demo

The runner uses `xelab --O0`. On this RTL, Vivado 2024.2 default `--O2`
produces X-valued control/data signals; the same plaintext test passes with
`--O0`. This workaround changes simulator optimization, not RTL arithmetic.

Tested toolchain: **Vivado 2024.2 on Linux**, plus Python 3 (standard library only).
Source your installation's `settings64.sh`, then run from the repository root:

```bash
python3 hardware/scripts/run_xsim.py
```

The command verifies encrypted-file checksums, compiles the protected implementation
and public testbench, and runs a deterministic 16 × 8, three-frame fixture with
synthetic coefficients and reduced channel counts. No GPU, FPGA board, dataset,
trained checkpoint, or encryption-authoring license is needed to run this demo.
The simulator must support the included AMD encryption key and rights.

On success, `build/public_xsim/validation.json` reports `PASS` and 256 checked RGB
pixels. The per-step logs and `rtl_output.hex` are stored in the same directory.
To exercise another AXI master data width:

```bash
python3 hardware/scripts/run_xsim.py --data-width 256 --output build/public_xsim_256
```

## Observe signals

```bash
make sim-waves
xsim build/public_xsim_waves/waves.wdb -gui
```

`make sim-waves` records `waves.wdb` and verifies a read of an internal encoder
coordinate exported through the simulation debug interface. Inspect the public
testbench's `debug_encoder_x`, `debug_encoder_y`, `debug_flow`, `debug_flow_valid`,
`debug_flow_ready`, `debug_cycles` and `debug_state` signals. The runner defines
`FIACCEL_SIM_DEBUG` and enables XSim debugging.

Vivado 2024.2 hides encrypted internal scopes even when schematic/probing rights
are enabled; its simulator rejects `runtime_visibility="all_names"`. The debug
interface exposes selected internal values without providing plaintext RTL.
Schematic visibility is enabled where supported by the tool, but a full Vivado
RTL schematic requires elaboration that the synthesis restriction can block.
The architecture diagrams remain available in the [architecture guide](../docs/architecture.md).

## What is checked

- AXI-Lite configuration, completion, error and counter register readback.
- All interpolated RGB pixels against the supplied golden fixture.
- Persistent feature representatives in external memory and selective-transfer counts.
- Coefficient reuse across frame commands and cache hit/miss accounting.
- Three AXI memory interfaces with a randomized public memory model.

The testbench observes external ports and public registers. The optional debug interface exposes selected internal signals. The included golden fixture is precomputed; this demo does
not independently regenerate it. It does not validate Full-HD throughput,
timing closure, paper PSNR, or internal flow values.

## Distribution boundaries

| Artifact or operation | Included / allowed |
|---|---|
| Encrypted implementation (`../rtl_vp/*.vp`) | Yes |
| Testbench, memory model, synthetic fixture | Readable |
| XSim behavioral simulation | Allowed |
| Simulation debug ports and WDB recording | Allowed |
| Synthesis and implementation | Disabled by encryption rights |
| Netlist export and bitstream generation | Disabled by encryption rights |
| RTL source, utilization reports, DCP, bitstream, license files | Not included |
| Verilator / Icarus execution of these encrypted files | Not supported by this package |

The published paper describes an Alveo U250 design. This evaluation package is
derived from the current U200 development RTL and is provided as a functional
evaluation, without publishing synthesis utilization. Paper figures and published
measurements remain visible in the research documentation.

Encryption restricts supported tool operations; it does not guarantee that all
implementation properties are impossible to infer from interfaces or behavior.

## Files

```text
rtl_vp/             # 29 protected Verilog modules, source list and SHA-256 manifest
hardware/
├── fixtures/smoke/  # deterministic inputs and expected results
├── scripts/         # XSim command-line runner
└── tb/              # readable interface-only verification
```

AMD documents the format and tool rights in
[UG1118](https://docs.amd.com/r/2024.2-English/ug1118-vivado-creating-packaging-custom-ip/Encrypting-IP-in-Vivado).
Encryption is a technical access restriction and does not itself grant a license
to reuse or redistribute the design.
