# Protected RTL hierarchy

[`files.f`](files.f) lists sources relative to the repository root;
[`manifest.json`](manifest.json) records their SHA-256 checksums.
Run `make sim` from the root. See the [simulation guide](../hardware/README.md).

| Path | Role |
|---|---|
| `FIACCEL.vp`, `core/` | Kernel wrapper and engine integration |
| `axi/` | AXI-Lite control, memory masters, DMA and response buffering |
| `network/` | Encoder, HDC, decoder and FMR selective transfer |
| `memory/` | Feature/weight banks, loaders and sample cache |
| `pe/` | MAC array, DSP arithmetic, reduction and quantization |
| `fifo/` | Vector FIFO, skid buffer and SLR credit transport |
| `synthesis/` | Flow geometry, frame sampling and RGB blending |
