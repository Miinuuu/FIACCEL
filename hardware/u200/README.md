# U200 board demo

[Project overview](../../README.md) · [Demo MP4](../../media/fiaccel_u200_demo.mp4) · [XSim evaluation](../README.md)

<img src="../../fig/alveo_u200.jpg" width="420" alt="AMD Alveo U200 accelerator card, manufacturer product image">

AMD Alveo U200 product image, © AMD. [Image source](https://www.amd.com/en/products/accelerators/alveo/u200/a-u200-p64g-pq-g.html).

## Validated on hardware

| Item | Result |
|---|---|
| Tools / platform | Vivado and Vitis 2024.2 / `xilinx_u200_gen3x16_xdma_2_202110_1` |
| Hardware build | Completed; 300 MHz kernel clock |
| Model | 10-epoch trained INT8 weights; 12 shift scales |
| Exact reference check | 3 input frames → 6 output frames; all 37,324,800 RGB bytes match |
| MP4 pipeline | 1920×1080, 30 fps / 60 frames → 60 fps / 120 frames |

The reference check models selective transfer with persistent feature
representatives. The video pipeline runs FFmpeg decode → XRT host → U200
→ FFmpeg encode. Output order is original A → midpoint A/B → original B;
the final original is repeated once to preserve duration.
FPS above describes the encoded video. [Validation summary](validation.json).

## Build the host and run a video

Requires Linux, a C++17 compiler, XRT development headers/libraries,
FFmpeg/FFprobe and a compatible FIACCEL `.xclbin` supplied separately.
Use an idle U200 and run from the repository root:

```bash
make -C sw host
python3 sw/video/interpolate_video.py input_1080p.mp4 output_1080p60.mp4 \
  --xclbin /path/to/FIACCEL.xclbin \
  --host build/u200/host --weights weights/u200_int8/config.hex
```

Frames pass through bounded pipes and recycled device buffers. The command
writes frame counts and execution metadata beside the output video.
The default fixed-pattern host self-test loads synthetic coefficients;
the video command above loads the published trained weights.

## Hardware build configuration

The completed development build used three 32-bit AXI masters, bursts up to
256 beats and up to eight outstanding requests. [`link.cfg`](link.cfg) maps
M00/M02 to DDR0 and M01 to DDR1, requests 300 MHz, and selects SLR balancing,
congestion-aware placement and physical optimization. Vitis frequency scaling
was enabled; the completed design retained 300 MHz.

For a development environment with an independently supplied FIACCEL kernel XO:

```bash
mkdir -p build/u200
v++ -t hw \
  --platform xilinx_u200_gen3x16_xdma_2_202110_1 \
  --config hardware/u200/link.cfg \
  -l /path/to/FIACCEL.xo -o build/u200/FIACCEL.xclbin
```

The encrypted RTL in this repository permits XSim evaluation and signal
inspection. It does not permit synthesis or bitstream generation. The command
above requires a separately authorized kernel XO; this release supplies host
sources and integration settings.
