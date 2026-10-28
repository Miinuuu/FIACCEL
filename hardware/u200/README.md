# U200 host software

[Project overview](../../README.md) · [XSim evaluation](../README.md)

The public `sw/` directory contains the Linux XRT C++ host, integer reference,
RGB24 streaming and FFmpeg video front end. `kernel.xml` and `link.cfg` describe
the expected FIACCEL kernel interface and U200 memory-bank connections.

## Build and run

Requires XRT development headers/libraries, a C++17 compiler, and a U200 with
a compatible FIACCEL `.xclbin` supplied separately. Run from the repository root:

```bash
make -C sw host
build/u200/host /path/to/FIACCEL.xclbin output.ppm
```

The fixed-pattern self-test loads deterministic coefficients and checks two
Full-HD RGB outputs and selective-transfer accounting. It reads the hardware
command counter at `0x44` and enforces a 6,220,800-clock frame limit. This source
release does not assert that a physical board run has passed.

## Video

With FFmpeg/FFprobe installed, process a 1920×1080 video:

```bash
python3 sw/video/interpolate_video.py input.mp4 output.mp4 \
  --xclbin /path/to/FIACCEL.xclbin --random-weights --seed 20261028
```

The host inserts one midpoint between consecutive source frames and doubles
FPS. The last source frame is held once to preserve duration. Random INT8
weights test the execution/data path; learned image quality requires trained
coefficients via `--weights config.hex`. Optional `--enforce-cycle-budget`
checks each interpolation command. Frames use bounded pipes and recycled
device buffers; the complete video is not buffered.

The protected RTL package remains simulation-only. These host tools operate
an independently supplied device binary; they do not enable synthesis or
bitstream generation from the encrypted files.
