# Model deployment bundles

For direct video inference, use the [U200 guide](../../hardware/u200/README.md)
and the published `weights/u200_int8/config.hex`.

`u200_model.py` packages an existing integer-model export, compiled host and
Full-HD RGB24 reference fixture into a checksum-verified bundle:

```bash
python3 sw/deploy/u200_model.py pack \
  --model weights/u200_int8 --host build/u200/host \
  --fixture /path/to/reference_fixture --output build/model_bundle
python3 sw/deploy/u200_model.py check build/model_bundle
python3 sw/deploy/u200_model.py self-test build/model_bundle \
  --xclbin hardware/u200/prebuilt/FIACCEL.xclbin
```

The fixture must provide `input.rgb`, `expected.rgb` and `manifest.json`,
with at least two Full-HD input frames and exact integer reference output.
Reference generation must retain selective-transfer feature representatives
across frames. Model-specific RGB fixtures are supplied separately; the U200
image is in `hardware/u200/prebuilt/`. The self-test checks device availability before programming and
compares every output RGB byte. It requires no training framework.
