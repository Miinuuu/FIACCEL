# Model

| Module | Role |
|---|---|
| `flow_estimation.py` | Encoder, HDC bottleneck, decoder and three-channel flow/mask head |
| `selective_transfer.py` | Persistent feature representatives and selective transfer |
| `pipeline.py` | Midpoint synthesis and streaming frame state |
| `hardware_qat.py` | Shift-only INT8 hardware-compatible training and export |
| `loss.py` | Training loss functions |

```python
from model.flow_estimation import flow_estimation
from model.pipeline import FIACCELPipeline

pipeline = FIACCELPipeline(flow_estimation()).eval()
```

See the [training guide](../docs/training.md) for checkpoints and configuration,
and [`weights/u200_int8/`](../weights/u200_int8/) for the deployed model.
