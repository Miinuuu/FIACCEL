"""RTL-compatible INT8 QAT without normalization or channel-gain layers."""
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .flow_estimation import _FeatureStream
from model_pipeline import FIACCELPipeline

NAMES = ("enc.0", "enc.2", "enc.4", "aspp1.0", "aspp2.0", "aspp3.0",
         "aspp4.0", "shrink.0", "dec.0", "dec.2", "dec.4", "output.0")


class _ReplaceForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, exact):
        return exact

    @staticmethod
    def backward(ctx, gradient):
        return gradient, None


def integer_ste(value, operation="round"):
    exact = value.detach().round() if operation == "round" else value.detach().floor()
    return _ReplaceForward.apply(value, exact)


class _HardwareLayer:
    def _init_hardware(self, shift, head, signed_input):
        self.register_buffer("right_shift", torch.tensor(shift, dtype=torch.long))
        self.head, self.signed_input = head, signed_input
        # Six-bit shift is a fixed deployment parameter, not a trainable scale.
        self.shift = int(shift)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        self.shift = int(self.right_shift.item())

    def integer_weight(self):
        return integer_ste(self.weight * (2 ** self.shift)).clamp(-128, 127)

    def forward(self, x):
        weight = self.integer_weight()
        # Bound every partial integer sum, not only the final accumulator.
        # FP32 is exact within 2**24 with TF32 disabled; use FP64 otherwise.
        rows = weight.transpose(0, 1) if self.transposed else weight
        bound = rows.detach().abs().flatten(1).sum(1).max().item() * (128 if self.signed_input else 255)
        dtype = torch.float32 if bound < 2 ** 24 else torch.float64
        if self.transposed:
            accumulator = F.conv_transpose2d(x.to(dtype), weight.to(dtype), stride=self.stride,
                padding=self.padding, output_padding=self.output_padding, dilation=self.dilation)
        else:
            accumulator = F.conv2d(x.to(dtype), weight.to(dtype), stride=self.stride,
                                   padding=self.padding, dilation=self.dilation)
        preactivation = accumulator / (2 ** self.shift)
        codes = integer_ste(preactivation, "floor")
        return codes.clamp(-16384, 16384).float() if self.head else codes.clamp(0, 255).float()


class HardwareConv(_HardwareLayer, nn.Conv2d):
    def __init__(self, ci, co, k=3, stride=1, dilation=1, shift=8, head=False, signed_input=False):
        super().__init__(ci, co, k, stride=stride, padding=(0, 0) if k == 1 else (1, dilation),
                         dilation=(1, dilation), bias=False)
        self._init_hardware(shift, head, signed_input)


class HardwareTranspose(_HardwareLayer, nn.ConvTranspose2d):
    def __init__(self, ci, co, shift=8):
        super().__init__(ci, co, 3, stride=2, padding=1, output_padding=1, bias=False)
        self._init_hardware(shift, False, False)


class HardwareFlow(_FeatureStream):
    def __init__(self, initial_mapping=None, initial_state=None, selective_transfer=True):
        super().__init__()
        # The integer-sum certificate requires full FP32 multiplication.
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        # Direct GEMM/col2im preserves integer sums. FFT/Winograd transforms
        # can introduce fractions even when the input and weights are integers.
        torch.backends.cudnn.enabled = False
        torch.backends.mkldnn.enabled = False
        self._init_stream(1.0, selective_transfer)
        self.enc = nn.Sequential(HardwareConv(3, 8, stride=2, signed_input=True), nn.Identity(),
                                 HardwareConv(8, 16, stride=2), nn.Identity(),
                                 HardwareConv(16, 32, stride=2), nn.Identity())
        for i in range(1, 5):
            setattr(self, f"aspp{i}", nn.Sequential(HardwareConv(64, 64, dilation=i), nn.Identity()))
        self.shrink = nn.Sequential(HardwareConv(256, 64, k=1), nn.Identity())
        self.dec = nn.Sequential(HardwareTranspose(64, 32), nn.Identity(),
                                 HardwareTranspose(32, 16), nn.Identity(),
                                 HardwareTranspose(16, 8), nn.Identity())
        self.output = nn.Sequential(HardwareConv(8, 3, head=True), nn.Identity())
        if initial_mapping:
            self.load_mapping(json.loads(Path(initial_mapping).read_text()))
        if initial_state:
            self.load_state_dict(torch.load(initial_state, map_location="cpu", weights_only=True))

    @torch.no_grad()
    def load_mapping(self, document):
        for name in NAMES:
            layer, data = self.get_submodule(name), document["layers"][name]
            if list(layer.weight.shape) != data["shape"]:
                raise ValueError(f"{name}: incompatible weight shape")
            shift = data["shift"]
            if type(shift) is not int or not 0 <= shift <= 63:
                raise ValueError(f"{name}: invalid shift")
            weight = torch.tensor(data["values"], dtype=torch.float32).reshape(data["shape"])
            if not torch.isfinite(weight).all() or (weight < -128).any() or (weight > 127).any() or not torch.equal(weight, weight.round()):
                raise ValueError(f"{name}: expected signed INT8 weights")
            layer.weight.copy_(weight / (2 ** shift))
            layer.right_shift.fill_(shift)
            layer.shift = shift

    @staticmethod
    def input_codes(image):
        return integer_ste(image * 255).clamp(0, 255) - 128

    def _encode(self, image):
        return self.enc(self.input_codes(image))

    def bottleneck(self, joined):
        return self.shrink(torch.cat([getattr(self, f"aspp{i}")(joined) for i in range(1, 5)], 1))

    def _decode(self, previous, current):
        return self.output(self.dec(self.bottleneck(torch.cat((previous, current), 1)))) / 16384

    def forward(self, previous, current, timestep=0.5):
        if timestep != 0.5:
            raise ValueError("Only midpoint interpolation is supported")
        encoded = self._encode(torch.cat((previous, current), 0))
        a, b = encoded.split(previous.shape[0], 0)
        return self._decode(a, b)

    @torch.no_grad()
    def export_document(self):
        if self.training:
            raise RuntimeError("Export requires model.eval()")
        for name in NAMES:
            if not torch.isfinite(self.get_submodule(name).integer_weight()).all():
                raise ValueError(f"{name}: nonfinite weights")
        return dict(depths=[8, 16, 32, 64, 32, 16, 8], output_channels=3, layers={
            name: dict(shape=list(self.get_submodule(name).weight.shape),
                       values=self.get_submodule(name).integer_weight().to(torch.int8).cpu().flatten().tolist(),
                       shift=self.get_submodule(name).shift) for name in NAMES})


def integer_synthesis(previous, current, flow):
    """Exact RGB integer forward path; all inputs here are integer tensors."""
    n, c, h, w = previous.shape
    x = torch.arange(w, device=flow.device).reshape(1, 1, w) * 256
    y = torch.arange(h, device=flow.device).reshape(1, h, 1) * 256
    sx, sy = (flow[:, 0]*(w-1)) >> 8, (flow[:, 1]*(h-1)) >> 8

    def sample(rgb, xx, yy):
        xx, yy = xx.clamp(0, (w-1)*256), yy.clamp(0, (h-1)*256)
        x0, y0, xf, yf = xx >> 8, yy >> 8, xx & 255, yy & 255
        x1, y1 = (x0+1).clamp(max=w-1), (y0+1).clamp(max=h-1)
        pixels = rgb.reshape(n, c, h*w)

        def gather(a, b):
            return pixels.gather(2, (b*w+a).reshape(n, 1, h*w).expand(n, c, h*w)).reshape(n, c, h, w)
        return (gather(x0, y0)*((256-xf)*(256-yf)).unsqueeze(1) +
                gather(x1, y0)*(xf*(256-yf)).unsqueeze(1) +
                gather(x0, y1)*((256-xf)*yf).unsqueeze(1) +
                gather(x1, y1)*(xf*yf).unsqueeze(1))
    a, b = sample(previous, x-sx, y-sy), sample(current, x+sx, y+sy)
    alpha = (16384+flow[:, 2]).unsqueeze(1)
    return ((a*alpha+b*(32768-alpha)+(1 << 30)) >> 31).clamp(0, 255)


class HardwarePipeline(FIACCELPipeline):
    @staticmethod
    def synthesize(previous, current, out):
        h, w = previous.shape[-2:]
        out = out[:, :, :h, :w]
        # Use snapped Q8 coordinates in the differentiable bilinear surrogate.
        dx = integer_ste(out[:, 0]*(w-1)*64, "floor")
        dy = integer_ste(out[:, 1]*(h-1)*64, "floor")
        snapped = torch.stack((dx/(64*max(w-1, 1)), dy/(64*max(h-1, 1)), out[:, 2]), 1)
        surrogate = FIACCELPipeline.synthesize(previous, current, snapped)
        with torch.no_grad():
            a = (previous*255).round().clamp(0, 255).long()
            b = (current*255).round().clamp(0, 255).long()
            flow = (out*16384).round().clamp(-16384, 16384).long()
            exact = integer_synthesis(a, b, flow).float()/255
        return _ReplaceForward.apply(surrogate, exact)
