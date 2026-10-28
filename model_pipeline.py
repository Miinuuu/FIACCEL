"""Three-channel midpoint flow/mask synthesis and explicit video-stream state."""
import torch
from torch import nn
from torch.nn import functional as F


def _grid(height, width, reference):
    x = torch.linspace(-1, 1, width, device=reference.device, dtype=reference.dtype)
    y = torch.linspace(-1, 1, height, device=reference.device, dtype=reference.dtype)
    gy, gx = torch.meshgrid(y, x, indexing='ij')
    return torch.stack((gx, gy), dim=-1).unsqueeze(0)


def warp(image, pixel_flow):
    """Compatibility helper for a displacement measured in pixels."""
    height, width = image.shape[-2:]
    denominator = image.new_tensor((max(width-1, 1)/2, max(height-1, 1)/2))
    coordinates = _grid(height, width, image) + pixel_flow.permute(0, 2, 3, 1) / denominator
    return F.grid_sample(image, coordinates, mode='bilinear', padding_mode='border', align_corners=True)


class FIACCELPipeline(nn.Module):
    """[flow_x, flow_y, mask] HardTanh output, matching the RTL synthesizer."""

    def __init__(self, flow_estimation):
        super().__init__()
        self.flow_estimation = flow_estimation
        self.register_buffer('_previous_frame', None, persistent=False)

    @staticmethod
    def meshgrid(height, width, device=None, dtype=None):
        reference = torch.empty((), device=device, dtype=dtype)
        grid = _grid(height, width, reference)
        return grid[..., 0], grid[..., 1]

    @staticmethod
    def synthesize(previous, current, out):
        if previous.shape != current.shape or previous.ndim != 4 or previous.shape[1] != 3:
            raise ValueError('Expected equally shaped RGB [B,3,H,W] frames')
        height, width = previous.shape[-2:]
        if (out.ndim != 4 or out.shape[1] != 3 or out.shape[0] != previous.shape[0]
                or out.shape[-2] < height or out.shape[-1] < width):
            raise ValueError('Expected three flow_x/flow_y/mask channels at input resolution')
        # Stride-2 layers round odd dimensions up; retain the original image.
        out = out[:, :, :height, :width]
        half_flow = 0.5 * out[:, :2].permute(0, 2, 3, 1)
        grid = _grid(height, width, previous)
        warped_previous = F.grid_sample(previous, grid-half_flow, mode='bilinear',
                                        padding_mode='border', align_corners=True)
        warped_current = F.grid_sample(current, grid+half_flow, mode='bilinear',
                                       padding_mode='border', align_corners=True)
        alpha = (1.0 + out[:, 2:3]) * 0.5
        return (warped_previous*alpha + warped_current*(1.0-alpha)).clamp(0, 1)

    def interpolate(self, previous, current):
        """Stateless pair path for training and independent clips."""
        return self.synthesize(previous, current, self.flow_estimation(previous, current))

    def forward(self, video):
        images = video.video_tensor
        return self.interpolate(images[:, 0], images[:, 2])

    def reset_stream(self):
        self.flow_estimation.reset_stream()
        self._previous_frame = None

    @torch.no_grad()
    def forward_frame(self, frame, timestep=0.5):
        """Warm one frame, then synthesize each adjacent frame pair using FMR."""
        out = self.flow_estimation.forward_frame(frame, timestep=timestep)
        previous = self._previous_frame
        self._previous_frame = frame.detach().clone()
        if out is None:
            return None
        return self.synthesize(previous, frame, out)

    @torch.no_grad()
    def inference(self, video, timestep=0.5):
        # Independent triplets must never inherit a different clip's cache.
        self.reset_stream()
        images = video.video_tensor
        self.forward_frame(images[:, 0], timestep=timestep)
        return self.forward_frame(images[:, 2], timestep=timestep)

    def train(self, mode=True):
        if mode and hasattr(self, '_previous_frame'):
            self._previous_frame = None
        return super().train(mode)
