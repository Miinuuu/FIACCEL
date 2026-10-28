"""The RTL FMR contract: compare 96 MSBs and retain complete 32-byte vectors."""
import torch
from torch import nn


class SelectiveTransfer(nn.Module):
    """Persistent UINT8 feature representatives, grouped by 32 channels.

    All three upper bits must match in every channel to suppress a write.
    A suppressed write keeps the OLD complete vector, including its low bits.
    ``step`` returns that old representative for [old, current] concatenation;
    the updated representative is used by the NEXT frame. State is per stream,
    excluded from checkpoints, and must be reset between unrelated clips.
    """

    def __init__(self, channels_per_vector=32, enabled=True):
        super().__init__()
        if channels_per_vector < 1:
            raise ValueError('channels_per_vector must be positive')
        self.channels_per_vector = channels_per_vector
        self.enabled = enabled
        self.register_buffer('representative', None, persistent=False)
        self.register_buffer('last_write_mask', None, persistent=False)
        self.reset()

    def reset(self):
        self.representative = None
        self.last_write_mask = None
        self.reads = self.writes = self.skips = 0

    @property
    def stats(self):
        size = self.channels_per_vector
        return dict(read_vectors=self.reads, written_vectors=self.writes,
                    skipped_vectors=self.skips, read_bytes=self.reads * size,
                    written_bytes=self.writes * size)

    @torch.no_grad()
    def step(self, current):
        if current.dtype != torch.uint8 or current.ndim != 4:
            raise ValueError('FMR expects a UINT8 [B,C,H,W] feature tensor')
        b, c, h, w = current.shape
        if min(b, c, h, w) < 1 or c % self.channels_per_vector:
            raise ValueError('Feature channels must contain complete feature vectors')
        old = self.representative
        if old is not None and (old.shape != current.shape or old.device != current.device):
            raise ValueError('Stream feature shape/device changed; call reset() first')
        groups = c // self.channels_per_vector
        if old is None:
            changed = torch.ones((b, groups, h, w), dtype=torch.bool, device=current.device)
        else:
            different = (old >> 5) != (current >> 5)
            changed = different.reshape(b, groups, self.channels_per_vector, h, w).any(dim=2)
            if not self.enabled:
                changed = torch.ones_like(changed)
            self.reads += changed.numel()
        channel_mask = changed.repeat_interleave(self.channels_per_vector, dim=1)
        self.representative = (current.clone() if old is None else torch.where(channel_mask, current, old))
        self.last_write_mask = changed
        writes = int(changed.sum().item())
        self.writes += writes
        self.skips += changed.numel() - writes
        return old
