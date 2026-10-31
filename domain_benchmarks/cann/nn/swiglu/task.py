"""SwiGLU with the official single-input half split.

Official semantics (CANN Bench): split the single input into two equal halves
along dim, then y = silu(x0) * x1. The size along dim must be even.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def _case_dim(default: int = -1) -> int:
    """Return the per-case dim attribute, defaulting when absent."""
    attrs = globals().get("CASE_ATTRS") or {}
    try:
        return int(attrs.get("dim", default))
    except (TypeError, ValueError):
        return default


class Model(nn.Module):
    """Official SwiGLU: split along dim, silu the first half, multiply."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        dim = _case_dim()
        half = x.shape[dim] // 2
        return F.silu(x.narrow(dim, 0, half)) * x.narrow(dim, half, half)


def get_inputs():
    """Return the default single input."""
    return [torch.randn(512, 2048)]


def get_init_inputs():
    """Return no constructor arguments."""
    return []
