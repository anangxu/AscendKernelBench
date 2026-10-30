"""SwiGLU with the official single-input half split (Ascend C kernel wrapper)."""

import torch
import torch.nn as nn


def _case_dim(default: int = -1) -> int:
    """Return the per-case dim attribute, defaulting when absent."""
    attrs = globals().get("CASE_ATTRS") or {}
    try:
        return int(attrs.get("dim", default))
    except (TypeError, ValueError):
        return default


class ModelNew(nn.Module):
    """Official SwiGLU: split along dim, silu the first half, multiply."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        dim = _case_dim()
        return torch.ops.custom_op.run(x, dim)
