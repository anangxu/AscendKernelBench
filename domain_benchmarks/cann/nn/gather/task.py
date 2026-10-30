"""Gather along the per-case dim attribute with integer indices."""

import torch
import torch.nn as nn

def _case_dim(default: int = -1) -> int:
    """Return the per-case dim attribute, defaulting when absent."""
    attrs = globals().get("CASE_ATTRS") or {}
    try:
        return int(attrs.get("dim", default))
    except (TypeError, ValueError):
        return default


class Model(nn.Module):
    """torch.gather over the case dim; the output follows the index shape."""

    def __init__(self):
        super().__init__()

    def forward(self, x, index):
        return torch.gather(x, _case_dim(), index)


def get_inputs():
    """Return a default input and an integer index tensor."""
    x = torch.randn(64, 128)
    index = torch.randint(0, 128, (64, 16), dtype=torch.int64)
    return [x, index]


def get_init_inputs():
    """Return no constructor arguments."""
    return []
