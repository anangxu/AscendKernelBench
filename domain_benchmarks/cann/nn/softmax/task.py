"""Softmax along the per-case dim attribute."""

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
    """Softmax over the case dim."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.softmax(x, dim=_case_dim())


def get_inputs():
    """Return the default single input."""
    return [torch.randn(32, 1024)]


def get_init_inputs():
    """Return no constructor arguments."""
    return []
