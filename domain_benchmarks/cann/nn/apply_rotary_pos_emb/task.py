"""Apply rotary position embedding with the official four-input contract.

Inputs: query, key, cos, sin. Attributes: layout and rotaryMode. Outputs:
(query_out, key_out), which the evaluator compares as an output tree.
"""

import torch
import torch.nn as nn


def _attr(name, default):
    """Return a case attribute, defaulting when absent."""
    attrs = globals().get("CASE_ATTRS") or {}
    return attrs.get(name, default)


def _rotate(x, cos, sin, layout):
    """Apply the rotary transform along the last dimension."""
    half = x.shape[-1] // 2
    if layout == "BSND" or layout == "BNSD":
        first, second = x[..., :half], x[..., half:]
    else:
        first, second = x[..., 0::2], x[..., 1::2]
    c = cos[..., :half] if cos.shape[-1] >= half else cos
    s = sin[..., :half] if sin.shape[-1] >= half else sin
    if layout in ("BSND", "BNSD"):
        return torch.cat([first * c - second * s, first * s + second * c], dim=-1)
    out = torch.empty_like(x)
    out[..., 0::2] = first * c - second * s
    out[..., 1::2] = first * s + second * c
    return out


class Model(nn.Module):
    """Reference rotary embedding; returns the two rotated tensors."""

    def __init__(self):
        super().__init__()

    def forward(self, query, key, cos, sin):
        layout = _attr("layout", "BSND")
        return _rotate(query, cos, sin, layout), _rotate(key, cos, sin, layout)


def get_inputs():
    """Return a default four-input set."""
    q = torch.randn(2, 8, 16, 64)
    k = torch.randn(2, 8, 16, 64)
    cos = torch.randn(1, 1, 16, 32)
    sin = torch.randn(1, 1, 16, 32)
    return [q, k, cos, sin]


def get_init_inputs():
    """Return no constructor arguments."""
    return []
