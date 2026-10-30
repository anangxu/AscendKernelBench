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
    """torch.gather over the case dim implemented via the custom Ascend C op."""

    def __init__(self):
        super().__init__()

    def forward(self, x, index):
        return torch.ops.custom_op.run(x, index, _case_dim())
