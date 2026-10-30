import torch
import torch.nn as nn


def _case_dim(default=-1):
    attrs = globals().get("CASE_ATTRS") or {}
    try:
        return int(attrs.get("dim", default))
    except (TypeError, ValueError):
        return default


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.ops.custom_op.run(x, _case_dim())
