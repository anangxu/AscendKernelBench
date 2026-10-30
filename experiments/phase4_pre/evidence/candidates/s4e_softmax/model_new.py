import torch
import torch.nn as nn

CASE_ATTRS = {"dim": -1}


class ModelNew(nn.Module):
    """Softmax over the per-case dim attribute."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        dim = int(CASE_ATTRS.get("dim", -1))
        return torch.ops.custom_op.run(x, dim)
