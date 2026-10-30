import torch
import torch.nn as nn

CASE_ATTRS = {}

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        attrs = CASE_ATTRS
        if attrs:
            dim = int(attrs.get("dim", -1))
        else:
            dim = -1
        return torch.ops.custom_op.run(x, dim)
