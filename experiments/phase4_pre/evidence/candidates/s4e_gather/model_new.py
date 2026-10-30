import torch
import torch.nn as nn

CASE_ATTRS = {}


class ModelNew(nn.Module):
    """torch.gather over the case dim; thin wrapper around the custom op."""

    def __init__(self):
        super().__init__()

    def forward(self, x, index):
        attrs = CASE_ATTRS
        dim = -1
        if attrs:
            dim = int(attrs.get("dim", -1))
        return torch.ops.custom_op.run(x, index, dim)
