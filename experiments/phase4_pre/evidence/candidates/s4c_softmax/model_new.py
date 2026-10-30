import torch

CASE_ATTRS = {}

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        dim = CASE_ATTRS.get("dim", -1)
        return torch.ops.custom_op.run(x, dim)
