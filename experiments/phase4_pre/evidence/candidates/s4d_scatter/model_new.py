import torch
import torch.nn as nn

CASE_ATTRS = {}


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *inputs, **attrs):
        x = inputs[0]
        gamma = inputs[1]
        eps = attrs.get("eps", None)
        if eps is None and len(inputs) > 2:
            eps = inputs[2]
        if eps is None:
            eps = CASE_ATTRS.get("eps", 1e-6)
        return torch.ops.custom_op.rms_norm(x, gamma, float(eps))
