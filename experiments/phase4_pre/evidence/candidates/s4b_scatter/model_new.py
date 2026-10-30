import torch
import torch.nn as nn

CASE_ATTRS = {}


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *inputs, **attrs):
        attrs = attrs or CASE_ATTRS

        x = attrs.get("x")
        if x is None:
            x = attrs.get("input")
        if x is None:
            x = inputs[0]

        gamma = attrs.get("gamma")
        if gamma is None:
            gamma = attrs.get("weight")
        if gamma is None:
            gamma = inputs[1]

        eps = attrs.get("eps")
        if eps is None:
            eps = attrs.get("epsilon", 1e-6)

        return torch.ops.custom_op.run(x, gamma, float(eps))
