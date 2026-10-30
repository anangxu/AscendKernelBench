import torch
import torch.nn as nn

CASE_ATTRS = {}


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *inputs, **attrs):
        merged = dict(CASE_ATTRS)
        merged.update(attrs)

        x = inputs[0]
        gamma = inputs[1]

        eps = merged.get("eps", merged.get("epsilon", None))
        if eps is None:
            if len(inputs) >= 3:
                eps = inputs[2]
            else:
                eps = 1e-6
        eps = float(eps)

        return torch.ops.custom_op.run(x, gamma, eps)
