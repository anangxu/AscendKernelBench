import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *inputs, **attrs):
        x = inputs[0]
        gamma = inputs[1]
        eps = None
        if isinstance(attrs, dict) and ('eps' in attrs):
            eps = attrs['eps']
        if eps is None:
            ca = globals().get('CASE_ATTRS') or {}
            eps = ca.get('eps', 1e-6)
        return torch.ops.custom_op.run(x, gamma, float(eps))
