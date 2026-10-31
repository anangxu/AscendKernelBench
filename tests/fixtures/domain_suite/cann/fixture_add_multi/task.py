"""Elementwise add over two tensors, used only for the case-adapter acceptance."""

import torch
import torch.nn as nn


class Model(nn.Module):
    """Reference add: y = x + z elementwise, dtype preserved."""

    def __init__(self):
        super().__init__()

    def forward(self, x, z):
        return x + z


def get_inputs():
    """Return the default fp32 pair; cases override this."""
    return [torch.randn(16, 16), torch.randn(16, 16)]


def get_init_inputs():
    """Return no constructor arguments."""
    return []
