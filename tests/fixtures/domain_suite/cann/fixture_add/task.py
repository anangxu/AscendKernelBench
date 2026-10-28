"""Elementwise add, used only to exercise the CANN suite offline."""

import torch
import torch.nn as nn


class Model(nn.Module):
    """Reference add; a candidate must match it within tolerance."""

    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return a + b


def get_inputs():
    """Return the fixed-shape operand pair."""
    return [torch.rand(16, 16), torch.rand(16, 16)]


def get_init_inputs():
    """Return no constructor arguments."""
    return []
