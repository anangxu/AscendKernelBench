import torch
import torch.nn as nn


class Model(nn.Module):
    """Sum of all elements of a contiguous fp32 tensor."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sum(x)


def get_inputs():
    # Zero-mean inputs: both signs, so a wrong sign or a dropped term shows up.
    return [torch.randn(1, 100003)]


def get_init_inputs():
    return []
