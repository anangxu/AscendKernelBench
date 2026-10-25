import torch
import torch.nn as nn


class Model(nn.Module):
    """Row-wise sum of a contiguous fp32 tensor: out[r] = sum_c x[r, c]."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sum(x, dim=-1)


def get_inputs():
    return [torch.rand(512, 64)]


def get_init_inputs():
    return []
