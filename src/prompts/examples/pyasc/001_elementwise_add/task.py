import torch
import torch.nn as nn


class Model(nn.Module):
    """Element-wise addition of two fp32 tensors: C = A + B."""

    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return A + B


def get_inputs():
    A = torch.rand(16, 4096)
    B = torch.rand(16, 4096)
    return [A, B]


def get_init_inputs():
    return []
