import torch
import torch.nn as nn


class ModelNew(nn.Module):
    """Thin wrapper; the same code serves every case."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return torch.ops.custom_op.run(x, z)
