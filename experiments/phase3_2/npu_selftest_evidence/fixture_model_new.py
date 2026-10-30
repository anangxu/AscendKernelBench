import torch
import torch.nn as nn


class ModelNew(nn.Module):
    """Thin wrapper; the fixture kernel reads the case dim from the shape."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.custom_op.run(x)
