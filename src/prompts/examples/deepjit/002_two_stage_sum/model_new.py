import torch
import torch.nn as nn


class ModelNew(nn.Module):
    """Thin wrapper around the compiled custom_op reduction kernel."""

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.custom_op.run(x)
