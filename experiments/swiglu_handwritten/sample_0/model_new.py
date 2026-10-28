import torch
import torch.nn as nn

HALF = 1024


class ModelNew(nn.Module):
    """Thin wrapper around the evaluation-loaded torch.ops.custom_op operator."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.custom_op.run(x)
