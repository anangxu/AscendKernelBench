import torch
import torch.nn as nn


class ModelNew(nn.Module):
    """Thin wrapper around the evaluation-loaded torch.ops.custom_op operator."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.ops.custom_op.run(x, weight)
