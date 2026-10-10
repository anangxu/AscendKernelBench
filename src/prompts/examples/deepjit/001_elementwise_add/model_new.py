import torch
import torch.nn as nn


class ModelNew(nn.Module):
    """Thin wrapper around the evaluator-loaded torch.ops.custom_op operator."""

    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return torch.ops.custom_op.run(A, B)
