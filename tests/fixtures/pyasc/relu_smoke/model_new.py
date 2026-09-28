import torch

import kernel


class ModelNew(torch.nn.Module):
    """Thin wrapper: all tensor compute happens in kernel.py's @asc.jit kernel."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return kernel.relu_launch(x)
