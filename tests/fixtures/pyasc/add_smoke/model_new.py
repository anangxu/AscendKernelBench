import kernel
import torch


class ModelNew(torch.nn.Module):
    """Thin wrapper: all tensor compute happens in kernel.py's @asc.jit kernel."""

    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return kernel.add_launch(A, B)
