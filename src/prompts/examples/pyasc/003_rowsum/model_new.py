import kernel
import torch


class ModelNew(torch.nn.Module):
    """Thin wrapper: the reduction runs in kernel.py's @asc.jit kernel."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return kernel.rowsum_launch(x)
