"""SwiGLU activation with the official single-input half split.

Contract (restricted subset of the CANN 9.1.X swiglu family): one contiguous
fp32 tensor of shape [512, 2048], dim -1, split into two halves along the last
dimension, then y = silu(x[..., :1024]) * x[..., 1024:] with shape [512, 1024].
FP16/BF16, other dims, other shapes, clipping, bias, and quantization are not
part of this subset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

SPLIT_DIM = -1
SHAPE = (512, 2048)


class Model(nn.Module):
    """Reference SwiGLU: split the last dim in half, silu the first half."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        half = x.shape[-1] // 2
        gate = x.narrow(SPLIT_DIM, half, half)
        activated = x.narrow(SPLIT_DIM, 0, half)
        return F.silu(activated) * gate


def get_inputs():
    """Return the fixed-shape contiguous input."""
    return [torch.randn(*SHAPE)]


def get_init_inputs():
    """Return no constructor arguments."""
    return []
