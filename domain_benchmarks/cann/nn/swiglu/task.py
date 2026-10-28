"""SwiGLU activation over an explicit pair: silu(x) * gate.

Semantics follow the CANN swiglu family (activateLeft=true, no clipping and no
quantization): the left operand is activated with silu and multiplied by the
right operand elementwise. The official op splits one tensor along dim and
takes halves; this task passes the two halves explicitly so the KernelBench
contract stays two inputs and one output.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """Reference SwiGLU: y = silu(x) * gate, elementwise, same shape."""

    def __init__(self):
        super().__init__()

    def forward(self, x, gate):
        return F.silu(x) * gate


def get_inputs():
    """Return the fixed-shape operand pair."""
    return [torch.randn(512, 1024), torch.randn(512, 1024)]


def get_init_inputs():
    """Return no constructor arguments."""
    return []
