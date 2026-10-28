"""RMSNorm over the last dimension, matching the official aclnnRmsNorm contract.

Contract (restricted subset): contiguous fp32 x of shape [512, 1024] and a fp32
gamma of shape [1024], epsilon 1e-6, last-dimension normalization. The official
op also returns rstd and accepts other dtypes and shapes; neither is part of
this subset. Semantics: y = x * rsqrt(mean(x^2) + eps) * gamma.
"""

import torch
import torch.nn as nn

EPS = 1e-6
SHAPE = (512, 1024)


class Model(nn.Module):
    """Reference RMSNorm: normalize the last dim, then scale by gamma."""

    def __init__(self):
        super().__init__()

    def forward(self, x, weight):
        rstd = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)
        return x * rstd * weight


def get_inputs():
    """Return the fixed-shape activation and scale."""
    return [torch.randn(*SHAPE), torch.randn(SHAPE[1])]


def get_init_inputs():
    """Return no constructor arguments."""
    return []
