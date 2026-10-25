"""Unaligned fp32 addition exercises padding in the shared pyasc example."""

from __future__ import annotations

import torch
import torch.nn as nn


class Model(nn.Module):
    """Add two tensors whose length is not a multiple of the copy alignment."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b


def get_inputs() -> list[torch.Tensor]:
    return [torch.rand(17, dtype=torch.float32), torch.rand(17, dtype=torch.float32)]


def get_init_inputs() -> list[object]:
    return []
