"""Apply rotary position embedding, delegating to the official golden.

Inputs: query, key, cos, sin. Attributes: layout, rotaryMode. Outputs:
(query_out, key_out), compared as an output tree.
"""

import importlib.util
from pathlib import Path

import torch
import torch.nn as nn

GOLDEN_PATH = Path(__file__).resolve().parent / "official" / "golden.py"


def _golden_module():
    """Load the vendored official golden module."""
    spec = importlib.util.spec_from_file_location("golden_rope", GOLDEN_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Model(nn.Module):
    """Official rotary embedding reference."""

    def __init__(self):
        super().__init__()

    def forward(self, *inputs, **attrs):
        attrs = attrs or (globals().get("CASE_ATTRS") or {})
        fn = _golden_module().apply_rotary_pos_emb
        import inspect

        params = [
            p
            for p in inspect.signature(fn).parameters.values()
            if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
        ]
        bound, used = {}, 0
        for p in params:
            if p.name in attrs:
                bound[p.name] = attrs[p.name]
            elif used < len(inputs):
                bound[p.name] = inputs[used]
                used += 1
        return fn(**bound)


def get_inputs():
    """Return a default four-input set for the baseline baseline."""
    q = torch.randn(2, 16, 4, 64)
    k = torch.randn(2, 16, 4, 64)
    cos = torch.randn(16, 32)
    sin = torch.randn(16, 32)
    return [q, k, cos, sin]


def get_init_inputs():
    """Return no constructor arguments."""
    return []
