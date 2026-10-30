"""Auto-converted CANN Bench task: lstm (L4)."""

import importlib.util
from pathlib import Path

import torch  # noqa: F401
import torch.nn as nn

GOLDEN_PATH = Path(__file__).resolve().parent / "official" / "golden.py"


def _golden_module():
    """Load the vendored official golden module once per process."""
    spec = importlib.util.spec_from_file_location("golden_lstm", GOLDEN_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _entry(module):
    """Return the official golden callable."""
    for candidate in ("lstm", "lstm", "golden", "run", "main"):
        fn = getattr(module, candidate, None)
        if callable(fn):
            return fn
    callables = [v for k, v in vars(module).items() if callable(v) and not k.startswith("_")]
    return callables[0] if callables else None


class Model(nn.Module):
    """Thin adapter onto the official golden implementation."""

    def __init__(self):
        super().__init__()

    def forward(self, *inputs, **attrs):
        attrs = attrs or (globals().get("CASE_ATTRS") or {})
        fn = _entry(_golden_module())
        if fn is None:
            raise RuntimeError("official golden has no callable entry")
        import inspect

        params = [
            p
            for p in inspect.signature(fn).parameters.values()
            if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
        ]
        # Tensors fill the leading parameters in order; attributes bind by name,
        # wherever they sit in the golden signature.
        bound, used = {}, 0
        for p in params:
            if p.name in attrs:
                # A declared attribute keeps its slot even when it sits between
                # tensor parameters, as dim does in scatter.
                bound[p.name] = attrs[p.name]
            elif used < len(inputs):
                bound[p.name] = inputs[used]
                used += 1
        return fn(**bound)


def get_inputs():
    """Default inputs; case evaluation supplies the official ones."""
    return [torch.randn(8, 16)]


def get_init_inputs():
    """Return no constructor arguments."""
    return []
