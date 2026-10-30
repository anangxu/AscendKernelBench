"""Auto-converted CANN Bench task: mla_prolog (L4)."""

import importlib.util
from pathlib import Path

import torch  # noqa: F401
import torch.nn as nn

GOLDEN_PATH = Path(__file__).resolve().parent / "official" / "golden.py"


def _golden_module():
    """Load the vendored official golden module once per process."""
    spec = importlib.util.spec_from_file_location("golden_mla_prolog", GOLDEN_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _entry(module):
    """Return the official golden callable."""
    for candidate in ("mla_prolog", "mla_prolog", "golden", "run", "main"):
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
        try:
            return fn(*inputs, **attrs)
        except TypeError:
            # Official goldens take the case attributes positionally, in their
            # own signature order, so bind them the same way here.
            import inspect

            names = [p.name for p in inspect.signature(fn).parameters.values()]
            ordered = [attrs[name] for name in names[len(inputs):] if name in attrs]
            return fn(*inputs, *ordered)


def get_inputs():
    """Default inputs; case evaluation supplies the official ones."""
    return [torch.randn(8, 16)]


def get_init_inputs():
    """Return no constructor arguments."""
    return []
