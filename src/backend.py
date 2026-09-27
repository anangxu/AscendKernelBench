"""Backend selection: which authoring language a run uses.

A run is written in exactly one backend. Ascend C stays the default so every
existing run, command, and result keeps its meaning; pyasc is opt-in per run.
"""

from __future__ import annotations

from enum import Enum

###########################################################################
# BACKEND REGISTRY
###########################################################################


class Backend(str, Enum):
    """Authoring language of one generation or evaluation run."""

    ASCENDC = "ascendc"
    PYASC = "pyasc"


DEFAULT_BACKEND = Backend.ASCENDC

# Kernel artifact per backend, relative to one sample directory. The wrapper
# name is shared: both backends call the compiled kernel from model_new.py.
KERNEL_FILE = {
    Backend.ASCENDC: "custom_op.asc",
    Backend.PYASC: "kernel.py",
}
WRAPPER_FILE = "model_new.py"

# Structured-output field names per backend. Ascend C keeps its historical
# name so archived responses and prompts stay valid; pyasc uses kernel_py.
KERNEL_FIELD = {
    Backend.ASCENDC: "custom_op_asc",
    Backend.PYASC: "kernel_py",
}
WRAPPER_FIELD = "model_new_py"

###########################################################################
# BACKEND REGISTRY
###########################################################################


def parse_backend(value: object) -> Backend:
    """Resolve a backend value, defaulting when absent.

    Args:
        value: A Backend member, a backend name, or None/empty for the default.

    Raises:
        ValueError: If the value is not a known backend.
    """
    if isinstance(value, Backend):
        return value
    if value is None:
        return DEFAULT_BACKEND
    text = str(value).strip().lower()
    if not text:
        return DEFAULT_BACKEND
    try:
        return Backend(text)
    except ValueError:
        supported = ", ".join(backend.value for backend in Backend)
        raise ValueError(
            f"Unknown backend {value!r}; supported backends are: {supported}"
        ) from None


def kernel_file(backend: Backend) -> str:
    """Return the kernel artifact filename for a backend."""
    return KERNEL_FILE[backend]


def kernel_field(backend: Backend) -> str:
    """Return the structured-output field name carrying the kernel source."""
    return KERNEL_FIELD[backend]


def sample_files(backend: Backend) -> tuple[str, str]:
    """Return the two artifact filenames that make a sample complete."""
    return kernel_file(backend), WRAPPER_FILE
