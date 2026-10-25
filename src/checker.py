"""Static anti-cheat checks; the policy lives in checks/."""

from __future__ import annotations

from .backend import Backend, parse_backend
from .checks import (
    check_custom_op_asc,
    check_model_new,
    check_pyasc_sources,
)

__all__ = [
    "check_custom_op_asc",
    "check_model_new",
    "check_pyasc_sources",
    "check_sample_sources",
]


def check_sample_sources(
    backend: Backend, kernel_source: str, wrapper_source: str
) -> list[str]:
    """Return static-check violations for one sample's two source files.

    Args:
        backend: Authoring backend of the sample; a backend name is
            accepted so an older caller can keep passing a string.
        kernel_source: Raw custom_op.asc for Ascend C, kernel.py for pyasc.
        wrapper_source: Raw model_new.py text, shared by both backends.

    Returns:
        Deduplicated human-readable violations; empty means pass.

    Raises:
        ValueError: If backend is not a known backend.
    """
    if parse_backend(backend) is Backend.PYASC:
        return check_pyasc_sources(kernel_source, wrapper_source)
    return check_custom_op_asc(kernel_source) + check_model_new(wrapper_source)
