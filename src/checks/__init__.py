"""Static anti-cheat checks for generated samples.

check_model_new checks the Python wrapper, check_custom_op_asc the Ascend C
source, check_pyasc_sources the pyasc pair; each returns violations. nn
layers may hold parameters, never compute.
"""

from .ascend_c import check_custom_op_asc
from .pyasc import (
    check_kernel_source,
    check_pyasc_model_new,
    check_pyasc_sources,
)
from .python_source import check_model_new

__all__ = [
    "check_custom_op_asc",
    "check_kernel_source",
    "check_model_new",
    "check_pyasc_model_new",
    "check_pyasc_sources",
]
