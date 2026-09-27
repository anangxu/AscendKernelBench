"""pyasc backend runtime: cache isolation, module loading, first-call facts.

asc is imported only inside the evaluation worker, so prompt building,
generation, and offline analysis keep working on a machine without pyasc or
an NPU.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

__all__ = [
    "PyAscError",
    "PyAscJITError",
    "PyAscLoadError",
    "PyAscUnavailableError",
    "cache_dir_for",
    "configure_platform",
    "import_asc",
    "load_kernel_module",
    "prepare_cache_dir",
    "runtime_facts",
    "time_first_call",
]

CACHE_DIR_NAME = "pyasc_cache"
# model_new.py imports the kernel under this name; the loader registers the
# real file-backed module under it so the import resolves.
KERNEL_MODULE_NAME = "kernel"


class PyAscError(RuntimeError):
    """Base class for pyasc backend failures."""


class PyAscUnavailableError(PyAscError):
    """pyasc is not installed in the worker environment."""


class PyAscLoadError(PyAscError):
    """kernel.py could not be imported as a real module."""


class PyAscJITError(PyAscError):
    """The first kernel call failed: trace, compile, or launch error."""


###########################################################################
# JIT CACHE ISOLATION
###########################################################################


def cache_dir_for(sample_dir: str | Path) -> Path:
    """Return the sample-local pyasc JIT cache directory."""
    return Path(sample_dir) / "build" / CACHE_DIR_NAME


def prepare_cache_dir(sample_dir: str | Path) -> Path:
    """Create and export the sample-local JIT cache directory.

    pyasc resolves PYASC_CACHE_DIR when it is imported, so this must run
    before import_asc. One directory per sample keeps cached kernels from
    bleeding between samples that reuse kernel function names.
    """
    path = cache_dir_for(sample_dir)
    path.mkdir(parents=True, exist_ok=True)
    os.environ["PYASC_CACHE_DIR"] = str(path)
    return path


###########################################################################
# RUNTIME SETUP
###########################################################################


def import_asc() -> ModuleType:
    """Import the pyasc runtime module.

    Raises:
        PyAscUnavailableError: If pyasc is absent from the environment.
    """
    try:
        return importlib.import_module("asc")
    except ModuleNotFoundError as exc:
        raise PyAscUnavailableError(
            "pyasc is not installed in this environment; install the pyasc "
            "package that matches the CANN version, or evaluate with "
            "--backend ascendc"
        ) from exc


def configure_platform(asc: ModuleType, device_index: int) -> str:
    """Select the pyasc execution backend and device, returning the SOC name.

    The SOC version is left to pyasc on the NPU path: it reads the live
    platform and rejects a mismatch, so passing the run's hardware profile
    value here would turn a profile typo into an evaluation failure.

    Set ASCEND_KERNEL_BENCH_PYASC_BACKEND=model to run on pyasc's simulator
    instead. The simulator executes the same kernel on CPU tensors with no
    NPU, which is useful for checking one candidate's correctness; it cannot
    replace the benchmark protocol, whose reference timing needs the device.
    The simulator runtime must be on LD_LIBRARY_PATH, see the pyasc guide.
    """
    config = importlib.import_module("asc.runtime.config")
    requested = os.environ.get(
        "ASCEND_KERNEL_BENCH_PYASC_BACKEND", "npu"
    ).strip().lower()
    if requested == "model":
        soc = os.environ.get("ASCEND_KERNEL_BENCH_PYASC_SOC", "Ascend910B4")
        config.set_platform(config.Backend.Model, config.Platform(soc))
        return soc
    config.set_platform(config.Backend.NPU, None, device_id=device_index)
    runtime = importlib.import_module("asc.lib.runtime")
    return str(runtime.current_platform())


def load_kernel_module(sample_dir: str | Path, kernel_path: str | Path) -> ModuleType:
    """Import kernel.py from its real path and register it for model_new.

    pyasc reads the decorated function's source with introspection, so the
    module must be imported from a file rather than exec'd from a string.
    The sample directory is importable for the duration of the import so a
    kernel can use sibling helper modules.

    Raises:
        PyAscLoadError: If the module cannot be imported.
    """
    sample = Path(sample_dir).resolve()
    kernel_path = Path(kernel_path).resolve()
    unique_name = f"ascend_kernel_bench_kernel_{abs(hash(str(sample))):x}"
    spec = importlib.util.spec_from_file_location(unique_name, kernel_path)
    if spec is None or spec.loader is None:
        raise PyAscLoadError(f"cannot import kernel module from {kernel_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = module
    sys.modules[KERNEL_MODULE_NAME] = module
    sys.path.insert(0, str(sample))
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(unique_name, None)
        sys.modules.pop(KERNEL_MODULE_NAME, None)
        raise PyAscLoadError(f"kernel.py import failed: {exc!r}") from exc
    finally:
        try:
            sys.path.remove(str(sample))
        except ValueError:
            pass
    return module


def time_first_call(call: Callable[[], Any]) -> tuple[float, Any]:
    """Run one call and return (seconds, result).

    The first pyasc call performs JIT trace, compile, and launch, so its
    duration is a compound figure. Report it as first_call_seconds rather
    than claiming a pure compile time.

    Raises:
        PyAscJITError: If the call fails.
    """
    started = time.monotonic()
    try:
        result = call()
    except Exception as exc:
        raise PyAscJITError(f"first pyasc kernel call failed: {exc!r}") from exc
    return time.monotonic() - started, result


def runtime_facts(sample_dir: str | Path | None = None) -> dict[str, Any]:
    """Return pyasc identity and cache facts for result metadata."""
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            package_version = version("pyasc")
        except PackageNotFoundError:
            package_version = "unknown"
    except Exception:  # pragma: no cover - importlib.metadata is stdlib
        package_version = "unknown"
    facts: dict[str, Any] = {"pyasc_version": package_version}
    cache_dir = os.environ.get("PYASC_CACHE_DIR")
    if cache_dir:
        facts["pyasc_cache_dir"] = cache_dir
    elif sample_dir is not None:
        facts["pyasc_cache_dir"] = str(cache_dir_for(sample_dir))
    facts["pyasc_cache_policy"] = "sample_local"
    return facts
