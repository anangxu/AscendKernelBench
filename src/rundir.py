"""Run directory layout and result persistence (docs/guide/results.md).

A run holds generation_config.yaml, one directory per sample with the two
deliverables and eval_result.json, and an aggregate eval_results.json. The
sample file names depend on the backend recorded for the run.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

from . import backend as backend_module
from ._paths import RUNS_DIR
from .backend import DEFAULT_BACKEND, Backend, parse_backend
from .io_util import read_json_object, write_json_atomic, write_yaml_atomic


def create_run(run_name: str, generation_config: dict) -> Path:
    """Create the run directory and stamp the generation config."""
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "generation_config.yaml"
    if generation_config:
        write_yaml_atomic(cfg_path, generation_config)
    return run_dir


def resolve_run(run: str | Path) -> Path:
    """Resolve a run name or directory path to an existing run directory.

    Args:
        run: Existing directory used as-is, or a bare name under the
            runs root.

    Raises:
        FileNotFoundError: If the resolved path is not a directory.
    """
    path = Path(run)
    if path.is_dir():
        return path.resolve()
    if path.is_absolute() or len(path.parts) > 1:
        raise FileNotFoundError(f"run dir not found: {path}")
    named = RUNS_DIR / path.name
    if named.is_dir():
        return named
    raise FileNotFoundError(f"run dir not found: {named}")


def _read_generation_config(run_dir: Path) -> dict[str, Any]:
    """Return generation_config.yaml, or an empty mapping when unusable."""
    path = Path(run_dir) / "generation_config.yaml"
    if not path.is_file():
        return {}
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    if not isinstance(loaded, dict):
        return {}
    return loaded


def generation_hardware_name(run_dir: Path) -> str | None:
    """Return the hardware profile name recorded at generation time."""
    name = _read_generation_config(run_dir).get("hardware")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return None


def generation_backend(run_dir: Path) -> Backend | None:
    """Return the backend recorded for a run, or None when it is absent.

    Runs older than the backend field carry no value; callers fall back to
    the default backend in that case.

    Raises:
        ValueError: If the recorded value is not a known backend.
    """
    value = _read_generation_config(run_dir).get("backend")
    if value is None or not str(value).strip():
        return None
    return parse_backend(value)


_HARNESS_KEYS = (
    "model",
    "prompt_mode",
    "temperature",
    "reasoning_effort",
    "max_tokens",
    "backend",
)


def generation_harness(run_dir: Path) -> dict[str, Any]:
    """Return the generation settings that define this run's harness."""
    loaded = _read_generation_config(run_dir)
    return {
        key: loaded[key]
        for key in _HARNESS_KEYS
        if key in loaded and loaded[key] is not None
    }


def sample_dir(run_dir: Path, task_id: str, sample_id: int) -> Path:
    """Return the sample directory path for a task id and sample index."""
    return Path(run_dir) / task_id / f"sample_{sample_id}"


def _backend_name(value: Backend | str) -> str:
    """Return the name parse_backend expects for a Backend or its value."""
    return value.value if isinstance(value, Backend) else value


def save_sample(
    run_dir: Path,
    task_id: str,
    sample_id: int,
    *,
    prompt: str,
    generation: object,
    raw_response: str = "",
    backend: Backend | str = DEFAULT_BACKEND,
) -> Path:
    """Persist one generated sample (prompt, both deliverables, raw text).

    Args:
        generation: An AscendCGeneration for the Ascend C backend, or a
            PyAscGeneration for pyasc.
        backend: Authoring language that fixes the kernel artifact name.

    Raises:
        TypeError: If generation does not match the backend.
    """
    resolved = parse_backend(_backend_name(backend))
    kernel_source = _kernel_source(generation, resolved)
    kernel_file, wrapper_file = backend_module.sample_files(resolved)
    out_dir = sample_dir(run_dir, task_id, sample_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    (out_dir / kernel_file).write_text(kernel_source, encoding="utf-8")
    (out_dir / wrapper_file).write_text(generation.model_new_py, encoding="utf-8")
    if raw_response:
        (out_dir / "response_raw.txt").write_text(raw_response, encoding="utf-8")
    return out_dir


def _kernel_source(generation: object, backend: Backend) -> str:
    """Return the kernel field of generation that matches the backend.

    Raises:
        TypeError: If generation is not the model of this backend.
    """
    if backend is Backend.PYASC:
        kernel_py = getattr(generation, "kernel_py", None)
        if not isinstance(kernel_py, str):
            raise TypeError("pyasc samples need a PyAscGeneration")
        return kernel_py
    custom_op_asc = getattr(generation, "custom_op_asc", None)
    if not isinstance(custom_op_asc, str):
        raise TypeError("ascendc samples need an AscendCGeneration")
    return custom_op_asc


def load_sample(
    sample_dir_path: Path, backend: Backend | str = DEFAULT_BACKEND
) -> tuple[str, str]:
    """Return the (kernel source, wrapper source) of one sample.

    Args:
        sample_dir_path: One sample directory.
        backend: Authoring language that fixes the kernel artifact name.

    Raises:
        FileNotFoundError: If either deliverable is missing.
    """
    kernel_file, wrapper_file = backend_module.sample_files(
        parse_backend(_backend_name(backend))
    )
    root = Path(sample_dir_path)
    return (
        (root / kernel_file).read_text(encoding="utf-8"),
        (root / wrapper_file).read_text(encoding="utf-8"),
    )


def sample_complete(
    sample_dir_path: Path, backend: Backend | str = DEFAULT_BACKEND
) -> bool:
    """Return True when a sample directory holds both deliverables."""
    kernel_file, wrapper_file = backend_module.sample_files(
        parse_backend(_backend_name(backend))
    )
    root = Path(sample_dir_path)
    return (root / kernel_file).is_file() and (root / wrapper_file).is_file()


def iter_sample_dirs(
    run_dir: Path,
    level: int | None = None,
    backend: Backend | str | None = None,
) -> Iterator[tuple[str, int, Path]]:
    """Yield (task_id, sample_id, dir) for every complete sample.

    A sample is complete when both of its backend's deliverables exist.
    With backend unset the run's recorded backend is used, so an old run
    without a backend field stays an Ascend C run. When the recorded
    backend yields no sample at all, the other backends are tried, so a
    directory copied out of a run still discovers its samples.
    """
    pattern = f"level{level}/*" if level is not None else "level*/*"
    task_dirs = [
        task_dir
        for task_dir in sorted(Path(run_dir).glob(pattern))
        if task_dir.is_dir()
    ]
    resolved = (
        parse_backend(_backend_name(backend))
        if backend is not None
        else generation_backend(run_dir)
    )
    candidates = _discovery_order(resolved, task_dirs)
    for index, candidate in enumerate(candidates):
        kernel_file, wrapper_file = backend_module.sample_files(candidate)
        found: list[tuple[str, int, Path]] = []
        for task_dir in task_dirs:
            task_id = f"{task_dir.parent.name}/{task_dir.name}"
            for sdir in sorted(task_dir.glob("sample_*")):
                if (sdir / kernel_file).is_file() and (sdir / wrapper_file).is_file():
                    found.append((task_id, int(sdir.name.split("_", 1)[1]), sdir))
        if found or index == len(candidates) - 1:
            yield from found
            return


def _discovery_order(resolved: Backend | None, task_dirs: list[Path]) -> list[Backend]:
    """Return the backends to try, most likely first, without duplicates."""
    first = resolved if resolved is not None else DEFAULT_BACKEND
    order = [first]
    for candidate in Backend:
        if candidate not in order:
            order.append(candidate)
    if resolved is None and not _any_sample(task_dirs, first):
        # No backend was recorded: prefer whichever backend the samples use.
        for candidate in order[1:]:
            if _any_sample(task_dirs, candidate):
                order.remove(candidate)
                order.insert(0, candidate)
                break
    return order


def _any_sample(task_dirs: list[Path], candidate: Backend) -> bool:
    """Return True when any sample dir is complete for one backend."""
    return any(
        sample_complete(sdir, candidate)
        for task_dir in task_dirs
        for sdir in task_dir.glob("sample_*")
    )


def load_eval_result(sample_dir_path: Path) -> dict[str, Any] | None:
    """Load eval_result.json from a sample directory, if present."""
    path = Path(sample_dir_path) / "eval_result.json"
    if not path.is_file():
        return None
    try:
        return read_json_object(path)
    except (json.JSONDecodeError, ValueError):
        return None


def collect_eval_results(run_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Assemble the KernelBench-compatible eval_results mapping for a run."""
    results: dict[str, list[dict]] = {}
    for task_id, sample_id, sdir in iter_sample_dirs(run_dir):
        result = load_eval_result(sdir)
        if result is None:
            continue
        results.setdefault(task_id, []).append({"sample_id": sample_id, **result})
    return results


def write_eval_results(run_dir: Path, results: dict[str, list[dict[str, Any]]]) -> Path:
    """Write eval_results.json atomically."""
    path = Path(run_dir) / "eval_results.json"
    write_json_atomic(path, results)
    return path


def write_pass_at_k(run_dir: Path, pass_at_k: dict) -> Path:
    """Write pass_at_k_results.json atomically."""
    path = Path(run_dir) / "pass_at_k_results.json"
    write_json_atomic(path, pass_at_k)
    return path
