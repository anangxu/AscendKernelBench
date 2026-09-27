"""Run directory layout and result persistence (docs/guide/results.md).

A run holds generation_config.yaml, one directory per sample with the two
deliverables and eval_result.json, and an aggregate eval_results.json.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

from ._paths import RUNS_DIR
from .io_util import (
    read_json_object,
    write_json_atomic,
    write_text_atomic,
    write_yaml_atomic,
)
from .llm import AscendCGeneration


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


def generation_hardware_file(run_dir: Path) -> str | None:
    """Return the hardware profile path recorded at generation time."""
    path = _read_generation_config(run_dir).get("hardware_file")
    if isinstance(path, str) and path.strip():
        return path.strip()
    return None


_HARNESS_KEYS = (
    "model",
    "prompt_mode",
    "temperature",
    "reasoning_effort",
    "max_tokens",
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


def save_sample(
    run_dir: Path,
    task_id: str,
    sample_id: int,
    *,
    prompt: str,
    generation: AscendCGeneration,
    raw_response: str = "",
) -> Path:
    """Persist one generated sample (prompt, both deliverables, raw text)."""
    out_dir = sample_dir(run_dir, task_id, sample_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_text_atomic(out_dir / "prompt.txt", prompt)
    write_text_atomic(out_dir / "custom_op.asc", generation.custom_op_asc)
    write_text_atomic(out_dir / "model_new.py", generation.model_new_py)
    if raw_response:
        write_text_atomic(out_dir / "response_raw.txt", raw_response)
    return out_dir


def iter_sample_dirs(
    run_dir: Path, level: int | None = None
) -> Iterator[tuple[str, int, Path]]:
    """Yield (task_id, sample_id, dir) for every complete sample.

    A sample is complete when both custom_op.asc and model_new.py exist.
    """
    pattern = f"level{level}/*" if level is not None else "level*/*"
    for task_dir in sorted(Path(run_dir).glob(pattern)):
        if not task_dir.is_dir():
            continue
        task_id = f"{task_dir.parent.name}/{task_dir.name}"
        for sdir in sorted(task_dir.glob("sample_*")):
            if (sdir / "custom_op.asc").is_file() and (sdir / "model_new.py").is_file():
                yield task_id, int(sdir.name.split("_", 1)[1]), sdir


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
