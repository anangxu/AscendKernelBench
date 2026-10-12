"""Run directory layout and result persistence (docs/guide/results.md).

A run holds generation_config.yaml, one directory per sample with the two
deliverables and eval_result.json, and an aggregate eval_results.json.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

from ._paths import RUNS_DIR
from .config import (
    DEFAULT_BUILD_BACKEND,
    BuildBackend,
    DeepJitConfig,
    parse_build_backend,
)
from .io_util import (
    read_json_object,
    write_json_atomic,
    write_text_atomic,
    write_yaml_atomic,
)
from .llm import AscendCGeneration

# Generation settings that identify one run's harness. A continuation may add
# tasks, but it may not change any of these, and it never rewrites a recorded
# value: the first request recorded for a key stays authoritative.
_RUN_IDENTITY_KEYS = (
    "build_backend",
    "hardware",
    "model",
    "prompt_mode",
    "temperature",
    "max_tokens",
    "reasoning_effort",
    "num_samples",
    "deepjit",
)


def _recorded_samples(run_dir: Path, recorded: dict) -> list[Path]:
    """Return the sample directories an existing record already holds."""
    task_ids = recorded.get("tasks") or []
    samples = int(recorded.get("num_samples") or 0)
    found: list[Path] = []
    for task_id in task_ids:
        for sample_id in range(samples):
            path = sample_dir(run_dir, task_id, sample_id)
            if (path / "custom_op.asc").is_file() and (path / "model_new.py").is_file():
                found.append(path)
            elif (path / "eval_result.json").is_file():
                found.append(path)
    return found


def _merge_generation_config(run_dir: Path, generation_config: dict) -> dict:
    """Return the record to write, keeping a run's history intact.

    A run is initialized once with its planned task list; a later request for
    the same run is a continuation that may add tasks. Recorded values win, so
    a continuation can never silently overwrite the model, the build backend,
    the generation parameters, or the task list. Identity fields are compared
    by presence, so an explicit None is a value like any other.

    Raises:
        ValueError: If a requested identity field conflicts with the record, or
            if the record predates a field the request executes while samples
            already exist.
    """
    recorded = _read_generation_config(run_dir)
    if not recorded:
        return generation_config
    interpreted = _interpreted_build_backend(recorded)
    if interpreted is not None:
        recorded["build_backend"] = interpreted.value
    existing = _recorded_samples(run_dir, recorded)
    filled: dict[str, Any] = {}
    for key in _RUN_IDENTITY_KEYS:
        if key in recorded and key in generation_config:
            left, right = recorded[key], generation_config[key]
            if key == "deepjit":
                left, right = _normalized_deepjit(left), _normalized_deepjit(right)
            if left != right:
                raise ValueError(
                    f"{run_dir} records {key}={recorded[key]!r} but this "
                    f"request uses {generation_config[key]!r}; one run keeps "
                    "one harness. Use a new run name."
                )
        elif key in generation_config:
            # The record predates this field. Adopting the request is only
            # safe while nothing was generated under the old record.
            if existing:
                raise ValueError(
                    f"{run_dir} predates the {key} field and already holds "
                    f"{len(existing)} sample(s), so this run cannot confirm "
                    f"{key}={generation_config[key]!r}; use a new run name."
                )
            filled[key] = generation_config[key]
    merged = dict(generation_config)
    merged.update(recorded)
    tasks = list(recorded.get("tasks") or [])
    for task in generation_config.get("tasks") or []:
        if task not in tasks:
            tasks.append(task)
    if tasks:
        merged["tasks"] = tasks
    if filled:
        merged["record_filled"] = {
            "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "fields": filled,
            "reason": "the record predates these fields and held no samples",
        }
    return merged


def _check_run_compatible(run_dir: Path, generation_config: dict) -> None:
    """Refuse to restamp a run that this generation request conflicts with.

    Raises:
        FileExistsError: If the requested tasks already hold samples.
    """
    existing = _recorded_samples(run_dir, generation_config)
    if existing:
        shown = ", ".join(str(path.relative_to(run_dir)) for path in existing[:3])
        more = "" if len(existing) <= 3 else f" and {len(existing) - 3} more"
        raise FileExistsError(
            f"{run_dir} already holds {len(existing)} candidate(s): {shown}{more}. "
            "Generation does not overwrite samples; use a new run name."
        )


def create_run(run_name: str, generation_config: dict) -> Path:
    """Create the run directory and stamp the generation config.

    A new directory records the request as given. An existing directory keeps
    its recorded harness and task list: the request is merged into it, so a
    per-task continuation accumulates tasks instead of replacing the record,
    and any conflict is refused rather than silently restamped.

    Raises:
        ValueError: If another identity field conflicts with this request.
        FileExistsError: If it already holds candidates for the planned tasks.
    """
    run_dir = RUNS_DIR / run_name
    record = generation_config
    if run_dir.is_dir() and generation_config:
        record = _merge_generation_config(run_dir, generation_config)
        # The candidate protection guards the tasks this request asks for, so a
        # continuation can still add a task the run has not planned yet.
        _check_run_compatible(run_dir, generation_config)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "generation_config.yaml"
    if record:
        write_yaml_atomic(cfg_path, record)
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
    "build_backend",
)

# Generation settings a continuation may inherit from a run's own record.
# Values are kept even when they are None: presence is what a continuation
# compares, so an empty recorded value is never replaced by a YAML default.
_CONTINUATION_KEYS = (
    "model",
    "prompt_mode",
    "temperature",
    "reasoning_effort",
    "max_tokens",
    "num_samples",
    "build_backend",
)


def recorded_generation_defaults(run_dir: Path) -> dict[str, Any]:
    """Return the generation settings an existing run already recorded.

    An empty mapping means the run has no record yet, so a caller keeps its
    own defaults. The deepjit block travels whole because its fields are
    machine configuration rather than flags.
    """
    loaded = _read_generation_config(run_dir)
    values: dict[str, Any] = {
        key: loaded[key] for key in _CONTINUATION_KEYS if key in loaded
    }
    if "build_backend" not in values:
        # A legacy record means cmake; inheriting it keeps the continuation on
        # the build path the run actually used, whatever the config defaults to.
        interpreted = _interpreted_build_backend(loaded)
        if interpreted is not None:
            values["build_backend"] = interpreted.value
    deepjit = loaded.get("deepjit")
    if isinstance(deepjit, dict):
        values["deepjit"] = dict(deepjit)
    return values


def _interpreted_build_backend(loaded: dict) -> BuildBackend | None:
    """Interpret a record's build backend, or None when it has no record.

    A record that predates the field only ever built through the CMake
    template, so it means cmake rather than whatever the configuration
    defaults to now. A record with an unknown value returns None.
    """
    if not loaded:
        return None
    value = loaded.get("build_backend")
    if value is None:
        return DEFAULT_BUILD_BACKEND
    try:
        return parse_build_backend(value)
    except ValueError:
        return None


def generation_build_backend(run_dir: Path) -> BuildBackend | None:
    """Return the build backend a run uses, or None when it has no record."""
    return _interpreted_build_backend(_read_generation_config(run_dir))


def _normalized_deepjit(value: object) -> dict[str, Any]:
    """Return a deepjit block as the current model sees it.

    Fields that no longer exist are dropped on both sides of a comparison, so
    removing a setting cannot make an older record look like a conflict.
    """
    raw = value if isinstance(value, dict) else {}
    return DeepJitConfig.model_validate(raw).model_dump()


def generation_deepjit(run_dir: Path) -> dict[str, Any]:
    """Return the DeepJIT settings recorded for a run, if any."""
    value = _read_generation_config(run_dir).get("deepjit")
    return dict(value) if isinstance(value, dict) else {}


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
