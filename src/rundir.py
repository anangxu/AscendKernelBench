"""Run directory layout and result persistence (docs/guide/results.md).

A run holds generation_config.yaml, one directory per sample with the two
deliverables and eval_result.json, and an aggregate eval_results.json. The
sample file names depend on the backend recorded for the run.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

from . import backend as backend_module
from . import suite as suite_module
from ._paths import RUNS_DIR
from .backend import DEFAULT_BACKEND, Backend, parse_backend, validate_backend_build
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

# Generation settings that identify one run's harness. A continuation may add
# tasks, but it may not change any of these, and it never rewrites a recorded
# value: the first request recorded for a key stays authoritative.
_RUN_IDENTITY_KEYS = (
    "backend",
    "hardware_file",
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
            if any(sample_complete(path, backend) for backend in Backend):
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
    recorded.setdefault("backend", DEFAULT_BACKEND.value)
    existing = _recorded_samples(run_dir, recorded)
    filled: dict[str, Any] = {}
    for key in _RUN_IDENTITY_KEYS:
        if key in recorded and key in generation_config:
            left, right = recorded[key], generation_config[key]
            if key == "deepjit":
                left, right = _normalized_deepjit(left), _normalized_deepjit(right)
            elif key == "hardware_file" and left and right:
                left, right = Path(left).resolve(), Path(right).resolve()
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
    provenance = _merged_task_meta(recorded, generation_config)
    if provenance:
        merged["task_meta"] = provenance
    if filled:
        merged["record_filled"] = {
            "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "fields": filled,
            "reason": "the record predates these fields and held no samples",
        }
    return merged


def _merged_task_meta(recorded: dict, request: dict) -> dict[str, Any]:
    """Combine task provenance, keeping the first record for a task.

    A continuation adds the provenance of the tasks it introduces and never
    rewrites what the run already recorded.
    """
    old = recorded.get("task_meta")
    new = request.get("task_meta")
    combined: dict[str, Any] = {}
    if isinstance(new, dict):
        combined.update(
            {key: dict(value) for key, value in new.items() if isinstance(value, dict)}
        )
    if isinstance(old, dict):
        combined.update(
            {key: dict(value) for key, value in old.items() if isinstance(value, dict)}
        )
    return combined


def create_run(run_name: str, generation_config: dict, *, resume: bool = False) -> Path:
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
    generation_config = dict(generation_config)
    if generation_config:
        validate_backend_build(
            generation_config.get("backend"), generation_config.get("build_backend")
        )
    record = generation_config
    if run_dir.is_dir() and generation_config:
        if (
            run_dir / "generation_config.yaml"
        ).exists() and not _read_generation_config(run_dir):
            raise ValueError(f"{run_dir} has an invalid generation record")
        record = _merge_generation_config(run_dir, generation_config)
        # The candidate protection guards the tasks this request asks for, so a
        # continuation can still add a task the run has not planned yet.
        _check_run_compatible(run_dir, generation_config, resume=resume)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "generation_config.yaml"
    if record:
        write_yaml_atomic(cfg_path, record)
    return run_dir


def _check_run_compatible(
    run_dir: Path, generation_config: dict, *, resume: bool = False
) -> None:
    """Refuse to restamp a run that this generation request conflicts with."""
    requested = parse_backend(
        _backend_name(generation_config.get("backend") or DEFAULT_BACKEND)
    )
    recorded = generation_backend(run_dir)
    if recorded is not None and recorded is not requested:
        raise ValueError(
            f"{run_dir} records backend {recorded.value} but this generation "
            f"request uses {requested.value}; one run is one backend. Use a "
            "new run name."
        )
    resolve_run_backend(run_dir)
    _check_recorded_task_meta(run_dir, generation_config)
    task_ids = generation_config.get("tasks") or []
    samples = int(generation_config.get("num_samples") or 0)
    if resume:
        for task_id in task_ids:
            for sample_id in range(samples):
                path = sample_dir(run_dir, task_id, sample_id)
                if sample_complete(path, requested):
                    continue
                _refuse_existing_artifacts(path)
        return
    existing = [
        sample_dir(run_dir, task_id, sample_id)
        for task_id in task_ids
        for sample_id in range(samples)
        if sample_complete(sample_dir(run_dir, task_id, sample_id), requested)
        or (sample_dir(run_dir, task_id, sample_id) / "eval_result.json").is_file()
    ]
    if existing:
        shown = ", ".join(str(path.relative_to(run_dir)) for path in existing[:3])
        more = "" if len(existing) <= 3 else f" and {len(existing) - 3} more"
        raise FileExistsError(
            f"{run_dir} already holds {len(existing)} candidate(s): {shown}{more}. "
            "Generation does not overwrite samples; use a new run name."
        )


def _check_recorded_task_meta(run_dir: Path, generation_config: dict) -> None:
    """Refuse a request whose task sources differ from the recorded ones."""
    recorded = recorded_task_meta(run_dir)
    requested = generation_config.get("task_meta")
    if not recorded or not isinstance(requested, dict):
        return
    for task_id, entry in requested.items():
        if not isinstance(entry, dict):
            continue
        old = recorded.get(task_id)
        if not isinstance(old, dict):
            continue
        conflict = _provenance_conflict(old, entry)
        if conflict:
            raise ValueError(
                f"{run_dir} recorded task {task_id} with {conflict}, but this "
                "request differs; the task source or its provenance changed. Use "
                "a new run name or restore it."
            )


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


def generation_backend(run_dir: Path) -> Backend | None:
    """Return the recorded source language, or None for an unrecorded run."""
    value = _read_generation_config(run_dir).get("backend")
    return None if value is None else parse_backend(value)


_HARNESS_KEYS = (
    "backend",
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
    "tasks",
    "backend",
    "hardware",
    "hardware_file",
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
    if loaded and "backend" not in values:
        values["backend"] = DEFAULT_BACKEND.value
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
    return parse_build_backend(value)


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


def recorded_task_meta(run_dir: Path) -> dict[str, Any]:
    """Return the per-task provenance recorded for a run, if any."""
    value = _read_generation_config(run_dir).get("task_meta")
    return dict(value) if isinstance(value, dict) else {}


# Provenance fields that make a task re-usable. A change to any of them is
# refused instead of silently evaluating different source.
_PROVENANCE_IDENTITY = ("suite", "source", "source_revision", "task_hash")


def _provenance_conflict(recorded: dict, current: dict) -> str | None:
    """Return the first provenance field that changed, if any."""
    for key in _PROVENANCE_IDENTITY:
        old, new = recorded.get(key), current.get(key)
        if old is None or new is None:
            continue
        if str(old) != str(new):
            return f"{key} {str(old)[:24]!r} -> {str(new)[:24]!r}"
    return None


def verify_task_meta(run_dir: Path, tasks: list[Any]) -> None:
    """Refuse a task whose recorded provenance no longer matches the corpus.

    Args:
        run_dir: Run directory that may record per-task provenance.
        tasks: Tasks about to be generated or evaluated.

    Raises:
        ValueError: If the suite, source, source revision, or content hash
            differs from the record.
    """
    recorded = recorded_task_meta(run_dir)
    if not recorded:
        return
    for task in tasks:
        entry = recorded.get(task.task_id)
        if not isinstance(entry, dict):
            continue
        conflict = _provenance_conflict(entry, task.provenance())
        if conflict:
            raise ValueError(
                f"{run_dir} recorded task {task.task_id} with {conflict}; the task "
                "source or its provenance changed, so this run cannot be reused. "
                "Restore it or use a new run name."
            )


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
        FileExistsError: If the sample directory already holds a candidate or a
            result. Generation never replaces a candidate by default: a new
            source must not be paired with an old verdict, so the run name or
            the sample directory has to change instead.
    """
    resolved = parse_backend(_backend_name(backend))
    recorded = generation_backend(run_dir)
    if recorded is not None and recorded is not resolved:
        raise ValueError(
            f"{run_dir} records backend {recorded.value}; refusing to write a "
            f"{resolved.value} sample into it. One run is one backend."
        )
    out_dir = sample_dir(run_dir, task_id, sample_id)
    if sample_complete(out_dir, resolved) or (out_dir / "eval_result.json").is_file():
        raise FileExistsError(
            f"{out_dir} already holds a candidate or its evaluation result; "
            "generation does not overwrite samples. Use a new run name, or "
            "remove that sample directory and generate it again."
        )
    _refuse_existing_artifacts(out_dir)
    kernel_source = _kernel_source(generation, resolved)
    kernel_file, wrapper_file = backend_module.sample_files(resolved)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_text_atomic(out_dir / "prompt.txt", prompt)
    write_text_atomic(out_dir / kernel_file, kernel_source)
    write_text_atomic(out_dir / wrapper_file, generation.model_new_py)
    if raw_response:
        write_text_atomic(out_dir / "response_raw.txt", raw_response)
    return out_dir


def _refuse_existing_artifacts(path: Path) -> None:
    """Refuse partial or stale samples before writing candidate files."""
    names = {"custom_op.asc", "kernel.py", "model_new.py", "eval_result.json"}
    if any((path / name).exists() for name in names):
        raise FileExistsError(f"{path} already holds candidate artifacts or results")


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


def _task_dirs(
    run_dir: Path,
    level: int | None = None,
    suite: str | None = None,
) -> list[tuple[str, Path]]:
    """Return (task id, directory) for every task directory holding samples."""
    resolved = Path(run_dir)
    found: dict[str, Path] = {}
    for sample_path in sorted(resolved.glob("**/sample_*")):
        if not sample_path.is_dir():
            continue
        task_path = sample_path.parent
        task_id = task_path.relative_to(resolved).as_posix()
        if "/" not in task_id:
            continue
        if level is not None and not task_id.startswith(f"level{level}/"):
            continue
        if suite is not None and suite_module.suite_of(task_id) != suite:
            continue
        found.setdefault(task_id, task_path)
    return sorted(found.items(), key=lambda item: (item[0].count("/"), item[0]))


def resolve_run_backend(run_dir: Path) -> Backend:
    """Resolve a run's source language and reject mixed candidate artifacts."""
    resolved = Path(run_dir)
    task_dirs = [path for _, path in _task_dirs(resolved)]
    present = {
        backend
        for backend in Backend
        if any(
            (sample / backend_module.kernel_file(backend)).is_file()
            for task in task_dirs
            for sample in task.glob("sample_*")
        )
    }
    recorded = generation_backend(resolved)
    if len(present) > 1 or (recorded is not None and present - {recorded}):
        names = ", ".join(sorted(item.value for item in present))
        raise ValueError(f"{resolved} has incompatible backend artifacts: {names}")
    return recorded or next(iter(present), DEFAULT_BACKEND)


def _complete_samples(
    task_dirs: list[tuple[str, Path]], kernel_file: str, wrapper_file: str
) -> list[tuple[str, int, Path]]:
    """Return (task id, sample id, dir) for every complete sample of a backend."""
    found: list[tuple[str, int, Path]] = []
    for task_id, task_dir in task_dirs:
        for sdir in sorted(task_dir.glob("sample_*")):
            if (sdir / kernel_file).is_file() and (sdir / wrapper_file).is_file():
                found.append((task_id, int(sdir.name.split("_", 1)[1]), sdir))
    return found


def iter_sample_dirs(
    run_dir: Path,
    level: int | None = None,
    backend: Backend | str | None = None,
    suite: str | None = None,
) -> Iterator[tuple[str, int, Path]]:
    """Yield complete samples whose artifacts agree with the run record."""
    resolved = resolve_run_backend(run_dir)
    if backend is not None and parse_backend(backend) is not resolved:
        raise ValueError(f"requested backend disagrees with {resolved.value} run")
    yield from _complete_samples(
        _task_dirs(run_dir, level=level, suite=suite),
        *backend_module.sample_files(resolved),
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
