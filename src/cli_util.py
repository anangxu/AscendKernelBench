"""Shared CLI helpers: settings resolution, task selection, and reports.

Reporting lives in .report; this module keeps the generation and evaluation
setting factories that the batch scripts share.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from . import rundir
from .backend import DEFAULT_BACKEND, Backend, parse_backend
from .config import (
    EvalConfig,
    HardwareProfile,
    load_eval_config,
    load_hardware_profile,
)
from .dataset import Task, discover_tasks, load_task
from .log import die
from .report import (
    cli_progress,
    eval_result_lines,
    print_eval_report,
    sample_status_label,
)

__all__ = [
    "EvalRuntime",
    "GenerationSettings",
    "cli_progress",
    "eval_result_lines",
    "generation_run_config",
    "load_eval_runtime",
    "print_eval_report",
    "read_task_ids",
    "resolve_backend",
    "resolve_generation_settings",
    "resolve_run_dir",
    "sample_status_label",
    "select_tasks",
]


def resolve_backend(
    cli_value: str | None = None, config_value: object = None
) -> Backend:
    """Resolve the authoring backend from a CLI flag, then the config.

    Args:
        cli_value: The --backend value, when the flag was given.
        config_value: The generation.backend value from the loaded config.

    Returns:
        The resolved backend; absent values resolve to the default backend.

    Raises:
        ValueError: If the resolved value is not a known backend.
    """
    value = cli_value if cli_value is not None else config_value
    return parse_backend(value.value if isinstance(value, Backend) else value)


class GenerationSettings(BaseModel):
    """Resolved generation settings after YAML defaults and CLI overrides."""

    model_config = ConfigDict(frozen=True)

    model: str
    prompt_mode: str
    temperature: float
    max_tokens: int
    num_samples: int
    reasoning_effort: str | None = None
    backend: Backend = DEFAULT_BACKEND


def resolve_generation_settings(
    config: EvalConfig,
    *,
    model: str | None = None,
    prompt_mode: str | None = None,
    temperature: float | None = None,
    num_samples: int | None = None,
    reasoning_effort: str | None = None,
    max_tokens: int | None = None,
    backend: str | None = None,
) -> GenerationSettings:
    """Resolve generation settings from YAML with optional CLI overrides."""
    gen_cfg = dict(config.generation)
    effort = reasoning_effort or str(gen_cfg.get("reasoning_effort", "")).strip()
    return GenerationSettings(
        model=model or str(gen_cfg.get("model", "deepseek-flash")),
        prompt_mode=prompt_mode or str(gen_cfg.get("prompt_mode", "one_shot")),
        temperature=float(
            gen_cfg.get("temperature", 0.0) if temperature is None else temperature
        ),
        max_tokens=(
            int(gen_cfg.get("max_tokens", 131072))
            if max_tokens is None
            else int(max_tokens)
        ),
        num_samples=(
            int(gen_cfg.get("num_samples", 1))
            if num_samples is None
            else int(num_samples)
        ),
        reasoning_effort=effort or None,
        backend=resolve_backend(backend, gen_cfg.get("backend")),
    )


class EvalRuntime(BaseModel):
    """Loaded evaluation config plus the resolved hardware profile."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    config: EvalConfig
    hardware: HardwareProfile


def load_eval_runtime(
    *,
    config_path: str | None = None,
    hardware: str | None = None,
) -> EvalRuntime:
    """Load YAML defaults and resolve the hardware profile.

    hardware names a profile or points at a YAML file, and overrides the
    config when given.

    Raises:
        FileNotFoundError: If the config or hardware YAML is missing.
    """
    config = load_eval_config(config_path)
    return EvalRuntime(
        config=config,
        hardware=load_hardware_profile(hardware or config.hardware),
    )


def read_task_ids(path: str | Path) -> list[str]:
    """Read task ids from a manifest file, one per line.

    Blank lines and lines starting with a hash are skipped, so a manifest
    can carry a header.
    """
    ids = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if entry and not entry.startswith("#"):
            ids.append(entry)
    return ids


def resolve_run_dir(run: str | Path) -> Path:
    """Resolve a run name or path, or exit if it is missing."""
    try:
        return rundir.resolve_run(run)
    except FileNotFoundError as exc:
        die(str(exc))


def select_tasks(
    *,
    level: int | None = None,
    task_ids: list[str] | None = None,
    tasks_file: str | Path | None = None,
) -> list[Task]:
    """Load tasks from explicit IDs, a manifest file, a level, or the corpus.

    Returns:
        Loaded tasks. Explicit IDs win over a manifest, which wins over
        level.
    """
    if task_ids is not None:
        return [load_task(item) for item in task_ids]
    if tasks_file is not None:
        return [load_task(item) for item in read_task_ids(tasks_file)]
    if level is not None:
        return discover_tasks(level=level)
    return discover_tasks()


def generation_run_config(
    settings: GenerationSettings,
    *,
    hardware_name: str,
    task_ids: list[str],
    **extra: object,
) -> dict[str, object]:
    """Return the mapping written to generation_config.yaml.

    Extra keyword arguments, such as device, are merged into the record.
    """
    payload: dict[str, object] = {
        "model": settings.model,
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
        "prompt_mode": settings.prompt_mode,
        "num_samples": settings.num_samples,
        "reasoning_effort": settings.reasoning_effort,
        "backend": settings.backend.value,
        "hardware": hardware_name,
        "tasks": list(task_ids),
    }
    payload.update(extra)
    return payload
