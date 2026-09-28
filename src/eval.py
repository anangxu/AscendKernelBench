"""Isolated evaluation of generated samples.

Static check runs on the host; build, correctness, and timing run in the
worker subprocess started by evaluate_run. One run uses one backend.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from loguru import logger

from . import rundir
from .backend import DEFAULT_BACKEND, Backend, sample_files
from .checker import check_sample_sources
from .config import (
    EvalConfig,
    HardwareProfile,
    load_eval_config,
    load_hardware_profile,
)
from .dataset import Task, load_task
from .eval_device import eval_sample_on_device
from .eval_result import fail_result
from .io_util import (
    load_cfg_argv,
    pop_required_path,
    write_json_atomic,
)
from .process import IsolatedJsonWorker
from .score import compute_pass_at_k
from .timing import l2_clear_bytes

__all__ = [
    "evaluate_run",
]


def _persist_eval_result(sample_dir: Path, result: dict[str, Any]) -> dict[str, Any]:
    """Write eval_result.json when sample_dir exists and return it."""
    if Path(sample_dir).is_dir():
        write_json_atomic(Path(sample_dir) / "eval_result.json", result)
    return result


def _load_run_settings(
    run_dir: Path,
) -> tuple[EvalConfig, HardwareProfile, Backend]:
    """Load the default protocol, the run's hardware, and its backend."""
    config = load_eval_config()
    hw_name = rundir.generation_hardware_name(run_dir) or config.hardware
    recorded = rundir.generation_backend(run_dir)
    backend = DEFAULT_BACKEND if recorded is None else recorded
    return config, load_hardware_profile(hw_name), backend


def eval_sample(
    task: Task,
    sample_dir: Path,
    *,
    hardware: HardwareProfile,
    config: EvalConfig,
    backend: Backend = DEFAULT_BACKEND,
) -> dict[str, Any]:
    """Host entry: static check, then isolated worker subprocess.

    sample_dir must already hold the backend's kernel artifact and
    model_new.py; the returned dict is also written to
    sample_dir/eval_result.json.
    """
    sample_dir = Path(sample_dir).resolve()
    kernel_name, wrapper_name = sample_files(backend)
    kernel_path = sample_dir / kernel_name
    wrapper_path = sample_dir / wrapper_name
    if not kernel_path.is_file() or not wrapper_path.is_file():
        return _persist_eval_result(
            sample_dir,
            fail_result(
                compilation_error=(
                    f"sample dir missing {kernel_name} or {wrapper_name}"
                ),
                backend=backend.value,
                failure_stage="static_check",
            ),
        )

    violations = check_sample_sources(
        backend,
        kernel_path.read_text(encoding="utf-8"),
        wrapper_path.read_text(encoding="utf-8"),
    )
    if violations:
        return _persist_eval_result(
            sample_dir,
            fail_result(
                compilation_error="; ".join(violations),
                static_check_error=violations,
                backend=backend.value,
                failure_stage="static_check",
            ),
        )

    timeout_s = config.eval_timeout + 2 * config.build_timeout
    logger.debug("eval {} ({}) in {}", task.task_id, backend.value, sample_dir)
    result = _run_eval_worker(
        {
            "task_py": task.task_py,
            "sample_dir": str(sample_dir),
            "backend": backend.value,
            "cmake_arch": hardware.cmake_arch,
            "hardware_name": hardware.name,
            "soc_version": hardware.soc_version,
            "device": "npu:0",
            "measure_performance": True,
            "seed": config.seed,
            "num_correct_trials": config.num_correct_trials,
            "num_perf_trials": config.num_perf_trials,
            "num_warmup": config.num_warmup,
            "precision": config.precision,
            "tolerances": config.tolerances,
            "excessive_speedup": config.excessive_speedup,
            "build_timeout": config.build_timeout,
            "memory_bandwidth_gbps": hardware.memory_bandwidth_gbps,
            "peak_tflops": hardware.peak_tflops_for(config.precision),
            "l2_clear_size": l2_clear_bytes(hardware.l2_cache_mb),
        },
        timeout_s=timeout_s,
    )
    return _persist_eval_result(sample_dir, result)


def evaluate_run(
    run: str | Path,
    level: int | None = None,
    *,
    on_sample: Callable[[str, int, dict[str, Any]], None] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Evaluate complete samples in a run and write aggregates.

    Args:
        run: Run directory name under runs/, or an existing path.
        level: When set, only evaluate samples under level{level}/.
        on_sample: Called with (task_id, sample_id, result) after each sample.

    Returns:
        KernelBench-compatible mapping of task id to sample result dicts.

    Raises:
        FileNotFoundError: If the run directory does not exist.
        ValueError: If the selection contains no complete samples, or the
            run records an unknown backend.
    """
    run_dir = rundir.resolve_run(run)
    config, hardware, backend = _load_run_settings(run_dir)
    samples = list(rundir.iter_sample_dirs(run_dir, level=level, backend=backend))
    if not samples:
        if level is not None:
            raise ValueError(f"no samples in {run_dir} for level {level}")
        raise ValueError(f"no samples in {run_dir}")

    harness = rundir.generation_harness(run_dir)
    for task_id, sample_id, sample_path in samples:
        result = eval_sample(
            load_task(task_id),
            sample_path,
            hardware=hardware,
            config=config,
            backend=backend,
        )
        metadata = result.get("metadata")
        if isinstance(metadata, dict):
            metadata["backend"] = backend.value
            if harness:
                metadata["harness"] = harness
            write_json_atomic(sample_path / "eval_result.json", result)
        if on_sample is not None:
            on_sample(task_id, sample_id, result)

    results = rundir.collect_eval_results(run_dir)
    rundir.write_eval_results(run_dir, results)
    rundir.write_pass_at_k(run_dir, compute_pass_at_k(results))
    return results


def worker_main(argv: list[str]) -> None:
    """Read cfg.json, evaluate one sample, and write result_path."""
    from .log import setup_logging

    setup_logging(rich_tracebacks=False)
    cfg = load_cfg_argv(argv)
    result_path = pop_required_path(cfg, "result_path")
    try:
        result = eval_sample_on_device(**cfg)
    except Exception as exc:
        # A crashed worker proves nothing about compilation, so the sample is
        # reported as not compiled with the stage recorded.
        logger.exception("eval worker crashed")
        result = fail_result(
            runtime_error=repr(exc),
            failure_stage="worker",
            backend=str(cfg.get("backend", DEFAULT_BACKEND.value)),
        )
    write_json_atomic(result_path, result)


def _run_eval_worker(cfg: dict[str, Any], timeout_s: int) -> dict[str, Any]:
    """Spawn the repo worker script and return its payload."""
    outcome = IsolatedJsonWorker.for_eval(timeout_s).run(cfg)
    if outcome.timed_out:
        return fail_result(
            runtime_error=f"eval timed out after {timeout_s}s",
            failure_stage="timeout",
            backend=str(cfg.get("backend", DEFAULT_BACKEND.value)),
        )
    if outcome.returncode != 0:
        err = outcome.stderr.strip()
        return fail_result(
            runtime_error=(
                err[-2000:] or f"worker exited with code {outcome.returncode}"
            ),
            failure_stage="worker",
            backend=str(cfg.get("backend", DEFAULT_BACKEND.value)),
        )
    if outcome.payload is not None:
        return outcome.payload
    return fail_result(
        runtime_error=outcome.parse_error or "worker produced no result.json",
        failure_stage="worker",
        backend=str(cfg.get("backend", DEFAULT_BACKEND.value)),
    )


if __name__ == "__main__":
    worker_main(sys.argv)
