#!/usr/bin/env python
"""Generate n samples per task into runs/{run_name}/.

Samples land on disk and can be evaluated later, on this or another machine.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import _bootstrap  # noqa: F401
from loguru import logger
from rich.console import Console

from src import rundir
from src.cli_util import (
    cli_progress,
    generation_run_config,
    load_eval_runtime,
    resolve_generation_settings,
    select_tasks,
)
from src.config import BuildBackend
from src.llm import LLMClient
from src.log import die, setup_logging
from src.prompt import build_prompt, system_prompt

console = Console()


def inherited_overrides(
    args: argparse.Namespace, recorded: dict[str, object]
) -> dict[str, object]:
    """Return settings overrides, filling omitted flags from a run record.

    A continuation can omit a flag: the recorded value then drives the
    generation call, so the run executes what it recorded instead of a YAML
    default. An explicitly passed flag still wins, and opening the run then
    checks it against the record.
    """
    deepjit = recorded.get("deepjit")
    deepjit = deepjit if isinstance(deepjit, dict) else {}
    numeric = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "num_samples": args.n_samples,
    }
    overrides: dict[str, object] = {
        key: value if value is not None else recorded.get(key)
        for key, value in numeric.items()
    }
    overrides.update(
        {
            "model": args.model or recorded.get("model"),
            "prompt_mode": args.prompt_mode or recorded.get("prompt_mode"),
            "reasoning_effort": (
                args.reasoning_effort or recorded.get("reasoning_effort")
            ),
            "build_backend": args.build_backend or recorded.get("build_backend"),
            "deepjit_root": args.deepjit_root or deepjit.get("root"),
            "deepjit_commit": args.deepjit_commit or deepjit.get("commit"),
        }
    )
    return overrides


def main() -> None:
    """Generate one or more samples per selected task."""
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", type=int, default=None)
    parser.add_argument(
        "--task",
        action="append",
        default=None,
        help="task id(s); overrides --level",
    )
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--hardware", default=None)
    parser.add_argument(
        "--tasks-file",
        default=None,
        help="manifest of task ids, one per line; see configs/subsets/",
    )
    parser.add_argument(
        "--prompt-mode",
        default=None,
        choices=["zero_shot", "one_shot", "few_shot"],
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        choices=["low", "medium", "high"],
        help="thinking depth; omitted when unset",
    )
    parser.add_argument(
        "--build-backend",
        default=None,
        choices=[item.value for item in BuildBackend],
        help="how custom_op.asc is built: cmake (default) or deepjit",
    )
    parser.add_argument(
        "--deepjit-root",
        default=None,
        help="DeepJIT checkout used by --build-backend deepjit",
    )
    parser.add_argument(
        "--deepjit-commit",
        default=None,
        help="pinned DeepJIT commit; a mismatch with the checkout is refused",
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    runtime = load_eval_runtime(
        config_path=args.config,
        hardware=args.hardware,
    )
    config, hardware = runtime.config, runtime.hardware
    run_name = args.run_name or f"gen_{datetime.now():%Y%m%d_%H%M%S}"
    recorded = rundir.recorded_generation_defaults(rundir.RUNS_DIR / run_name)
    try:
        settings = resolve_generation_settings(
            config, **inherited_overrides(args, recorded)
        )
    except ValueError as exc:
        die(str(exc))

    tasks = select_tasks(
        level=args.level,
        task_ids=args.task,
        tasks_file=args.tasks_file,
    )
    if not tasks:
        die("no tasks found")

    hardware_file = None
    hardware_spec = args.hardware or config.hardware
    candidate = Path(hardware_spec)
    if candidate.is_file():
        hardware_file = str(candidate.resolve())

    try:
        run_dir = rundir.create_run(
            run_name,
            generation_run_config(
                settings,
                hardware_name=hardware.name,
                task_ids=[task.task_id for task in tasks],
                hardware_file=hardware_file,
            ),
        )
    except (ValueError, FileExistsError) as exc:
        die(str(exc))
    logger.info(
        "run dir: {} ({} tasks x {} samples)",
        run_dir,
        len(tasks),
        settings.num_samples,
    )

    client = LLMClient(
        settings.model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        reasoning_effort=settings.reasoning_effort,
    )
    system = system_prompt(settings.build_backend)
    total = len(tasks) * settings.num_samples
    failures = 0
    with cli_progress(console) as progress:
        bar = progress.add_task("generating", total=total)
        for task in tasks:
            prompt = build_prompt(
                task,
                hardware,
                mode=settings.prompt_mode,
                build_backend=settings.build_backend,
            )
            for sample_id in range(settings.num_samples):
                progress.update(bar, description=f"{task.task_id} s{sample_id}")
                try:
                    result = client.generate(prompt, system=system)
                    rundir.save_sample(
                        run_dir,
                        task.task_id,
                        sample_id,
                        prompt=prompt,
                        generation=result.generation,
                        raw_response=result.raw_text,
                    )
                except Exception as exc:
                    failures += 1
                    logger.error(
                        "generate failed {} sample {}: {}",
                        task.task_id,
                        sample_id,
                        exc,
                    )
                progress.advance(bar)
    logger.info("done: {}/{} samples saved to {}", total - failures, total, run_dir)
    raise SystemExit(1 if failures == total else 0)


if __name__ == "__main__":
    main()
