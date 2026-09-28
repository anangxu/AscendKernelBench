#!/usr/bin/env python
"""Generate n samples per task into runs/{run_name}/.

Samples land on disk and can be evaluated later, on this or another machine.
"""

from __future__ import annotations

import argparse
from datetime import datetime

import _bootstrap  # noqa: F401
from loguru import logger
from rich.console import Console

from src import rundir
from src.backend import Backend
from src.cli_util import (
    cli_progress,
    generation_run_config,
    load_eval_runtime,
    resolve_generation_settings,
    select_tasks,
)
from src.llm import LLMClient
from src.log import die, setup_logging
from src.prompt import PYASC_SYSTEM_PROMPT, SYSTEM_PROMPT, build_prompt

console = Console()


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
        "--backend",
        default=None,
        choices=[backend.value for backend in Backend],
        help="authoring language: ascendc (default) or pyasc",
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    runtime = load_eval_runtime(
        config_path=args.config,
        hardware=args.hardware,
    )
    config, hardware = runtime.config, runtime.hardware
    try:
        settings = resolve_generation_settings(
            config,
            model=args.model,
            prompt_mode=args.prompt_mode,
            temperature=args.temperature,
            num_samples=args.n_samples,
            reasoning_effort=args.reasoning_effort,
            max_tokens=args.max_tokens,
            backend=args.backend,
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

    run_name = args.run_name or f"gen_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = rundir.create_run(
        run_name,
        generation_run_config(
            settings,
            hardware_name=hardware.name,
            task_ids=[task.task_id for task in tasks],
        ),
    )
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
        backend=settings.backend,
    )
    system_prompt = (
        PYASC_SYSTEM_PROMPT if settings.backend is Backend.PYASC else SYSTEM_PROMPT
    )
    total = len(tasks) * settings.num_samples
    failures = 0
    with cli_progress(console) as progress:
        bar = progress.add_task("generating", total=total)
        for task in tasks:
            prompt = build_prompt(
                task,
                hardware,
                mode=settings.prompt_mode,
                backend=settings.backend,
            )
            for sample_id in range(settings.num_samples):
                progress.update(bar, description=f"{task.task_id} s{sample_id}")
                try:
                    result = client.generate(prompt, system=system_prompt)
                    rundir.save_sample(
                        run_dir,
                        task.task_id,
                        sample_id,
                        prompt=prompt,
                        generation=result.generation,
                        raw_response=result.raw_text,
                        backend=settings.backend,
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
