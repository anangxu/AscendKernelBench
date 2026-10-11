"""Call-chain test: a continuation executes the harness a run recorded.

Drives scripts/generate.py with a recording stand-in for the LLM client, so no
model is called and no NPU is involved. The assertions are on the parameters
that reach the generation boundary, not on the record merger:

    python tests/test_generate_call_chain.py

The run is a real one: an old record is stamped with explicit generation
settings, then a continuation request omits those flags and must still execute
the recorded values while the record stays consistent.
"""

from __future__ import annotations

import io
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import generate  # noqa: E402
from src import rundir  # noqa: E402
from src.backend import Backend, BuildBackend  # noqa: E402
from src.cli_util import generation_run_config, resolve_generation_settings  # noqa: E402
from src.config import load_eval_config  # noqa: E402
from src.llm import AscendCGeneration, GenerationResult  # noqa: E402

TASK = "level1/88_MinGPTNewGelu"
RUN = "chain_run"
RECORDED = {
    "model": "record-model",
    "prompt_mode": "few_shot",
    "temperature": 0.3,
    "max_tokens": 4096,
    "num_samples": 1,
    "reasoning_effort": "medium",
    "backend": "ascendc",
    "build_backend": "deepjit",
    "hardware": "ascend910b2",
    "tasks": [TASK],
}
DEEPJIT_ROOT = "/tmp/deepjit-checkout"
PIN = "3732a3b9a0a2e5126396908a362ba5abdf043b2f"


def check(name: str, condition: bool, detail: str = "") -> None:
    """Record one check result."""
    print(
        f"  {'ok  ' if condition else 'FAIL'} - {name}{'' if condition else ' :: ' + detail}"
    )
    if not condition:
        raise AssertionError(f"{name}: {detail}")


class RecordingClient:
    """Stand-in for LLMClient that captures its own construction arguments."""

    calls: list[dict[str, Any]] = []

    def __init__(self, model: str, **kwargs: object) -> None:
        """Record the parameters the generator passes at the call boundary."""
        RecordingClient.calls.append({"model": model, **kwargs})

    def generate(self, prompt: str, *, system: str | None = None) -> GenerationResult:
        """Return a fixed candidate instead of calling a model."""
        return GenerationResult(
            generation=AscendCGeneration(
                custom_op_asc="#include <kernel_operator.h>\n",
                model_new_py="import torch\n",
            ),
            raw_text="recorded by the test",
            model=self.calls[-1]["model"],
            usage={},
        )


def recording_build_prompt(calls: list[dict[str, Any]]) -> Any:
    """Return a build_prompt replacement that records its arguments."""

    def build_prompt(task: object, hardware: object, **kwargs: object) -> str:
        calls.append({"task": getattr(task, "task_id", ""), **kwargs})
        return "prompt"

    return build_prompt


@contextmanager
def generation_call(argv: list[str]) -> Any:
    """Run generate.main() with recording seams and a temporary runs root."""
    original_argv, original_client = sys.argv, generate.LLMClient
    original_prompt, original_dir = generate.build_prompt, rundir.RUNS_DIR
    original_progress, original_stdout = generate.cli_progress, sys.stdout
    prompt_calls: list[dict[str, Any]] = []
    RecordingClient.calls = []
    with tempfile.TemporaryDirectory() as tmp:
        rundir.RUNS_DIR = Path(tmp)
        sys.argv = ["generate.py", *argv]
        generate.LLMClient = RecordingClient
        generate.build_prompt = recording_build_prompt(prompt_calls)
        generate.cli_progress = lambda console: _null_progress()
        sys.stdout = io.StringIO()
        try:
            yield rundir.RUNS_DIR, prompt_calls, RecordingClient.calls
        finally:
            sys.argv, generate.LLMClient = original_argv, original_client
            generate.build_prompt, rundir.RUNS_DIR = original_prompt, original_dir
            generate.cli_progress, sys.stdout = original_progress, original_stdout


class NullProgress:
    """Progress stand-in that draws nothing and tracks nothing."""

    def add_task(self, description: str, total: int | None = None) -> int:
        """Return a fixed task id."""
        return 0

    def update(self, task_id: int, **kwargs: object) -> None:
        """Ignore progress updates."""

    def advance(self, task_id: int, advance: int = 1) -> None:
        """Ignore progress advances."""


@contextmanager
def _null_progress() -> Any:
    """Yield a progress object without a terminal."""
    yield NullProgress()


def _run_main() -> int:
    """Run generate.main() and return the exit code it raises."""
    try:
        generate.main()
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


def stamp_old_run(runs_dir: Path) -> Path:
    """Create the old run record through the same path the CLI uses."""
    settings = resolve_generation_settings(
        load_eval_config(),
        model=RECORDED["model"],
        prompt_mode=RECORDED["prompt_mode"],
        temperature=RECORDED["temperature"],
        num_samples=RECORDED["num_samples"],
        reasoning_effort=RECORDED["reasoning_effort"],
        max_tokens=RECORDED["max_tokens"],
        backend=RECORDED["backend"],
        build_backend=RECORDED["build_backend"],
        deepjit_root=DEEPJIT_ROOT,
        deepjit_commit=PIN,
    )
    payload = generation_run_config(
        settings, hardware_name=RECORDED["hardware"], task_ids=[TASK]
    )
    return rundir.create_run(RUN, payload)


def test_omitted_flags_execute_recorded_values() -> None:
    """An omitted flag reaches the model boundary as the recorded value."""
    with generation_call(["--task", TASK, "--run-name", RUN]) as (
        runs_dir,
        prompts,
        calls,
    ):
        stamp_old_run(runs_dir)
        before = rundir._read_generation_config(runs_dir / RUN)
        exit_code = _run_main()
        recorded_after = rundir._read_generation_config(runs_dir / RUN)
        sample = rundir.sample_dir(runs_dir / RUN, TASK, 0)
        sample_written = (sample / "custom_op.asc").is_file()
    check("the generation run succeeded", exit_code == 0, str(exit_code))

    check("the model was called once", len(calls) == 1, str(len(calls)))
    boundary = calls[0]
    check(
        "reasoning_effort reaches the model as the recorded value",
        boundary.get("reasoning_effort") == RECORDED["reasoning_effort"],
        str(boundary.get("reasoning_effort")),
    )
    check(
        "model reaches the model as the recorded value",
        boundary.get("model") == RECORDED["model"],
        str(boundary.get("model")),
    )
    check(
        "temperature reaches the model as the recorded value",
        boundary.get("temperature") == RECORDED["temperature"],
        str(boundary.get("temperature")),
    )
    check(
        "max_tokens reaches the model as the recorded value",
        boundary.get("max_tokens") == RECORDED["max_tokens"],
        str(boundary.get("max_tokens")),
    )
    check(
        "the backend reaches the model as the recorded value",
        boundary.get("backend") == Backend.ASCENDC,
        str(boundary.get("backend")),
    )
    check("the prompt was built once", len(prompts) == 1, str(len(prompts)))
    check(
        "the prompt uses the recorded prompt_mode",
        prompts[0].get("mode") == RECORDED["prompt_mode"],
        str(prompts[0].get("mode")),
    )
    check(
        "the prompt uses the recorded build backend",
        prompts[0].get("build_backend") == BuildBackend.DEEPJIT,
        str(prompts[0].get("build_backend")),
    )
    check(
        "the record keeps the harness it recorded",
        all(recorded_after.get(key) == value for key, value in RECORDED.items()),
        str({key: recorded_after.get(key) for key in RECORDED}),
    )
    check(
        "the record keeps its deepjit root and pin",
        recorded_after.get("deepjit", {}).get("root") == DEEPJIT_ROOT
        and recorded_after.get("deepjit", {}).get("commit") == PIN,
        str(recorded_after.get("deepjit")),
    )
    check(
        "the record is not restamped by the continuation",
        recorded_after == before,
        str(sorted(set(recorded_after) ^ set(before))),
    )
    check("the sample was written", sample_written)


def test_explicit_flag_that_differs_is_refused() -> None:
    """An explicitly different flag is refused instead of executed."""
    with generation_call(
        ["--task", TASK, "--run-name", RUN, "--reasoning-effort", "high"]
    ) as (runs_dir, prompts, calls):
        run_dir = stamp_old_run(runs_dir)
        before = (run_dir / "generation_config.yaml").read_bytes()
        try:
            generate.main()
        except SystemExit as exc:
            check("the conflicting request exits cleanly", exc.code == 1, str(exc.code))
        else:
            check("the conflicting request exits cleanly", False, "no exit")
        after = (run_dir / "generation_config.yaml").read_bytes()

    check("no model call was made", not calls, str(len(calls)))
    check("no prompt was built", not prompts, str(len(prompts)))
    check("the record is unchanged after the refusal", before == after)


def test_a_new_run_keeps_the_configured_defaults() -> None:
    """Without a record the CLI still uses the configured defaults."""
    with generation_call(["--task", TASK, "--run-name", "fresh_run"]) as (
        runs_dir,
        prompts,
        calls,
    ):
        exit_code = _run_main()
        recorded = rundir._read_generation_config(runs_dir / "fresh_run")
    check("the fresh run succeeded", exit_code == 0, str(exit_code))

    check(
        "the default model is used",
        calls[0].get("model") == "deepseek-flash",
        str(calls[0]),
    )
    check(
        "the default reasoning effort is used",
        calls[0].get("reasoning_effort") == "high",
        str(calls[0].get("reasoning_effort")),
    )
    check(
        "the fresh record matches the executed defaults",
        recorded.get("reasoning_effort") == calls[0].get("reasoning_effort")
        and recorded.get("model") == calls[0].get("model"),
        str({key: recorded.get(key) for key in ("model", "reasoning_effort")}),
    )
    check(
        "the fresh run is recorded as the YAML build backend",
        recorded.get("build_backend") == BuildBackend.CMAKE.value,
        str(recorded.get("build_backend")),
    )


def main() -> int:
    """Run every call-chain check."""
    print("generate call-chain checks")
    test_omitted_flags_execute_recorded_values()
    test_explicit_flag_that_differs_is_refused()
    test_a_new_run_keeps_the_configured_defaults()
    print("\nall generate call-chain checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
