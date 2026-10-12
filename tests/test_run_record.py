"""No-device tests for the run record's continuation rules.

Runs with plain asserts and no test framework:

    python tests/test_run_record.py

A new run records the planned task list once. A later request for the same run
is a continuation: it accumulates tasks and keeps every recorded value, and a
conflicting harness field is refused instead of silently replacing the record.
Nothing here imports torch or touches an NPU.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import rundir  # noqa: E402
from src.cli_util import (  # noqa: E402
    generation_run_config,
    resolve_generation_settings,
)
from src.config import BuildBackend, load_eval_config  # noqa: E402

PLANNED = ["level1/a_task", "level1/b_task", "level1/c_task"]


def check(name: str, condition: bool, detail: str = "") -> None:
    """Record one check result."""
    suffix = "" if condition else " :: " + detail
    print(f"  {'ok  ' if condition else 'FAIL'} - {name}{suffix}")
    if not condition:
        raise AssertionError(f"{name}: {detail}")


def config(**overrides: object) -> dict:
    """Return a generation config with optional overrides."""
    base = {
        "model": "deepseek-flash",
        "temperature": 0.0,
        "max_tokens": 131072,
        "prompt_mode": "one_shot",
        "num_samples": 1,
        "reasoning_effort": "high",
        "build_backend": "deepjit",
        "hardware": "ascend910b2",
        "tasks": [],
        "deepjit": {"root": "/tmp/deepjit", "commit": "deadbeef"},
    }
    base.update(overrides)
    return base


def with_runs_dir(root: Path):
    """Point rundir at a temporary runs directory."""
    return root


def test_new_run_records_the_plan() -> None:
    """A new run records the whole planned task list in one call."""
    with tempfile.TemporaryDirectory() as tmp:
        rundir.RUNS_DIR = Path(tmp)
        run_dir = rundir.create_run("one_shot_plan", config(tasks=PLANNED))
        recorded = rundir._read_generation_config(run_dir)
        check(
            "planned tasks recorded",
            recorded["tasks"] == PLANNED,
            str(recorded["tasks"]),
        )
        check("build backend recorded", recorded["build_backend"] == "deepjit")
        check(
            "harness reads back",
            rundir.generation_harness(run_dir)["model"] == "deepseek-flash",
        )
        check(
            "build backend reads back",
            rundir.generation_build_backend(run_dir).value == "deepjit",
        )
        check(
            "hardware reads back",
            rundir.generation_hardware_name(run_dir) == "ascend910b2",
        )
        check(
            "deepjit settings read back",
            rundir.generation_deepjit(run_dir)["commit"] == "deadbeef",
        )


def test_per_task_continuation_accumulates() -> None:
    """One create_run per task keeps every task and every recorded value."""
    with tempfile.TemporaryDirectory() as tmp:
        rundir.RUNS_DIR = Path(tmp)
        for task in PLANNED:
            run_dir = rundir.create_run("per_task", config(tasks=[task]))
        recorded = rundir._read_generation_config(run_dir)
        check(
            "all three tasks survive",
            recorded["tasks"] == PLANNED,
            str(recorded["tasks"]),
        )
        check("model survives", recorded["model"] == "deepseek-flash")
        check("temperature survives", recorded["temperature"] == 0.0)
        check("build backend survives", recorded["build_backend"] == "deepjit")
        check("deepjit block survives", recorded["deepjit"]["commit"] == "deadbeef")
        check(
            "the record is the one evaluation reads",
            rundir.generation_build_backend(run_dir) is BuildBackend.DEEPJIT,
        )


def test_continuation_keeps_non_identity_values() -> None:
    """A continuation adds tasks and keeps already recorded non-identity keys."""
    with tempfile.TemporaryDirectory() as tmp:
        rundir.RUNS_DIR = Path(tmp)
        rundir.create_run("keep", config(tasks=["level1/a_task"], run_notes="first"))
        run_dir = rundir.create_run(
            "keep",
            config(tasks=["level1/b_task"], run_notes="second", extra_key="added"),
        )
        recorded = rundir._read_generation_config(run_dir)
        check(
            "task list grew",
            recorded["tasks"] == ["level1/a_task", "level1/b_task"],
            str(recorded["tasks"]),
        )
        check(
            "recorded value wins",
            recorded["run_notes"] == "first",
            str(recorded["run_notes"]),
        )
        check("a new key is filled in", recorded["extra_key"] == "added")
        check("build backend still recorded", recorded["build_backend"] == "deepjit")


def test_conflicting_harness_is_refused() -> None:
    """A continuation that changes the harness is refused, not merged."""
    for label, overrides in (
        ("model", {"model": "other-model"}),
        ("build backend", {"build_backend": "cmake"}),
        ("num samples", {"num_samples": 4}),
        ("hardware", {"hardware": "ascend950pr"}),
        ("deepjit root", {"deepjit": {"root": "/other/root", "commit": "deadbeef"}}),
    ):
        with tempfile.TemporaryDirectory() as tmp:
            rundir.RUNS_DIR = Path(tmp)
            rundir.create_run("conflict", config(tasks=["level1/a_task"]))
            try:
                rundir.create_run(
                    "conflict", config(tasks=["level1/b_task"], **overrides)
                )
            except ValueError:
                check(f"conflicting {label} refused", True)
            else:
                check(f"conflicting {label} refused", False, str(overrides))
            recorded = rundir._read_generation_config(rundir.RUNS_DIR / "conflict")
            check(
                f"conflicting {label} left the record alone",
                recorded["tasks"] == ["level1/a_task"],
                str(recorded["tasks"]),
            )


def test_completed_candidate_is_protected_across_tasks() -> None:
    """A continuation that re-plans a finished task still runs on the merged plan."""
    with tempfile.TemporaryDirectory() as tmp:
        rundir.RUNS_DIR = Path(tmp)
        run_dir = rundir.create_run("protect", config(tasks=["level1/a_task"]))
        sample = rundir.sample_dir(run_dir, "level1/a_task", 0)
        sample.mkdir(parents=True)
        (sample / "custom_op.asc").write_text(
            "#include <kernel_operator.h>\n", encoding="utf-8"
        )
        (sample / "model_new.py").write_text("import torch\n", encoding="utf-8")
        try:
            rundir.create_run("protect", config(tasks=["level1/a_task"]))
        except FileExistsError:
            check("a finished task cannot be regenerated", True)
        else:
            check("a finished task cannot be regenerated", False)
        run_dir = rundir.create_run("protect", config(tasks=["level1/b_task"]))
        recorded = rundir._read_generation_config(run_dir)
        check(
            "a new task still appends",
            recorded["tasks"] == ["level1/a_task", "level1/b_task"],
            str(recorded["tasks"]),
        )


def test_explicit_none_transitions() -> None:
    """An explicit None is compared like any other identity value."""
    for label, recorded_value, requested_value, expected in (
        ("None to high", None, "high", "refused"),
        ("high to None", "high", None, "refused"),
        ("None to None", None, None, "accepted"),
        ("high to high", "high", "high", "accepted"),
        ("high to medium", "high", "medium", "refused"),
    ):
        with tempfile.TemporaryDirectory() as tmp:
            rundir.RUNS_DIR = Path(tmp)
            run_dir = rundir.create_run(
                "effort",
                config(tasks=["level1/a_task"], reasoning_effort=recorded_value),
            )
            before = (run_dir / "generation_config.yaml").read_bytes()
            try:
                rundir.create_run(
                    "effort",
                    config(tasks=["level1/b_task"], reasoning_effort=requested_value),
                )
            except ValueError:
                outcome = "refused"
            else:
                outcome = "accepted"
            check(f"{label} is {expected}", outcome == expected, outcome)
            recorded = rundir._read_generation_config(run_dir)
            check(
                f"{label} keeps the recorded value",
                recorded["reasoning_effort"] == recorded_value,
                str(recorded["reasoning_effort"]),
            )
            if outcome == "refused":
                check(
                    f"{label} leaves the record file untouched",
                    (run_dir / "generation_config.yaml").read_bytes() == before,
                )
            expected_tasks = (
                ["level1/a_task"]
                if outcome == "refused"
                else ["level1/a_task", "level1/b_task"]
            )
            check(
                f"{label} leaves the task list as {expected_tasks}",
                recorded["tasks"] == expected_tasks,
                str(recorded["tasks"]),
            )


def test_missing_identity_field_rule() -> None:
    """A record that predates a field is only continued before samples exist."""
    with tempfile.TemporaryDirectory() as tmp:
        rundir.RUNS_DIR = Path(tmp)
        old = config(tasks=["level1/a_task"])
        old.pop("reasoning_effort")
        run_dir = rundir.create_run("old_record", old)
        recorded_keys = rundir._read_generation_config(run_dir)
        check(
            "the record has no reasoning_effort",
            "reasoning_effort" not in recorded_keys,
        )
        run_dir = rundir.create_run(
            "old_record", config(tasks=["level1/b_task"], reasoning_effort="high")
        )
        recorded = rundir._read_generation_config(run_dir)
        check(
            "a sample-free record adopts the executed value",
            recorded["reasoning_effort"] == "high",
            str(recorded.get("reasoning_effort")),
        )
        check(
            "the fill is documented",
            recorded["record_filled"]["fields"] == {"reasoning_effort": "high"},
            str(recorded.get("record_filled")),
        )
        check(
            "the tasks still merge",
            recorded["tasks"] == ["level1/a_task", "level1/b_task"],
        )
    with tempfile.TemporaryDirectory() as tmp:
        rundir.RUNS_DIR = Path(tmp)
        old = config(tasks=["level1/a_task"])
        old.pop("reasoning_effort")
        run_dir = rundir.create_run("with_samples", old)
        sample = rundir.sample_dir(run_dir, "level1/a_task", 0)
        sample.mkdir(parents=True)
        (sample / "custom_op.asc").write_text(
            "#include <kernel_operator.h>\n", encoding="utf-8"
        )
        (sample / "model_new.py").write_text("import torch\n", encoding="utf-8")
        before = (run_dir / "generation_config.yaml").read_bytes()
        try:
            rundir.create_run(
                "with_samples", config(tasks=["level1/b_task"], reasoning_effort="high")
            )
        except ValueError as exc:
            check(
                "a record with samples refuses the new field",
                "predates the reasoning_effort field" in str(exc),
                str(exc),
            )
        else:
            check("a record with samples refuses the new field", False, "accepted")
        check(
            "the refusal left the record alone",
            (run_dir / "generation_config.yaml").read_bytes() == before,
        )
    with tempfile.TemporaryDirectory() as tmp:
        rundir.RUNS_DIR = Path(tmp)
        rundir.create_run(
            "request_omits", config(tasks=["level1/a_task"], reasoning_effort="high")
        )
        run_dir = rundir.create_run("request_omits", config(tasks=["level1/b_task"]))
        recorded = rundir._read_generation_config(run_dir)
        check(
            "an omitted request field keeps the record",
            recorded["reasoning_effort"] == "high",
            str(recorded.get("reasoning_effort")),
        )


def test_refusal_keeps_candidates() -> None:
    """A refused continuation cannot touch the samples already on disk."""
    with tempfile.TemporaryDirectory() as tmp:
        rundir.RUNS_DIR = Path(tmp)
        run_dir = rundir.create_run("keep_candidate", config(tasks=["level1/a_task"]))
        sample = rundir.sample_dir(run_dir, "level1/a_task", 0)
        sample.mkdir(parents=True)
        candidate = sample / "custom_op.asc"
        candidate.write_text(
            "#include <kernel_operator.h>\n// keeps this\n", encoding="utf-8"
        )
        record = run_dir / "generation_config.yaml"
        before_record, before_candidate = record.read_bytes(), candidate.read_bytes()
        try:
            rundir.create_run(
                "keep_candidate", config(tasks=["level1/a_task"], model="other")
            )
        except ValueError:
            pass
        check(
            "the record is untouched after a refusal",
            record.read_bytes() == before_record,
        )
        check(
            "the candidate is untouched after a refusal",
            candidate.read_bytes() == before_candidate,
        )


def test_recorded_keys_are_all_identity_checked() -> None:
    """Every harness key a run records is compared on continuation."""
    settings = resolve_generation_settings(
        load_eval_config(),
        build_backend="deepjit",
        deepjit_root="/tmp/d",
    )
    record = generation_run_config(
        settings, hardware_name="ascend910b2", task_ids=["level1/a_task"]
    )
    compared = set(rundir._RUN_IDENTITY_KEYS)
    # The backend is compared by _check_run_compatible, which create_run always
    # calls on an existing run and which keeps the established message.
    covered = compared | {"backend"}
    uncovered = {key for key in record if key not in covered and key != "tasks"}
    check(
        "recorded harness keys are all identity-checked",
        not uncovered,
        str(sorted(uncovered)),
    )
    generator_keys = {
        "model",
        "temperature",
        "max_tokens",
        "prompt_mode",
        "num_samples",
        "reasoning_effort",
        "build_backend",
        "deepjit",
    }
    check(
        "every generator parameter is compared on continuation",
        generator_keys <= covered,
        str(sorted(generator_keys - covered)),
    )
    check(
        "the recorded build backend is present",
        record["build_backend"] == BuildBackend.DEEPJIT.value,
    )


def main() -> int:
    """Run every run-record check."""
    print("run record checks")
    for test in (
        test_new_run_records_the_plan,
        test_per_task_continuation_accumulates,
        test_continuation_keeps_non_identity_values,
        test_conflicting_harness_is_refused,
        test_completed_candidate_is_protected_across_tasks,
        test_explicit_none_transitions,
        test_missing_identity_field_rule,
        test_refusal_keeps_candidates,
        test_recorded_keys_are_all_identity_checked,
    ):
        test()
    print("\nall run record checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
