"""No-device tests for the deepjit evaluation wiring.

Runs with plain asserts and no test framework:

    python tests/test_deepjit_run_flow.py

The worker subprocess is replaced by a recorder, so the whole host-side chain
runs offline: the generation helpers stamp a run, the evaluation resolves the
build path from that record, the worker payload carries it, and the result
metadata reflects what actually ran. No torch, no NPU, no model call.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import eval as eval_module  # noqa: E402
from src import rundir  # noqa: E402
from src.cli_util import (  # noqa: E402
    generation_run_config,
    resolve_generation_settings,
)
from src.config import (  # noqa: E402
    BuildBackend,
    load_eval_config,
    load_hardware_profile,
)

TASK = "level1/88_MinGPTNewGelu"
EXAMPLE = ROOT / "src" / "prompts" / "examples" / "deepjit" / "001_elementwise_add"
DEEPJIT_ROOT = "/tmp/deepjit-checkout"


def check(name: str, condition: bool, detail: str = "") -> None:
    """Record one check result."""
    suffix = "" if condition else " :: " + detail
    print(f"  {'ok  ' if condition else 'FAIL'} - {name}{suffix}")
    if not condition:
        raise AssertionError(f"{name}: {detail}")


def stamp_run(run_name: str, *, build_backend: str, deepjit_root: str = "") -> Path:
    """Create a run record through the generation helpers."""
    settings = resolve_generation_settings(
        load_eval_config(),
        build_backend=build_backend,
        deepjit_root=deepjit_root,
    )
    run_dir = rundir.create_run(
        run_name,
        generation_run_config(settings, hardware_name="ascend910b2", task_ids=[TASK]),
    )
    sample = rundir.sample_dir(run_dir, TASK, 0)
    sample.mkdir(parents=True, exist_ok=True)
    (sample / "custom_op.asc").write_text(
        (EXAMPLE / "custom_op.asc").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (sample / "model_new.py").write_text(
        (EXAMPLE / "model_new.py").read_text(encoding="utf-8"), encoding="utf-8"
    )
    return run_dir


def evaluate_with_recorder(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate a run while recording the worker payload and result."""
    calls: list[dict[str, Any]] = []
    original = eval_module._run_eval_worker

    def recorder(cfg: dict[str, Any], timeout_s: int) -> dict[str, Any]:
        calls.append(cfg)
        return {
            "compiled": True,
            "correctness": True,
            "runtime": 0.5,
            "ref_runtime": 1.0,
            "metadata": {"build_seconds": 0.1},
        }

    eval_module._run_eval_worker = recorder
    try:
        results = eval_module.evaluate_run(run_dir)
    finally:
        eval_module._run_eval_worker = original
    check("the worker was called once", len(calls) == 1, str(len(calls)))
    sample = rundir.sample_dir(run_dir, TASK, 0)
    stored = rundir.load_eval_result(sample) or {}
    check("the run produced one result", bool(results), str(sorted(results)))
    return calls[0], stored


def test_deepjit_run_reaches_the_worker() -> None:
    """A deepjit record drives the evaluation payload and the result."""
    with tempfile.TemporaryDirectory() as tmp:
        rundir.RUNS_DIR = Path(tmp)
        run_dir = stamp_run(
            "dj_run", build_backend="deepjit", deepjit_root=DEEPJIT_ROOT
        )
        payload, stored = evaluate_with_recorder(run_dir)
    check(
        "the worker payload carries deepjit",
        payload.get("build_backend") == BuildBackend.DEEPJIT.value,
        str(payload.get("build_backend")),
    )
    check(
        "the worker payload carries the recorded checkout",
        (payload.get("deepjit") or {}).get("root") == DEEPJIT_ROOT,
        str(payload.get("deepjit")),
    )
    check(
        "the worker payload carries a SoC field",
        "soc_version" in payload,
        str(sorted(payload)),
    )
    check(
        "the worker keeps the top-level build timeout",
        payload.get("build_timeout") == load_eval_config().build_timeout,
        str(payload.get("build_timeout")),
    )
    hardware = load_hardware_profile(load_eval_config().hardware)
    check(
        "the payload keeps the recorded arch",
        payload.get("cmake_arch") == hardware.cmake_arch,
        str(payload.get("cmake_arch")),
    )
    metadata = stored.get("metadata") or {}
    check(
        "the stored result names the build path",
        metadata.get("build_backend") == BuildBackend.DEEPJIT.value,
        str(metadata.get("build_backend")),
    )


def test_cmake_run_is_the_default_path() -> None:
    """A run without a build backend evaluates through cmake."""
    with tempfile.TemporaryDirectory() as tmp:
        rundir.RUNS_DIR = Path(tmp)
        run_dir = stamp_run("cmake_run", build_backend="cmake")
        payload, stored = evaluate_with_recorder(run_dir)
    check(
        "the worker payload defaults to cmake",
        payload.get("build_backend") == BuildBackend.CMAKE.value,
        str(payload.get("build_backend")),
    )
    deepjit_block = payload.get("deepjit") or {}
    check(
        "an unselected path carries an unconfigured deepjit block",
        deepjit_block.get("root") == "" and deepjit_block.get("commit") == "",
        str(deepjit_block),
    )
    metadata = stored.get("metadata") or {}
    check(
        "the stored result names cmake",
        metadata.get("build_backend") == BuildBackend.CMAKE.value,
        str(metadata.get("build_backend")),
    )


def test_deepjit_without_a_checkout_is_refused() -> None:
    """A deepjit run with no checkout to build against fails explicitly."""
    with tempfile.TemporaryDirectory() as tmp:
        rundir.RUNS_DIR = Path(tmp)
        run_dir = stamp_run("dj_empty", build_backend="deepjit")
        try:
            eval_module._load_run_settings(run_dir)
        except ValueError as exc:
            check(
                "the missing checkout is named",
                "needs a DeepJIT checkout" in str(exc),
                str(exc),
            )
        else:
            check("the missing checkout is named", False, "accepted")


def main() -> int:
    """Run every evaluation-wiring check."""
    print("deepjit run-flow checks")
    test_deepjit_run_reaches_the_worker()
    test_cmake_run_is_the_default_path()
    test_deepjit_without_a_checkout_is_refused()
    print("\nall deepjit run-flow checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
