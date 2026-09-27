"""No-device tests for the pyasc evaluation wiring.

Runs with plain asserts and no test framework:

    python tests/test_eval_backend.py

Nothing here imports torch, pyasc, or touches an NPU: the pyasc runtime is
stubbed, so the classification logic is exercised on any machine.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import eval_device  # noqa: E402
from src.backend import Backend  # noqa: E402
from src.eval_device import DeviceEvalRequest, SampleEvaluator  # noqa: E402

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    """Record one assertion result."""
    if condition:
        print(f"ok - {label}")
    else:
        print(f"FAIL - {label} {detail}")
        FAILURES.append(label)


def make_request(sample_dir: Path, backend: str) -> DeviceEvalRequest:
    """Build a minimal worker request for one sample directory."""
    return DeviceEvalRequest(
        task_py="",
        sample_dir=str(sample_dir),
        cmake_arch="dav-2201",
        hardware_name="ascend910b2",
        device="npu:0",
        measure_performance=False,
        seed=42,
        num_correct_trials=1,
        num_perf_trials=1,
        num_warmup=0,
        precision="fp32",
        tolerances={},
        excessive_speedup=10.0,
        build_timeout=1,
        backend=backend,
    )


def test_request_backend_field() -> None:
    """The request carries a backend and defaults to Ascend C."""
    with tempfile.TemporaryDirectory() as tmp:
        default = DeviceEvalRequest(
            task_py="",
            sample_dir=tmp,
            cmake_arch="dav-2201",
            hardware_name="ascend910b2",
            device="npu:0",
            measure_performance=False,
            seed=1,
            num_correct_trials=1,
            num_perf_trials=1,
            num_warmup=0,
            precision="fp32",
            tolerances={},
            excessive_speedup=10.0,
            build_timeout=1,
        )
        check("request default backend is ascendc", default.backend == "ascendc")
        check(
            "request accepts pyasc",
            make_request(Path(tmp), "pyasc").backend == "pyasc",
        )
        evaluator = SampleEvaluator(make_request(Path(tmp), "pyasc"))
        check(
            "evaluator resolves the Backend enum",
            evaluator.backend is Backend.PYASC,
            repr(evaluator.backend),
        )
        try:
            SampleEvaluator(make_request(Path(tmp), "nonsense"))
            check("unknown backend rejected", False, "no error raised")
        except ValueError as exc:
            check("unknown backend rejected", "nonsense" in str(exc), str(exc))


def test_pyasc_missing_kernel_file() -> None:
    """A missing kernel.py is a classified module-load failure, not a crash."""
    with tempfile.TemporaryDirectory() as tmp:
        sample = Path(tmp)
        (sample / "model_new.py").write_text("class ModelNew: pass\n")
        evaluator = SampleEvaluator(make_request(sample, "pyasc"))
        result = evaluator._load_pyasc()
        check("missing kernel.py returns a failure payload", result is not None)
        assert result is not None
        check("missing kernel.py is not compiled", result["compiled"] is False)
        metadata = result["metadata"]
        check(
            "missing kernel.py is a module_load failure",
            metadata.get("failure_stage") == "module_load",
            repr(metadata.get("failure_stage")),
        )
        check(
            "failure records the backend",
            metadata.get("backend") == "pyasc",
            repr(metadata.get("backend")),
        )


def test_pyasc_load_with_stubs() -> None:
    """A stub asc module plus a stub loader means load succeeds cheaply."""
    with tempfile.TemporaryDirectory() as tmp:
        sample = Path(tmp)
        (sample / "kernel.py").write_text("# stub\n")
        evaluator = SampleEvaluator(make_request(sample, "pyasc"))

        original = (
            eval_device.prepare_cache_dir,
            eval_device.import_asc,
            eval_device.configure_platform,
            eval_device.load_kernel_module,
        )
        eval_device.prepare_cache_dir = lambda path: Path(path) / "cache"
        eval_device.import_asc = lambda: SimpleNamespace(name="asc")
        eval_device.configure_platform = lambda asc, index: "Ascend910B4"
        eval_device.load_kernel_module = lambda path, kernel: SimpleNamespace()
        try:
            result = evaluator._load_pyasc()
        finally:
            (
                eval_device.prepare_cache_dir,
                eval_device.import_asc,
                eval_device.configure_platform,
                eval_device.load_kernel_module,
            ) = original
        check("stubbed load succeeds", result is None, repr(result))
        check(
            "load records build_mode=pyasc",
            evaluator.metadata.get("build_mode") == "pyasc",
            repr(evaluator.metadata.get("build_mode")),
        )
        check(
            "load records the detected SOC version",
            evaluator.metadata.get("pyasc_soc_version") == "Ascend910B4",
            repr(evaluator.metadata.get("pyasc_soc_version")),
        )
        check(
            "import alone does not mark the sample compiled",
            "compiled" not in evaluator.metadata,
        )


def test_warm_up_is_a_noop_for_ascendc() -> None:
    """Ascend C never runs the pyasc warm-up."""
    with tempfile.TemporaryDirectory() as tmp:
        evaluator = SampleEvaluator(make_request(Path(tmp), "ascendc"))
        check("ascendc warm-up is a no-op", evaluator._warm_up_pyasc() is None)


def test_first_call_failure_is_a_jit_failure() -> None:
    """A failing first call is classified as jit and never as compiled."""
    with tempfile.TemporaryDirectory() as tmp:
        evaluator = SampleEvaluator(make_request(Path(tmp), "pyasc"))
        evaluator.torch = SimpleNamespace(npu=SimpleNamespace(synchronize=lambda: None))
        evaluator.get_inputs = lambda: []
        evaluator._process_input = lambda value: value

        def boom(*args: object, **kwargs: object) -> None:
            raise RuntimeError("kernel trace failed")

        evaluator.new_model = boom
        result = evaluator._warm_up_pyasc()
        check("first-call failure returns a payload", result is not None)
        assert result is not None
        metadata = result["metadata"]
        check("first-call failure is not compiled", result["compiled"] is False)
        check(
            "first-call failure stage is jit",
            metadata.get("failure_stage") == "jit",
            repr(metadata.get("failure_stage")),
        )
        check(
            "first-call failure records jit_error",
            "kernel trace failed" in str(metadata.get("jit_error")),
            repr(metadata.get("jit_error")),
        )


def test_first_call_success_records_seconds_and_rebuilds() -> None:
    """A good first call records first_call_seconds and rebuilds the models."""
    with tempfile.TemporaryDirectory() as tmp:
        evaluator = SampleEvaluator(make_request(Path(tmp), "pyasc"))
        evaluator.torch = SimpleNamespace(npu=SimpleNamespace(synchronize=lambda: None))
        evaluator.get_inputs = lambda: []
        evaluator._process_input = lambda value: value
        evaluator.new_model = lambda *args: None
        rebuilt = {"count": 0}

        def fake_construct() -> None:
            rebuilt["count"] += 1
            evaluator.new_model = "rebuilt"
            return None

        evaluator._construct_models = fake_construct
        result = evaluator._warm_up_pyasc()
        check("successful warm-up returns None", result is None, repr(result))
        check(
            "first_call_seconds recorded",
            isinstance(evaluator.metadata.get("first_call_seconds"), float),
            repr(evaluator.metadata.get("first_call_seconds")),
        )
        check("models were rebuilt", rebuilt["count"] == 1)
        check("rebuild replaced the model", evaluator.new_model == "rebuilt")


def test_pyasc_timing_validation() -> None:
    """The event-versus-wall cross-check gates the reported speedup."""
    import time as _time

    def evaluator_with(runtime: float | None, sleep_s: float) -> SampleEvaluator:
        tmp = tempfile.mkdtemp()
        ev = SampleEvaluator(make_request(Path(tmp), "pyasc"))
        ev.torch = SimpleNamespace(
            no_grad=lambda: __import__("contextlib").nullcontext(),
            npu=SimpleNamespace(synchronize=lambda: None),
        )
        ev.get_inputs = lambda: []
        ev._process_input = lambda value: value
        ev.new_model = lambda *args: _time.sleep(sleep_s)
        ev.runtime = runtime
        ev.runtime_stats = {"mean": runtime}
        ev.ref_runtime = 1.0
        ev.ref_runtime_stats = {"mean": 1.0}
        ev.metadata = {"speedup": 3.0}
        return ev

    ascendc = SampleEvaluator(make_request(Path(tempfile.mkdtemp()), "ascendc"))
    ascendc.metadata = {}
    ascendc._validate_pyasc_timing()
    check("timing validation is a no-op for ascendc", "timing_valid" not in ascendc.metadata)

    no_runtime = evaluator_with(None, 0.001)
    no_runtime._validate_pyasc_timing()
    check("no runtime means nothing to validate", "timing_valid" not in no_runtime.metadata)

    healthy = evaluator_with(2.0, 0.002)
    healthy._validate_pyasc_timing()
    check("healthy event/wall ratio stays valid", healthy.metadata.get("timing_valid") is True)
    check(
        "healthy sample records the ratio",
        healthy.metadata.get("timing_event_wall_ratio") is not None,
    )
    check("healthy sample keeps its runtime", healthy.runtime == 2.0)

    collapsed = evaluator_with(0.001, 0.005)
    collapsed._validate_pyasc_timing()
    check(
        "collapsed event time invalidates timing",
        collapsed.metadata.get("timing_valid") is False,
        repr(collapsed.metadata.get("timing_valid")),
    )
    check("invalid timing drops the runtime", collapsed.runtime is None)
    check("invalid timing drops the reference runtime", collapsed.ref_runtime is None)
    check("invalid timing drops the speedup", "speedup" not in collapsed.metadata)
    check(
        "invalid timing records a reason",
        bool(collapsed.metadata.get("timing_invalid_reason")),
    )


def test_ascendc_build_path_untouched() -> None:
    """The Ascend C branch still reads custom_op.asc as before."""
    with tempfile.TemporaryDirectory() as tmp:
        evaluator = SampleEvaluator(make_request(Path(tmp), "ascendc"))
        try:
            evaluator._build_and_load()
        except FileNotFoundError:
            check("ascendc path still reads custom_op.asc", True)
        else:
            check(
                "ascendc path still reads custom_op.asc",
                False,
                "expected FileNotFoundError for a missing custom_op.asc",
            )


def main() -> int:
    test_request_backend_field()
    test_pyasc_missing_kernel_file()
    test_pyasc_load_with_stubs()
    test_warm_up_is_a_noop_for_ascendc()
    # The warm-up path seeds the RNG, which needs a real torch.npu; stub it so
    # the classification logic stays testable on a machine without a device.
    original_seed = eval_device.seed_torch
    eval_device.seed_torch = lambda value: None
    try:
        test_first_call_failure_is_a_jit_failure()
        test_first_call_success_records_seconds_and_rebuilds()
    finally:
        eval_device.seed_torch = original_seed
    test_ascendc_build_path_untouched()
    test_pyasc_timing_validation()
    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} -> {FAILURES}")
        return 1
    print("all eval-backend checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
