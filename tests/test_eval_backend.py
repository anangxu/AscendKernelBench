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
from src.eval_result import runtime_failure  # noqa: E402
from src.score import summarize_eval_results  # noqa: E402

FAILURES: list[str] = []


class OutOfMemoryError(RuntimeError):
    """Stand-in carrying the class name a device raises, without importing torch."""


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


def test_first_call_correctness_is_a_diagnostic() -> None:
    """The first call's output is recorded, and never fails the sample."""
    import contextlib

    with tempfile.TemporaryDirectory() as tmp:
        evaluator = SampleEvaluator(make_request(Path(tmp), "pyasc"))
        evaluator.torch = SimpleNamespace(
            npu=SimpleNamespace(synchronize=lambda **kwargs: None),
            no_grad=contextlib.nullcontext,
        )
        evaluator.get_inputs = lambda: []
        evaluator._process_input = lambda value: value
        evaluator.new_model = lambda *args: "same"
        evaluator._run_reference = lambda inputs, raw, trial, stage=None: "same"
        evaluator._outputs_ok = lambda ref, new: ref == new

        def fake_construct() -> None:
            evaluator.new_model = lambda *args: "same"
            return None

        evaluator._construct_models = fake_construct

        original_diff = eval_device.max_abs_diff
        original_seed = eval_device.seed_torch
        eval_device.max_abs_diff = lambda ref, new: 0.0
        eval_device.seed_torch = lambda value: None
        try:
            result = evaluator._warm_up_pyasc()
            check("a matching first call still returns None", result is None, repr(result))
            check(
                "first_call_correct is recorded as True",
                evaluator.metadata.get("first_call_correct") is True,
                repr(evaluator.metadata.get("first_call_correct")),
            )

            evaluator._outputs_ok = lambda ref, new: False
            evaluator.metadata = {}
            result = evaluator._warm_up_pyasc()
            check(
                "a wrong first call is recorded, not enforced",
                result is None and evaluator.metadata.get("first_call_correct") is False,
                repr((result, evaluator.metadata.get("first_call_correct"))),
            )

            def explode(*args, **kwargs):
                raise RuntimeError("reference unavailable")

            evaluator._run_reference = explode
            evaluator.metadata = {}
            result = evaluator._warm_up_pyasc()
            check(
                "a failed check is recorded as a diagnostic error",
                result is None
                and evaluator.metadata.get("first_call_correct") is None
                and "reference unavailable" in str(evaluator.metadata.get("first_call_check_error")),
                repr(evaluator.metadata.get("first_call_check_error")),
            )
        finally:
            eval_device.max_abs_diff = original_diff
            eval_device.seed_torch = original_seed


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

    # The wall-clock cross-check can fail before a ratio exists. That path used
    # to keep the runtime and the SOL score, so an untrusted measurement stayed
    # eligible for fast_p, the geometric mean and the SOL average.
    def explode(*args: object) -> None:
        raise RuntimeError("wall clock unavailable")

    broken = evaluator_with(2.0, 0.002)
    broken.new_model = explode
    broken.metadata = {"speedup": 3.0, "sol_score": 0.9, "sol_bound_ms": 0.01}
    broken._validate_pyasc_timing()
    check(
        "a failed wall-clock check also invalidates timing",
        broken.metadata.get("timing_valid") is False,
        repr(broken.metadata.get("timing_valid")),
    )
    check("a failed wall-clock check drops the runtime", broken.runtime is None)
    check(
        "a failed wall-clock check drops the reference runtime",
        broken.ref_runtime is None,
    )
    check(
        "a failed wall-clock check drops the speedup",
        "speedup" not in broken.metadata,
    )
    check(
        "a failed wall-clock check drops the SOL score",
        "sol_score" not in broken.metadata,
        repr(broken.metadata.get("sol_score")),
    )


def test_invalid_timing_is_excluded_from_performance() -> None:
    """A rejected timing keeps correctness credit and loses every perf number."""
    from src.score import (
        fast_p,
        geometric_mean_speedup,
        sample_speedup,
        summarize_eval_results,
    )
    from src.sol import mean_sol_score

    def sample(valid: bool | None) -> dict:
        metadata: dict = {}
        if valid is not None:
            metadata["timing_valid"] = valid
        return {
            "compiled": True,
            "correctness": True,
            "runtime": 0.5,
            "ref_runtime": 5.0,
            "metadata": {**metadata, "sol_score": 0.8},
        }

    for label, valid in (("absent", None), ("true", True)):
        check(
            f"an {label} timing flag keeps its speedup",
            sample_speedup(sample(valid)) == 10.0,
            repr(sample_speedup(sample(valid))),
        )
    rejected = sample(False)
    check("a rejected timing reports no speedup", sample_speedup(rejected) is None)
    check(
        "a rejected timing is still counted in fast_0",
        fast_p([rejected])["fast_0"] == 1.0,
        repr(fast_p([rejected])["fast_0"]),
    )
    check(
        "a rejected timing is excluded above fast_0",
        all(
            value == 0.0
            for key, value in fast_p([rejected]).items()
            if key != "fast_0"
        ),
        repr(fast_p([rejected])),
    )
    check(
        "a rejected timing is excluded from the geometric mean",
        geometric_mean_speedup([rejected]) == 0.0,
    )
    check(
        "a rejected timing is excluded from the mean SOL score",
        mean_sol_score([rejected]) is None,
        repr(mean_sol_score([rejected])),
    )
    summary = summarize_eval_results({"level1/5_x": [rejected]})
    check(
        "the summary counts invalid timings",
        summary.get("timing_invalid") == 1,
        repr(summary.get("timing_invalid")),
    )
    check(
        "an invalid timing stays in the fast_0 denominator",
        summary["fast_p"]["fast_0"] == 1.0 and summary["correct"] == 1,
        repr((summary["fast_p"]["fast_0"], summary["correct"])),
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


def test_unsupported_dtype_is_reported_not_casted() -> None:
    """A dtype pyasc cannot marshal is an explicit backend-wide failure."""
    from src.eval_device import _pyasc_failure_kind
    from src.score import summarize_eval_results

    class FakeTensor:
        def __init__(self, dtype: str) -> None:
            self.dtype = dtype

    def evaluator_for(dtype: str):
        ev = SampleEvaluator(make_request(Path(tempfile.mkdtemp()), "pyasc"))
        ev.torch = SimpleNamespace(npu=SimpleNamespace(synchronize=lambda: None))
        ev.get_inputs = lambda: [FakeTensor(dtype)]
        # Stand in for move_value_to_device: record every cast so a rejection
        # that happens after the cast cannot pass as pre-cast.
        cast = {"n": 0}

        def process(value: object) -> object:
            cast["n"] += 1
            return value

        ev._process_input = process
        called = {"n": 0}

        def model(*args: object) -> None:
            called["n"] += 1

        ev.new_model = model
        return ev, called, cast

    original_seed = eval_device.seed_torch
    eval_device.seed_torch = lambda value: None
    try:
        ev, called, cast = evaluator_for("torch.bfloat16")
        result = ev._warm_up_pyasc()
        assert result is not None
        metadata = result["metadata"]
        check("bf16 is not compiled", result["compiled"] is False)
        check(
            "bf16 reports unsupported_dtype",
            metadata.get("failure_stage") == "unsupported_dtype",
            repr(metadata.get("failure_stage")),
        )
        check(
            "bf16 is a backend-wide limitation",
            metadata.get("limitation_scope") == "backend",
            repr(metadata.get("limitation_scope")),
        )
        check(
            "bf16 names the dtype",
            metadata.get("unsupported_dtypes") == ["bfloat16"],
            repr(metadata.get("unsupported_dtypes")),
        )
        check("bf16 never reaches the kernel", called["n"] == 0, repr(called))
        check(
            "bf16 is rejected before any input is cast",
            cast["n"] == 0,
            f"move_value_to_device ran {cast['n']} time(s) first",
        )

        ev_ok, _, cast_ok = evaluator_for("torch.float32")
        ev_ok.new_model = lambda *args: None
        ev_ok._construct_models = lambda: None
        result_ok = ev_ok._warm_up_pyasc()
        check(
            "fp32 is not classified as unsupported",
            result_ok is None
            or result_ok["metadata"].get("failure_stage") != "unsupported_dtype",
        )
        check(
            "fp32 still goes through the cast path",
            cast_ok["n"] > 0,
            repr(cast_ok),
        )
    finally:
        eval_device.seed_torch = original_seed

    check(
        "UnsupportedSyntaxError is a codegen error, not a backend limit",
        _pyasc_failure_kind(RuntimeError("x UnsupportedSyntaxError y"))
        == ("codegen_error", "code"),
    )
    check(
        "other first-call failures stay jit",
        _pyasc_failure_kind(RuntimeError("boom")) == ("jit", "none"),
    )

    summary = summarize_eval_results(
        {
            "level1/901_add_smoke": [
                {
                    "compiled": False,
                    "correctness": False,
                    "metadata": {
                        "failure_stage": "unsupported_dtype",
                        "limitation_scope": "backend",
                    },
                },
                {
                    "compiled": True,
                    "correctness": True,
                    "metadata": {"reference": "npu"},
                },
            ]
        }
    )
    check(
        "summary keeps the unsupported count",
        summary.get("unsupported_dtype_backend_wide") == 1,
        repr(summary.get("unsupported_dtype_backend_wide")),
    )
    check(
        "unsupported sample is not counted correct",
        summary.get("correct") == 1,
        repr(summary.get("correct")),
    )


def test_runtime_failure_is_classified_without_assigning_blame() -> None:
    """A runtime failure keeps its class, stage, and message summary."""
    exc = RuntimeError("NPU out of memory. Tried to allocate 6.00 GiB " + "x" * 400)
    classified = runtime_failure(exc, stage="candidate_trial")
    check(
        "the exception class is recorded",
        classified.get("runtime_error_class") == "RuntimeError",
        repr(classified),
    )
    check(
        "the stage that raised it is recorded",
        classified.get("runtime_error_stage") == "candidate_trial",
        repr(classified),
    )
    summary = str(classified.get("runtime_error_summary"))
    check("the message is summarized on one line", "\n" not in summary, repr(summary))
    check("the summary is bounded", len(summary) <= 240, repr(len(summary)))
    check(
        "the classification never decides who is at fault",
        "limitation_scope" not in classified and "failure_stage" not in classified,
        repr(sorted(classified)),
    )

    oom = {
        "compiled": True,
        "correctness": False,
        "metadata": {
            "runtime_error": "trial 1: candidate runtime error: OutOfMemoryError()",
            **runtime_failure(RuntimeError("out of memory"), stage="candidate_trial"),
        },
    }
    legacy = {
        "compiled": True,
        "correctness": False,
        "metadata": {"runtime_error": "trial 1: candidate runtime error"},
    }
    good = {"compiled": True, "correctness": True, "metadata": {"reference": "npu"}}
    aggregated = summarize_eval_results({"level1/1_x": [oom, legacy, good]})
    check(
        "classes are counted generically",
        aggregated.get("runtime_error_classes") == {"RuntimeError": 1, "unclassified": 1},
        repr(aggregated.get("runtime_error_classes")),
    )
    check(
        "stages are counted",
        aggregated.get("runtime_error_stages") == {"candidate_trial": 1},
        repr(aggregated.get("runtime_error_stages")),
    )
    check(
        "the scoring denominator is unchanged",
        aggregated.get("total_samples") == 3
        and aggregated.get("correct") == 1
        and aggregated["fast_p"]["fast_0"] == 1 / 3,
        repr((aggregated.get("total_samples"), aggregated.get("correct"))),
    )


def test_timing_input_preparation_failure_discards_timing() -> None:
    """A timing-input failure keeps correctness and invalidates performance."""
    import contextlib

    def build(source: str) -> SampleEvaluator:
        evaluator = SampleEvaluator(make_request(Path("/nonexistent"), "pyasc"))
        evaluator.torch = SimpleNamespace(
            npu=SimpleNamespace(synchronize=lambda **kwargs: None),
            no_grad=contextlib.nullcontext,
        )
        evaluator.runtime = 2.0
        evaluator.runtime_stats = {"mean": 2.0}
        evaluator.ref_runtime = 4.0
        evaluator.ref_runtime_stats = {"mean": 4.0}
        evaluator.new_model = lambda *args: None
        if source == "get_inputs":

            def boom_inputs() -> list:
                raise OutOfMemoryError("tried to allocate 6.00 GiB")

            evaluator.get_inputs = boom_inputs
            evaluator._process_input = lambda value: value
        else:
            evaluator.get_inputs = lambda: [object()]

            def boom_process(value: object) -> object:
                raise OutOfMemoryError("tried to allocate 6.00 GiB")

            evaluator._process_input = boom_process
        return evaluator

    for source in ("get_inputs", "_process_input"):
        evaluator = build(source)
        escaped = ""
        try:
            evaluator._validate_pyasc_timing()
        except Exception as exc:  # noqa: BLE001 - the pre-fix behaviour
            escaped = f"{type(exc).__name__}: {exc}"
        check(f"{source} failure does not escape timing validation", escaped == "", escaped)
        metadata = evaluator.metadata
        check(
            f"{source} failure invalidates the timing",
            metadata.get("timing_valid") is False,
            repr(metadata.get("timing_valid")),
        )
        check(
            f"{source} failure leaves no performance number",
            evaluator.runtime is None
            and evaluator.runtime_stats is None
            and evaluator.ref_runtime is None
            and evaluator.ref_runtime_stats is None,
            repr((evaluator.runtime, evaluator.ref_runtime)),
        )
        check(
            f"{source} failure records the class",
            metadata.get("runtime_error_class") == "OutOfMemoryError",
            repr(metadata.get("runtime_error_class")),
        )
        check(
            f"{source} failure records the stage",
            metadata.get("runtime_error_stage") == "timing_inputs",
            repr(metadata.get("runtime_error_stage")),
        )
        check(
            f"{source} failure records a summary",
            "6.00 GiB" in str(metadata.get("runtime_error_summary")),
            repr(metadata.get("runtime_error_summary")),
        )


def test_construct_models_reports_the_compile_state() -> None:
    """An init failure reports the real compile state, not an assumed one."""

    class Boom:
        def __init__(self, *args: object) -> None:
            raise RuntimeError("model init failed")

    def fail_init(evaluator: SampleEvaluator) -> dict:
        evaluator.model_new_cls = Boom
        evaluator.init_inputs = []
        evaluator.torch_device = None
        evaluator.dtype = None
        result = evaluator._construct_models()
        assert result is not None
        return result

    with tempfile.TemporaryDirectory() as tmp:
        before = SampleEvaluator(make_request(Path(tmp), "pyasc"))
        result = fail_init(before)
        check(
            "pyasc init failure before the first call is not compiled",
            result["compiled"] is False,
            repr(result["compiled"]),
        )
        check(
            "pyasc init failure before the first call is an execution failure",
            result["metadata"].get("failure_stage") == "execution"
            and result["metadata"].get("runtime_error_stage") == "model_init",
            repr(result["metadata"].get("runtime_error_stage")),
        )

        after = SampleEvaluator(make_request(Path(tmp), "pyasc"))
        after.torch = SimpleNamespace(
            npu=SimpleNamespace(synchronize=lambda **kwargs: None)
        )
        after.get_inputs = lambda: []
        after._process_input = lambda value: value
        after.new_model = lambda *args: None
        real_construct = after._construct_models
        after._construct_models = lambda: None
        original_seed = eval_device.seed_torch
        eval_device.seed_torch = lambda value: None
        try:
            check("pyasc warm-up succeeds", after._warm_up_pyasc() is None)
        finally:
            eval_device.seed_torch = original_seed
        after._construct_models = real_construct
        result = fail_init(after)
        check(
            "pyasc init failure after a successful first call is compiled",
            result["compiled"] is True,
            repr(result["compiled"]),
        )

        built = SampleEvaluator(make_request(Path(tmp), "ascendc"))
        (Path(tmp) / "custom_op.asc").write_text("// legacy kernel\n")
        original_build = eval_device.build_custom_op
        original_load = eval_device.load_custom_op
        eval_device.build_custom_op = lambda *a, **k: Path(tmp) / "libcustom_op.so"
        eval_device.load_custom_op = lambda *a, **k: None
        try:
            check("ascendc build returns no failure", built._build_and_load() is None)
        finally:
            eval_device.build_custom_op = original_build
            eval_device.load_custom_op = original_load
        result = fail_init(built)
        check(
            "ascendc init failure after a successful build is compiled",
            result["compiled"] is True,
            repr(result["compiled"]),
        )

        loader = SampleEvaluator(make_request(Path(tmp), "pyasc"))
        loader.req = loader.req.model_copy(update={"task_py": "raise RuntimeError('boom')"})
        result = loader._load_python_modules()
        assert result is not None
        check(
            "pyasc module-load failure before the first call is not compiled",
            result["compiled"] is False,
            repr(result["compiled"]),
        )


def test_late_trial_failure_keeps_diagnostics() -> None:
    """A trial failure after the first call keeps the diagnostics it produced."""
    import contextlib

    with tempfile.TemporaryDirectory() as tmp:
        sample = Path(tmp)
        (sample / "model_new.py").write_text("class ModelNew: pass\n")
        evaluator = SampleEvaluator(make_request(sample, "pyasc"))
        evaluator.torch = SimpleNamespace(
            npu=SimpleNamespace(synchronize=lambda **kwargs: None),
            no_grad=contextlib.nullcontext,
        )
        evaluator.get_inputs = lambda: []
        evaluator._process_input = lambda value: value
        evaluator.new_model = lambda *args: "same"
        evaluator._run_reference = lambda inputs, raw, trial, stage=None: "same"
        evaluator._outputs_ok = lambda ref, new: ref == new
        evaluator._construct_models = lambda: None

        original_diff = eval_device.max_abs_diff
        original_seed = eval_device.seed_torch
        original_facts = eval_device.runtime_facts
        eval_device.max_abs_diff = lambda ref, new: 0.0
        eval_device.seed_torch = lambda value: None
        eval_device.runtime_facts = lambda path: {
            "pyasc_version": "1.1.1",
            "platform": "Ascend910B4",
        }
        try:
            check("warm-up records no failure", evaluator._warm_up_pyasc() is None)

            def boom(*args: object) -> object:
                raise OutOfMemoryError("tried to allocate 6.00 GiB")

            evaluator.new_model = boom
            evaluator.torch = SimpleNamespace(
                npu=SimpleNamespace(synchronize=lambda **kwargs: None),
                no_grad=contextlib.nullcontext,
                randint=lambda *a, **k: SimpleNamespace(item=lambda: 7),
            )
            failed = evaluator._run_correctness_trials()
        finally:
            eval_device.max_abs_diff = original_diff
            eval_device.seed_torch = original_seed
            eval_device.runtime_facts = original_facts

        check("late trial failure returns a payload", failed is not None)
        assert failed is not None
        result = evaluator._with_build_facts(failed)
        metadata = result["metadata"]
        check("late trial failure is compiled", result["compiled"] is True, repr(result["compiled"]))
        check(
            "late trial failure is not correct",
            result["correctness"] is False,
            repr(result["correctness"]),
        )
        check(
            "late trial failure reports no timing",
            result["runtime"] is None
            and result["runtime_stats"] is None
            and result["ref_runtime"] is None
            and result["ref_runtime_stats"] is None,
            repr(result["runtime"]),
        )
        check(
            "first_call_seconds survives the later failure",
            isinstance(metadata.get("first_call_seconds"), float),
            repr(metadata.get("first_call_seconds")),
        )
        check(
            "first_call_correct survives the later failure",
            metadata.get("first_call_correct") is True,
            repr(metadata.get("first_call_correct")),
        )
        check(
            "runtime facts survive the later failure",
            metadata.get("pyasc_version") == "1.1.1"
            and metadata.get("platform") == "Ascend910B4",
            repr((metadata.get("pyasc_version"), metadata.get("platform"))),
        )
        check(
            "the failure's own error fields win",
            metadata.get("failure_stage") == "execution"
            and metadata.get("runtime_error_class") == "OutOfMemoryError"
            and metadata.get("runtime_error_stage") == "candidate_trial"
            and "trial 0" in str(metadata.get("runtime_error")),
            repr(metadata.get("runtime_error")),
        )


def test_hidden_input_preparation_failure_is_classified() -> None:
    """A hidden-gate input failure fails the gate instead of escaping."""
    import contextlib

    with tempfile.TemporaryDirectory() as tmp:
        sample = Path(tmp)
        (sample / "model_new.py").write_text("class ModelNew: pass\n")
        evaluator = SampleEvaluator(make_request(sample, "pyasc"))
        evaluator.torch = SimpleNamespace(
            npu=SimpleNamespace(synchronize=lambda **kwargs: None),
            no_grad=contextlib.nullcontext,
        )
        evaluator.new_model = lambda *args: None
        evaluator._outputs_ok = lambda ref, new: True

        def boom_inputs() -> list:
            raise OutOfMemoryError("tried to allocate 6.00 GiB")

        evaluator.get_inputs = boom_inputs
        evaluator._process_input = lambda value: value

        original_seed = eval_device.seed_torch
        eval_device.seed_torch = lambda value: None
        try:
            escaped = ""
            result: dict | None = None
            try:
                result = evaluator._run_hidden_distributions()
            except Exception as exc:  # noqa: BLE001 - the pre-fix behaviour
                escaped = f"{type(exc).__name__}: {exc}"
        finally:
            eval_device.seed_torch = original_seed

        check("hidden input failure does not escape the gate", escaped == "", escaped)
        check("hidden input failure returns a payload", result is not None)
        if result is not None:
            check(
                "hidden input failure is compiled but not correct",
                result["compiled"] is True and result["correctness"] is False,
                repr((result["compiled"], result["correctness"])),
            )
            metadata = result["metadata"]
            check(
                "hidden input failure records the class and stage",
                metadata.get("runtime_error_class") == "OutOfMemoryError"
                and metadata.get("runtime_error_stage") == "hidden_inputs",
                repr((metadata.get("runtime_error_class"), metadata.get("runtime_error_stage"))),
            )
            check(
                "hidden input failure reports no timing",
                result["runtime"] is None and result["ref_runtime"] is None,
                repr(result["runtime"]),
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
        test_first_call_correctness_is_a_diagnostic()
    finally:
        eval_device.seed_torch = original_seed
    test_ascendc_build_path_untouched()
    test_pyasc_timing_validation()
    test_invalid_timing_is_excluded_from_performance()
    test_unsupported_dtype_is_reported_not_casted()
    test_runtime_failure_is_classified_without_assigning_blame()
    test_timing_input_preparation_failure_discards_timing()
    test_construct_models_reports_the_compile_state()
    test_late_trial_failure_keeps_diagnostics()
    test_hidden_input_preparation_failure_is_classified()
    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} -> {FAILURES}")
        return 1
    print(
        "skip - device check that a bfloat16 task produces correct output: "
        "pyasc 1.1.1 dtype table rejects bfloat16 and bool, so the harness "
        "reports unsupported_dtype instead of running it"
    )
    print("all eval-backend checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
