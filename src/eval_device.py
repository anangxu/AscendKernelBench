"""Worker-side build, correctness, and timing for one sample.

Runs inside the isolated subprocess started by eval.evaluate_run; the
stages live on SampleEvaluator so shared trial state is explicit.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from .backend import Backend, kernel_file, parse_backend
from .build import (
    BuildError,
    LoadError,
    build_custom_op,
    load_custom_op,
    split_asc_source,
)
from .compare import (
    HIDDEN_DISTRIBUTIONS,
    compare_candidate_outputs,
    inputs_were_mutated,
    max_abs_diff,
    move_value_to_device,
    perturb_floating_inputs,
    resolve_tolerances,
    snapshot_inputs,
    tensor_nbytes,
)
from .eval_result import (
    compiled_result,
    eval_protocol_metadata,
    fail_result,
    runtime_failure,
)
from .pyasc_runtime import (
    PyAscError,
    PyAscJITError,
    PyAscLoadError,
    PyAscUnavailableError,
    configure_platform,
    import_asc,
    load_kernel_module,
    prepare_cache_dir,
    runtime_facts,
    time_first_call,
)
from .runtime import (
    npu_device_index,
    npu_runtime_metadata,
    seed_torch,
    torch_dtype_for,
)
from .sol import attach_sol_metadata, task_declared_flops
from .timing import (
    REFRESH_INPUT_BYTES_LIMIT,
    get_timing_stats,
    time_execution_with_npu_event,
)

__all__ = ["DeviceEvalRequest", "eval_sample_on_device", "exec_python_source"]

###########################################################################
# PYASC TIMING VALIDATION
###########################################################################
# pyasc launches on its own runtime stream, so an NPU event pair recorded on
# torch's stream could in principle measure almost nothing. Five unmeasured
# calls give a wall-clock figure to compare against the event-measured mean.
# On the validated host the healthy ratio is 0.30 to 0.35 in harness: the
# event captures the device execution while wall clock also pays pyasc's
# per-call host overhead (kernel re-registration and a synchronize), which is
# not on the device timeline. A stream that is genuinely not covered yields
# roughly zero, so the gate sits well below the healthy band.
PYASC_TIMING_WALL_TRIALS = 5
PYASC_MIN_EVENT_WALL_RATIO = 0.05

# pyasc's dtype table has no factory for these, so a sample whose inputs or
# parameters use them cannot be marshalled at all. That is a backend-wide
# limitation, not a per-operator one, and it is reported as such: the harness
# never casts the tensor to a supported dtype, because that would measure a
# different task than the one the reference model defines.
PYASC_UNSUPPORTED_DTYPES = frozenset({"bfloat16", "bool", "complex64", "complex128"})
###########################################################################
###########################################################################


def _pyasc_failure_kind(exc: BaseException) -> tuple[str, str]:
    """Return (failure_stage, limitation_scope) for a first-call failure.

    UnsupportedSyntaxError means the traced body used something the codegen
    rejects, which is attributable to the generated source, so it is reported
    as a codegen error and never as a backend capability limit. Only a
    positively identified operator-capability limit would use
    unsupported_operator/operator, and nothing here infers that from an
    exception: a backend limit is reported by the dtype pre-check, which can
    name the missing capability. The class name is read from the wrapped repr
    because eval_device imports asc only inside the worker.
    """
    if "UnsupportedSyntaxError" in repr(exc):
        return "codegen_error", "code"
    return "jit", "none"


class DeviceEvalRequest(BaseModel):
    """Validated worker payload for one on-device sample evaluation."""

    model_config = ConfigDict(extra="ignore")

    task_py: str
    sample_dir: str
    cmake_arch: str
    hardware_name: str
    device: str
    measure_performance: bool
    seed: int
    num_correct_trials: int
    num_perf_trials: int
    num_warmup: int
    precision: str
    tolerances: dict[str, dict[str, float]]
    excessive_speedup: float
    build_timeout: int
    memory_bandwidth_gbps: float = 0.0
    peak_tflops: float = 0.0
    l2_clear_size: int = 256 * 1024 * 1024
    backend: str = "ascendc"
    soc_version: str = ""


def exec_python_source(
    source: str,
    filename: str,
    namespace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute source and return the namespace it populated."""
    populated: dict[str, Any] = {} if namespace is None else namespace
    exec(compile(source, filename, "exec"), populated)
    return populated


def cpu_reference_inputs(raw_inputs: Sequence[Any], torch_mod: Any) -> list[Any]:
    """Cast floating CPU-reference inputs to float32; leave others as-is."""
    return [
        item.float()
        if isinstance(item, torch_mod.Tensor) and torch_mod.is_floating_point(item)
        else item
        for item in raw_inputs
    ]


def eval_sample_on_device(**kwargs: Any) -> dict[str, Any]:
    """Build, check correctness, and time a sample in the current process."""
    request = DeviceEvalRequest.model_validate(kwargs)
    return SampleEvaluator(request).run()


class SampleEvaluator:
    """Stateful worker that evaluates one generated sample on the NPU."""

    def __init__(self, request: DeviceEvalRequest) -> None:
        """Bind the request; runtime objects are filled during run."""
        self.req = request
        self.backend = parse_backend(request.backend)
        self.kernel_module: Any = None
        self.sample_path = Path(request.sample_dir)
        self.so_path: Path | None = None
        # Whether this sample has really compiled. Set by a successful Ascend C
        # build and by a successful pyasc first call; never inferred from the
        # backend name, because a pyasc failure before the first call has not
        # compiled anything.
        self.compiled = False
        self.torch: Any = None
        self.torch_device: Any = None
        self.dtype: Any = None
        self.ref_globals: dict[str, Any] = {}
        self.model_cls: Any = None
        self.model_new_cls: Any = None
        self.get_init_inputs: Any = None
        self.get_inputs: Any = None
        self.custom_check: Any = None
        self.init_inputs: Any = None
        self.atol = 0.0
        self.rtol = 0.0
        self.new_model: Any = None
        self.ref_model: Any = None
        self.ref_model_cpu: Any = None
        self.ref_mode = "npu"
        self.ref_npu_error: str | None = None
        self.pass_count = 0
        self.max_diff = 0.0
        self.correctness_error = ""
        self.last_new_out: Any = None
        self.metadata: dict[str, Any] = {}
        self.runtime: Any = None
        self.runtime_stats: Any = None
        self.ref_runtime: Any = None
        self.ref_runtime_stats: Any = None

    def run(self) -> dict[str, Any]:
        """Execute build, correctness, optional timing, and the re-check."""
        import torch
        import torch_npu  # noqa: F401

        self.torch = torch
        failed = self._build_and_load()
        if failed is not None:
            return self._with_build_facts(failed)
        failed = self._load_python_modules()
        if failed is not None:
            return self._with_build_facts(failed)
        self._prepare_device()
        failed = self._construct_models()
        if failed is not None:
            return self._with_build_facts(failed)
        failed = self._warm_up_pyasc()
        if failed is not None:
            return self._with_build_facts(failed)
        failed = self._run_correctness_trials()
        if failed is not None:
            return self._with_build_facts(failed)
        self._fill_metadata()
        if self.pass_count != self.req.num_correct_trials:
            self.metadata["correctness_error"] = self.correctness_error
            return compiled_result(correctness=False, metadata=self.metadata)
        hidden_failure = self._run_hidden_distributions()
        if hidden_failure is not None:
            return hidden_failure
        if self.req.measure_performance:
            self._time_both_models()
            self._validate_pyasc_timing()
            if self._post_timing_recheck() is not None:
                return self._compiled(correctness=False)
        return self._compiled(correctness=True)

    def _with_build_facts(self, result: dict[str, Any]) -> dict[str, Any]:
        """Merge the diagnostics collected so far into a failure payload.

        Everything recorded before the failure survives, and the payload's own
        fields win: a later failure must not be overwritten by an earlier value,
        and a recorded diagnostic must not be replaced by an empty one.
        """
        metadata = result.get("metadata")
        if isinstance(metadata, dict):
            merged = dict(self.metadata)
            for key, value in metadata.items():
                if value is None and merged.get(key) is not None:
                    continue
                merged[key] = value
            result["metadata"] = merged
        return result

    def _compiled(self, *, correctness: bool) -> dict[str, Any]:
        """Return a post-build payload from the fields collected so far."""
        return compiled_result(
            correctness=correctness,
            metadata=self.metadata,
            runtime=self.runtime,
            runtime_stats=self.runtime_stats,
            ref_runtime=self.ref_runtime,
            ref_runtime_stats=self.ref_runtime_stats,
        )

    def _build_and_load(self) -> dict[str, Any] | None:
        """Compile and load the backend's kernel; fail-payload on error."""
        if self.backend is Backend.PYASC:
            return self._load_pyasc()
        asc_source = (self.sample_path / "custom_op.asc").read_text(encoding="utf-8")
        self.metadata["build_mode"] = (
            "split" if split_asc_source(asc_source) is not None else "legacy"
        )
        started = time.monotonic()
        try:
            self.so_path = build_custom_op(
                asc_source,
                self.sample_path,
                cmake_arch=self.req.cmake_arch,
                timeout_s=self.req.build_timeout,
            )
            # The build produced a library: for Ascend C that is the compile
            # step, and a later load failure still reports compiled=True.
            self.compiled = True
            load_custom_op(self.so_path, asc_source)
        except (BuildError, OSError) as exc:
            return fail_result(compilation_error=str(exc))
        except LoadError as exc:
            return fail_result(
                compiled=True,
                runtime_error=f"shared library load failed: {exc}",
            )
        finally:
            self.metadata["build_seconds"] = round(time.monotonic() - started, 3)
        return None

    def _load_pyasc(self) -> dict[str, Any] | None:
        """Import asc, isolate the JIT cache, and load kernel.py.

        Importing the module does not compile it: pyasc traces, compiles, and
        launches on the first subscripted call, so a successful import must
        never be reported as a successful compile. The JIT cache directory is
        exported before asc is imported because pyasc reads it at import time.
        """
        backend = self.backend.value
        kernel_path = self.sample_path / kernel_file(Backend.PYASC)
        if not kernel_path.is_file():
            return fail_result(
                compilation_error="sample dir missing kernel.py",
                failure_stage="module_load",
                backend=backend,
            )
        started = time.monotonic()
        self.metadata["build_mode"] = "pyasc"
        try:
            prepare_cache_dir(self.sample_path)
            asc = import_asc()
            self.metadata["pyasc_soc_version"] = configure_platform(
                asc, npu_device_index(self.req.device)
            )
            self.kernel_module = load_kernel_module(self.sample_path, kernel_path)
        except (PyAscUnavailableError, PyAscLoadError, PyAscError) as exc:
            return fail_result(
                compilation_error=str(exc),
                failure_stage="module_load",
                backend=backend,
            )
        finally:
            self.metadata["build_seconds"] = round(time.monotonic() - started, 3)
        return None

    def _warm_up_pyasc(self) -> dict[str, Any] | None:
        """Run the first pyasc call: trace, compile, and launch.

        Only this call decides whether the sample compiled. Its cost is a
        compound figure and is reported as first_call_seconds, never as a pure
        compile time. The models are rebuilt afterwards so the preparation
        call leaves no RNG or parameter state in the seeded protocol.
        """
        if self.backend is not Backend.PYASC:
            return None
        backend = self.backend.value
        seed_torch(self.req.seed)
        try:
            raw_inputs = self.get_inputs()
        except Exception as exc:
            return fail_result(
                compilation_error=f"input preparation failed: {exc!r}",
                failure_stage="execution",
                backend=backend,
                **runtime_failure(exc, stage="input_preparation"),
            )
        # Reject before anything casts: move_value_to_device turns every
        # floating tensor into the configured precision, which would hide a
        # task whose declared dtype pyasc cannot marshal. Nothing is converted
        # while this is unresolved, so the rejection is genuinely pre-cast.
        unsupported = self._unsupported_pyasc_dtypes(raw_inputs)
        if not unsupported:
            try:
                inputs = [self._process_input(item) for item in raw_inputs]
            except Exception as exc:
                return fail_result(
                    compilation_error=f"input preparation failed: {exc!r}",
                    failure_stage="execution",
                    backend=backend,
                    **runtime_failure(exc, stage="input_preparation"),
                )
            # Backstop on the tensors the kernel is actually handed.
            unsupported = self._unsupported_pyasc_dtypes(inputs)
        if unsupported:
            return fail_result(
                compilation_error=(
                    "pyasc cannot marshal these tensor dtypes: "
                    + ", ".join(unsupported)
                ),
                failure_stage="unsupported_dtype",
                limitation_scope="backend",
                unsupported_dtypes=unsupported,
                backend=backend,
            )
        try:
            seconds, first_out = time_first_call(lambda: self.new_model(*inputs))
            self.torch.npu.synchronize()
        except PyAscJITError as exc:
            stage, scope = _pyasc_failure_kind(exc)
            return fail_result(
                compilation_error=str(exc),
                jit_error=str(exc),
                failure_stage=stage,
                limitation_scope=scope,
                backend=backend,
            )
        except Exception as exc:
            return fail_result(
                compilation_error=f"pyasc first call failed: {exc!r}",
                failure_stage="execution",
                backend=backend,
                **runtime_failure(exc, stage="first_call"),
            )
        self.compiled = True
        self.metadata["first_call_seconds"] = round(seconds, 3)
        # Diagnostic only: a first call can succeed and still produce wrong
        # output, which the compiled flag cannot express. Never fails a sample.
        self._record_first_call_correctness(first_out, inputs, raw_inputs)
        self.metadata.update(runtime_facts(self.sample_path))
        # Rebuild so the warm-up call cannot influence measured behaviour.
        self.new_model = None
        self.ref_model = None
        self.ref_model_cpu = None
        return self._construct_models()

    def _load_python_modules(self) -> dict[str, Any] | None:
        """Exec the task and model_new.py; fail-payload on error."""
        try:
            self.ref_globals = exec_python_source(self.req.task_py, "<task.py>")
            self.model_cls = self.ref_globals["Model"]
            self.get_init_inputs = self.ref_globals["get_init_inputs"]
            self.get_inputs = self.ref_globals["get_inputs"]
            self.custom_check = self.ref_globals.get("custom_check")
            model_new_path = self.sample_path / "model_new.py"
            custom_globals = exec_python_source(
                model_new_path.read_text(encoding="utf-8"),
                str(model_new_path),
                {"__file__": str(model_new_path)},
            )
            self.model_new_cls = custom_globals["ModelNew"]
        except Exception as exc:
            return fail_result(
                compiled=self.compiled, runtime_error=f"module load failed: {exc!r}"
            )
        task_tolerance = self.ref_globals.get("TOLERANCE")
        self.atol, self.rtol = resolve_tolerances(
            self.req.precision,
            self.req.tolerances,
            task_tolerance if isinstance(task_tolerance, dict) else None,
        )
        return None

    def _prepare_device(self) -> None:
        """Select the NPU, dtype, and constructor seed."""
        self.torch_device = self.torch.device(self.req.device)
        self.torch.npu.set_device(npu_device_index(self.req.device))
        self.dtype = torch_dtype_for(self.req.precision)
        seed_torch(self.req.seed)
        self.init_inputs = self.get_init_inputs()

    def _construct_models(self) -> dict[str, Any] | None:
        """Build ModelNew and the NPU (or CPU-fallback) reference."""
        try:
            seed_torch(self.req.seed)
            self.new_model = self.model_new_cls(*self.init_inputs).to(
                device=self.torch_device, dtype=self.dtype
            )
            self.new_model.eval()
        except Exception as exc:
            return fail_result(
                compiled=self.compiled,
                runtime_error=f"candidate model init failed: {exc!r}",
                failure_stage="execution",
                **runtime_failure(exc, stage="model_init"),
            )
        try:
            seed_torch(self.req.seed)
            self.ref_model = self.model_cls(*self.init_inputs).to(
                device=self.torch_device, dtype=self.dtype
            )
            self.ref_model.eval()
        except Exception as exc:
            self.ref_mode = "cpu"
            self.ref_npu_error = repr(exc)
        return None

    def _process_input(self, value: Any) -> Any:
        """Move one argument onto the evaluation device."""
        return move_value_to_device(value, self.torch_device, self.dtype)

    def _outputs_ok(self, ref: Any, new: Any) -> bool:
        """Compare candidate outputs under the active reference mode."""
        return compare_candidate_outputs(
            ref,
            new,
            ref_mode=self.ref_mode,
            atol=self.atol,
            rtol=self.rtol,
            custom_check=(self.custom_check if callable(self.custom_check) else None),
        )

    def _run_ref_cpu(self, raw_inputs: Sequence[Any]) -> Any:
        """Lazily construct and run the CPU reference model."""
        if self.ref_model_cpu is None:
            seed_torch(self.req.seed)
            self.ref_model_cpu = self.model_cls(*self.init_inputs)
            self.ref_model_cpu.eval()
        return self.ref_model_cpu(*cpu_reference_inputs(raw_inputs, self.torch))

    def _run_reference(
        self,
        inputs: Sequence[Any],
        raw_inputs: Sequence[Any],
        trial: int,
        *,
        stage: str | None = None,
    ) -> Any:
        """Run the NPU reference, falling back to CPU after a device error."""
        if self.ref_mode != "npu":
            return self._run_ref_cpu(raw_inputs)
        try:
            ref_out = self.ref_model(*inputs)
            self.torch.npu.synchronize(device=self.req.device)
            return ref_out
        except Exception as exc:
            self.ref_mode = "cpu"
            label = stage or f"trial {trial}"
            self.ref_npu_error = f"{label}: {exc!r}"
            return self._run_ref_cpu(raw_inputs)

    ################################# TRIALS #################################
    def _run_correctness_trials(self) -> dict[str, Any] | None:
        """Run seeded correctness trials; hard-fail on runtime or mutation."""
        seed_torch(self.req.seed)
        trial_seeds = [
            self.torch.randint(0, 2**32 - 1, (1,)).item()
            for _ in range(self.req.num_correct_trials)
        ]
        with self.torch.no_grad():
            for trial, trial_seed in enumerate(trial_seeds):
                failed = self._one_correctness_trial(trial, trial_seed)
                if failed is not None:
                    return failed
        return None

    def _run_hidden_distributions(self) -> dict[str, Any] | None:
        """Gate on four value transforms; timing keeps the original draw."""
        seed_torch(self.req.seed)
        try:
            raw_inputs = self.get_inputs()
        except Exception as exc:
            # The gate cannot run at all. Fail it, and keep the correctness
            # verdict and the diagnostics already recorded.
            self.metadata["hidden_passed"] = []
            self.metadata["runtime_error"] = f"hidden inputs: {exc!r}"
            self.metadata["failure_stage"] = "execution"
            self.metadata.update(runtime_failure(exc, stage="hidden_inputs"))
            return self._compiled(correctness=False)
        passed: list[str] = []
        with self.torch.no_grad():
            for name, scale in HIDDEN_DISTRIBUTIONS:
                failed = self._one_hidden_distribution(name, scale, raw_inputs)
                if failed is not None:
                    self.metadata["hidden_passed"] = passed
                    self.metadata["hidden_failed"] = name
                    self.metadata.update(failed)
                    return self._compiled(correctness=False)
                passed.append(name)
        self.metadata["hidden_passed"] = passed
        self.metadata["hidden_failed"] = None
        return None

    def _one_hidden_distribution(
        self,
        name: str,
        scale: float,
        raw_inputs: Sequence[Any],
    ) -> dict[str, Any] | None:
        """Return an error mapping when one hidden transform fails."""
        try:
            raw = perturb_floating_inputs(raw_inputs, scale)
            inputs = [self._process_input(item) for item in raw]
        except Exception as exc:
            return {
                "runtime_error": f"hidden {name}: input preparation failed: {exc!r}",
                "failure_stage": "execution",
                **runtime_failure(exc, stage="hidden_inputs"),
            }
        self.torch.npu.synchronize(device=self.req.device)
        try:
            ref_out = self._run_reference(inputs, raw, 0, stage=f"hidden {name}")
        except Exception as exc:
            return {
                "runtime_error": (f"hidden {name}: reference runtime error: {exc!r}"),
                "failure_stage": "execution",
                **runtime_failure(exc, stage="hidden_reference"),
            }
        ref_snapshot = snapshot_inputs(inputs)
        try:
            new_out = self.new_model(*inputs)
            self.torch.npu.synchronize(device=self.req.device)
        except Exception as exc:
            return {
                "runtime_error": (f"hidden {name}: candidate runtime error: {exc!r}"),
                "failure_stage": "execution",
                **runtime_failure(exc, stage="hidden_candidate"),
            }
        if inputs_were_mutated(inputs, ref_snapshot):
            return {"runtime_error": f"hidden {name}: candidate mutated its inputs"}
        if self._outputs_ok(ref_out, new_out):
            return None
        self.max_diff = max(self.max_diff, max_abs_diff(ref_out, new_out))
        self.metadata["max_difference"] = self.max_diff
        return {
            "correctness_error": (
                f"hidden {name}: output mismatch on value-transformed inputs"
            )
        }

    def _one_correctness_trial(
        self, trial: int, trial_seed: int
    ) -> dict[str, Any] | None:
        """Run one correctness trial; return a fail payload on hard errors."""
        seed_torch(trial_seed)
        raw_inputs = self.get_inputs()
        inputs = [self._process_input(item) for item in raw_inputs]
        seed_torch(trial_seed)
        self.torch.npu.synchronize(device=self.req.device)
        ref_out = self._run_reference(inputs, raw_inputs, trial)
        ref_snapshot = snapshot_inputs(inputs)
        try:
            new_out = self.new_model(*inputs)
            self.torch.npu.synchronize(device=self.req.device)
        except Exception as exc:
            return fail_result(
                compiled=True,
                runtime_error=(f"trial {trial}: candidate runtime error: {exc!r}"),
                failure_stage="execution",
                **runtime_failure(exc, stage="candidate_trial"),
            )
        if inputs_were_mutated(inputs, ref_snapshot):
            return fail_result(
                compiled=True,
                runtime_error=(f"trial {trial}: candidate mutated its inputs"),
            )
        if self._outputs_ok(ref_out, new_out):
            self.pass_count += 1
        else:
            self.max_diff = max(self.max_diff, max_abs_diff(ref_out, new_out))
            self.correctness_error = (
                f"trial {trial}: output mismatch "
                f"(passed {self.pass_count}/{self.req.num_correct_trials} "
                "so far)"
            )
        self.last_new_out = new_out
        return None

    def _record_first_call_correctness(
        self,
        first_out: Any,
        inputs: Sequence[Any],
        raw_inputs: Sequence[Any],
    ) -> None:
        """Record whether the warm-up call's output matched, as a diagnostic.

        The warm-up call already decides whether the sample compiled. This adds
        the separate question of whether that first output was right, which
        matters when a backend can lose synchronisation on a first call. It is
        recorded, never enforced: the scored protocol stays the correctness
        trials. Any failure here is recorded as a diagnostic error.
        """
        if first_out is None:
            self.metadata["first_call_correct"] = None
            self.metadata["first_call_check_error"] = "first call returned no output"
            return
        try:
            with self.torch.no_grad():
                ref_out = self._run_reference(inputs, raw_inputs, 0, stage="first call")
                self.torch.npu.synchronize(device=self.req.device)
            self.metadata["first_call_correct"] = self._outputs_ok(ref_out, first_out)
            self.metadata["first_call_max_difference"] = max_abs_diff(
                ref_out, first_out
            )
        except Exception as exc:
            self.metadata["first_call_correct"] = None
            self.metadata["first_call_check_error"] = repr(exc)

    ################################# TRIALS #################################

    def _fill_metadata(self) -> None:
        """Record protocol, reference mode, and runtime-stack facts."""
        self.metadata = {
            # Preserve the build facts recorded by _build_and_load.
            **self.metadata,
            **eval_protocol_metadata(
                hardware_name=self.req.hardware_name,
                precision=self.req.precision,
                seed=self.req.seed,
                num_correct_trials=self.req.num_correct_trials,
                num_warmup=self.req.num_warmup,
                num_perf_trials=self.req.num_perf_trials,
                l2_clear_size=self.req.l2_clear_size,
                atol=self.atol,
                rtol=self.rtol,
            ),
            "reference": self.ref_mode,
            "max_difference": self.max_diff,
            "correctness_passed": self.pass_count,
            "shared_library": str(self.so_path),
            "backend": self.backend.value,
            **npu_runtime_metadata(self.req.device),
        }
        if self.ref_npu_error:
            self.metadata["reference_npu_error"] = self.ref_npu_error
        if self.backend is Backend.PYASC:
            # Timing validity is decided by _validate_pyasc_timing, which
            # compares the event-measured mean against wall-clock time.
            self.metadata.setdefault("stream_mode", "pyasc_current_stream")

    def _unsupported_pyasc_dtypes(self, inputs: Sequence[Any]) -> list[str]:
        """Return dtypes pyasc cannot marshal, across inputs and parameters."""
        found: list[str] = []

        def consider(value: Any) -> None:
            dtype = getattr(value, "dtype", None)
            if dtype is None:
                return
            name = str(dtype).replace("torch.", "")
            if name in PYASC_UNSUPPORTED_DTYPES and name not in found:
                found.append(name)

        for value in inputs:
            consider(value)
        if self.new_model is not None:
            with contextlib.suppress(Exception):
                for parameter in self.new_model.parameters():
                    consider(parameter)
        return found

    ################################# TIMING #################################
    def _validate_pyasc_timing(self) -> None:
        """Check that NPU events really cover the candidate's execution.

        Compares the event-measured mean against a wall-clock measurement of
        the same calls. When the events do not cover the kernel the candidate
        is left with no speedup rather than a number that cannot be trusted.
        """
        if self.backend is not Backend.PYASC or self.runtime is None:
            return
        # Preparing the cross-check inputs can fail on its own, and it must
        # not escape: correctness is already established and only the
        # performance numbers are in question.
        try:
            inputs = [self._process_input(item) for item in self.get_inputs()]
        except Exception as exc:
            self._discard_timing(f"timing input preparation failed: {exc!r}")
            self.metadata.update(runtime_failure(exc, stage="timing_inputs"))
            return
        try:
            with self.torch.no_grad():
                for _ in range(2):
                    self.new_model(*inputs)
                self.torch.npu.synchronize()
                started = time.perf_counter()
                for _ in range(PYASC_TIMING_WALL_TRIALS):
                    self.new_model(*inputs)
                self.torch.npu.synchronize()
                wall_ms = (
                    (time.perf_counter() - started) * 1000.0 / PYASC_TIMING_WALL_TRIALS
                )
        except Exception as exc:
            self._discard_timing(f"wall-clock cross-check failed: {exc!r}")
            return
        ratio = self.runtime / wall_ms if wall_ms > 0 else 0.0
        self.metadata["timing_event_ms"] = float(f"{self.runtime:.6g}")
        self.metadata["timing_wall_ms"] = float(f"{wall_ms:.6g}")
        self.metadata["timing_event_wall_ratio"] = float(f"{ratio:.4g}")
        if ratio < PYASC_MIN_EVENT_WALL_RATIO:
            self._discard_timing(
                f"event time {self.runtime:.6g}ms is far below wall time "
                f"{wall_ms:.6g}ms (ratio {ratio:.3f}); the timed stream does "
                "not cover the kernel, so no speedup is reported"
            )
            return
        self.metadata["timing_valid"] = True

    def _discard_timing(self, reason: str) -> None:
        """Drop every timing-derived result so nothing scores an untrusted one.

        Correctness is untouched: the sample still counts in fast_0 and
        pass@k. What goes is the runtime, the trial statistics, the speedup and
        the SOL score, because a runtime the event/wall cross-check rejected
        must not reach a performance threshold or a geometric mean.
        """
        self.metadata["timing_valid"] = False
        self.metadata["timing_invalid_reason"] = reason
        self.runtime = None
        self.runtime_stats = None
        self.ref_runtime = None
        self.ref_runtime_stats = None
        self.metadata.pop("speedup", None)
        self.metadata["excessive_speedup"] = False
        for key in ("sol_score", "sol_bound_ms", "sol_bound_kind", "bytes_moved"):
            self.metadata.pop(key, None)

    def _time_both_models(self) -> None:
        """Time candidate and NPU reference; attach speedup and SOL metadata."""

        def draw_inputs() -> list[Any]:
            return [self._process_input(item) for item in self.get_inputs()]

        seed_torch(self.req.seed)
        probe_inputs = draw_inputs()
        input_bytes = tensor_nbytes(probe_inputs)
        fresh_per_trial = input_bytes <= REFRESH_INPUT_BYTES_LIMIT
        self.metadata["timing_fresh_inputs"] = bool(fresh_per_trial)
        perf_box: list[Any] = [probe_inputs]
        prev_box: list[Any] = [None]

        def refresh_inputs() -> None:
            prev_box[0] = perf_box[0]
            perf_box[0] = draw_inputs()

        def timed(fn: Any) -> list[float]:
            seed_torch(self.req.seed)
            return time_execution_with_npu_event(
                lambda: fn(*perf_box[0]),
                [],
                num_warmup=self.req.num_warmup,
                num_trials=self.req.num_perf_trials,
                device=self.torch_device,
                setup=refresh_inputs if fresh_per_trial else None,
                l2_clear_size=self.req.l2_clear_size,
            )

        try:
            with self.torch.no_grad():
                self.runtime_stats = get_timing_stats(timed(self.new_model))
                self.runtime = self.runtime_stats["mean"]
                if self.ref_mode == "npu":
                    self.ref_runtime_stats = get_timing_stats(timed(self.ref_model))
                    self.ref_runtime = self.ref_runtime_stats["mean"]
                    speedup = self.ref_runtime / self.runtime if self.runtime else 0.0
                    self.metadata["speedup"] = float(f"{speedup:.4g}")
                    self.metadata["excessive_speedup"] = bool(
                        speedup > self.req.excessive_speedup
                    )
            attach_sol_metadata(
                self.metadata,
                kernel_ms=(
                    self.runtime if isinstance(self.runtime, int | float) else None
                ),
                baseline_ms=(
                    self.ref_runtime
                    if isinstance(self.ref_runtime, int | float)
                    else None
                ),
                bytes_moved=input_bytes + tensor_nbytes(self.last_new_out),
                bandwidth_gbps=float(self.req.memory_bandwidth_gbps),
                flops=task_declared_flops(self.ref_globals),
                peak_tflops=(
                    float(self.req.peak_tflops) if self.req.peak_tflops else None
                ),
            )
        except Exception as exc:
            self.metadata["runtime_error"] = f"timing failed: {exc!r}"
            self.metadata.update(runtime_failure(exc, stage="timing"))

    ################################# TIMING #################################

    ################################ RECHECK #################################
    def _post_timing_recheck(self) -> dict[str, Any] | None:
        """Fail the sample if a fresh-input re-check mismatches or raises."""
        try:
            seed_torch(self.req.seed + 1)
            recheck_raw = self.get_inputs()
            recheck_inputs = [self._process_input(item) for item in recheck_raw]
            self.torch.npu.synchronize(device=self.req.device)
            with self.torch.no_grad():
                if self.ref_mode == "npu":
                    recheck_ref = self.ref_model(*recheck_inputs)
                    self.torch.npu.synchronize(device=self.req.device)
                else:
                    recheck_ref = self._run_ref_cpu(recheck_raw)
                recheck_new = self.new_model(*recheck_inputs)
                self.torch.npu.synchronize(device=self.req.device)
            if not self._outputs_ok(recheck_ref, recheck_new):
                self.metadata["correctness_error"] = (
                    "post-timing fresh-input re-check failed: outputs are not "
                    "a pure function of current inputs (caching or state drift)"
                )
                return self.metadata
        except Exception as exc:
            self.metadata["runtime_error"] = f"post-timing re-check failed: {exc!r}"
            self.metadata.update(runtime_failure(exc, stage="post_timing_recheck"))
            return self.metadata
        return None

    ################################ RECHECK #################################
