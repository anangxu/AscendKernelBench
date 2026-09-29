"""KernelBench-compatible evaluation result payloads."""

from __future__ import annotations

from typing import Any

from .compare import HIDDEN_DISTRIBUTIONS

__all__ = [
    "compiled_result",
    "eval_protocol_metadata",
    "fail_result",
    "runtime_failure",
    "summarize_runtime_error",
]

_TOP_LEVEL_RESULT_KEYS = {
    "compiled",
    "correctness",
    "runtime",
    "runtime_stats",
    "ref_runtime",
    "ref_runtime_stats",
    "metadata",
}

_RUNTIME_SUMMARY_LIMIT = 240


def summarize_runtime_error(exc: BaseException) -> str:
    """Return a one-line summary of a runtime exception message."""
    text = " ".join(str(exc).split())
    if len(text) > _RUNTIME_SUMMARY_LIMIT:
        return text[: _RUNTIME_SUMMARY_LIMIT - 3] + "..."
    return text


def runtime_failure(exc: BaseException, *, stage: str) -> dict[str, object]:
    """Return the classification metadata of one runtime exception.

    The class and the stage say what happened and where. Nothing here decides
    whether the model or the environment caused it: an out-of-memory error can
    come from a bad tiling plan or from a device limit, and the summary keeps
    the message the runtime produced so a reader can tell them apart.

    Args:
        exc: The exception that ended the step.
        stage: Where it was raised, for example candidate_trial or timing.
    """
    return {
        "runtime_error_class": type(exc).__name__,
        "runtime_error_stage": stage,
        "runtime_error_summary": summarize_runtime_error(exc),
    }


def fail_result(
    *,
    compiled: bool = False,
    compilation_error: str | None = None,
    runtime_error: str | None = None,
    **extra_metadata: object,
) -> dict[str, Any]:
    """Return a failed evaluation payload with empty timing fields.

    Extra metadata keys merge into metadata; top-level result keys are
    ignored so callers cannot overwrite the payload schema.
    """
    extras = {
        key: value
        for key, value in extra_metadata.items()
        if key not in _TOP_LEVEL_RESULT_KEYS
    }
    return {
        "compiled": compiled,
        "correctness": False,
        "runtime": None,
        "runtime_stats": None,
        "ref_runtime": None,
        "ref_runtime_stats": None,
        "metadata": {
            "compilation_error": compilation_error,
            "runtime_error": runtime_error,
            **extras,
        },
    }


def compiled_result(
    *,
    correctness: bool,
    metadata: dict[str, Any],
    runtime: float | None = None,
    runtime_stats: dict[str, Any] | None = None,
    ref_runtime: float | None = None,
    ref_runtime_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a post-build result payload with compiled True.

    runtime and ref_runtime are mean latencies in milliseconds, or None.
    """
    return {
        "compiled": True,
        "correctness": correctness,
        "runtime": runtime,
        "runtime_stats": runtime_stats,
        "ref_runtime": ref_runtime,
        "ref_runtime_stats": ref_runtime_stats,
        "metadata": metadata,
    }


def eval_protocol_metadata(
    *,
    hardware_name: str,
    precision: str,
    seed: int,
    num_correct_trials: int,
    num_warmup: int,
    num_perf_trials: int,
    l2_clear_size: int,
    atol: float,
    rtol: float,
) -> dict[str, object]:
    """Return the evaluation-protocol snapshot stored on scored samples.

    precision is a floating dtype key: fp32, fp16, or bf16. l2_clear_size
    is in bytes and is allocated to flush L2 before each timed call.
    """
    return {
        "hardware": hardware_name,
        "precision": precision,
        "seed": seed,
        "correctness_trials": num_correct_trials,
        "num_warmup": num_warmup,
        "num_perf_trials": num_perf_trials,
        "operator_mode": "aclnn",
        "l2_clear_size": int(l2_clear_size),
        "atol": atol,
        "rtol": rtol,
        "hidden_distributions": [name for name, _scale in HIDDEN_DISTRIBUTIONS],
    }
