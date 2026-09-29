"""Scoring: fast_p and pass@k, KernelBench-compatible schema.

fast_p counts every collected sample in the denominator, including
compile failures. Format compatibility does not imply comparability.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from statistics import geometric_mean
from typing import Any

from .sol import mean_sol_score

FAST_P_THRESHOLDS = (0.0, 0.5, 0.8, 1.0, 1.5, 2.0)


def _threshold_name(t: float) -> str:
    """Return the KernelBench-style key, for example fast_0.5."""
    return f"fast_{t:g}"


def _by_count(counts: Mapping[str, int]) -> dict[str, int]:
    """Return a count map ordered by descending count, then by key."""
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def sample_speedup(sample: Mapping[str, Any]) -> float | None:
    """Return speedup (ref mean / kernel mean), or None if unusable.

    A sample whose timing the backend rejected is excluded here, and this is
    the only gate the performance numbers read: the thresholds above fast_0,
    the geometric mean and the per-problem best speedup all come through it.
    Correctness rates are computed from the correctness flag instead, so the
    sample still counts there.
    """
    if not sample.get("correctness"):
        return None
    metadata = sample.get("metadata") or {}
    if metadata.get("excessive_speedup"):
        return None
    if metadata.get("timing_valid") is False:
        return None
    runtime = sample.get("runtime")
    ref_runtime = sample.get("ref_runtime")
    if runtime and ref_runtime and runtime > 0:
        return ref_runtime / runtime
    return None


################################## SCORING ##################################
def fast_p(samples: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Return fast_p over a flat sample list, counting every sample."""
    total = len(samples)
    if total == 0:
        return {_threshold_name(t): 0.0 for t in FAST_P_THRESHOLDS}
    result: dict[str, float] = {}
    for t in FAST_P_THRESHOLDS:
        count = 0
        for sample in samples:
            if not sample.get("correctness"):
                continue
            if t == 0.0:
                count += 1
                continue
            speedup = sample_speedup(sample)
            if speedup is not None and speedup > t:
                count += 1
        result[_threshold_name(t)] = count / total
    return result


################################## SCORING ##################################


def geometric_mean_speedup(samples: Sequence[Mapping[str, Any]]) -> float:
    """Return the geometric mean speedup over correct, non-flagged samples."""
    speedups = [s for s in (sample_speedup(x) for x in samples) if s is not None]
    if not speedups:
        return 0.0
    return float(geometric_mean(speedups))


def pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    """Return the standard unbiased pass@k estimator (KernelBench)."""
    if num_samples < k:
        return float(num_correct > 0)
    if num_samples - num_correct < k:
        return 1.0
    return 1.0 - math.comb(num_samples - num_correct, k) / math.comb(num_samples, k)


def summarize_eval_results(
    eval_results: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Aggregate an eval_results.json mapping into headline metrics."""
    all_samples = [s for samples in eval_results.values() for s in samples]
    compiled = sum(1 for s in all_samples if s.get("compiled"))
    correct = sum(1 for s in all_samples if s.get("correctness"))

    def _meta(sample: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return sample metadata when it is a mapping, else an empty dict."""
        metadata = sample.get("metadata") or {}
        return metadata if isinstance(metadata, Mapping) else {}

    cpu_reference = sum(
        1 for sample in all_samples if _meta(sample).get("reference") == "cpu"
    )
    npu_reference = sum(
        1 for sample in all_samples if _meta(sample).get("reference") == "npu"
    )
    flagged = sum(1 for sample in all_samples if _meta(sample).get("excessive_speedup"))
    hidden_failed = sum(
        1 for sample in all_samples if _meta(sample).get("hidden_failed")
    )
    # Samples the backend cannot run at all stay in every rate above; these
    # counters keep them visible so a success rate cannot hide the gap.
    unsupported_dtype = sum(
        1
        for sample in all_samples
        if _meta(sample).get("failure_stage") == "unsupported_dtype"
    )
    unsupported_operator = sum(
        1
        for sample in all_samples
        if _meta(sample).get("failure_stage") == "unsupported_operator"
    )
    timing_invalid = sum(
        1 for sample in all_samples if _meta(sample).get("timing_valid") is False
    )
    # Runtime failures are classified generically, by exception class and by
    # the stage that raised. The harness records what happened; it never
    # decides on its own whether the model or the environment caused it, so
    # these counters report both without moving a sample out of any rate.
    runtime_classes: dict[str, int] = {}
    runtime_stages: dict[str, int] = {}
    for sample in all_samples:
        meta = _meta(sample)
        name = meta.get("runtime_error_class")
        if name:
            runtime_classes[str(name)] = runtime_classes.get(str(name), 0) + 1
        elif meta.get("runtime_error"):
            key = "unclassified"
            runtime_classes[key] = runtime_classes.get(key, 0) + 1
        stage = meta.get("runtime_error_stage")
        if stage:
            runtime_stages[str(stage)] = runtime_stages.get(str(stage), 0) + 1
    per_problem: dict[str, dict[str, Any]] = {}
    for problem_id, samples in eval_results.items():
        n = len(samples)
        c = sum(1 for s in samples if s.get("correctness"))
        per_problem[str(problem_id)] = {
            "num_samples": n,
            "num_correct": c,
            "any_correct": c > 0,
        }
    summary: dict[str, Any] = {
        "total_samples": len(all_samples),
        "total_problems": len(eval_results),
        "compiled": compiled,
        "correct": correct,
        "cpu_reference": cpu_reference,
        "npu_reference": npu_reference,
        "excessive_speedup": flagged,
        "hidden_failed": hidden_failed,
        "unsupported_dtype": unsupported_dtype,
        "unsupported_dtype_backend_wide": sum(
            1
            for sample in all_samples
            if _meta(sample).get("failure_stage") == "unsupported_dtype"
            and _meta(sample).get("limitation_scope") == "backend"
        ),
        "unsupported_operator": unsupported_operator,
        "timing_invalid": timing_invalid,
        "runtime_error_classes": _by_count(runtime_classes),
        "runtime_error_stages": _by_count(runtime_stages),
        "fast_p": fast_p(all_samples),
        "geometric_mean_speedup_correct_only": geometric_mean_speedup(all_samples),
        "mean_sol_score": mean_sol_score(all_samples),
        "per_problem": per_problem,
    }
    return summary


def compute_pass_at_k(
    eval_results: dict[str, list[dict]], ks: Sequence[int] = (1, 5, 10)
) -> dict[str, dict[str, float]]:
    """Compute pass@k per problem and the unweighted average."""
    per_problem: dict[str, dict[str, float]] = {}
    for problem_id, samples in eval_results.items():
        n = len(samples)
        c = sum(1 for s in samples if s.get("correctness"))
        per_problem[problem_id] = {
            f"pass@{k}": pass_at_k(n, c, k) for k in ks if k <= n or k == 1
        }
    if not per_problem:
        return {"per_problem": {}, "average": {}}
    average = {}
    for k in ks:
        key = f"pass@{k}"
        values = [v[key] for v in per_problem.values() if key in v]
        if values:
            average[key] = sum(values) / len(values)
    return {"per_problem": per_problem, "average": average}
