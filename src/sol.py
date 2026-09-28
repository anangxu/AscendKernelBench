"""Hardware Speed-of-Light (SOL) scoring after NVIDIA SOL-ExecBench.

The score measures how much of the gap between a software baseline and an
analytic roofline bound a kernel closes: 0.5 matches the baseline.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence

_MS_PER_S = 1_000.0
_BYTES_PER_GB = 1e9
_FLOPS_PER_TFLOP = 1e12


def roofline_bound_ms(
    *,
    bytes_moved: int,
    bandwidth_gbps: float,
    flops: float | None = None,
    peak_tflops: float | None = None,
) -> float | None:
    """Return a roofline lower bound in milliseconds, or None if undefined."""
    if bytes_moved <= 0 or bandwidth_gbps <= 0:
        return None
    t_mem_ms = (bytes_moved / (bandwidth_gbps * _BYTES_PER_GB)) * _MS_PER_S
    if flops is None or peak_tflops is None or flops <= 0 or peak_tflops <= 0:
        return t_mem_ms
    t_compute_ms = (flops / (peak_tflops * _FLOPS_PER_TFLOP)) * _MS_PER_S
    return max(t_mem_ms, t_compute_ms)


################################## SCORING ##################################
def sol_score(
    kernel_ms: float,
    baseline_ms: float,
    sol_ms: float,
) -> float | None:
    """Return the SOL-ExecBench-style score in [0, 1], or None.

    The bound is clamped below both measured times so a noisy bound
    cannot invert the score.
    """
    if kernel_ms <= 0 or baseline_ms <= 0 or sol_ms <= 0:
        return None
    sol_ms = min(sol_ms, kernel_ms * 0.999, baseline_ms * 0.999)
    gap = baseline_ms - sol_ms
    if gap <= 0:
        return None
    score = gap / ((kernel_ms - sol_ms) + gap)
    return max(0.0, min(1.0, score))


################################## SCORING ##################################


def attach_sol_metadata(
    metadata: MutableMapping[str, object],
    *,
    kernel_ms: float | None,
    baseline_ms: float | None,
    bytes_moved: int,
    bandwidth_gbps: float,
    flops: float | None = None,
    peak_tflops: float | None = None,
) -> None:
    """Record the roofline bound and SOL score on metadata when defined."""
    bound = roofline_bound_ms(
        bytes_moved=bytes_moved,
        bandwidth_gbps=bandwidth_gbps,
        flops=flops,
        peak_tflops=peak_tflops,
    )
    if bound is None:
        return
    metadata["bytes_moved"] = int(bytes_moved)
    metadata["sol_bound_ms"] = float(f"{bound:.6g}")
    metadata["sol_bound_kind"] = (
        "roofline" if flops and peak_tflops else "roofline_memory"
    )
    if not isinstance(kernel_ms, (int, float)):
        return
    if not isinstance(baseline_ms, (int, float)):
        return
    score = sol_score(float(kernel_ms), float(baseline_ms), bound)
    if score is not None:
        metadata["sol_score"] = float(f"{score:.6g}")


def task_declared_flops(namespace: Mapping[str, object]) -> float | None:
    """Return a positive FLOP count declared by the task, if any.

    The vendored KernelBench corpus declares none, so the default SOL
    bound stays memory-only unless a task author adds that constant.
    """
    for key in ("FLOPS", "NUM_FLOPS"):
        value = namespace.get(key)
        if isinstance(value, (int, float)) and float(value) > 0:
            return float(value)
    return None


def mean_sol_score(samples: Sequence[Mapping[str, object]]) -> float | None:
    """Return the mean of per-sample metadata.sol_score values."""
    scores: list[float] = []
    for sample in samples:
        metadata = sample.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            continue
        if metadata.get("timing_valid") is False:
            continue
        value = metadata.get("sol_score")
        if isinstance(value, (int, float)):
            scores.append(float(value))
    if not scores:
        return None
    return sum(scores) / len(scores)
