"""NPU event timing with per-trial L2 clearing (docs/guide/evaluation.md).

Warmup empties the allocator cache; each trial times one call with an event
pair, thrashes L2 outside the event window, and drops the first trial.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from .compare import tensor_nbytes
from .runtime import seed_torch

DEFAULT_L2_CLEAR_BYTES = 256 * 1024 * 1024
# Input sets larger than this reuse one allocation during timing (KernelBench
# L1 already ships multi-gigabyte cases). Smaller sets are redrawn per trial.
REFRESH_INPUT_BYTES_LIMIT = 256 * 1024 * 1024


def l2_clear_bytes(l2_cache_mb: int) -> int:
    """Return a flush-buffer size in bytes: at least 256 MiB and 2x L2.

    Args:
        l2_cache_mb: Profile L2 capacity in mebibytes; 0 keeps the default.
    """
    if l2_cache_mb <= 0:
        return DEFAULT_L2_CLEAR_BYTES
    twice = 2 * int(l2_cache_mb) * 1024 * 1024
    return max(DEFAULT_L2_CLEAR_BYTES, twice)


def clear_l2_cache(
    device: Any,
    size_bytes: int = DEFAULT_L2_CLEAR_BYTES,
) -> None:
    """Thrash device memory so the next kernel misses in L2.

    size_bytes is rounded down to whole int64 elements; a non-positive
    size is a no-op.
    """
    import torch

    n_elements = max(int(size_bytes), 0) // 8
    if n_elements <= 0:
        return
    dummy = torch.empty((n_elements,), dtype=torch.int64, device=device)
    dummy.fill_(42)
    del dummy


class AdaptiveInputTimer:
    """Seed, draw inputs, and time calls under the refresh policy.

    The first draw decides whether each trial gets fresh inputs. The same
    instance must time the candidate and reference so both start from the
    same seed and keep the previous allocation alive.
    """

    def __init__(
        self,
        draw_inputs: Callable[[], Sequence[Any]],
        *,
        seed: int,
        num_warmup: int,
        num_trials: int,
        device: Any,
        l2_clear_size: int,
    ) -> None:
        """Draw the probe input set and record the refresh decision."""
        self._draw_inputs = draw_inputs
        self._seed = seed
        self._num_warmup = num_warmup
        self._num_trials = num_trials
        self._device = device
        self._l2_clear_size = l2_clear_size
        seed_torch(seed)
        probe = list(draw_inputs())
        self.input_bytes = tensor_nbytes(probe)
        self.fresh_per_trial = self.input_bytes <= REFRESH_INPUT_BYTES_LIMIT
        self._perf_box: list[Sequence[Any]] = [probe]
        self._prev_box: list[Sequence[Any] | None] = [None]

    def _refresh(self) -> None:
        """Replace the live input set and keep the previous allocation."""
        self._prev_box[0] = self._perf_box[0]
        self._perf_box[0] = list(self._draw_inputs())

    def time(self, kernel_fn: Callable[..., Any]) -> list[float]:
        """Time kernel_fn(*inputs) with the configured warmup and trials."""
        seed_torch(self._seed)
        return time_execution_with_npu_event(
            lambda: kernel_fn(*self._perf_box[0]),
            [],
            num_warmup=self._num_warmup,
            num_trials=self._num_trials,
            device=self._device,
            setup=self._refresh if self.fresh_per_trial else None,
            l2_clear_size=self._l2_clear_size,
        )


def time_execution_with_npu_event(
    kernel_fn: Callable[..., Any],
    args: Sequence[Any],
    *,
    num_warmup: int,
    num_trials: int,
    device: Any,
    l2_clear_size: int,
    setup: Callable[[], None] | None = None,
    discard_first: int = 1,
) -> list[float]:
    """Time kernel_fn(*args) with torch.npu.Event, in milliseconds.

    num_trials counts retained measurements and the first discard_first
    trials are dropped. setup runs before every warmup and timed call,
    outside the event window, so a trial can use fresh inputs.
    """
    import torch

    previous_device = torch.npu.current_device()
    torch.npu.set_device(device)
    try:
        for _ in range(num_warmup):
            if setup is not None:
                setup()
            kernel_fn(*args)
            torch.npu.synchronize(device=device)

        # Releases PyTorch's caching allocator, not the device L2 cache.
        torch.npu.empty_cache()

        elapsed_times: list[float] = []
        ##################### TIMED LOOP #####################
        for trial in range(num_trials + discard_first):
            if setup is not None:
                setup()
            torch.npu.synchronize(device=device)

            start_event = torch.npu.Event(enable_timing=True)
            end_event = torch.npu.Event(enable_timing=True)

            clear_l2_cache(device=device, size_bytes=l2_clear_size)

            start_event.record()
            kernel_fn(*args)
            end_event.record()

            torch.npu.synchronize(device=device)

            elapsed_time_ms = start_event.elapsed_time(end_event)
            if trial >= discard_first:
                elapsed_times.append(elapsed_time_ms)
        ##################### TIMED LOOP #####################
    finally:
        torch.npu.set_device(previous_device)

    return elapsed_times


def get_timing_stats(elapsed_times: Sequence[float]) -> dict[str, float | int]:
    """Return mean/std/min/max/num_trials rounded to 3 significant digits.

    Raises:
        ValueError: If elapsed_times is empty.
    """
    if not elapsed_times:
        raise ValueError("elapsed_times must be a non-empty sequence")
    values = np.asarray(elapsed_times, dtype=np.float64)
    return {
        "mean": float(f"{float(np.mean(values)):.3g}"),
        "std": float(f"{float(np.std(values)):.3g}"),
        "min": float(f"{float(np.min(values)):.3g}"),
        "max": float(f"{float(np.max(values)):.3g}"),
        "num_trials": len(elapsed_times),
    }
