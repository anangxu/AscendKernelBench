"""Isolated eager-reference timing worker for archived baselines.

Evaluation always re-measures the live NPU reference in the same process
as the candidate; this worker only archives torch_npu eager timings.
"""

from __future__ import annotations

import sys
from typing import Any

from .compare import move_value_to_device
from .eval_device import exec_python_source
from .io_util import load_cfg_argv, pop_required_path, write_json_atomic
from .runtime import npu_device_index, seed_torch, torch_dtype_for
from .timing import (
    DEFAULT_L2_CLEAR_BYTES,
    AdaptiveInputTimer,
    get_timing_stats,
)


def measure_reference(cfg: dict[str, Any]) -> dict[str, Any]:
    """Time the task Model on NPU using the evaluation timing protocol.

    Returns:
        Timing statistics plus supported_on_npu; error string when unsupported.
    """
    import torch
    import torch_npu  # noqa: F401

    ref_globals = exec_python_source(cfg["task_py"], "<task.py>")
    model_cls = ref_globals["Model"]
    get_init_inputs = ref_globals["get_init_inputs"]
    get_inputs = ref_globals["get_inputs"]

    device = torch.device(cfg["device"])
    torch.npu.set_device(device.index or npu_device_index(cfg["device"]))
    dtype = torch_dtype_for(cfg["precision"])
    l2_clear_size = int(cfg.get("l2_clear_size", DEFAULT_L2_CLEAR_BYTES))

    def process_input(value: Any) -> Any:
        """Move a tensor onto the baseline device."""
        return move_value_to_device(value, device, dtype)

    def draw() -> list[Any]:
        """Draw one processed input set."""
        return [process_input(item) for item in get_inputs()]

    try:
        seed_torch(cfg["seed"])
        model = model_cls(*get_init_inputs()).to(device=device, dtype=dtype)
        model.eval()
        timer = AdaptiveInputTimer(
            draw,
            seed=cfg["seed"],
            num_warmup=cfg["num_warmup"],
            num_trials=cfg["num_perf_trials"],
            device=device,
            l2_clear_size=l2_clear_size,
        )
        ############################### TIMING ###############################
        torch.npu.synchronize(device=device)
        with torch.no_grad():
            times = timer.time(model)
        ############################### TIMING ###############################
        stats = get_timing_stats(times)
        stats["supported_on_npu"] = True
        stats["timing_fresh_inputs"] = bool(timer.fresh_per_trial)
        return stats
    except Exception as exc:
        return {"supported_on_npu": False, "error": repr(exc)}


def main(argv: list[str]) -> None:
    """Read cfg.json, time the reference, and write out_path."""
    from .log import setup_logging

    setup_logging(rich_tracebacks=False)
    cfg = load_cfg_argv(argv)
    out_path = pop_required_path(cfg, "out_path")
    stats = measure_reference(cfg)
    write_json_atomic(out_path, stats)


if __name__ == "__main__":
    main(sys.argv)
