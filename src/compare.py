"""Dtype-aware comparison of reference and candidate outputs.

Floating and complex tensors use allclose, integer and Boolean tensors
require exact equality. CPU-reference mode compares on host memory.
"""

from __future__ import annotations

import os

from collections.abc import Callable, Mapping, Sequence
from typing import Any

CompareFn = Callable[[Any, Any], bool]


def resolve_tolerances(
    precision: str,
    tolerances: Mapping[str, Mapping[str, float]],
    task_tolerance: Mapping[str, Any] | None,
) -> tuple[float, float]:
    """Resolve atol and rtol from a task override or the eval config."""
    if task_tolerance:
        return (
            float(task_tolerance.get("atol", 1e-4)),
            float(task_tolerance.get("rtol", 1e-4)),
        )
    tol = tolerances.get(precision, {"atol": 1e-4, "rtol": 1e-4})
    return float(tol["atol"]), float(tol["rtol"])


def is_discrete_tensor(value: Any) -> bool:
    """Return True if value is a non-floating, non-complex tensor."""
    import torch

    return isinstance(value, torch.Tensor) and not (
        torch.is_floating_point(value) or torch.is_complex(value)
    )


################################## MATCHING ##################################
def tensor_match(ref: Any, new: Any, *, atol: float, rtol: float) -> bool:
    """Return True if two tensors match in shape and values."""
    import torch

    if not isinstance(ref, torch.Tensor) or not isinstance(new, torch.Tensor):
        return False
    if ref.shape != new.shape:
        return False
    if is_discrete_tensor(ref) and is_discrete_tensor(new):
        return bool(torch.equal(ref, new))
    try:
        if os.environ.get("AKB_CASE_COMPARE_TASK"):
            from .cases import compare_outputs

            ok, _ = compare_outputs(ref, new, atol=atol, rtol=rtol)
            return ok
        return bool(torch.allclose(ref, new, atol=atol, rtol=rtol))
    except RuntimeError:
        return False


################################## MATCHING ##################################


def outputs_match(ref: Any, new: Any, *, atol: float, rtol: float) -> bool:
    """Recursively compare a reference output tree to a candidate tree."""
    import torch

    if isinstance(ref, torch.Tensor):
        return tensor_match(ref, new, atol=atol, rtol=rtol)
    if isinstance(ref, (tuple, list)):
        return (
            isinstance(new, (tuple, list))
            and len(ref) == len(new)
            and all(
                outputs_match(left, right, atol=atol, rtol=rtol)
                for left, right in zip(ref, new, strict=True)
            )
        )
    return bool(ref == new)


def to_cpu_compare_value(value: Any) -> Any:
    """Move a tensor to CPU, casting floating values to float32."""
    import torch

    if not isinstance(value, torch.Tensor):
        return value
    if torch.is_floating_point(value):
        return value.float().cpu()
    return value.cpu()


def to_cpu_compare_tree(value: Any) -> Any:
    """Apply to_cpu_compare_value to a tensor or a flat sequence."""
    if isinstance(value, (tuple, list)):
        return [to_cpu_compare_value(item) for item in value]
    return to_cpu_compare_value(value)


def compare_candidate_outputs(
    ref: Any,
    new: Any,
    *,
    ref_mode: str,
    atol: float,
    rtol: float,
    custom_check: CompareFn | None,
) -> bool:
    """Compare candidate outputs under the npu or cpu reference mode."""
    if ref_mode == "cpu":
        ref = to_cpu_compare_tree(ref)
        new = to_cpu_compare_tree(new)
    if custom_check is not None:
        return bool(custom_check(ref, new))
    return outputs_match(ref, new, atol=atol, rtol=rtol)


def max_abs_diff(ref: Any, new: Any) -> float:
    """Return a finite max-abs difference for equal-shaped top-level tensors."""
    import torch

    if not (
        isinstance(ref, torch.Tensor)
        and isinstance(new, torch.Tensor)
        and ref.shape == new.shape
    ):
        return 0.0
    if ref.device != new.device:
        ref = ref.cpu()
        new = new.cpu()
    if torch.is_floating_point(ref) or torch.is_floating_point(new):
        ref = ref.float()
        new = new.float()
    try:
        value = float(torch.abs(ref - new).max().item())
    except (RuntimeError, TypeError):
        return 0.0
    if value != value:  # NaN
        return 0.0
    return value


################################ HIDDEN INPUTS ################################
# Value-only transforms from KernelBench-Verified. Shapes stay fixed so
# shape specialization remains a legal optimization.
HIDDEN_DISTRIBUTIONS: tuple[tuple[str, float], ...] = (
    ("d1", 1.0),
    ("d2", 3.0),
    ("d3", 0.01),
    ("d4", -1.0),
)
################################ HIDDEN INPUTS ################################


def perturb_floating_inputs(inputs: Sequence[Any], scale: float) -> list[Any]:
    """Clone inputs and scale nonzero floating values by scale.

    Integer, Boolean, and exact-zero entries are left unchanged, as are
    shapes. scale 1 is a clone of the original draw.
    """
    return [_perturb_floating(item, scale) for item in inputs]


def _perturb_floating(value: Any, scale: float) -> Any:
    """Return a scaled clone of one input node."""
    import torch

    if isinstance(value, torch.Tensor):
        out = value.detach().clone()
        if scale != 1.0 and (torch.is_floating_point(out) or torch.is_complex(out)):
            nonzero = out != 0
            out = torch.where(nonzero, out * scale, out)
        return out
    if isinstance(value, list):
        return [_perturb_floating(item, scale) for item in value]
    if isinstance(value, tuple):
        return tuple(_perturb_floating(item, scale) for item in value)
    return value


def snapshot_inputs(inputs: Sequence[Any]) -> list[Any]:
    """Clone top-level tensors so later mutations can be detected."""
    import torch

    return [item.clone() if isinstance(item, torch.Tensor) else item for item in inputs]


def inputs_were_mutated(inputs: Sequence[Any], snapshot: Sequence[Any]) -> bool:
    """Return True if any top-level tensor differs from its snapshot."""
    import torch

    return any(
        isinstance(left, torch.Tensor)
        and isinstance(right, torch.Tensor)
        and not torch.equal(left, right)
        for left, right in zip(inputs, snapshot, strict=True)
    )


def move_value_to_device(value: Any, device: Any, dtype: Any) -> Any:
    """Move a tensor to device, casting only floating dtypes."""
    import torch

    if not isinstance(value, torch.Tensor):
        return value
    if torch.is_floating_point(value):
        return value.to(device=device, dtype=dtype)
    return value.to(device=device)


def tensor_nbytes(value: Any) -> int:
    """Return the number of bytes in tensors in value (one nesting level)."""
    import torch

    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return sum(tensor_nbytes(item) for item in value)
    return 0
