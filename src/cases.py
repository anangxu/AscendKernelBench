"""Case specs for multi-case industry tasks.

One task is one operator; one generation is one sample; a sample is checked
against every required case. This module only describes cases and counts
outcomes, so a suite can add cases without touching the evaluator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

CASE_STATUSES = ("pending", "not_run", "passed", "failed", "unsupported", "error")


@dataclass(frozen=True)
class CaseSpec:
    """One official test case: shapes, dtypes, attrs, value range, and seed."""

    case_id: int
    input_shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[str, ...]
    attrs: Mapping[str, Any]
    value_range: tuple[float, float]
    seed: int = 0
    source_revision: str = ""
    note: str = ""
    required: bool = True
    unsupported: str = ""

    @property
    def epsilon(self) -> float:
        """Return the case epsilon when the case carries one."""
        return float(self.attrs.get("epsilon", 1e-6))

    def record(self) -> dict[str, Any]:
        """Return the JSON-ready description stored with a run."""
        return {
            "case_id": self.case_id,
            "input_shapes": [list(shape) for shape in self.input_shapes],
            "dtypes": list(self.dtypes),
            "attrs": dict(self.attrs),
            "value_range": list(self.value_range),
            "seed": self.seed,
            "source_revision": self.source_revision,
            "required": self.required,
            "unsupported": self.unsupported,
        }


@dataclass
class CaseSummary:
    """Per-candidate case outcome counts; a candidate counts once."""

    total: int = 0
    executed: int = 0
    passed: int = 0
    failed: int = 0
    unsupported: int = 0
    errors: int = 0
    not_run: int = 0
    statuses: dict[int, str] = field(default_factory=dict)

    @property
    def task_correct(self) -> bool:
        """A task passes when every required case passed."""
        return self.total > 0 and self.passed == self.total

    def record(self) -> dict[str, Any]:
        """Return the JSON-ready summary."""
        return {
            "cases_total": self.total,
            "cases_executed": self.executed,
            "cases_passed": self.passed,
            "cases_failed": self.failed,
            "cases_unsupported": self.unsupported,
            "cases_error": self.errors,
            "cases_not_run": self.not_run,
            "task_correct": self.task_correct,
            "statuses": {
                str(key): value for key, value in sorted(self.statuses.items())
            },
        }


def load_cases(
    path: str | Path,
    *,
    source_revision: str = "",
    seed_base: int = 0,
) -> tuple[CaseSpec, ...]:
    """Load case specs from a YAML file with a top-level cases list.

    Args:
        path: YAML file holding ``cases``.
        source_revision: Provenance recorded on every case.
        seed_base: Added to the case id to form the case seed.

    Raises:
        ValueError: If the file is malformed or a case lacks fields.
    """
    import yaml

    loaded: Any = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    items = loaded.get("cases") if isinstance(loaded, dict) else None
    if not isinstance(items, list) or not items:
        raise ValueError(f"{path}: expected a nonempty 'cases' list")
    specs: list[CaseSpec] = []
    for raw in items:
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: every case must be a mapping")
        try:
            case_id = int(raw["case_id"])
            shapes = tuple(
                tuple(int(item) for item in shape) for shape in raw["input_shape"]
            )
            dtypes = tuple(str(item) for item in raw["dtype"])
            attrs = dict(raw.get("attrs") or {})
            low, high = (float(item) for item in raw.get("value_range", [0.0, 1.0]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}: malformed case {raw!r}: {exc}") from exc
        specs.append(
            CaseSpec(
                case_id=case_id,
                input_shapes=shapes,
                dtypes=dtypes,
                attrs=attrs,
                value_range=(low, high),
                seed=seed_base + case_id,
                source_revision=source_revision,
                note=str(raw.get("note", "")),
            )
        )
    if len({spec.case_id for spec in specs}) != len(specs):
        raise ValueError(f"{path}: duplicate case ids")
    return tuple(specs)


def generate_inputs(case: CaseSpec, torch) -> list:
    """Build the case inputs from its shapes, dtypes, range, and seed.

    The value range also encodes special values: an infinite range mixes
    infinities with finite values, a NaN range fills NaN, and an all-zero range
    stays zero. No dtype is widened: tensors keep the case dtype.
    """
    torch.manual_seed(case.seed)
    low, high = case.value_range
    if math.isnan(low) or math.isnan(high):
        values = [float("nan")]
    elif math.isinf(low) or math.isinf(high):
        values = [float("-inf"), float("inf"), 1.0, -1.0]
    elif low == high:
        values = [low]
    else:
        values = None
    tensors = []
    for shape, name in zip(case.input_shapes, case.dtypes):
        dtype = getattr(torch, name)
        if values is None:
            tensor = torch.rand(shape, dtype=torch.float32) * (high - low) + low
            tensor = tensor.to(dtype)
        elif len(values) == 1:
            tensor = torch.full(shape, values[0], dtype=dtype)
        else:
            base = torch.rand(shape, dtype=torch.float32) * 2.0 - 1.0
            tensor = base.to(dtype)
            inf = torch.full_like(tensor, float("inf"))
            tensor = torch.where(base > 0.5, inf, tensor)
            ninf = torch.full_like(tensor, float("-inf"))
            tensor = torch.where(base < -0.5, ninf, tensor)
        tensors.append(tensor)
    return tensors


def compare_outputs(reference, candidate, *, atol: float, rtol: float):
    """Compare a case output, matching NaN masks and Inf signs.

    Finite elements compare within atol/rtol. A NaN in the reference must be a
    NaN in the candidate at the same position, and a candidate NaN where the
    reference is finite always fails. All-NaN outputs only pass when the
    reference is all-NaN, so skipping every non-finite element cannot pass.

    Returns:
        (ok, detail) where detail explains the first failure class.
    """
    import torch

    ref = reference.detach().to(torch.float32)
    cand = candidate.detach().to(torch.float32)
    if tuple(ref.shape) != tuple(cand.shape):
        return False, f"shape {tuple(cand.shape)} != {tuple(ref.shape)}"
    if ref.numel() == 0:
        return False, "empty output"
    ref_nan = torch.isnan(ref)
    cand_nan = torch.isnan(cand)
    if bool((ref_nan != cand_nan).any()):
        return False, "NaN mask mismatch"
    finite = ~ref_nan
    if int(finite.sum()) == 0:
        return True, "all non-finite and matching"
    ref_f, cand_f = ref[finite], cand[finite]
    ref_inf, cand_inf = torch.isinf(ref_f), torch.isinf(cand_f)
    if bool((ref_inf != cand_inf).any()):
        return False, "Inf position mismatch"
    if bool(ref_inf.any()):
        if bool((torch.sign(ref_f[ref_inf]) != torch.sign(cand_f[cand_inf])).any()):
            return False, "Inf sign mismatch"
    keep = ~ref_inf
    if bool(keep.any()):
        if not bool(torch.allclose(ref_f[keep], cand_f[keep], rtol=rtol, atol=atol)):
            return False, "finite values out of tolerance"
    return True, "ok"


def aggregate_speedup(records: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Return the task-level speedup and the per-case diagnostics.

    The formal speedup is the geometric mean of the per-case speedups, and only
    when every case passed with a valid speedup. Otherwise it is None and the
    collected values stay diagnostic.
    """
    import math

    values: dict[int, float] = {}
    for key, record in records.items():
        if record.get("status") != "passed":
            continue
        speedup = (record.get("metadata") or {}).get("speedup")
        if isinstance(speedup, (int, float)) and speedup > 0:
            values[int(key)] = float(speedup)
    expected = sum(1 for record in records.values() if record.get("status") == "passed")
    valid = len(values) == len(records) and expected == len(records) and bool(values)
    if not valid:
        return {"speedup": None, "diagnostic_speedups": values}
    mean = math.exp(sum(math.log(value) for value in values.values()) / len(values))
    return {"speedup": mean, "diagnostic_speedups": values}


def summarize(cases: tuple[CaseSpec, ...], statuses: Mapping[int, str]) -> CaseSummary:
    """Count one candidate's per-case outcomes.

    A candidate counts once regardless of how many cases it was checked
    against, so adding cases never inflates the candidate count.

    Raises:
        ValueError: If a status is not a known classification.
    """
    summary = CaseSummary(total=len(cases))
    for case in cases:
        status = statuses.get(case.case_id, "pending")
        if status not in CASE_STATUSES:
            raise ValueError(f"case {case.case_id}: unknown status {status!r}")
        summary.statuses[case.case_id] = status
        if status == "passed":
            summary.passed += 1
            summary.executed += 1
        elif status == "failed":
            summary.failed += 1
            summary.executed += 1
        elif status == "unsupported":
            summary.unsupported += 1
        elif status == "error":
            summary.errors += 1
            summary.executed += 1
        elif status == "not_run":
            summary.not_run += 1
    return summary
