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

CASE_STATUSES = ("pending", "passed", "failed", "unsupported", "error")


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
            "task_correct": self.task_correct,
            "statuses": {str(key): value for key, value in sorted(self.statuses.items())},
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
            shapes = tuple(tuple(int(item) for item in shape) for shape in raw["input_shape"])
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
            tensor = torch.where(base > 0.5, torch.full_like(tensor, float("inf")), tensor)
            tensor = torch.where(base < -0.5, torch.full_like(tensor, float("-inf")), tensor)
        tensors.append(tensor)
    return tensors


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
    return summary
