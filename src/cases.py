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
    value_range: tuple[float, float] | tuple[tuple[float, float], ...]
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
            "value_range": [
                list(item) if isinstance(item, (list, tuple)) else item
                for item in self.value_range
            ]
            if isinstance(self.value_range[0], (list, tuple))
            else list(self.value_range),
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
                None
                if shape is None
                else tuple(tuple(int(v) for v in inner) for inner in shape)
                if shape and isinstance(shape[0], (list, tuple))
                else tuple(int(item) for item in shape)
                for shape in raw["input_shape"]
            )
            dtypes = tuple(
                None
                if item is None
                else tuple(str(v) for v in item)
                if isinstance(item, (list, tuple))
                else str(item)
                for item in raw["dtype"]
            )
            attrs = dict(raw.get("attrs") or {})
            raw_range = raw.get("value_range", [0.0, 1.0])
            if isinstance(raw_range, str):
                # Semantic range such as "unique"; recorded, applied at generation.
                value_range = (raw_range,)
            elif raw_range and isinstance(raw_range[0], str):
                parts = list(raw_range)
                value_range = (tuple(parts),) if len(parts) > 1 else (str(parts[0]),)
            elif raw_range and isinstance(raw_range[0], (list, tuple)):
                parsed = []
                for item in raw_range:
                    if item is None:
                        parsed.append(None)
                    elif isinstance(item[0], (list, tuple)):
                        parsed.append(
                            tuple(
                                (float(inner[0]), float(inner[1]))
                                if not isinstance(inner[0], str)
                                else tuple([str(inner[0])] + [float(v) for v in inner[1:]])
                                for inner in item
                            )
                        )
                    elif isinstance(item[0], str):
                        parsed.append(
                            tuple([str(item[0])] + [float(v) for v in item[1:]])
                        )
                    else:
                        parsed.append((float(item[0]), float(item[1])))
                value_range = tuple(parsed) + (None,) * (
                    len(raw["input_shape"]) - len(parsed)
                )
            else:
                value_range = tuple(float(item) for item in raw_range)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}: malformed case {raw!r}: {exc}") from exc
        specs.append(
            CaseSpec(
                case_id=case_id,
                input_shapes=shapes,
                dtypes=dtypes,
                attrs=attrs,
                value_range=value_range,
                seed=seed_base + case_id,
                source_revision=source_revision,
                note=str(raw.get("note", "")),
            )
        )
    if len({spec.case_id for spec in specs}) != len(specs):
        raise ValueError(f"{path}: duplicate case ids")
    return tuple(specs)


def _semantic_tensor(spec, shape, name, torch):
    """Build a tensor for a semantic range such as unique indices."""
    kind = spec[0]
    lo = int(spec[1]) if len(spec) > 1 else 0
    hi = int(spec[2]) if len(spec) > 2 else lo + 1
    if kind != "unique":
        raise ValueError(f"unsupported value_range semantics: {kind}")
    total = 1
    for dim in shape:
        total *= int(dim)
    capacity = hi - lo + 1
    if total > capacity:
        raise ValueError(f"unique range {lo}..{hi} cannot fill {total} values")
    values = torch.randperm(capacity)[:total] + lo
    return values.reshape(tuple(int(v) for v in shape)).to(getattr(torch, name))


def generate_inputs(case: CaseSpec, torch) -> list:
    """Build the case inputs from their shapes, dtypes, ranges, and seed.

    A case may carry one range per input or a single shared range. Integer
    dtypes are generated as index-like values and are never widened to float.
    A NaN or infinite range is materialized as those special values.
    """
    torch.manual_seed(case.seed)
    first = case.value_range[0] if case.value_range else None
    per_input = not isinstance(first, (int, float))
    ranges = (
        [
            span if span is None or isinstance(span, str) else tuple(span)
            for span in case.value_range
        ]
        if per_input
        else [tuple(case.value_range)] * len(case.input_shapes)
    )
    tensors = []
    for shape, name, span in zip(case.input_shapes, case.dtypes, ranges):
        if shape is None or name is None:
            # Optional input the official case marks absent.
            tensors.append(None)
            continue
        if shape and isinstance(shape[0], (list, tuple)):
            # TensorList input: one tensor per listed shape, order preserved.
            tensors.append([torch.randn(*tuple(inner)) for inner in shape])
            continue
        if isinstance(span, tuple) and span and isinstance(span[0], str):
            tensors.append(_semantic_tensor(span, shape, name, torch))
            continue
        if span is None:
            span = (0.0, 1.0)
        dtype = getattr(torch, name)
        low, high = float(span[0]), float(span[1])
        if dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.bool):
            upper = int(high) + 1 if high >= 0 else 1
            tensors.append(torch.randint(0, max(upper, 1), shape, dtype=dtype))
            continue
        if math.isnan(low) or math.isnan(high):
            tensors.append(torch.full(shape, float("nan"), dtype=dtype))
        elif math.isinf(low) or math.isinf(high):
            base = torch.rand(shape) * 2.0 - 1.0
            tensor = base.to(dtype)
            tensor = torch.where(
                base > 0.5, torch.full_like(tensor, float("inf")), tensor
            )
            tensor = torch.where(
                base < -0.5, torch.full_like(tensor, float("-inf")), tensor
            )
            tensors.append(tensor)
        elif low == high:
            tensors.append(torch.full(shape, low, dtype=dtype))
        else:
            tensors.append((torch.rand(shape) * (high - low) + low).to(dtype))
    return tensors


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
