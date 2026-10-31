"""Offline checks for the generic case adapter and the RMSNorm cases."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import cases as cases_module  # noqa: E402
from src.dataset import discover_tasks, load_task  # noqa: E402

OFFICIAL = ROOT / "domain_benchmarks" / "cann" / "nn" / "rmsnorm" / "official"
FIXTURE = ROOT / "tests" / "fixtures" / "domain_suite" / "cann"
FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    """Record one check result."""
    suffix = "" if condition else " :: " + detail
    print(f"  {'ok  ' if condition else 'FAIL'} - {name}{suffix}")
    if not condition:
        FAILURES.append(name)


def yaml_vs_csv() -> None:
    """The official YAML and CSV must describe the same 20 cases."""
    specs = {spec.case_id: spec for spec in load_task("cann/nn/rmsnorm").cases}
    rows = list(csv.DictReader((OFFICIAL / "cases.csv").open(encoding="utf-8")))
    check("YAML has 20 cases", len(specs) == 20, str(len(specs)))
    check("CSV has 20 rows", len(rows) == 20, str(len(rows)))
    mismatch = []
    for row in rows:
        spec = specs.get(int(row["case_id"]))
        if spec is None:
            mismatch.append(f"missing {row['case_id']}")
            continue
        shapes = tuple(tuple(json_shape) for json_shape in eval(row["input_shape"]))
        dtypes = tuple(eval(row["dtype"]))
        if shapes != spec.input_shapes or dtypes != spec.dtypes:
            mismatch.append(f"case {row['case_id']}")
        if abs(float(eval(row["attrs"])["epsilon"]) - spec.epsilon) > 1e-12:
            mismatch.append(f"case {row['case_id']} epsilon")
    check("YAML and CSV agree case by case", not mismatch, str(mismatch[:4]))
    check(
        "every case carries a source revision",
        all(spec.source_revision for spec in specs.values()),
    )
    check("case ids are 1..20", [spec.case_id for spec in specs.values()] == list(range(1, 21)))


def special_values() -> None:
    """Cases 15, 16, 17 must really carry inf, NaN, and zeros."""
    import torch

    specs = {spec.case_id: spec for spec in load_task("cann/nn/rmsnorm").cases}
    inf_inputs = cases_module.generate_inputs(specs[15], torch)
    nan_inputs = cases_module.generate_inputs(specs[16], torch)
    zero_inputs = cases_module.generate_inputs(specs[17], torch)
    check("case 15 keeps its dtype", str(inf_inputs[0].dtype) == "torch.bfloat16", str(inf_inputs[0].dtype))
    check("case 15 contains both infinities", bool(torch.isinf(inf_inputs[0]).any()))
    check("case 15 has no NaN", not bool(torch.isnan(inf_inputs[0]).any()))
    check("case 16 contains NaN", bool(torch.isnan(nan_inputs[0]).all()))
    check("case 17 is exactly zero", bool((zero_inputs[0] == 0).all()))
    check("case 17 is float32", str(zero_inputs[0].dtype) == "torch.float32")
    for case_id, spec in specs.items():
        generated = cases_module.generate_inputs(spec, torch)
        if len(generated) != len(spec.input_shapes):
            check(f"case {case_id} input count", False)
            break
        if any(tuple(t.shape) != tuple(s) for t, s in zip(generated, spec.input_shapes)):
            check(f"case {case_id} shapes preserved", False)
            break
    else:
        check("every case keeps its shapes and dtypes", True)


def classification() -> None:
    """A candidate counts once; every outcome class is distinct."""
    specs = load_task("cann/nn/rmsnorm").cases
    partial = cases_module.summarize(specs, {case.case_id: "passed" for case in specs[:10]})
    check("partial: 10 of 20 passed", (partial.passed, partial.executed, partial.total) == (10, 10, 20))
    check("partial is not task-correct", not partial.task_correct)
    all_pass = cases_module.summarize(specs, {case.case_id: "passed" for case in specs})
    check("all pass is task-correct", all_pass.task_correct and all_pass.passed == 20)
    none = cases_module.summarize(specs, {case.case_id: "failed" for case in specs})
    check("all fail is not task-correct", none.failed == 20 and not none.task_correct)
    mixed = cases_module.summarize(
        specs,
        {specs[0].case_id: "passed", specs[1].case_id: "failed",
         specs[2].case_id: "unsupported", specs[3].case_id: "error"},
    )
    check("unsupported and error are not passes", mixed.passed == 1 and mixed.unsupported == 1)
    check("error counts as executed but not passed", mixed.errors == 1 and mixed.executed == 3)
    record = all_pass.record()
    check("record keeps one candidate", record["cases_total"] == 20)
    check("record has no candidate duplication key", "candidates" not in record or record["candidates"] == 1)
    try:
        cases_module.summarize(specs, {specs[0].case_id: "bogus"})
    except ValueError:
        check("unknown status is refused", True)
    else:
        check("unknown status is refused", False)


def backward_compatibility() -> None:
    """Old tasks carry no cases and the corpus is untouched."""
    check("kernelbench task has no cases", load_task("level1/19_ReLU").cases == ())
    check("kernelbench still has 270 tasks", len(discover_tasks()) == 270)
    fixture = load_task("cann/fixture_add", roots={"cann": FIXTURE})
    check("fixture task carries two cases", len(fixture.cases) == 2)
    check(
        "suite holds the five official tasks",
        {"rmsnorm", "swiglu", "softmax", "gather", "apply_rotary_pos_emb"}
        <= {p.parent.name for p in (ROOT / "domain_benchmarks/cann/nn").glob("*/meta.yaml")},
    )


def main() -> int:
    """Run every case-adapter check."""
    for section in (yaml_vs_csv, special_values, classification, backward_compatibility):
        print(f"== {section.__name__} ==")
        section()
    print(f"\n{'all case adapter checks passed' if not FAILURES else 'FAILURES: ' + ', '.join(FAILURES)}")
    return int(bool(FAILURES))


if __name__ == "__main__":
    raise SystemExit(main())
