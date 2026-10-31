"""Regressions for the CANN-native case path: comparison, binding, inputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.cann_baseline import BASELINE_PATH, baseline_case, build_inputs  # noqa: E402
from src.cases import CaseSpec, compare_outputs, generate_inputs  # noqa: E402
from src.compare import outputs_match, perturb_floating_inputs  # noqa: E402

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    """Record one check result."""
    suffix = "" if condition else " :: " + detail
    print(f"  {'ok  ' if condition else 'FAIL'} - {name}{suffix}")
    if not condition:
        FAILURES.append(name)


def comparison_checks() -> None:
    """compare_outputs must handle NaN, Inf, shape, dtype and tolerance."""
    ref = torch.tensor([1.0, float("nan"), float("inf"), 2.0])
    check("identical tensors pass", compare_outputs(ref, ref.clone(), atol=1e-5, rtol=1e-5)[0])
    missing_nan = torch.tensor([1.0, 0.0, float("inf"), 2.0])
    check("missing NaN fails", not compare_outputs(ref, missing_nan, atol=1e-5, rtol=1e-5)[0])
    flipped = torch.tensor([1.0, float("nan"), float("-inf"), 2.0])
    check("flipped Inf sign fails", not compare_outputs(ref, flipped, atol=1e-5, rtol=1e-5)[0])
    check(
        "shape mismatch fails",
        not compare_outputs(ref, ref[:2], atol=1e-5, rtol=1e-5)[0],
    )
    check(
        "dtype change is compared in float",
        compare_outputs(ref[0:2], ref[0:2].to(torch.float16), atol=1e-3, rtol=1e-3)[0],
    )
    check(
        "tolerance boundary respected",
        compare_outputs(torch.tensor([1.0]), torch.tensor([1.001]), atol=1e-4, rtol=1e-4)[0]
        is False,
    )


def binding_checks() -> None:
    """A golden whose attribute sits between tensors binds by name."""
    import domain_benchmarks.cann.auto.scatter.task as scatter_task

    data = torch.zeros(4, 3)
    index = torch.tensor([[0], [1], [2], [3]])
    updates = torch.ones(4, 1)
    # The adapter reads CASE_ATTRS from its own module globals, exactly as the
    # worker does before each case.
    scatter_task.CASE_ATTRS = {"dim": 0, "reduce": None}
    try:
        out = scatter_task.Model()(data, index, updates)
    finally:
        scatter_task.CASE_ATTRS = {}
    check("scatter binds dim by name", tuple(out.shape) == (4, 3), str(tuple(out.shape)))


def structure_checks() -> None:
    """TensorList and optional inputs survive the baseline spec."""
    data = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))["tasks"]
    check("baseline covers sixty tasks", len(data) == 60, str(len(data)))
    lstm = data.get("cann/auto/lstm")
    check("lstm baseline exists", lstm is not None)
    if lstm:
        nested = [s for s in lstm["shapes"] if s and isinstance(s[0], list)]
        check("lstm declares tensor lists", bool(nested), str(lstm["shapes"][:3]))
    emb = data.get("cann/auto/categorical_embedding_bag")
    if emb:
        check("optional inputs recorded as null", None in emb["shapes"], str(emb["shapes"]))
    spec = baseline_case("cann/nn/swiglu", data["cann/nn/swiglu"], seed=0)
    check("baseline case is a CaseSpec", isinstance(spec, CaseSpec))
    check("baseline case carries attrs", bool(spec.attrs))


def multi_output_checks() -> None:
    """Two-output trees compare recursively."""
    q = torch.randn(2, 2, 4)
    k = torch.randn(2, 2, 4)
    check("both outputs match", outputs_match((q, k), (q, k), atol=1e-5, rtol=1e-5))
    check("second output wrong fails", not outputs_match((q, k), (q, k + 1), atol=1e-5, rtol=1e-5))
    check("output count differs fails", not outputs_match((q, k), (q,), atol=1e-5, rtol=1e-5))


def unique_and_hidden_checks() -> None:
    """Unique ranges stay distinct; hidden scaling never touches integers."""
    case = CaseSpec(
        case_id=1,
        input_shapes=((16,),),
        dtypes=("int64",),
        attrs={},
        value_range=(("unique", 0.0, 31.0),),
        seed=7,
    )
    values = generate_inputs(case, torch)[0]
    check("unique values are distinct", len(set(values.tolist())) == values.numel())
    check("unique values respect the bounds", int(values.min()) >= 0 and int(values.max()) <= 31)

    data = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))["tasks"]
    for task_id, config in data.items():
        raw = build_inputs(task_id, config, torch, 0)
        scaled = perturb_floating_inputs(raw, 1.5)
        for before, after in zip(raw, scaled):
            if isinstance(before, torch.Tensor) and before.dtype in (
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            ):
                check(f"{task_id} keeps integer inputs", torch.equal(before, after))
                break
        break


def main() -> int:
    """Run the CANN-native regressions."""
    for section in (
        comparison_checks,
        binding_checks,
        structure_checks,
        multi_output_checks,
        unique_and_hidden_checks,
    ):
        print(f"== {section.__name__} ==")
        section()
    print(f"\n{'all cann native checks passed' if not FAILURES else 'FAILURES: ' + ', '.join(FAILURES)}")
    return int(bool(FAILURES))


if __name__ == "__main__":
    raise SystemExit(main())
