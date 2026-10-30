"""Offline checks for the official single-input SwiGLU task."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import discover_tasks, load_task  # noqa: E402

TASK_ID = "cann/nn/swiglu"
FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    """Record one check result."""
    suffix = "" if condition else " :: " + detail
    print(f"  {'ok  ' if condition else 'FAIL'} - {name}{suffix}")
    if not condition:
        FAILURES.append(name)


def contract_checks() -> None:
    """Official identity and provenance."""
    task = load_task(TASK_ID)
    check("suite is cann", task.suite == "cann", task.suite)
    check("level is absent", task.level is None, str(task.level))
    check("source is cann-ops-nn", task.source == "cann-ops-nn", task.source)
    check("revision names swi_glu", "swi_glu" in task.source_revision, task.source_revision)
    check("category is gated activation", task.category == "gated-activation", task.category)
    check("docstring documents dim", "dim" in task.task_py)
    check("content hash", len(task.task_hash) == 64)
    check("official cases loaded", len(task.cases) == 20, str(len(task.cases)))
    check(
        "dim values cover the official spread",
        sorted({c.attrs.get("dim") for c in task.cases if "dim" in c.attrs}) == [-1, 0, 1, 2],
        str(sorted({c.attrs.get("dim") for c in task.cases if "dim" in c.attrs})),
    )
    check("suite holds five tasks", len(discover_tasks(suite="cann")) == 5)
    check("kernelbench unchanged", len(discover_tasks()) == 270)


def numerical_checks() -> None:
    """The reference follows silu(x0) * x1 with the case dim."""
    import torch
    import torch.nn.functional as F

    task = load_task(TASK_ID)
    for dim in (-1, 0, 1):
        case = next((c for c in task.cases if c.attrs.get("dim") == dim), None)
        if case is None:
            continue
        scope = {"CASE_ATTRS": dict(case.attrs)}
        exec(compile(task.task_py, "task.py", "exec"), scope)
        shape = tuple(int(v) for v in case.input_shapes[0])
        x = torch.randn(*shape)
        got = scope["Model"]().forward(x)
        half = x.shape[dim] // 2
        want = F.silu(x.narrow(dim, 0, half)) * x.narrow(dim, half, half)
        check(f"dim={dim} output shape", tuple(got.shape) == tuple(want.shape), str(tuple(got.shape)))
        check(f"dim={dim} matches the official split", torch.allclose(got, want))
    scope = {"CASE_ATTRS": {"dim": -1}}
    exec(compile(task.task_py, "task.py", "exec"), scope)
    x = torch.randn(4, 8)
    got = scope["Model"]().forward(x)
    check("default dim is the last one", tuple(got.shape) == (4, 4), str(tuple(got.shape)))


def main() -> int:
    """Run contract checks always and numerical checks with torch."""
    print("== contract ==")
    contract_checks()
    try:
        import torch  # noqa: F401
    except Exception:
        print("== numerical: SKIP (torch unavailable) ==")
    else:
        print("== numerical ==")
        numerical_checks()
    print(f"\n{'all swiglu task checks passed' if not FAILURES else 'FAILURES: ' + ', '.join(FAILURES)}")
    return int(bool(FAILURES))


if __name__ == "__main__":
    raise SystemExit(main())
