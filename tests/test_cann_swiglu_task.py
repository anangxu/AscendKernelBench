"""Golden and boundary checks for the CANN SwiGLU task.

Runs offline for the contract and provenance checks; the numerical golden
needs torch, so it runs on the Ascend host and reports a skip elsewhere.
"""

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
    """Task identity, provenance, and discovery."""
    task = load_task(TASK_ID)
    check("suite is cann", task.suite == "cann", task.suite)
    check("no fabricated level", task.level is None, str(task.level))
    check("name is the directory", task.name == "swiglu", task.name)
    check("source is recorded", task.source == "cann-ops-nn", task.source)
    check("revision is recorded", "9.1.0" in task.source_revision, task.source_revision)
    check("category is recorded", task.category == "nn/activation", task.category)
    check("content hash present", len(task.task_hash) == 64, task.task_hash)
    for symbol in ("Model", "get_inputs", "get_init_inputs"):
        check(f"contract has {symbol}", symbol in task.task_py)
    check(
        "discovery finds exactly the real task",
        [item.task_id for item in discover_tasks(suite="cann")] == [TASK_ID],
        str([item.task_id for item in discover_tasks(suite="cann")]),
    )
    check(
        "the kernelbench corpus still has 270 tasks",
        len(discover_tasks()) == 270,
    )


def numerical_checks() -> None:
    """Golden values and boundaries against an independent formula."""
    import torch
    import torch.nn.functional as F

    task_module = {}
    exec(compile(load_task(TASK_ID).task_py, TASK_ID, "exec"), task_module)
    model = task_module["Model"]()
    check("init arguments are empty", task_module["get_init_inputs"]() == [])
    inputs = task_module["get_inputs"]()
    check("two inputs", len(inputs) == 2)
    x, gate = inputs
    check("fixed shape", tuple(x.shape) == (512, 1024), str(tuple(x.shape)))
    check("float32 inputs", x.dtype == torch.float32 and gate.dtype == torch.float32)

    torch.manual_seed(0)
    xs = torch.randn(512, 1024)
    gs = torch.randn(512, 1024)
    with torch.no_grad():
        got = model(xs, gs)
        golden = xs * torch.sigmoid(xs) * gs
        reference = F.silu(xs) * gs
    check(
        "matches the independent sigmoid formula",
        torch.allclose(got, golden, rtol=1e-6, atol=1e-7),
    )
    check("matches F.silu", torch.equal(got, reference))
    check("shape preserved", tuple(got.shape) == (512, 1024), str(tuple(got.shape)))
    check("dtype preserved", got.dtype == torch.float32, str(got.dtype))
    check("all finite for random inputs", bool(torch.isfinite(got).all()))

    spot = torch.tensor([[0.0, 1.0, -1.0, 10.0, -10.0, 100.0, -100.0]])
    ones = torch.ones_like(spot)
    with torch.no_grad():
        out = model(spot, ones)
    silu = spot * torch.sigmoid(spot)
    check("silu(0) is 0", out[0, 0].item() == 0.0, str(out[0, 0].item()))
    check("silu(1) ~ 0.7310586", abs(out[0, 1].item() - 0.7310586) < 1e-6)
    check("silu(-1) ~ -0.2689414", abs(out[0, 2].item() + 0.2689414) < 1e-6)
    check("silu(10) ~ 9.999546", abs(out[0, 3].item() - 9.999546) < 1e-4)
    check("silu(-10) ~ -4.54e-4", abs(out[0, 4].item() + 4.54e-4) < 1e-5)
    check("silu(100) ~ 100", abs(out[0, 5].item() - 100.0) < 1e-9)
    check("silu(-100) ~ 0", abs(out[0, 6].item()) < 1e-30)
    check("no NaN at the extremes", bool(torch.isfinite(out).all()))
    check("spot values follow silu exactly", torch.equal(out, silu))

    zeros = torch.zeros(4, 4)
    with torch.no_grad():
        check(
            "zero gate yields zero",
            torch.equal(model(torch.ones(4, 4), zeros), torch.zeros(4, 4)),
        )

    noncontig = torch.randn(64, 128).t()
    with torch.no_grad():
        check(
            "non-contiguous input works",
            torch.allclose(model(noncontig, noncontig), F.silu(noncontig) * noncontig),
        )


def main() -> int:
    """Run contract checks always and the golden checks when torch exists."""
    print("== contract ==")
    contract_checks()
    try:
        import torch  # noqa: F401
    except Exception:
        print("== numerical: SKIP (torch unavailable on this host) ==")
    else:
        print("== numerical ==")
        numerical_checks()
    print(f"\n{'all swiglu task checks passed' if not FAILURES else 'FAILURES: ' + ', '.join(FAILURES)}")
    return int(bool(FAILURES))


if __name__ == "__main__":
    raise SystemExit(main())
