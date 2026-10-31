"""Golden, boundary, and official cross-checks for the CANN RMSNorm task."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import discover_tasks, load_task  # noqa: E402

TASK_ID = "cann/nn/rmsnorm"
EPS = 1e-6
FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    """Record one check result."""
    suffix = "" if condition else " :: " + detail
    print(f"  {'ok  ' if condition else 'FAIL'} - {name}{suffix}")
    if not condition:
        FAILURES.append(name)


def contract_checks() -> None:
    """Identity, provenance, discovery, and the documented overlap."""
    task = load_task(TASK_ID)
    check("suite is cann", task.suite == "cann", task.suite)
    check("no fabricated level", task.level is None, str(task.level))
    check("source is cann-ops-nn", task.source == "cann-ops-nn", task.source)
    check("revision names aclnn_rms_norm", "aclnn_rms_norm" in task.source_revision)
    check("category is nn/norm", task.category == "nn/norm", task.category)
    check("content hash present", len(task.task_hash) == 64, task.task_hash)
    for symbol in ("Model", "get_inputs", "get_init_inputs"):
        check(f"contract has {symbol}", symbol in task.task_py)
    check(
        "the five pilot tasks are still registered",
        {"rmsnorm", "swiglu", "softmax", "gather", "apply_rotary_pos_emb"}
        <= {p.parent.name for p in (ROOT / "domain_benchmarks/cann/nn").glob("*/meta.yaml")},
    )
    check("the kernelbench corpus still has 270 tasks", len(discover_tasks()) == 270)
    overlap = [
        item.task_id for item in discover_tasks() if "rmsnorm" in item.task_id.lower()
    ]
    print(f"  note - KernelBench overlap: {overlap}")


def numerical_checks() -> None:
    """Golden values against two independent formulas."""
    import torch

    scope: dict = {}
    exec(compile(load_task(TASK_ID).task_py, TASK_ID, "exec"), scope)
    model = scope["Model"]()
    check("init arguments are empty", scope["get_init_inputs"]() == [])
    inputs = scope["get_inputs"]()
    check("two inputs", len(inputs) == 2)
    x, weight = inputs
    check("x shape [512, 1024]", tuple(x.shape) == (512, 1024), str(tuple(x.shape)))
    check("weight shape [1024]", tuple(weight.shape) == (1024,), str(tuple(weight.shape)))
    check("both float32", x.dtype == torch.float32 and weight.dtype == torch.float32)
    check("both contiguous", bool(x.is_contiguous() and weight.is_contiguous()))

    torch.manual_seed(0)
    xs = torch.randn(512, 1024)
    ws = torch.randn(1024)
    with torch.no_grad():
        got = model(xs, ws)
        mean_sq = xs.pow(2).mean(dim=-1, keepdim=True)
        golden = xs * torch.rsqrt(mean_sq + EPS) * ws
        second = xs / torch.sqrt(mean_sq + EPS) * ws
    check("output shape [512, 1024]", tuple(got.shape) == (512, 1024), str(tuple(got.shape)))
    check("matches the rsqrt formula", torch.equal(got, golden))
    check(
        "matches the div-by-sqrt formula",
        torch.allclose(got, second, rtol=1e-6, atol=1e-7),
    )
    check("all finite", bool(torch.isfinite(got).all()))

    probe = torch.tensor([[[1.0, -1.0, 0.0, 2.0], [3.0, 3.0, 3.0, 3.0]]])
    gamma = torch.ones(4)
    with torch.no_grad():
        out = model(probe, gamma)
        exp = probe / torch.sqrt(probe.pow(2).mean(dim=-1, keepdim=True) + EPS)
    check("probe follows the reference", torch.allclose(out, exp))
    check("sign is preserved", (out[0, 0, 0] > 0) and (out[0, 0, 1] < 0))
    check("uniform rows normalize to ones", torch.allclose(out[0, 1], torch.ones(4)))
    zeros = torch.zeros(8, 1024)
    with torch.no_grad():
        check("all-zero input yields zeros", torch.equal(model(zeros, ws), torch.zeros(8, 1024)))
    with torch.no_grad():
        huge = torch.full((4, 1024), 1e6)
        check(
            "large values stay finite",
            bool(torch.isfinite(model(huge, ws)).all()),
        )


def official_checks() -> None:
    """Cross-check the golden against the installed official aclnn op."""
    import torch

    try:
        import torch_npu  # noqa: F401
    except Exception:
        print("  skip - official cross-check needs torch_npu")
        return
    if not torch.npu.is_available():
        print("  skip - official cross-check needs an NPU")
        return
    scope: dict = {}
    exec(compile(load_task(TASK_ID).task_py, TASK_ID, "exec"), scope)
    model = scope["Model"]()
    torch.manual_seed(2)
    xs = torch.randn(512, 1024, dtype=torch.float32)
    ws = torch.randn(1024, dtype=torch.float32)
    with torch.no_grad():
        golden = model(xs, ws)
        official, rstd = torch_npu.npu_rms_norm(xs.npu(), ws.npu(), EPS)
        official = official.cpu()
    check("official op returns [512, 1024]", tuple(official.shape) == (512, 1024))
    check("official rstd shape [512, 1]", tuple(rstd.shape) == (512, 1), str(tuple(rstd.shape)))
    check(
        "golden matches the official aclnn op",
        bool(torch.allclose(golden, official, rtol=1e-5, atol=1e-5)),
        f"max abs diff {(golden - official).abs().max().item():.3e}",
    )


def main() -> int:
    """Run contract checks always, numerical with torch, official on NPU."""
    print("== contract ==")
    contract_checks()
    try:
        import torch  # noqa: F401
    except Exception:
        print("== numerical: SKIP (torch unavailable) ==")
    else:
        print("== numerical ==")
        numerical_checks()
        print("== official ==")
        official_checks()
    print(f"\n{'all rmsnorm task checks passed' if not FAILURES else 'FAILURES: ' + ', '.join(FAILURES)}")
    return int(bool(FAILURES))


if __name__ == "__main__":
    raise SystemExit(main())
