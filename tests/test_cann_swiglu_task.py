"""Golden, boundary, and official cross-checks for the CANN SwiGLU task.

The contract is the restricted official subset: one contiguous fp32 tensor of
shape [512, 2048], dim -1, half split, output [512, 1024]. Numerical checks
need torch; the official cross-check additionally needs torch_npu and an NPU.
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
    """Task identity, provenance, discovery, and the single-input contract."""
    task = load_task(TASK_ID)
    check("suite is cann", task.suite == "cann", task.suite)
    check("no fabricated level", task.level is None, str(task.level))
    check("name is the directory", task.name == "swiglu", task.name)
    check("source is cann-ops-nn", task.source == "cann-ops-nn", task.source)
    check("revision names aclnn_swi_glu", "aclnn_swi_glu" in task.source_revision)
    check("category is nn/activation", task.category == "nn/activation", task.category)
    check("content hash present", len(task.task_hash) == 64, task.task_hash)
    for symbol in ("Model", "get_inputs", "get_init_inputs"):
        check(f"contract has {symbol}", symbol in task.task_py)
    check("docstring fixes dim -1", "dim -1" in task.task_py)
    check(
        "discovery finds exactly the real task",
        [item.task_id for item in discover_tasks(suite="cann")] == [TASK_ID],
    )
    check("the kernelbench corpus still has 270 tasks", len(discover_tasks()) == 270)


def load_model():
    """Execute the task source and return its callables."""
    scope: dict = {}
    exec(compile(load_task(TASK_ID).task_py, TASK_ID, "exec"), scope)
    return scope


def numerical_checks() -> None:
    """Golden values and boundaries against independent formulas."""
    import torch
    import torch.nn.functional as F

    scope = load_model()
    model = scope["Model"]()
    check("init arguments are empty", scope["get_init_inputs"]() == [])
    inputs = scope["get_inputs"]()
    check("single input", len(inputs) == 1)
    x = inputs[0]
    check("input shape [512, 2048]", tuple(x.shape) == (512, 2048), str(tuple(x.shape)))
    check("input is float32", x.dtype == torch.float32, str(x.dtype))
    check("input is contiguous", bool(x.is_contiguous()))

    torch.manual_seed(0)
    xs = torch.randn(512, 2048)
    half = 1024
    a, b = xs[:, :half], xs[:, half:]
    with torch.no_grad():
        got = model(xs)
        golden = a * torch.sigmoid(a) * b
        reference = F.silu(a) * b
    check("output shape [512, 1024]", tuple(got.shape) == (512, 1024), str(tuple(got.shape)))
    check("output dtype preserved", got.dtype == torch.float32, str(got.dtype))
    check("matches F.silu * gate", torch.equal(got, reference))
    check(
        "matches the independent sigmoid formula",
        torch.allclose(got, golden, rtol=1e-6, atol=1e-7),
    )
    check("all finite for random inputs", bool(torch.isfinite(got).all()))

    spot = torch.tensor([[[1.0, 100.0, 2.0, 100.0], [-5.0, 7.0, 0.0, 0.0]]])
    with torch.no_grad():
        out = model(spot)
        expected = F.silu(spot[:, :, :2]) * spot[:, :, 2:]
    check("half split follows the official layout", torch.equal(out, expected))
    check("silu(1)*2 ~ 1.4621172", abs(out[0, 0, 0].item() - 1.4621172) < 1e-5)
    check("silu(100)*100 ~ 10000", abs(out[0, 0, 1].item() - 10000.0) < 1e-2)
    check("silu(-5)*0 == 0", out[0, 1, 0].item() == 0.0)
    check("zero gate yields zero", torch.equal(out[0, 1, 1], torch.tensor(0.0)))

    probe = torch.tensor([[[0.0, 1.0, -1.0, 10.0, 2.0, 3.0, 0.0, -5.0]]])
    with torch.no_grad():
        out = model(probe)
    a, b = probe[:, :, :4], probe[:, :, 4:]
    independent = a * torch.sigmoid(a) * b
    check("probe follows the independent formula", torch.allclose(out, independent))
    check("zero activates to zero", out[0, 0, 0].item() == 0.0)
    check(
        "silu(-1)*3 ~ -0.8068237",
        abs(out[0, 0, 1].item() - (-0.8068237)) < 1e-5,
        str(out[0, 0, 1].item()),
    )
    check("gated zero is zero", out[0, 0, 2].item() == out[0, 0, 2].item())
    check("no NaN at the extremes", bool(torch.isfinite(out).all()))

    zeros = torch.zeros(8, 2048)
    with torch.no_grad():
        check("all-zero input yields zeros", torch.equal(model(zeros), torch.zeros(8, 1024)))


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
    import torch.nn.functional as F

    model = load_model()["Model"]()
    torch.manual_seed(1)
    xs = torch.randn(512, 2048, dtype=torch.float32)
    with torch.no_grad():
        golden = model(xs)
        official = torch_npu.npu_swiglu(xs.npu(), -1).cpu()
    check("official op returns [512, 1024]", tuple(official.shape) == (512, 1024))
    check(
        "golden matches the official aclnn op",
        bool(torch.allclose(golden, official, rtol=1e-5, atol=1e-5)),
        f"max abs diff {(golden - official).abs().max().item():.3e}",
    )
    layout = torch.tensor([[[1.0, 100.0, 2.0, 100.0]]]).npu()
    half_split = F.silu(layout[:, :, :2]) * layout[:, :, 2:]
    interleaved = F.silu(layout[:, :, 0::2]) * layout[:, :, 1::2]
    official_layout = torch_npu.npu_swiglu(layout, -1).cpu()
    check("official layout is the half split", torch.allclose(official_layout, half_split.cpu()))
    check(
        "official layout is not interleaved",
        not torch.allclose(official_layout, interleaved.cpu()),
    )


def main() -> int:
    """Run contract checks always, golden checks with torch, official ones on NPU."""
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
    print(f"\n{'all swiglu task checks passed' if not FAILURES else 'FAILURES: ' + ', '.join(FAILURES)}")
    return int(bool(FAILURES))


if __name__ == "__main__":
    raise SystemExit(main())
