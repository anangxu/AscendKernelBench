"""Official CANN Bench RMSNorm cases: load, golden, and coverage classification."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TASK_ID = "cann/nn/rmsnorm"
CASES = ROOT / "domain_benchmarks" / "cann" / "nn" / "rmsnorm" / "official" / "cases.yaml"
SEED = 1234
# The current AKB task contract is fixed: fp32, [512, 1024], gamma [1024].
TASK_DTYPE = "float32"
TASK_SHAPE = [512, 1024]
FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    """Record one check result."""
    suffix = "" if condition else " :: " + detail
    print(f"  {'ok  ' if condition else 'FAIL'} - {name}{suffix}")
    if not condition:
        FAILURES.append(name)


def load_cases() -> list[dict]:
    """Return the vendored official cases."""
    return yaml.safe_load(CASES.read_text(encoding="utf-8"))["cases"]


def run_case(case: dict) -> tuple[bool, str]:
    """Build the case and compare the golden against an independent formula."""
    import torch
    import torch.nn.functional as F

    shapes = case["input_shape"]
    dtypes = [getattr(torch, name) for name in case["dtype"]]
    eps = float(case["attrs"]["epsilon"])
    low, high = case["value_range"]
    torch.manual_seed(SEED + int(case["case_id"]))
    x = (torch.rand(shapes[0], dtype=torch.float32) * (high - low) + low).to(dtypes[0])
    gamma = (torch.rand(shapes[1], dtype=torch.float32) * (high - low) + low).to(dtypes[1])
    reference = F.rms_norm(x, (shapes[1][-1],), gamma, eps).to(torch.float32)
    independent = x.to(torch.float32) * torch.rsqrt(
        x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True) + eps
    ) * gamma.to(torch.float32)
    ok = bool(torch.allclose(reference, independent, rtol=1e-4, atol=1e-5))
    detail = f"max diff {(reference - independent).abs().max().item():.2e}"
    return ok, detail


def akb_support(case: dict) -> tuple[bool, str]:
    """Whether the fixed AKB task contract can execute this case as-is."""
    shapes = case["input_shape"]
    dtypes = case["dtype"]
    if dtypes[0] != TASK_DTYPE or dtypes[1] != TASK_DTYPE:
        return False, f"dtype {dtypes[0]} is outside the fixed fp32 task"
    if list(shapes[0]) != TASK_SHAPE:
        return False, f"shape {shapes[0]} is outside the fixed {TASK_SHAPE} task"
    return True, "matches the fixed fp32 [512, 1024] contract"


def main() -> int:
    """Load every official case, run its golden, and classify coverage."""
    cases = load_cases()
    check("official case count is 20", len(cases) == 20, str(len(cases)))
    check(
        "case ids are 1..20",
        [int(item["case_id"]) for item in cases] == list(range(1, 21)),
    )
    print("\n  case | dtype              | shape                | golden | AKB task")
    golden_pass = 0
    executable = 0
    for case in cases:
        ok, detail = run_case(case)
        golden_pass += int(ok)
        usable, reason = akb_support(case)
        executable += int(usable)
        print(
            f"  {int(case['case_id']):>4} | {case['dtype'][0]:<18} | "
            f"{str(case['input_shape'][0]):<20} | {'pass' if ok else 'FAIL'}   | "
            f"{'yes' if usable else 'no'} ({reason})"
        )
        check(f"case {case['case_id']} golden", ok, detail)
    print(
        f"\n  official={len(cases)} golden_pass={golden_pass} "
        f"akb_executable={executable} akb_passed=0 (no candidate ran here)"
    )
    check("every official case golden passes", golden_pass == len(cases), str(golden_pass))
    print(f"\n{'all official-case checks passed' if not FAILURES else 'FAILURES: ' + ', '.join(FAILURES)}")
    return int(bool(FAILURES))


if __name__ == "__main__":
    raise SystemExit(main())
