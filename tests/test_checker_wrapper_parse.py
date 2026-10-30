"""Regression: masking must not turn a valid wrapper into a SyntaxError."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.checks.python_source import check_model_new  # noqa: E402

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    """Record one check result."""
    suffix = "" if condition else " :: " + detail
    print(f"  {'ok  ' if condition else 'FAIL'} - {name}{suffix}")
    if not condition:
        FAILURES.append(name)


LEGAL = '''import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def forward(self, x):
        dim = CASE_ATTRS.get("dim", -1)
        return torch.ops.custom_op.run(x, dim)
'''


def main() -> int:
    """Run the wrapper parse regressions."""
    violations = check_model_new(LEGAL)
    check("CASE_ATTRS.get wrapper has no parse error",
          not any("does not parse" in v for v in violations), str(violations))
    try_except = "class ModelNew:\n    def forward(self, x):\n        try:\n            return x\n        except Exception:\n            return x\n"
    hits = check_model_new(try_except)
    check("legal try/except still triggers the rule",
          any("try-except" in v for v in hits), str(hits))
    broken = "class ModelNew:\n    def forward(self, x)\n        return x\n"
    hits = check_model_new(broken)
    check("a real syntax error is still rejected",
          any("does not parse" in v for v in hits), str(hits))
    print(f"\n{'all wrapper parse checks passed' if not FAILURES else 'FAILURES: ' + ', '.join(FAILURES)}")
    return int(bool(FAILURES))


if __name__ == "__main__":
    raise SystemExit(main())
