"""Strategy objects for regex-based static checks.

Patterns are matched against prepared source: comments and string literal
content are blanked first, so text inside strings cannot trip a rule.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class PatternRule:
    """One regex catalog and the violation it reports."""

    patterns: tuple[str, ...]
    message: str
    include_match: bool = False

    def check(self, code: str) -> list[str]:
        """Return one violation for the first matching pattern, or []."""
        for pattern in self.patterns:
            match = re.search(pattern, code)
            if match:
                if self.include_match:
                    return [f"{self.message}: {match.group(0)}"]
                return [self.message]
        return []


def run_rules(code: str, rules: Sequence[PatternRule]) -> list[str]:
    """Return violations from rules applied in order; not deduplicated."""
    violations: list[str] = []
    for rule in rules:
        violations.extend(rule.check(code))
    return violations


# Tensor methods that count as compute on either the host wrapper or the
# generated C++ glue. The Python checker adds a few array-only names.
TENSOR_COMPUTE_METHODS = (
    "matmul",
    "mm",
    "bmm",
    "addmm",
    "baddbmm",
    "mv",
    "ger",
    "outer",
    "relu",
    "relu_",
    "sigmoid",
    "sigmoid_",
    "tanh",
    "tanh_",
    "gelu",
    "silu",
    "softmax",
    "log_softmax",
    "leaky_relu",
    "elu",
    "selu",
    "add",
    "add_",
    "sub",
    "sub_",
    "mul",
    "mul_",
    "div",
    "div_",
    "pow",
    "sum",
    "mean",
    "amax",
    "amin",
    "argmax",
    "argmin",
    "prod",
    "norm",
    "var",
    "std",
    "cumsum",
    "topk",
    "sort",
    "clamp",
    "gather",
    "scatter",
    "index_select",
    "conv1d",
    "conv2d",
    "conv3d",
)
