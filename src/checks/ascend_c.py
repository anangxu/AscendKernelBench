"""Static checks for generated custom_op.asc source."""

from __future__ import annotations

import re

from .rules import TENSOR_COMPUTE_METHODS, PatternRule, run_rules
from .text import (
    HOST_SECTION_MARKER,
    HOST_SECTION_MARKER_RE,
    dedupe,
    strip_cpp_comments,
)

_ASC_KERNEL_MARKERS = ("__global__", "__vector__")

_ASC_AT_ALLOWED_CALLS = {
    "Tensor",
    "empty",
    "empty_like",
    "zeros",
    "zeros_like",
    "ones",
    "ones_like",
    "full",
    "full_like",
    "empty_strided",
    "from_blob",
    "scalar_tensor",
    "tensor",
}

_ASC_COMPUTE_METHODS = TENSOR_COMPUTE_METHODS

########################### BANNED PATTERN CATALOG ###########################
ASC_BANNED_RULES: tuple[PatternRule, ...] = (
    PatternRule(
        (r"\baclnn[A-Z]\w*",),
        "vendor prebuilt operator (aclnn*) — implement the kernel yourself",
        include_match=True,
    ),
    PatternRule(
        (r"\baclop\w*",),
        "legacy vendor operator API (aclop*)",
        include_match=True,
    ),
    PatternRule(
        (r"\bsystem\s*\(",),
        "host process execution",
        include_match=True,
    ),
    PatternRule(
        (r"\bpopen\s*\(|\bexecl\w*\s*\(|\bexecv\w*\s*\(|\bfork\s*\(",),
        "host process execution",
        include_match=True,
    ),
    PatternRule(
        (r"\bsocket\s*\(|\bconnect\s*\(",),
        "network access",
        include_match=True,
    ),
    PatternRule(
        (r"\bdlopen\s*\(|\bdlsym\s*\(",),
        "dynamic loading",
        include_match=True,
    ),
    PatternRule(
        (r"#\s*include\s*<ATen/ops/",),
        "ATen operator headers",
        include_match=True,
    ),
    PatternRule(
        (r"\bstd::thread\b|\bpthread_create\b|\bstd::async\b",),
        "host threads (timing manipulation)",
        include_match=True,
    ),
)
########################### BANNED PATTERN CATALOG ###########################


def _check_asc_binding(code: str) -> list[str]:
    """Require kernel markers and a process-local torch.library binding."""
    violations: list[str] = []
    for marker in _ASC_KERNEL_MARKERS:
        if marker not in code:
            violations.append(
                f"custom_op.asc missing {marker!r}: not an Ascend C kernel "
                "(host-only ATen implementations are not allowed)"
            )
    if not re.search(r"\bTORCH_LIBRARY\s*\(", code):
        violations.append(
            "custom_op.asc missing TORCH_LIBRARY(...): register the operator "
            "schema so the evaluator can torch.ops.load_library the "
            "process-local .so"
        )
    if not re.search(r"\bTORCH_LIBRARY_IMPL\s*\(", code):
        violations.append(
            "custom_op.asc missing TORCH_LIBRARY_IMPL(...): bind the NPU "
            "implementation (PrivateUse1) for torch.ops.custom_op"
        )
    return violations


def _check_pybind_ban(code: str) -> list[str]:
    """Ban pybind11 in either layout; registration is torch.library only."""
    masked = strip_cpp_comments(code, mask_literals=True)
    if "PYBIND11_MODULE" in masked or re.search(r'#\s*include\s*[<"]pybind11/', code):
        return [
            "pybind11 is not used; register with TORCH_LIBRARY / "
            "TORCH_LIBRARY_IMPL and let the evaluator call "
            "torch.ops.load_library"
        ]
    return []


def _check_split_layout(source: str) -> list[str]:
    """Validate the two-section layout of a new-style split source."""
    lines = source.splitlines()
    markers = [i for i, line in enumerate(lines) if HOST_SECTION_MARKER_RE.match(line)]
    if len(markers) != 1:
        return [
            f"custom_op.asc must carry exactly one comment line containing "
            f"{HOST_SECTION_MARKER!r} separating device and host sections "
            f"({len(markers)} found)"
        ]
    device_raw = "\n".join(lines[: markers[0]])
    host_raw = "\n".join(lines[markers[0] + 1 :])
    device = strip_cpp_comments(device_raw)
    device_code = strip_cpp_comments(device_raw, mask_literals=True)
    host_code = strip_cpp_comments(host_raw, mask_literals=True)

    violations: list[str] = []
    for marker in _ASC_KERNEL_MARKERS:
        if marker not in device_code:
            violations.append(
                f"device section missing {marker!r}: kernels belong above "
                f"the {HOST_SECTION_MARKER} line"
            )
    if re.search(r"#\s*include\s*[<\"](?:torch|ATen)/", device) or re.search(
        r"#\s*include\s*[<\"]torch_npu/", device
    ):
        violations.append(
            "device section must not include torch headers; the fast build "
            "compiles it without them (torch includes belong in the host "
            "section)"
        )
    for marker in _ASC_KERNEL_MARKERS:
        if marker in host_code:
            violations.append(
                f"host section contains {marker!r}: kernel definitions "
                f"belong above the {HOST_SECTION_MARKER} line"
            )
            break
    if not re.search(r"\bTORCH_LIBRARY\s*\(", host_code):
        violations.append(
            "host section missing TORCH_LIBRARY(...): register the operator "
            "schema so the evaluator can torch.ops.load_library the "
            "process-local .so"
        )
    if not re.search(r"\bTORCH_LIBRARY_IMPL\s*\(", host_code):
        violations.append(
            "host section missing TORCH_LIBRARY_IMPL(...): bind the NPU "
            "implementation (PrivateUse1) for torch.ops.custom_op"
        )
    kernels = {
        match.group(1)
        for match in re.finditer(
            r"__global__\s+__vector__\s+void\s+(\w+)\s*\(", device_code
        )
    }
    calls = {match.group(1) for match in re.finditer(r"\b(\w+)_launch\s*\(", host_code)}
    for name in sorted(calls - kernels):
        violations.append(
            f"host section calls {name}_launch(...) but no __global__ "
            f"__vector__ kernel named {name} exists in the device section"
        )
    return violations


def _check_asc_aten(code: str) -> list[str]:
    """Flag host-side ATen/libtorch compute calls."""
    violations: list[str] = []
    bad_aten = sorted(
        {
            match.group(1)
            for match in re.finditer(r"\bat::(\w+)\s*\(", code)
            if match.group(1) not in _ASC_AT_ALLOWED_CALLS
        }
        | {
            match.group(1)
            for match in re.finditer(r"\btorch::(\w+)\s*\(", code)
            if match.group(1) not in _ASC_AT_ALLOWED_CALLS
        }
    )
    for name in bad_aten:
        violations.append(
            f"host-side ATen call at::{name}(...) — compute must live in the "
            "Ascend C kernel, not in libtorch"
        )
    if re.search(r"\bat::native::", code):
        violations.append("at::native:: call — direct ATen kernel reuse is not allowed")
    return violations


def _check_asc_methods(code: str) -> list[str]:
    """Flag host-side tensor compute methods (.relu(), ->matmul())."""
    method_re = r"(?:\.|->)(" + "|".join(_ASC_COMPUTE_METHODS) + r")\s*\("
    bad_methods = sorted({match.group(1) for match in re.finditer(method_re, code)})
    return [
        f"host-side tensor method .{name}(...) — compute must live in the "
        "Ascend C kernel"
        for name in bad_methods
    ]


def check_custom_op_asc(source: str) -> list[str]:
    """Return static-check violations for generated custom_op.asc source.

    Args:
        source: Raw custom_op.asc text.

    Returns:
        Deduplicated human-readable violations; empty means pass.
    """
    code = strip_cpp_comments(source)
    rule_code = strip_cpp_comments(source, mask_literals=True)
    violations: list[str] = []
    if HOST_SECTION_MARKER_RE.search(source):
        violations.extend(_check_split_layout(source))
    else:
        violations.extend(_check_asc_binding(rule_code))
    violations.extend(_check_pybind_ban(code))
    violations.extend(_check_asc_aten(rule_code))
    violations.extend(_check_asc_methods(rule_code))
    violations.extend(run_rules(rule_code, ASC_BANNED_RULES))
    return dedupe(violations)
