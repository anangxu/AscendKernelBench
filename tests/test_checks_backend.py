"""No-device tests for the pyasc static checks.

    python tests/test_checks_backend.py

The two positive samples are the hand-written ones verified on a real NPU;
the mutations are the cheating shapes the checks must reject.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.backend import Backend  # noqa: E402
from src.checker import check_sample_sources  # noqa: E402

FIXTURES = Path(
    "/Users/anangxu/文件/论文/local_pass10_20260909/pyasc_fixtures/harness"
)
FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    """Record one assertion result."""
    if condition:
        print(f"ok - {label}")
    else:
        print(f"FAIL - {label} {detail}")
        FAILURES.append(label)


def violations(backend: Backend, kernel: str, wrapper: str) -> list[str]:
    """Return the violations for one sample pair."""
    return check_sample_sources(backend, kernel, wrapper)


def expect_clean(label: str, backend: Backend, kernel: str, wrapper: str) -> None:
    """Assert a legal sample produces no violations."""
    found = violations(backend, kernel, wrapper)
    check(label, not found, "; ".join(item[:90] for item in found[:3]))


def expect_rejected(label: str, kernel: str, wrapper: str) -> None:
    """Assert a cheating sample produces at least one violation."""
    found = violations(Backend.PYASC, kernel, wrapper)
    check(label, bool(found), "no violation reported")


def main() -> int:
    asc = (ROOT / "src/prompts/examples/001_elementwise_add/custom_op.asc").read_text()
    asc_wrapper = (
        ROOT / "src/prompts/examples/001_elementwise_add/model_new.py"
    ).read_text()
    expect_clean("ascendc official example stays clean", Backend.ASCENDC, asc, asc_wrapper)

    add_kernel = (FIXTURES / "runs_pyasc_add/level1/901_add_smoke/sample_0/kernel.py").read_text()
    add_wrapper = (FIXTURES / "runs_pyasc_add/level1/901_add_smoke/sample_0/model_new.py").read_text()
    relu_kernel = (FIXTURES / "runs_pyasc_relu/level1/902_relu_smoke/sample_0/kernel.py").read_text()
    relu_wrapper = (FIXTURES / "runs_pyasc_relu/level1/902_relu_smoke/sample_0/model_new.py").read_text()

    expect_clean("pyasc Add sample is accepted", Backend.PYASC, add_kernel, add_wrapper)
    expect_clean("pyasc ReLU sample is accepted", Backend.PYASC, relu_kernel, relu_wrapper)

    never_launches = re.sub(
        r"^\s*add_kernel\[[^\n]*\n", "        pass\n", add_kernel, flags=re.M
    )
    assert "add_kernel[" not in never_launches
    expect_rejected("launcher that never launches", never_launches, add_wrapper)

    hidden = add_kernel.replace(
        "    return max(1, BYTE_ALIGN // element_size)",
        "    return int(torch.relu(torch.tensor(float(element_size))).item())",
    )
    assert "torch.relu" in hidden
    expect_rejected("torch compute hidden in a host helper", hidden, add_wrapper)

    expect_rejected(
        "torch compute in the launcher",
        add_kernel.replace("    z = torch.empty_like(x)\n", "    z = torch.relu(x) + y\n", 1),
        add_wrapper,
    )
    expect_rejected(
        "torch compute in the wrapper",
        add_kernel,
        add_wrapper.replace("return kernel.add_launch(A, B)", "return torch.relu(A) + B"),
    )
    expect_rejected(
        "no @asc.jit device function",
        add_kernel.replace("@asc.jit\n", ""),
        add_wrapper,
    )
    expect_rejected("wrapper without ModelNew", add_kernel, "import torch\n")
    expect_rejected(
        "vendor native op shortcut",
        add_kernel.replace("import asc\n", "import torch_npu\nimport aclnn_ops\nimport asc\n"),
        add_wrapper,
    )
    expect_rejected(
        "result caching",
        add_kernel.replace("import asc\n", "import functools\nimport asc\n").replace(
            "def add_launch(", "@functools.lru_cache(maxsize=None)\ndef add_launch("
        ),
        add_wrapper,
    )
    expect_rejected(
        "dynamic execution",
        add_kernel.replace("import asc\n", "import importlib\nimport asc\n").replace(
            "    z = torch.empty_like(x)\n",
            "    importlib.import_module('os').system('true')\n    z = torch.empty_like(x)\n",
            1,
        ),
        add_wrapper,
    )
    expect_rejected(
        "timing tamper",
        add_kernel.replace("import asc\n", "import torch\nimport asc\n").replace(
            "    z = torch.empty_like(x)\n",
            "    torch.npu.Event.record = None\n    z = torch.empty_like(x)\n",
            1,
        ),
        add_wrapper,
    )

    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} -> {FAILURES}")
        return 1
    print("all static-check cases passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
