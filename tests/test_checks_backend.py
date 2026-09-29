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

FIXTURES = ROOT / "tests" / "fixtures" / "pyasc"
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


# Two legal pyasc shapes that the checks once rejected. Found while evaluating
# generated samples: a kernel inside a module-level conditional, and a launcher
# that delegates the launch to a helper while taking int-annotated sizes.
NESTED_AND_DELEGATED_KERNEL = """\
import torch
import torch_npu  # noqa: F401

import asc
import asc.lib.runtime as rt

BLOCK = 8


if hasattr(asc, "mul"):

    @asc.jit
    def mul_kernel(x: asc.GlobalAddress, y: asc.GlobalAddress, n: asc.ConstExpr[int]):
        x_gm = asc.GlobalTensor()
        y_gm = asc.GlobalTensor()
        x_gm.set_global_buffer(x, n)
        y_gm.set_global_buffer(y, n)
        pipe = asc.TPipe()
        q = asc.TQue(asc.TPosition.VECIN, 1)
        pipe.init_buffer(q, 1, n * 4)
        t = q.alloc_tensor(asc.float32)
        asc.data_copy(t, x_gm, n)
        q.enque(t)
        out = q.deque(asc.float32)
        asc.data_copy(y_gm, out, n)
        q.free_tensor(out)


def _launch(x, z, total: int) -> None:
    block = total // BLOCK
    mul_kernel[1, rt.current_stream()](x, z, block)


def mul_launch(x: torch.Tensor, z: torch.Tensor) -> None:
    _launch(x, z, x.numel())
"""

NESTED_AND_DELEGATED_WRAPPER = """\
import kernel


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> None:
        return kernel.mul_launch(A, B)
"""


# Two more legal shapes: the kernel is chosen at run time through a local name,
# and through a loop over a module-level collection of kernels.
ALIASED_KERNEL = """\
import torch
import torch_npu  # noqa: F401

import asc
import asc.lib.runtime as rt


@asc.jit
def a_kernel(x: asc.GlobalAddress, y: asc.GlobalAddress, n: asc.ConstExpr[int]):
    x_gm = asc.GlobalTensor()
    y_gm = asc.GlobalTensor()
    x_gm.set_global_buffer(x, n)
    y_gm.set_global_buffer(y, n)
    pipe = asc.TPipe()
    q = asc.TQue(asc.TPosition.VECIN, 1)
    pipe.init_buffer(q, 1, n * 4)
    t = q.alloc_tensor(asc.float32)
    asc.data_copy(t, x_gm, n)
    q.enque(t)
    out = q.deque(asc.float32)
    asc.data_copy(y_gm, out, n)
    q.free_tensor(out)


@asc.jit
def b_kernel(x: asc.GlobalAddress, y: asc.GlobalAddress, n: asc.ConstExpr[int]):
    x_gm = asc.GlobalTensor()
    y_gm = asc.GlobalTensor()
    x_gm.set_global_buffer(x, n)
    y_gm.set_global_buffer(y, n)
    pipe = asc.TPipe()
    q = asc.TQue(asc.TPosition.VECIN, 1)
    pipe.init_buffer(q, 1, n * 4)
    t = q.alloc_tensor(asc.float32)
    asc.data_copy(t, x_gm, n)
    q.enque(t)
    out = q.deque(asc.float32)
    asc.data_copy(y_gm, out, n)
    q.free_tensor(out)


_KERNELS = (a_kernel, b_kernel)


def run(x: torch.Tensor, y: torch.Tensor, slope: float = 0.01) -> None:
    chosen = a_kernel if slope <= 1.0 else b_kernel
    chosen[1, rt.current_stream()](x, y, x.numel())


def run_collection(x: torch.Tensor, y: torch.Tensor) -> None:
    for kernel in _KERNELS:
        kernel[1, rt.current_stream()](x, y, x.numel())
"""

ALIASED_WRAPPER = """\
import kernel


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> None:
        return kernel.run(A, B)
"""


# Two shapes generated for real. A module-level dict of kernels picked by a
# module-level mode string subscripts the collection before the launch
# subscript; a dispatcher walks a module-level tuple of launch variants and
# keeps the first that runs, which is how one generated matmul fell back
# between two kernels.
DICT_KERNEL = ALIASED_KERNEL.replace(
    "_KERNELS = (a_kernel, b_kernel)",
    '_KERNELS = {"a": a_kernel, "b": b_kernel}\nMODE = "a"',
).replace(
    """def run_collection(x: torch.Tensor, y: torch.Tensor) -> None:
    for kernel in _KERNELS:
        kernel[1, rt.current_stream()](x, y, x.numel())""",
    """def run_dict(x: torch.Tensor, y: torch.Tensor) -> None:
    _KERNELS[MODE][1, rt.current_stream()](x, y, x.numel())""",
)

DISPATCH_KERNEL = ALIASED_KERNEL.replace(
    """def run_collection(x: torch.Tensor, y: torch.Tensor) -> None:
    for kernel in _KERNELS:
        kernel[1, rt.current_stream()](x, y, x.numel())""",
    """def _launch_a(x: torch.Tensor, y: torch.Tensor) -> None:
    a_kernel[1, rt.current_stream()](x, y, x.numel())


def _launch_b(x: torch.Tensor, y: torch.Tensor) -> None:
    b_kernel[1, rt.current_stream()](x, y, x.numel())


_LAUNCHERS = (_launch_a, _launch_b)


def dispatch(x: torch.Tensor, y: torch.Tensor) -> None:
    for index in range(len(_LAUNCHERS)):
        try:
            _LAUNCHERS[index](x, y)
            return
        except Exception:
            pass""",
)


BASE_KERNEL = re.split(r"\n_KERNELS = \(a_kernel, b_kernel\)", ALIASED_KERNEL)[0]

RUN_HEAD = (
    "def run(x: torch.Tensor, y: torch.Tensor, slope: float = 0.01) -> None:\n"
    "    chosen = a_kernel if slope <= 1.0 else b_kernel\n"
)


def _helper_cases() -> list[tuple[str, bool, str]]:
    """Return (label, expect_clean, host source) for helper reachability."""
    launch = "    chosen[1, rt.current_stream()](x, y, total)"
    return [
        (
            "a helper called with scalars keeps its planning arithmetic",
            True,
            "def plan(rows: int, cols: int) -> int:\n"
            "    return (rows * cols) % 8\n\n\n"
            + RUN_HEAD + "    total = plan(x.numel(), 4)\n" + launch,
        ),
        (
            "an unannotated helper whose call sites pass scalars is accepted",
            True,
            "def plan(rows, cols) -> int:\n"
            "    return (rows * cols) % 8\n\n\n"
            + RUN_HEAD + "    total = plan(x.numel(), 4)\n" + launch,
        ),
        (
            "tensor arithmetic in a helper the launcher calls is rejected",
            False,
            "def helper(x: torch.Tensor) -> int:\n"
            "    scaled = x * 2\n"
            "    return scaled.numel()\n\n\n"
            + RUN_HEAD + "    total = helper(x)\n" + launch,
        ),
        (
            "the same helper reached by an attribute call is rejected",
            False,
            "class Runner:\n"
            "    @staticmethod\n"
            "    def helper(x: torch.Tensor) -> int:\n"
            "        scaled = x * 2\n"
            "        return scaled.numel()\n\n\n"
            "RUNNER = Runner()\n\n\n"
            + RUN_HEAD + "    total = RUNNER.helper(x)\n" + launch,
        ),
        (
            "a helper reached by getattr with a constant name is rejected",
            False,
            "class Holder:\n"
            "    pass\n\n\n"
            "def helper(x: torch.Tensor) -> int:\n"
            "    scaled = x * 2\n"
            "    return scaled.numel()\n\n\n"
            "Holder.helper = helper\n\n\n"
            + RUN_HEAD + "    total = getattr(Holder, 'helper')(x)\n" + launch,
        ),
        (
            "getattr with a scalar argument stays accepted",
            True,
            "class Holder:\n"
            "    pass\n\n\n"
            "def plan(rows: int) -> int:\n"
            "    return (rows + 1) % 4\n\n\n"
            "Holder.plan = plan\n\n\n"
            + RUN_HEAD + "    total = getattr(Holder, 'plan')(x.numel())\n" + launch,
        ),
        (
            "known blind spot: a computed getattr name is not resolved",
            True,
            "class Holder:\n"
            "    pass\n\n\n"
            "def helper(x: torch.Tensor) -> int:\n"
            "    scaled = x * 2\n"
            "    return scaled.numel()\n\n\n"
            "Holder.helper = helper\n"
            "_NAME = 'helper'\n\n\n"
            + RUN_HEAD + "    total = getattr(Holder, _NAME)(x)\n" + launch,
        ),
        (
            "an unknown argument is not treated as a scalar",
            False,
            "_BOUND = torch.empty(2)\n\n\n"
            "def plan(rows) -> int:\n"
            "    return (rows + 1) % 4\n\n\n"
            + RUN_HEAD + "    total = plan(_BOUND)\n" + launch,
        ),
    ]


def _collection_scope_cases() -> list[tuple[str, bool, str]]:
    """Return (label, expect_clean, host source) for collection scope."""
    run = (
        "def run(x: torch.Tensor, y: torch.Tensor, slope: float = 0.01) -> None:\n"
        "    total = x.numel()\n"
        "    _KERNELS[0][1, rt.current_stream()](x, y, total)"
    )
    return [
        (
            "a kernel collection at module level is accepted",
            True,
            "_KERNELS = (a_kernel, b_kernel)\n\n\n" + run,
        ),
        (
            "a kernel collection inside a module-level conditional is accepted",
            True,
            "if hasattr(asc, 'mul'):\n"
            "    _KERNELS = (a_kernel, b_kernel)\n\n\n" + run,
        ),
        (
            "a class-scoped collection is not a module-level collection",
            False,
            "class Holder:\n"
            "    _KERNELS = (a_kernel, b_kernel)\n\n\n"
            "def run(x: torch.Tensor, y: torch.Tensor, slope: float = 0.01) -> None:\n"
            "    total = x.numel()\n"
            "    Holder._KERNELS[0][1, rt.current_stream()](x, y, total)",
        ),
    ]



def main() -> int:
    asc = (ROOT / "src/prompts/examples/001_elementwise_add/custom_op.asc").read_text()
    asc_wrapper = (
        ROOT / "src/prompts/examples/001_elementwise_add/model_new.py"
    ).read_text()
    expect_clean("ascendc official example stays clean", Backend.ASCENDC, asc, asc_wrapper)

    add_kernel = (FIXTURES / "add_smoke" / "kernel.py").read_text()
    add_wrapper = (FIXTURES / "add_smoke" / "model_new.py").read_text()
    relu_kernel = (FIXTURES / "relu_smoke" / "kernel.py").read_text()
    relu_wrapper = (FIXTURES / "relu_smoke" / "model_new.py").read_text()

    expect_clean("pyasc Add sample is accepted", Backend.PYASC, add_kernel, add_wrapper)
    expect_clean("pyasc ReLU sample is accepted", Backend.PYASC, relu_kernel, relu_wrapper)

    never_launches = re.sub(
        r"^\s*add_kernel\[[^\n]*\n", "        pass\n", add_kernel, flags=re.M
    )
    assert "add_kernel[" not in never_launches
    for label, clean, host_source in _helper_cases():
        source = BASE_KERNEL + host_source
        if clean:
            expect_clean(label, Backend.PYASC, source, ALIASED_WRAPPER)
        else:
            expect_rejected(label, source, ALIASED_WRAPPER)
    for label, clean, host_source in _collection_scope_cases():
        source = BASE_KERNEL + host_source
        if clean:
            expect_clean(label, Backend.PYASC, source, ALIASED_WRAPPER)
        else:
            expect_rejected(label, source, ALIASED_WRAPPER)
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
    expect_clean(
        "kernel inside a module-level conditional is accepted",
        Backend.PYASC,
        NESTED_AND_DELEGATED_KERNEL,
        NESTED_AND_DELEGATED_WRAPPER,
    )
    expect_clean(
        "launcher delegating the launch to a helper is accepted",
        Backend.PYASC,
        NESTED_AND_DELEGATED_KERNEL,
        NESTED_AND_DELEGATED_WRAPPER,
    )
    # An unannotated parameter is scalar only when its call sites pass
    # scalars: the same function is accepted when it receives `x.numel()` and
    # rejected when it receives the tensor itself.
    expect_clean(
        "an unannotated launcher parameter fed a scalar is accepted",
        Backend.PYASC,
        NESTED_AND_DELEGATED_KERNEL.replace(
            "def _launch(x, z, total: int) -> None:",
            "def _launch(x, z, total) -> None:",
        ),
        NESTED_AND_DELEGATED_WRAPPER,
    )
    expect_rejected(
        "tensor arithmetic on an unannotated launcher parameter is still rejected",
        NESTED_AND_DELEGATED_KERNEL.replace(
            "def _launch(x, z, total: int) -> None:",
            "def _launch(x, z, total) -> None:",
        ).replace("    _launch(x, z, x.numel())", "    _launch(x, z, x)"),
        NESTED_AND_DELEGATED_WRAPPER,
    )
    expect_clean(
        "a kernel chosen through a local alias is accepted",
        Backend.PYASC,
        ALIASED_KERNEL,
        ALIASED_WRAPPER,
    )
    expect_clean(
        "a kernel chosen by looping over a module-level collection is accepted",
        Backend.PYASC,
        ALIASED_KERNEL,
        ALIASED_WRAPPER.replace("kernel.run(A, B)", "kernel.run_collection(A, B)"),
    )
    expect_clean(
        "constants bound by tuple unpacking count as scalars",
        Backend.PYASC,
        ALIASED_KERNEL.replace(
            "def run(x: torch.Tensor, y: torch.Tensor, slope: float = 0.01) -> None:",
            "N, W = 16, 32\n\n\ndef run(x: torch.Tensor, y: torch.Tensor, slope: float = 0.01) -> None:",
        ).replace(
            "    chosen[1, rt.current_stream()](x, y, x.numel())",
            "    total = N * W\n    chosen[1, rt.current_stream()](x, y, total)",
        ),
        ALIASED_WRAPPER,
    )
    expect_rejected(
        "a bare call through the alias is still rejected",
        ALIASED_KERNEL.replace(
            "    chosen[1, rt.current_stream()](x, y, x.numel())",
            "    chosen(x, y, x.numel())",
        ),
        ALIASED_WRAPPER,
    )
    expect_clean(
        "a kernel picked from a module-level dict is accepted",
        Backend.PYASC,
        DICT_KERNEL,
        ALIASED_WRAPPER.replace("kernel.run(A, B)", "kernel.run_dict(A, B)"),
    )
    expect_rejected(
        "a subscripted collection holding no kernel is still rejected",
        DICT_KERNEL.replace(
            '"a": a_kernel, "b": b_kernel', '"a": _noop, "b": _noop'
        ).replace(
            "def run_dict(",
            "def _noop(x: torch.Tensor, y: torch.Tensor) -> None:\n"
            "    return None\n\n\n"
            "def run_dict(",
        ),
        ALIASED_WRAPPER.replace("kernel.run(A, B)", "kernel.run_dict(A, B)"),
    )
    expect_clean(
        "a dispatcher over a module-level tuple of launchers is accepted",
        Backend.PYASC,
        DISPATCH_KERNEL,
        ALIASED_WRAPPER.replace("kernel.run(A, B)", "kernel.dispatch(A, B)"),
    )
    expect_rejected(
        "a dispatcher over helpers that never launch is still rejected",
        DISPATCH_KERNEL.replace(
            "_LAUNCHERS = (_launch_a, _launch_b)", "_LAUNCHERS = (_noop_a, _noop_b)"
        ).replace(
            "def dispatch(",
            "def _noop_a(x: torch.Tensor, y: torch.Tensor) -> None:\n"
            "    return None\n\n\n"
            "def _noop_b(x: torch.Tensor, y: torch.Tensor) -> None:\n"
            "    return None\n\n\n"
            "def dispatch(",
        ),
        ALIASED_WRAPPER.replace("kernel.run(A, B)", "kernel.dispatch(A, B)"),
    )
    expect_clean(
        "inline element_size() in shape arithmetic is accepted",
        Backend.PYASC,
        ALIASED_KERNEL.replace(
            "    chosen[1, rt.current_stream()](x, y, x.numel())",
            "    align = max(1, 32 // x.element_size())\n"
            "    total = ((x.numel() + align - 1) // align) * align\n"
            "    chosen[1, rt.current_stream()](x, y, total)",
        ),
        ALIASED_WRAPPER,
    )
    expect_rejected(
        "tensor arithmetic in a launcher is still rejected",
        ALIASED_KERNEL.replace(
            "    chosen[1, rt.current_stream()](x, y, x.numel())",
            "    total = x * 2\n    chosen[1, rt.current_stream()](x, y, total)",
        ),
        ALIASED_WRAPPER,
    )
    expect_clean(
        "a launcher published under a second name is accepted",
        Backend.PYASC,
        ALIASED_KERNEL.replace(
            "def run(x: torch.Tensor, y: torch.Tensor, slope: float = 0.01) -> None:",
            "def run_impl(x: torch.Tensor, y: torch.Tensor, slope: float = 0.01) -> None:",
        ).replace(
            "        kernel[1, rt.current_stream()](x, y, x.numel())\n",
            "        kernel[1, rt.current_stream()](x, y, x.numel())\n\n\nrun = run_impl\n",
            1,
        ),
        ALIASED_WRAPPER,
    )
    expect_rejected(
        "an alias of a helper that never launches is still rejected",
        ALIASED_KERNEL.replace(
            "    chosen[1, rt.current_stream()](x, y, x.numel())",
            "    return None",
        ).replace(
            "def run(",
            "def _noop(x: torch.Tensor, y: torch.Tensor) -> None:\n"
            "    return None\n\n\n"
            "def run(",
        )
        + "\n\nrun_alias = _noop",
        ALIASED_WRAPPER.replace("kernel.run(A, B)", "kernel.run_alias(A, B)"),
    )
    expect_clean(
        "an integer bit_length() stays scalar shape arithmetic",
        Backend.PYASC,
        ALIASED_KERNEL.replace(
            "    chosen[1, rt.current_stream()](x, y, x.numel())",
            "    steps = (x.numel() // 8).bit_length() - 1\n"
            "    total = x.numel() + steps\n"
            "    chosen[1, rt.current_stream()](x, y, total)",
        ),
        ALIASED_WRAPPER,
    )
    # The shapes above are accepted for their call graph only. None of them may
    # become a way to move tensor compute onto the host.
    expect_rejected(
        "host compute behind a dict-selected kernel is still rejected",
        DICT_KERNEL.replace(
            "    _KERNELS[MODE][1, rt.current_stream()](x, y, x.numel())",
            "    total = x * 2\n"
            "    _KERNELS[MODE][1, rt.current_stream()](x, y, total)",
        ),
        ALIASED_WRAPPER.replace("kernel.run(A, B)", "kernel.run_dict(A, B)"),
    )
    expect_rejected(
        "host compute behind a launcher alias is still rejected",
        ALIASED_KERNEL.replace(
            "def run(",
            "def run_impl(x: torch.Tensor, y: torch.Tensor) -> None:\n"
            "    total = torch.abs(x).numel()\n"
            "    a_kernel[1, rt.current_stream()](x, y, total)\n\n\n"
            "run = run_impl\n\n\n"
            "def run(",
        ),
        ALIASED_WRAPPER,
    )
    expect_rejected(
        "host compute behind a launcher dispatch is still rejected",
        DISPATCH_KERNEL.replace(
            "    for index in range(len(_LAUNCHERS)):",
            "    total = x * 2\n    for index in range(len(_LAUNCHERS)):",
        ),
        ALIASED_WRAPPER.replace("kernel.run(A, B)", "kernel.dispatch(A, B)"),
    )
    expect_rejected(
        "a metadata call cannot launder tensor arithmetic",
        ALIASED_KERNEL.replace(
            "    chosen[1, rt.current_stream()](x, y, x.numel())",
            "    total = (x * 2).numel()\n"
            "    chosen[1, rt.current_stream()](x, y, total)",
        ),
        ALIASED_WRAPPER,
    )
    expect_rejected(
        "a dispatch collection of tensors is still rejected",
        DISPATCH_KERNEL.replace(
            "_LAUNCHERS = (_launch_a, _launch_b)", "_LAUNCHERS = (y, y)"
        ),
        ALIASED_WRAPPER.replace("kernel.run(A, B)", "kernel.dispatch(A, B)"),
    )
    expect_clean(
        "a module-level list of scalars stays scalar inside a dispatcher",
        Backend.PYASC,
        DISPATCH_KERNEL.replace("def dispatch(", "_GOOD = [0]\n\n\ndef dispatch(").replace(
            "    for index in range(len(_LAUNCHERS)):",
            "    index = _GOOD[0]\n"
            "    while index < len(_LAUNCHERS):\n"
            "        index = index + 1\n"
            "    for index in range(len(_LAUNCHERS)):",
        ),
        ALIASED_WRAPPER.replace("kernel.run(A, B)", "kernel.dispatch(A, B)"),
    )
    expect_rejected(
        "a module-level list holding a tensor is not scalar",
        ALIASED_KERNEL.replace(
            "    chosen[1, rt.current_stream()](x, y, x.numel())",
            "    span = _BAD[0] + 1\n    chosen[1, rt.current_stream()](x, y, span)",
        ).replace(
            "_KERNELS = (a_kernel, b_kernel)",
            "_KERNELS = (a_kernel, b_kernel)\n_BAD = [x]",
        ),
        ALIASED_WRAPPER,
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
