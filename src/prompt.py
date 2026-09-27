"""Component-based prompt construction (docs/reference/configuration.md).

Sections assemble in order: problem statement, hardware block, examples block,
output contract, instruction. Modes: zero_shot, one_shot, few_shot. The
backend selects the authoring language: Ascend C or pyasc.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from ._paths import PROMPT_EXAMPLES_DIR
from .backend import DEFAULT_BACKEND, Backend, parse_backend, sample_files
from .config import HardwareProfile
from .dataset import Task

SYSTEM_PROMPT = (
    "You are an expert Ascend C kernel engineer. You write correct, "
    "high-performance Ascend C operators for Huawei Ascend NPUs. The "
    "benchmark compiles your operator with a fixed modern CMake project into "
    "a process-local shared library (libcustom_op.so) and loads it with "
    "torch.ops.load_library inside the evaluating PyTorch process. Register "
    "the operator with TORCH_LIBRARY / TORCH_LIBRARY_IMPL. Do not use "
    "pybind11. The library is never installed into site-packages or the "
    "global CANN OPP path."
)

OUTPUT_CONTRACT = """\
## Output Contract

The benchmark already owns the CMake operator project. You must NOT emit
CMakeLists.txt, build.sh, op_host / op_kernel trees, framework plugins,
custom_opp packages, or any install / pip / OPP deployment step. Output
exactly two fenced code blocks, tagged with their filenames:

1. ```custom_op.asc — one Ascend C source file in TWO sections, separated by
   exactly one marker line (copy it verbatim):

   // ==================== ASCEND_HOST_SECTION ====================

   DEVICE SECTION (above the marker) — compiled without torch headers:
   a. `#include "kernel_operator.h"`, then the kernel class: `Init` (data
      partition across cores, GM buffers) and `Process` (UB allocation,
      copy-in, compute, copy-out);
   b. kernel functions annotated `__global__ __vector__`, calling
      `AscendC::InitSocState()`, `Init`, `Process`,
      `AscendC::PipeBarrier<PIPE_ALL>()`.
   Never include torch, ATen, or torch_npu headers in the device section.

   HOST SECTION (below the marker) — compiled as plain C++17:
   c. exactly these includes: `#include <torch/library.h>`,
      `#include <ATen/ATen.h>`, and
      `#include "torch_npu/csrc/core/npu/NPUStream.h"`;
   d. host wrapper taking `const at::Tensor&` arguments, fetching the current
      NPU stream via `c10_npu::getCurrentNPUStream().stream(false)`,
      allocating outputs, and launching each kernel through its generated
      stub: for a kernel `foo(...)` call
      `foo_launch(numBlocks, (void*)aclStream, <kernel args in order>)`.
      The build auto-generates `foo_launch` from your kernel signature, so
      never declare or define it yourself. Kernel pointer parameters
      (`__gm__ T*`) become plain `T*` at the call site;
   e. a process-local torch.library binding. Register the host function with
      `TORCH_LIBRARY(custom_op, m)` and bind the NPU implementation with
      `TORCH_LIBRARY_IMPL(custom_op, PrivateUse1, m)`. Do NOT use
      `PYBIND11_MODULE` or any pybind11 header. The evaluator will
      `torch.ops.load_library` the resulting `libcustom_op.so`.
   The library namespace MUST be `custom_op`. Exported function names are
   free (`run` is the convention); multi-kernel tasks may export several
   entries. Schema strings must match the host function (for example
   `run(Tensor x, Tensor y) -> Tensor`). The host wrapper may only allocate
   memory and launch kernels — all compute must happen inside the Ascend C
   kernel, never in host-side ATen calls (`at::matmul`, `tensor.relu()`,
   ...) or vendor prebuilt ops (`aclnn*`). Do not call `cmake --install` or
   write files outside this source.
2. ```model_new.py — class `ModelNew` with the SAME `__init__` and `forward`
   signatures as the reference `Model`. It is a thin wrapper: call
   `torch.ops.custom_op.run(...)` (or the names you exported). The evaluator
   already loaded `libcustom_op.so` with `torch.ops.load_library`. Do not
   `import custom_op`, do not call `torch.ops.load_library`, do not change
   `sys.path`, and do not install anything. Keep all optimisation work in
   the `.asc` file.
   - Parameters: if the reference `Model` has parameters (e.g. `nn.Conv2d`,
     `nn.Linear`, norm layers), you MAY instantiate the same `nn` modules in
     `ModelNew.__init__` as parameter containers — the evaluator seeds the
     RNG identically before constructing `Model` and `ModelNew`, so identical
     construction yields identical weights — but you must NEVER call them;
     pass their `.weight`/`.bias` tensors into your custom op.

Do not output any test code, `if __name__ == "__main__"` blocks, or prose
between the two code blocks. In model_new.py, ALL tensor compute must go
through `torch.ops.custom_op`: no torch native operators in any form —
free functions
(`torch.matmul`), tensor methods (`x.softmax(...)`), operators (`A @ B`,
`A + B` on tensors), or comparisons on tensor data — and no nn.functional,
torch_npu/aclnn shortcuts, CPU/NumPy fallbacks, try/except, dynamic imports
(`importlib`, `__import__`, `getattr` on torch), or result caching. Integer
shape arithmetic (shapes, strides, counts) is of course allowed.
"""

INSTRUCTION = """\
## Instruction

Implement the operator defined by the reference model above in Ascend C.
Generate real, compilable code: every API you use must exist in the new-style
Ascend C API shown in the example. The evaluator will compile this file with
its fixed CMake project into libcustom_op.so and load that library with
torch.ops.load_library; you only write the two code blocks defined by the
Output Contract.
"""

PYASC_SYSTEM_PROMPT = (
    "You are an expert Ascend kernel engineer. You write correct, "
    "high-performance operators for Huawei Ascend NPUs with pyasc, the "
    "official CANN Python DSL (package pyasc, import name asc). The "
    "benchmark launches your kernels from plain Python inside the evaluating "
    "PyTorch process: kernel.py holds one or more @asc.jit kernels plus a "
    "plain-Python host launcher function, and model_new.py calls that "
    "launcher. All tensor compute runs on the device inside the @asc.jit "
    "kernels; the host side only prepares arguments and launches."
)

PYASC_OUTPUT_CONTRACT = """\
## Output Contract

Output exactly two fenced code blocks, tagged with their filenames:

1. ```kernel.py — the pyasc kernel module. It has two parts:
   a. one or more kernels decorated with `@asc.jit`. A kernel parameter is
      an `asc.GlobalAddress` when it is a device tensor, a plain `int` /
      `float` / `bool` when it is a runtime scalar, and
      `asc.ConstExpr[int]` / `asc.ConstExpr[float]` when it is a compile-time
      constant. Calling a kernel never returns a value: each output is
      written into a tensor that is passed in;
   b. a plain-Python host launcher function (no `@asc.jit`) that prepares
      arguments, allocates every output with `torch.empty` / `torch.empty_like`,
      launches the kernel as `kernel[core_num, rt.current_stream()](x, y, out, ...)`
      with `import asc.lib.runtime as rt`, and returns the output tensors. The
      launcher ends with `rt.synchronize()`, so a launched kernel is
      synchronous. The evaluator selects the execution platform and the NPU
      device before the launcher runs, so the module must not call
      `asc.runtime.config.set_platform` itself.
   Inside a `@asc.jit` kernel use only the real pyasc device API, for
   example:
   - `asc.GlobalTensor()`, `gm.set_global_buffer(addr[, length])`, and
     `gm[offset:]` slicing to address a device buffer;
   - `asc.get_block_idx()` with one core-count constant to partition work
     across cores;
   - `asc.LocalTensor(dtype, asc.TPosition.VECIN, offset, length)`,
     `asc.data_copy(dst, src, count)`, and vector ops such as
     `asc.add(dst, a, b, count)`. A `data_copy` count is in elements; keep
     it to whole 32-byte blocks, because a shorter write-back can silently
     leave the destination unchanged on the target device;
   - `asc.set_flag(asc.HardEvent.MTE2_V, id)` and
     `asc.wait_flag(asc.HardEvent.MTE2_V, id)` when explicit pipe
     synchronization is needed, choosing the event that matches the
     consumer's pipe (MTE2_V before a vector op, MTE2_MTE3 before a
     write-back copy), or the TPipe style
     (`pipe = asc.TPipe()`, `asc.TQue(asc.TPosition.VECIN, BUFFER_NUM)`,
     `pipe.init_buffer(q, BUFFER_NUM, length * dtype.sizeof())`,
     `q.alloc_tensor(dtype)`, `q.enque(t)`, `q.deque(dtype)`,
     `q.free_tensor(t)`).
   The dtype comes from the tensor annotation (`x.dtype`) and
   `dtype.sizeof()` gives its element size. A helper kernel may itself be
   decorated with `@asc.jit`.
2. ```model_new.py — class `ModelNew` with the SAME `__init__` and `forward`
   signatures as the reference `Model`. Import the launcher from kernel.py
   (`from kernel import run`) and call it; shape bookkeeping is fine, but
   every tensor computation must happen inside the `@asc.jit` kernels.
   - Parameters: if the reference `Model` has parameters (e.g. `nn.Conv2d`,
     `nn.Linear`, norm layers), you MAY instantiate the same `nn` modules in
     `ModelNew.__init__` as parameter containers — the evaluator seeds the
     RNG identically before constructing `Model` and `ModelNew`, so identical
     construction yields identical weights — but you must NEVER call them;
     pass their `.weight` / `.bias` tensors into the launcher.

The host side may only do shape arithmetic, argument preparation, output
allocation, and kernel launch. All tensor compute must happen in the
`@asc.jit` kernels: no PyTorch, NumPy, CPU, or vendor-native compute may be
used as a substitute for the device kernels, and no `torch` operator may
appear in a launcher or in `ModelNew.forward` other than allocation and
integer shape arithmetic (shapes, strides, counts). Do not output any test
code, `if __name__ == "__main__"` blocks, or prose between the two code
blocks.
"""

PYASC_INSTRUCTION = """\
## Instruction

Implement the operator defined by the reference model above with pyasc, the
official CANN Python DSL, on the target NPU. Generate real, runnable code:
every API you use must exist in the pyasc API shown in the example. The
evaluator imports kernel.py and model_new.py, calls the launcher, and
compares the device result against the reference model; you only write the
two code blocks defined by the Output Contract.
"""


class PromptMode(str, Enum):
    """How many bundled examples the user prompt includes."""

    ZERO_SHOT = "zero_shot"
    ONE_SHOT = "one_shot"
    FEW_SHOT = "few_shot"

    def chosen_examples(self, pool: list[PromptExample]) -> list[PromptExample]:
        """Return the example slice this mode should embed."""
        if self is PromptMode.ZERO_SHOT:
            return []
        if self is PromptMode.ONE_SHOT:
            return pool[:1]
        return list(pool)


class PromptExample(BaseModel):
    """A verified example pair (task input -> expected answer)."""

    model_config = ConfigDict(frozen=True)

    name: str
    task_py: str
    kernel_src: str
    kernel_tag: str
    model_new_py: str


# Backends whose examples live in a subdirectory, named after the backend.
_BACKEND_EXAMPLE_SUBDIR = {Backend.PYASC: Backend.PYASC.value}


def _backend_name(value: Backend | str) -> str:
    """Return the name parse_backend expects for a Backend or its value."""
    return value.value if isinstance(value, Backend) else value


def _resolve_backend(value: Backend | str) -> Backend:
    """Return the Backend a name or Backend value resolves to."""
    return parse_backend(_backend_name(value))


def _example_pool_dir(backend: Backend) -> Path:
    """Return the prompt-example directory that serves one backend."""
    subdir = _BACKEND_EXAMPLE_SUBDIR.get(backend)
    return PROMPT_EXAMPLES_DIR / subdir if subdir else PROMPT_EXAMPLES_DIR


def load_examples(backend: Backend | str = DEFAULT_BACKEND) -> list[PromptExample]:
    """Load verified few-shot example assets shipped with the engine.

    Args:
        backend: Authoring language whose examples are loaded. The Ascend C
            pool is the top level of the examples directory; a backend with
            its own subdirectory reads that subdirectory.

    Raises:
        OSError: If an example directory is missing a required file.
    """
    examples: list[PromptExample] = []
    for example_dir in sorted(_example_pool_dir(_resolve_backend(backend)).iterdir()):
        if not example_dir.is_dir() or not (example_dir / "task.py").is_file():
            continue
        kernel_file, wrapper_file = sample_files(_resolve_backend(backend))
        examples.append(
            PromptExample(
                name=example_dir.name,
                task_py=(example_dir / "task.py").read_text(encoding="utf-8"),
                kernel_src=(example_dir / kernel_file).read_text(encoding="utf-8"),
                kernel_tag=kernel_file,
                model_new_py=(example_dir / wrapper_file).read_text(encoding="utf-8"),
            )
        )
    return examples


# pyasc twin of 002_leaky_relu is TODO: it needs a verified signature for
# the scalar/activation ops (asc.leaky_relu, asc.adds) before it can ship.
def load_pyasc_examples() -> list[PromptExample]:
    """Load the pyasc few-shot examples."""
    return load_examples(Backend.PYASC)


class PromptBuilder:
    """Assemble the generation prompt as an ordered list of sections."""

    def __init__(
        self,
        task: Task,
        hardware: HardwareProfile,
        *,
        backend: Backend | str = DEFAULT_BACKEND,
        examples: list[PromptExample] | None = None,
    ) -> None:
        """Bind the task, hardware profile, backend, and example override."""
        self.task = task
        self.hardware = hardware
        self.backend = _resolve_backend(backend)
        self._examples = examples

    def build(self, mode: str | PromptMode = PromptMode.ONE_SHOT) -> str:
        """Return the complete English user prompt (no system message).

        Raises:
            ValueError: If mode is unknown or examples are missing.
        """
        try:
            if isinstance(mode, PromptMode):
                resolved = mode
            else:
                resolved = PromptMode(mode)
        except ValueError as exc:
            raise ValueError(f"Unknown prompt mode: {mode}") from exc
        sections = [self._problem_statement(), self._hardware_block()]
        chosen = resolved.chosen_examples(self._example_pool(resolved))
        if chosen:
            sections.append(self._examples_block(chosen))
        sections.extend(self._wording())
        return "\n".join(sections)

    def _wording(self) -> tuple[str, str]:
        """Return the (output contract, instruction) pair for this backend."""
        if self.backend is Backend.PYASC:
            return PYASC_OUTPUT_CONTRACT, PYASC_INSTRUCTION
        return OUTPUT_CONTRACT, INSTRUCTION

    def _example_pool(self, mode: PromptMode) -> list[PromptExample]:
        """Load bundled examples when the mode needs them."""
        if mode is PromptMode.ZERO_SHOT:
            return []
        pool = (
            self._examples
            if self._examples is not None
            else load_examples(self.backend)
        )
        if not pool:
            raise ValueError("no prompt examples available")
        return pool

    def _problem_statement(self) -> str:
        """Return the English problem-statement block."""
        intro = (
            "Implement the operator defined by the reference PyTorch model "
            "below as\na pyasc kernel (the CANN Python DSL) on the target NPU."
            if self.backend is Backend.PYASC
            else "Implement the operator defined by the reference PyTorch "
            "model below as an\nAscend C kernel on the target NPU."
        )
        return f"""\
## Problem Statement

{intro}

```python
{self.task.task_py}
```
"""

    def _hardware_block(self) -> str:
        """Return the English hardware-contract block."""
        hw = self.hardware
        dtypes = ", ".join(hw.supported_dtypes)
        cores = f"{hw.ai_core_num}"
        if hw.cube_core_num or hw.vector_core_num:
            cores = (
                f"{hw.ai_core_num} (cube={hw.cube_core_num}, "
                f"vector={hw.vector_core_num})"
            )
        return f"""\
## Target Hardware Contract

- SoC: {hw.soc_version} (CMake arch `{hw.cmake_arch}`)
- AI cores: {cores}; UB budget per core: {hw.ub_size_kb} KB
- HBM: {hw.hbm_gb} GB, bandwidth ~{hw.memory_bandwidth_gbps} GB/s
- Supported dtypes: {dtypes}

{self._api_style_block()}
"""

    def _api_style_block(self) -> str:
        """Return the API-style note for this backend."""
        if self.backend is Backend.PYASC:
            return (
                "### pyasc API style (mandatory)\n\n"
                "Use the pyasc Python DSL (import name asc) as shown in the "
                "example. Do not emit Ascend C, CMake, or C++ sources."
            )
        return f"### Ascend C API style (mandatory)\n\n{self.hardware.api_style}"

    def _examples_block(self, examples: list[PromptExample]) -> str:
        """Render verified example pairs as prompt markdown."""
        parts = ["## Example\n"]
        for example in examples:
            parts.append(
                f"### Example task: {example.name}\n\n"
                f"Reference Model:\n\n```python\n{example.task_py}\n```\n\n"
                f"Expected answer:\n\n"
                f"```{example.kernel_tag}\n{example.kernel_src}\n```\n\n"
                f"```model_new.py\n{example.model_new_py}\n```\n"
            )
        return "\n".join(parts)


def build_prompt(
    task: Task,
    hardware: HardwareProfile,
    *,
    mode: str = "one_shot",
    backend: Backend | str = DEFAULT_BACKEND,
    examples: list[PromptExample] | None = None,
) -> str:
    """Assemble the full generation prompt for one task.

    Args:
        backend: Authoring language; a name or a Backend value.

    Raises:
        ValueError: If mode or backend is unknown, or examples are missing.
    """
    resolved = _resolve_backend(backend)
    return PromptBuilder(task, hardware, backend=resolved, examples=examples).build(
        mode
    )
