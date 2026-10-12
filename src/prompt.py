"""Component-based prompt construction (docs/reference/configuration.md).

Sections assemble in order: problem statement, hardware block, examples block,
output contract, instruction. Modes: zero_shot, one_shot, few_shot.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from ._paths import PROMPT_EXAMPLES_DIR
from .config import (
    DEFAULT_BUILD_BACKEND,
    BuildBackend,
    HardwareProfile,
    parse_build_backend,
)
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


DEEPJIT_SYSTEM_PROMPT = (
    "You are an expert Ascend C kernel engineer. You write correct, "
    "high-performance Ascend C operators for Huawei Ascend NPUs. The "
    "benchmark compiles the device units of your source with the DeepJIT "
    "runtime and loads the resulting process-local shared library with "
    "torch.ops.load_library inside the evaluating PyTorch process. Register "
    "the operator with TORCH_LIBRARY / TORCH_LIBRARY_IMPL. Do not use "
    "pybind11. The library is never installed into site-packages or the "
    "global CANN OPP path."
)

DEEPJIT_OUTPUT_CONTRACT = """\
## Output Contract

The benchmark owns the build. You must NOT emit CMakeLists.txt, build
scripts, op_host / op_kernel trees, framework plugins, deployment steps, or
any file other than the two below. Output exactly two fenced code blocks,
tagged with their filenames:

1. ```custom_op.asc — one Ascend C source file in two sections, separated by
   exactly one marker line (copy it verbatim):

   // ==================== ASCEND_HOST_SECTION ====================

   DEVICE SECTION (above the marker) — compiled without torch headers:
   a. start with `#include "kernel_operator.h"`, then the kernel class
      (`Init` for the data partition and GM buffers, `Process` for the UB
      allocation, copy-in, compute, and copy-out), then the kernel entry;
   b. entry functions are annotated `__global__ __vector__`, call
      `AscendC::InitSocState()`, run `Init` and `Process`, and end with
      `AscendC::PipeBarrier<PIPE_ALL>()`;
   c. exactly one entry per compiled unit. For more than one kernel, separate
      consecutive entries with this marker line (copy it verbatim):

      // ==================== ASCEND_KERNEL_SECTION ====================

      Everything before the first kernel marker is a shared preamble
      (includes, constants, helper structs, `__aicore__` helper functions)
      that is prepended to every unit; every segment after a marker holds
      exactly one entry, and entry names must be distinct. A source with no
      kernel marker is one unit;
   d. keep every vector operand 32-byte aligned: index local tensors by
      multiples of eight fp32 elements and size the partial buffers to match;
   e. never include torch, ATen, or torch_npu headers in the device section.

   HOST SECTION (below the marker) — compiled as plain C++17 and as a
   separate translation unit from each device unit:
   f. exactly these includes: `#include <torch/library.h>`,
      `#include <ATen/ATen.h>`, and
      `#include "torch_npu/csrc/core/npu/NPUStream.h"`;
   g. repeat in the host section every constant it needs (tile sizes, lane
      counts): the device section is a different translation unit and its
      constants are not visible here;
   h. host wrapper taking `const at::Tensor&` arguments, fetching the current
      NPU stream via `c10_npu::getCurrentNPUStream().stream(false)`,
      allocating every output and intermediate tensor itself, and launching
      each kernel through its generated stub: for a kernel `foo(...)` call
      `foo_launch(numBlocks, (void*)aclStream, <kernel args in order>)`.
      The framework generates `foo_launch` from your signature, so never
      declare or define it. Kernel pointer parameters (`__gm__ T*`) become
      plain `T*` at the call site. When one kernel consumes what another
      produced, launch both on that same stream in order and keep the
      intermediate tensor alive until both launches are enqueued;
   i. a process-local torch.library binding. Register the host function with
      `TORCH_LIBRARY(custom_op, m)` and bind the NPU implementation with
      `TORCH_LIBRARY_IMPL(custom_op, PrivateUse1, m)`. Do NOT use
      `PYBIND11_MODULE` or any pybind11 header. The evaluator will
      `torch.ops.load_library` the resulting library. The library namespace
      MUST be `custom_op`. Exported function names are free (`run` is the
      convention). Schema strings must match the host function (for example
      `run(Tensor x) -> Tensor`). The host wrapper may only allocate memory
      and launch kernels — all compute must happen inside the Ascend C
      kernel, never in host-side ATen calls (`at::matmul`, `tensor.relu()`,
      ...) or vendor prebuilt ops (`aclnn*`).
   Supported device parameter types: `__gm__ uint8_t*`, `__gm__ int8_t*`,
   `__gm__ float*`, `__gm__ int32_t*`, `__gm__ uint32_t*`, and the scalars
   `int`, `int32_t`, `uint32_t`, `int64_t`, `float`. Array parameters are not
   supported. Write includes in angle form, except the standard quoted
   `#include "kernel_operator.h"` for that one header.

   The build, the kernel cache, and the preparation step belong to the
   framework: it compiles each entry before your wrapper runs, reuses cached
   artifacts in later processes, and reports a compile failure per kernel.
   Never emit your own compile, cache, or library-loading step, and never
   assume the device source is compiled more than once per process.
2. ```model_new.py — class `ModelNew` with the SAME `__init__` and `forward`
   signatures as the reference `Model`. It is a thin wrapper: call
   `torch.ops.custom_op.run(...)` (or the names you exported). The evaluator
   already loaded the library with `torch.ops.load_library`. Do not
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
free functions (`torch.matmul`), tensor methods (`x.softmax(...)`), operators
(`A @ B`, `A + B` on tensors), or comparisons on tensor data — and no
nn.functional, torch_npu/aclnn shortcuts, CPU/NumPy fallbacks, try/except,
dynamic imports (`importlib`, `__import__`, `getattr` on torch), or result
caching. Integer shape arithmetic (shapes, strides, counts) is of course
allowed.
"""

DEEPJIT_INSTRUCTION = """\
## Instruction

Implement the operator defined by the reference model above in Ascend C.
Generate real, compilable code: every API you use must exist in the new-style
Ascend C API shown in the example. The framework compiles each device entry
with DeepJIT and loads the resulting library with torch.ops.load_library; you
only write the two code blocks defined by the Output Contract.
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
    custom_op_asc: str
    model_new_py: str


# The Ascend C CMake pool is the top level of the examples directory; the
# deepjit path has its own pool because its host rules differ.
DEEPJIT_EXAMPLE_SUBDIR = "deepjit"


def _example_pool_dir(
    build_backend: BuildBackend = DEFAULT_BUILD_BACKEND,
) -> Path:
    """Return the prompt-example directory that serves one build path."""
    if build_backend is BuildBackend.DEEPJIT:
        return PROMPT_EXAMPLES_DIR / DEEPJIT_EXAMPLE_SUBDIR
    return PROMPT_EXAMPLES_DIR


def load_examples(
    build_backend: BuildBackend | str = DEFAULT_BUILD_BACKEND,
) -> list[PromptExample]:
    """Load verified few-shot example assets shipped with the engine.

    Args:
        build_backend: Build path whose example pool is used.

    Raises:
        OSError: If an example directory is missing a required file.
    """
    examples: list[PromptExample] = []
    pool_dir = _example_pool_dir(parse_build_backend(build_backend))
    for example_dir in sorted(pool_dir.iterdir()):
        if not example_dir.is_dir() or not (example_dir / "task.py").is_file():
            continue
        examples.append(
            PromptExample(
                name=example_dir.name,
                task_py=(example_dir / "task.py").read_text(encoding="utf-8"),
                custom_op_asc=(example_dir / "custom_op.asc").read_text(
                    encoding="utf-8"
                ),
                model_new_py=(example_dir / "model_new.py").read_text(encoding="utf-8"),
            )
        )
    return examples


class PromptBuilder:
    """Assemble the generation prompt as an ordered list of sections."""

    def __init__(
        self,
        task: Task,
        hardware: HardwareProfile,
        *,
        build_backend: BuildBackend | str = DEFAULT_BUILD_BACKEND,
        examples: list[PromptExample] | None = None,
    ) -> None:
        """Bind the task, hardware, build path, and optional example override."""
        self.task = task
        self.hardware = hardware
        self.build_backend = parse_build_backend(build_backend)
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
        contract, instruction = self._wording()
        sections.extend([contract, instruction])
        return "\n".join(sections)

    def _wording(self) -> tuple[str, str]:
        """Return the (output contract, instruction) pair for this path."""
        if self.build_backend is BuildBackend.DEEPJIT:
            return DEEPJIT_OUTPUT_CONTRACT, DEEPJIT_INSTRUCTION
        return OUTPUT_CONTRACT, INSTRUCTION

    def _example_pool(self, mode: PromptMode) -> list[PromptExample]:
        """Load bundled examples when the mode needs them."""
        if mode is PromptMode.ZERO_SHOT:
            return []
        pool = (
            self._examples
            if self._examples is not None
            else load_examples(self.build_backend)
        )
        if not pool:
            raise ValueError("no prompt examples available")
        return pool

    def _problem_statement(self) -> str:
        """Return the English problem-statement block."""
        return f"""\
## Problem Statement

Implement the operator defined by the reference PyTorch model below as an
Ascend C kernel on the target NPU.

```python
{self.task.task_py}
```
"""

    def _arch_label(self) -> str:
        """Return the arch field label for this build path."""
        if self.build_backend is BuildBackend.DEEPJIT:
            return "target arch"
        return "CMake arch"

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

- SoC: {hw.soc_version} ({self._arch_label()} `{hw.cmake_arch}`)
- AI cores: {cores}; UB budget per core: {hw.ub_size_kb} KB
- HBM: {hw.hbm_gb} GB, bandwidth ~{hw.memory_bandwidth_gbps} GB/s
- Supported dtypes: {dtypes}

### Ascend C API style (mandatory)

{hw.api_style}
"""

    def _examples_block(self, examples: list[PromptExample]) -> str:
        """Render verified example pairs as prompt markdown."""
        parts = ["## Example\n"]
        for example in examples:
            parts.append(
                f"### Example task: {example.name}\n\n"
                f"Reference Model:\n\n```python\n{example.task_py}\n```\n\n"
                f"Expected answer:\n\n"
                f"```custom_op.asc\n{example.custom_op_asc}\n```\n\n"
                f"```model_new.py\n{example.model_new_py}\n```\n"
            )
        return "\n".join(parts)


def build_prompt(
    task: Task,
    hardware: HardwareProfile,
    *,
    mode: str = "one_shot",
    build_backend: BuildBackend | str = DEFAULT_BUILD_BACKEND,
    examples: list[PromptExample] | None = None,
) -> str:
    """Assemble the full generation prompt for one task.

    Args:
        build_backend: Build path whose wording and example pool are used.

    Raises:
        ValueError: If mode or build backend is unknown, or examples are
            missing.
    """
    return PromptBuilder(
        task, hardware, build_backend=build_backend, examples=examples
    ).build(mode)


def system_prompt(
    build_backend: BuildBackend | str = DEFAULT_BUILD_BACKEND,
) -> str:
    """Return the system message for one build path."""
    if parse_build_backend(build_backend) is BuildBackend.DEEPJIT:
        return DEEPJIT_SYSTEM_PROMPT
    return SYSTEM_PROMPT
