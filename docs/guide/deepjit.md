# DeepJIT build backend

An optional, opt-in build path for the Ascend C backend. A candidate keeps
its normal shape: `custom_op.asc` with the host-section marker and a
`model_new.py` that calls `torch.ops.custom_op`. Only the build path differs:
instead of the CMake template, a PyTorch extension embeds the DeepJIT runtime
and compiles the device kernel at prepare time.

`cmake` stays the default. A run records its build backend in
`generation_config.yaml`, and evaluation reads it from the run, so one run
keeps one build path.

## Configure

Global defaults live in `configs/eval_default.yaml`:

```yaml
build_backend: cmake        # cmake | deepjit
deepjit:
  root: ""                  # DeepJIT checkout holding include/deep_jit
  commit: ""                # optional pin; a mismatch is refused
  cxx: ""                   # host compiler; needs std::format (GCC 13+)
  include_dirs: []          # extra -I paths for the host extension
  library_dirs: []          # extra -L and -rpath paths
  env: {}                   # environment applied in the worker process
  extra_cflags: []
  extra_ldflags: []
  cache_dir: ""             # empty means <sample>/deepjit/cache
  build_timeout: 1800
```

Every path is local machine configuration; nothing machine-specific is
committed. Empty `root`, `commit`, `cxx`, and `cache_dir` fall back to
`DEEPJIT_ROOT`, `DEEPJIT_COMMIT`, `DEEPJIT_CXX`, and `DEEPJIT_CACHE_DIR`;
`DEEPJIT_INCLUDE_DIRS` and `DEEPJIT_LIBRARY_DIRS` are colon-separated.

Generation records the choice and the DeepJIT settings in the run:

```bash
python scripts/generate.py --task level1/19_ReLU --model <model> \
    --build-backend deepjit --deepjit-root /path/to/DeepJIT \
    --deepjit-commit <sha> --run-name relu_deepjit
```

`scripts/evaluate.py` takes no build flags: it reads the run record. A run
with `build_backend: deepjit` and `backend: pyasc` is refused, because pyasc
never builds a shared library.

## How one sample is built

1. Static checks run on the host exactly as for `cmake`.
2. The worker builds a host extension with `torch.utils.cpp_extension`, using
   `-DDJ_DISABLE_GIL=1` as the DeepJIT integration guide requires for code
   called through `torch.ops`. Nothing is installed globally.
3. The worker loads it with `torch.ops.load_library`.
4. The framework prepare entry `akb_deepjit_prepare()` compiles the device
   kernel through DeepJIT. Nothing is launched and no candidate code runs
   before it. Only this step sets `compiled`. The entry is called across a C
   ABI, so it catches every C++ exception and returns a status code plus a
   message instead of unwinding out of the boundary.
5. The candidate's `model_new.py` calls the operator; the generated
   `<kernel>_launch(...)` stub launches through DeepJIT.

Files the framework generates under `<sample>/deepjit/`:

| File | Role |
| --- | --- |
| `akb_deepjit_kernel/kernel_{i}.hpp` | the device source actually compiled for kernel i |
| `akb_deepjit_runtime.hpp` | framework runtime: Config, create_lazy_jit, compile, launch |
| `akb_deepjit_launcher.cpp` | prepare entry plus one launch stub per kernel |
| `akb_deepjit_launch_decls.h` | declarations force-included into the host TU |
| `custom_op_host.cpp` | the candidate host section with the framework prologue |
| `build_facts.json` | commit, signature, cache dir, kernel hash (copied to metadata) |
| `cache/` | DeepJIT artifact cache for this sample |

The candidate never runs a build command, never chooses the DeepJIT root, and
never sees the runtime headers.

The launch stub resolves its kernel handle from a per-process cache that
prepare fills, so a timed launch cannot reach the compiler or the include
parser. The stub passes the stream it is handed into the launch options; the
generated sources contain no device-wide synchronize, so a wrong stream
cannot be hidden by one.

## Generation

Generation follows the same three steps as the CMake path, with the deepjit
wording and example pool:

```bash
python scripts/generate.py --task level1/19_ReLU --model <model> \
    --build-backend deepjit --deepjit-root <checkout> --run-name relu_deepjit
```

`build_prompt`, `PromptBuilder`, and `system_prompt` take `build_backend`, so
the prompt states the conventions this page describes: the two markers, one
entry per compiled unit, the shared preamble, per-unit scope with host-side
constant repetition, the supported parameter types, the generated launcher
call, host-owned outputs and intermediates on the current stream, and the
framework-owned build, cache, and prepare step. The cmake and pyasc wordings
are unchanged, and each path reads its own example pool under
`src/prompts/examples/`: the Ascend C pool at the top level, `pyasc/`, and
`deepjit/`. The pools hold the verified assets only, and the existing
`prompt_mode` rules still decide how many examples reach the prompt, so
`one_shot` stays a single example.

## Source rules

The device section may use angle-bracket includes and the standard quoted
`#include "kernel_operator.h"`, which is rewritten as `#include
<kernel_operator.h>` because the DeepJIT include parser refuses quoted
includes. The rewritten file is what gets compiled. Any other quoted include
is refused rather than rewritten.

Pointer parameters must be `__gm__` pointers to `uint8_t`, `int8_t`, `float`,
`int32_t`, or `uint32_t`; scalar parameters must be `int`, `int32_t`,
`uint32_t`, `int64_t`, or `float`. Anything else, a source without the
host-section marker, or a path containing whitespace or shell metacharacters
is reported as a build error. The path never falls back to CMake.

## More than one kernel

DeepJIT requires exactly one kernel entry per compiled artifact, so a
multi-kernel sample is compiled as one unit per entry. The device section is
split by an explicit marker line, never by a pattern that could cut a
function body:

```cpp
#include <kernel_operator.h>
constexpr uint32_t kTileElements = 2048;   // shared preamble

// ==================== ASCEND_KERNEL_SECTION ====================
__global__ __vector__ void partial_sum(...) { ... }

// ==================== ASCEND_KERNEL_SECTION ====================
__global__ __vector__ void finalize_sum(...) { ... }
```

Rules:

* The text before the first marker is a shared preamble and is prepended to
  every unit. `#include` lines, constants, helper structs, and `__aicore__`
  helper functions belong there.
* Every unit is `shared preamble + one segment` and must hold exactly one
  `__global__ __vector__` entry. A segment with no entry, a unit with two
  entries, or an entry before the first marker is refused with a build error
  that names what was found.
* Without any marker the whole device section is one unit, so single-kernel
  samples are unchanged.
* Entry names must be distinct; they are also the DeepJIT cache names, so
  each kernel gets its own cache entry and its own bisheng and ld.lld calls.
* The host section is unchanged and calls one `<kernel>_launch(...)` stub per
  entry, in the order the two kernels must run.

`akb_deepjit_prepare()` prepares every unit in order and returns the status
of the first failure with a message that names the kernel, so `compiled` is
only true when every kernel is ready. The launch stubs then start only the
handles prepare filled; a timed launch never compiles. A change inside the
shared preamble invalidates every unit, while a change inside one segment
invalidates only that unit.

## ACL include order

DeepJIT's Ascend headers include `torch_npu/csrc/core/npu/NPUStream.h`, which
expands torch_npu's trimmed `acl/acl_rt.h` first; CANN's full copy is then
suppressed by the identical include guard, and the symbols DeepJIT loads are
missing. The generated runtime header and host TU therefore include
`<acl/acl.h>` before any DeepJIT or torch_npu header, and CANN's include
directories come before torch_npu's. Do not reorder them.

## Verified facts

Measured on Ascend 910B4 (20 AI cores, 40 vector cores, 29.5 GiB HBM),
CANN 9.1.0, torch 2.10.0 + torch_npu 2.10.0, DeepJIT commit
`3732a3b9a0a2e5126396908a362ba5abdf043b2f`:

* `ACL_DEV_ATTR_NPU_ARCH` is `2201`, so DeepJIT compiles with
  `bisheng -O2 --npu-arch=dav-2201 --cce-aicore-only` and links with
  `ld.lld -m aicorelinux -Ttext 0`.
* A host build costs about 23 s cold and the device compile about 6 s per
  kernel cold and 0.00 s on a warm cache. A two-kernel sample therefore
  shows two bisheng and two ld.lld invocations on a cold cache and none on a
  warm one.
* The benchmark Add kernel, compiled through this path, returns exact results
  for 1, 2, 10, and 40 blocks, including a per-core remainder and a tail tile
  whose length is not 32-byte aligned.
* `ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE` reports `0` on this device. It is
  recorded as a known limit; no value is hardcoded.

## Known limits

* The official DeepJIT SIMD sample (`__cce_simd`, `__simd_vf__`) does not
  compile against CANN 9.1.0 headers: the intrinsic header chain is not
  reachable and `tikcpp` does not surface `__cce_simd`. This is unrelated to
  the Ascend C path above, which uses `AscendC::` interfaces.
* Level-1 activation tasks hold 6 GiB per tensor. On a 29.5 GiB device the
  harness needs more than that during the hidden-distribution stage, so such
  a sample ends with `OutOfMemoryError` in `hidden_candidate`. That is a
  device-capacity limit, not a build-path failure.
* A run whose sample path contains whitespace is refused, because DeepJIT
  passes compiler and flag strings through a shell command.
