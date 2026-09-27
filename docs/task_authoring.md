# Authoring tasks and candidates

A benchmark task is a self-contained Python reference file under `KernelBench/level{L}/`. The evaluator uses the original KernelBench contract, and the generator receives the complete source as the problem statement. There is no separate `spec.md` or per-task build project.

## Task identity and discovery

The file `KernelBench/level1/19_ReLU.py` has task ID `level1/19_ReLU`. This ID also identifies the task in run directories and in `eval_results.json`.

Discovery reads `*.py` files directly inside each `level*` directory and sorts filenames by their leading numeric identifier. The loader parses the source with Python's AST and checks for top-level symbols named `Model`, `get_inputs`, and `get_init_inputs`. It does not import PyTorch or execute task code during discovery.

An explicit task load also accepts the legacy `level{L}/{task}/task.py` layout, but automatic discovery only scans the single-file layout. Add new tasks as single files.

## Reference contract

| Symbol | Contract |
| --- | --- |
| `Model` | A `torch.nn.Module` whose `forward` implements the reference computation. |
| `get_init_inputs()` | Returns a list of positional constructor arguments for `Model` and `ModelNew`. Prefer deterministic values that do not consume RNG state. |
| `get_inputs()` | Returns a fresh list of positional forward arguments. Create tensors on CPU; the evaluator moves top-level tensor arguments to the target NPU. |
| `TOLERANCE` | Optional, nonempty dictionary such as `{"atol": 1e-3, "rtol": 1e-3}`. Overrides the configured floating-point comparison tolerances. |
| `custom_check(ref_out, out)` | Optional predicate replacing the default output comparison. Return a Boolean-compatible value. |
| `FLOPS` or `NUM_FLOPS` | Optional positive number. When set, and the hardware profile has `peak_tflops` for the active precision, the SOL bound includes a compute term. The vendored corpus does not set this. |

Use a tensor or a tuple/list of tensors for outputs. The default matcher recursively handles tuples and lists, but CPU fallback only transfers top-level tensor outputs and one level of tuple/list members to CPU. Nested tensor input containers are also not recursively transferred. Keep tensor arguments and returned collections flat if the task must support both reference modes.

This complete example defines a small parameter-free task:

```python
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return a + b


def get_init_inputs():
    return []


def get_inputs():
    return [torch.randn(1024), torch.randn(1024)]
```

This is a contract example, not a hardware-validated benchmark result.

## Determinism and state

The evaluator uses the same base seed before constructing the candidate and reference. A candidate can reproduce random parameter initialization by constructing the same layers in the same order. It does not receive a copied reference `state_dict`.

Both models run in `eval()` mode, and forward calls run under `torch.no_grad()`. Design the reference as a deterministic function of its current arguments and initialized parameters. Do not depend on counters, cached results, wall-clock time, file contents, network responses, or mutable state carried between calls. Task code should not perform I/O, launch processes, or invoke compilation.

These are authoring requirements, not guarantees established by the AST contract check. That check verifies symbol presence and Python syntax only. The task module is executed later inside the evaluation worker.

`get_inputs()` is called repeatedly with different seeds for correctness and, for input sets up to 256 MiB, for every timing call. Large tensor shapes can dominate allocation time or exceed device memory. Choose shapes deliberately and preserve their intent when modifying existing tasks.

## Precision and output checks

The evaluator casts floating-point input tensors and model parameters to the selected precision (`fp32`, `fp16`, or `bf16`). Integer and Boolean input tensors retain their dtype. Non-tensor positional arguments pass through unchanged.

By default, tensor shapes must match. Floating-point and complex outputs use `torch.allclose` with the resolved tolerances; integer/Boolean outputs use `torch.equal` when both outputs are discrete tensors. Tuple/list lengths must match, and their members are checked recursively. Other values are compared using equality.

A nonempty task `TOLERANCE` takes precedence over the evaluation configuration. If either tolerance key is omitted from that dictionary, its fallback is `1e-4`. A `custom_check` takes precedence over the default matcher and is used for both the initial trials and the post-timing check. It should validate the whole result, including structure and shape where relevant; the default shape check does not run first.

The primary reference is eager PyTorch with `torch_npu` on the selected NPU. A reference construction failure or an exception from reference execution/synchronization during an initial correctness trial switches reference execution to CPU. CPU inputs use float32 for floating-point tensors, and no NPU reference speedup is reported. A CPU fallback result therefore does not, by itself, prove that the operator is unsupported by `torch_npu`: inspect `metadata.reference_npu_error` for the actual cause. See the [evaluation protocol](guide/evaluation.md).

## Candidate contract

Each generated sample contains the two implementation files used by the build/evaluation pipeline:

| File | Required content |
| --- | --- |
| `custom_op.asc` | A two-section Ascend C implementation: device kernels above the `ASCEND_HOST_SECTION` marker, the host launch wrapper and a process-local `TORCH_LIBRARY(custom_op, ...)` / `TORCH_LIBRARY_IMPL(custom_op, PrivateUse1, ...)` binding below it. |
| `kernel.py` | pyasc backend only: one or more `@asc.jit` device kernels plus a plain-Python host launcher. The launcher allocates outputs, launches with `kernel[core_num, rt.current_stream()](...)`, and must not select the platform. |
| `model_new.py` | A `ModelNew` class with constructor and forward signatures matching `Model`, calling `torch.ops.custom_op`. |

The `.asc` source must contain `__global__`, `__vector__`, `TORCH_LIBRARY`, and `TORCH_LIBRARY_IMPL`, plus exactly one `// ==================== ASCEND_HOST_SECTION ====================` separator line. The device section includes only `kernel_operator.h`; the host section includes `torch/library.h`, `ATen/ATen.h`, and the `torch_npu` stream header, and launches each kernel through the generated `<kernel>_launch(numBlocks, (void*)stream, args...)` stub. Follow the bundled [elementwise-add example](https://github.com/wuzhenqing/AscendKernelBench/tree/main/src/prompts/examples/001_elementwise_add) for the kernel class, current NPU stream, launch, and binding structure. The namespace is fixed to `custom_op`; exported entry point names are free, with `run` used by convention. The build system supplies the CMake project and target architecture and writes `libcustom_op.so` into the sample directory. Do not emit pybind11 code or an OPP / `custom_opp` install project; the evaluator loads that `.so` with `torch.ops.load_library`.

The wrapper should remain small:

```python
import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return torch.ops.custom_op.run(a, b)
```

For parameterized tasks, `ModelNew.__init__` may construct layers such as `nn.Linear` as parameter containers. Pass their weights and biases to the custom operator. Calling those layers to perform the computation is prohibited.

## Candidate rules and checks

All tensor computation belongs in the Ascend C kernel. The Python wrapper may handle allocation, supported data movement/layout operations, and scalar shape arithmetic. The static checker rejects recognizable forms of:

- PyTorch compute functions, tensor compute methods, tensor arithmetic/comparisons, and calls to constructed neural-network layers.
- `torch_npu` or vendor operator shortcuts, CPU/NumPy fallbacks, and host-side ATen computation in the `.asc` source.
- `try`/`except`, `pass`, dynamic Python execution/imports, and known result-caching patterns in the wrapper.
- Timing event patches, stream manipulation, threading, subprocess execution, and several native host side effects.

Candidates must not mutate their inputs, and their output must depend on the current input values. Correctness trials check top-level tensor inputs for candidate mutation after reference execution. Timing uses fresh inputs where practical and follows with an additional output check.

The [checker implementation](https://github.com/wuzhenqing/AscendKernelBench/blob/main/src/checker.py) is the exact source of accepted/rejected syntax. Its pattern and AST checks are heuristics and may reject unfamiliar legitimate forms. Passing these checks does not prove that a candidate is safe or honest. Subprocess separation and timeout handling are not a security sandbox.

## Add and review a task

1. Add a self-contained file to the appropriate `KernelBench/level{L}/` directory, using an unused numeric prefix and a descriptive name.
2. Confirm that discovery loads it without executing the model:

   ```bash
   python -c 'from src.dataset import load_task; print(load_task("level1/101_Example").task_id)'
   ```

   Replace the example ID with your new task ID. Run it from the checkout root; it does not require an NPU.

3. Review initialization, input generation, shapes, output structure, and optional comparison overrides.
4. On a configured Ascend machine, evaluate candidate correctness and timing using the [evaluation workflow](guide/evaluation.md).
5. Preserve the task source, configuration, hardware/software environment, and results when publishing measurements.

Changing an existing task's shapes, precision requirements, or reference semantics changes what is measured. Record the repository revision when comparing results across runs.
