# Troubleshooting

Start with the stage that failed: generation, static checking, compilation, module loading, correctness, or timing. A successful generation or documentation build does not validate an Ascend kernel.

## Working on macOS

You can generate candidates through a remote endpoint, inspect task and prompt source, run checks that do not import the NPU runtime, analyze existing JSON results, and build this documentation on macOS. Kernel compilation and evaluation require a Linux Ascend machine with the appropriate CANN, PyTorch, and `torch_npu` environment.

The CPU reference fallback also requires a working NPU for the candidate; it is not a CPU-only evaluation mode.

## Find the diagnostic files

For a sample such as `runs/my_run/level1/19_ReLU/sample_0/`, inspect:

| File | What it tells you |
| --- | --- |
| `prompt.txt` | The benchmark prompt used for generation. |
| `response_raw.txt` | Raw text of a successful response, when nonempty. |
| `custom_op.asc`, `model_new.py` | The saved candidate being checked and built. |
| `build/configure.log` | CMake configuration output after the command finishes. |
| `build/build.log` | Compiler and linker output after the command finishes. |
| `eval_result.json` | Per-sample status and error metadata. |

The aggregate file is `runs/my_run/eval_results.json`. Failed generation attempts are reported on the console and have no corresponding saved failure record. Build logs may be absent if the command could not start or timed out.

Read the metadata rather than relying on the CLI's `COMPILE-FAIL` / `WRONG` label or the `compiled` flag alone. Static rejection and several worker failures also use `compiled: false`. Conversely, an exception escaping the worker, including an early dependency import error, may be reported with `compiled: true`.

## Dependencies and repository paths

### `No module named src`

User scripts and the isolated eval worker add the checkout root to `sys.path` (`scripts/_bootstrap.py`). Run commands from a complete checkout so `scripts/_eval_worker.py` exists, and use the same Python interpreter that has the third-party dependencies installed.

### Missing `torch` or `torch_npu`

`requirements.txt` installs PyTorch and `torch_npu` on Linux, but never the Ascend runtime stack itself. On a Linux host with CANN 9.1.0, recreate the experiment environment with `conda create -n AscendKernelBench python=3.12 -y`, then `conda activate AscendKernelBench`, `pip install -r requirements.txt`, and source `set_env.sh` (see [getting started](getting-started.md)). Use that same interpreter for evaluation; the CMake build receives it as `Python3_EXECUTABLE` and discovers PyTorch / `torch_npu` through it.

Do not try to resolve a missing `torch_npu` import on macOS by substituting a CPU evaluation path. Move the saved sources to the prepared Ascend host instead. Some tasks also import optional libraries; inspect the task's imports when a reference module cannot load.

### `Task not found`, `no tasks found`, or missing configuration

Task IDs use the form `level1/19_ReLU`, relative to `KernelBench/`. Check the exact filename and level. The checkout must contain `KernelBench/`, `configs/`, and `build_template/`.

Paths are inferred from `src/_paths.py` (the parent of `src/`). That also chooses the `runs/` and `results/` directories. Changing the current directory alone does not redirect them.

## LLM connection and response errors

See [Connect an LLM service](../deploy_llm_service.md) for the endpoint contract.

| Symptom | What to check |
| --- | --- |
| Authentication error | `OPENAI_API_KEY` is available in the same shell and is accepted by that endpoint. Do not print or share the key. |
| Connection, DNS, or TLS error | The generation host can reach the API base URL, including any required network or proxy configuration. |
| HTTP 404 or unknown model | The base URL is an API root rather than a full completion URL, and `--model` matches a served identifier. |
| Unsupported parameter or response format | The endpoint accepts Chat Completions with `temperature` and `max_tokens`. Structured-output failure triggers a plain-request fallback. |
| `no JSON fields and no fenced code blocks` | The response must carry either a JSON object with the two deliverable fields or two fenced source blocks. A truncated or prose-only answer has neither. |
| Missing `__vector__`, `TORCH_LIBRARY` / `TORCH_LIBRARY_IMPL`, `class ModelNew`, or `torch.ops.custom_op` | The model returned incomplete or incorrectly formatted file contents. Inspect the response where available and the console error. |
| Truncated source | Review the service's output limit and `generation.max_tokens` in the selected YAML configuration. |

`generation failed validation` can wrap service or transport errors as well as content errors. Read the nested message. `response_raw.txt` is saved only after generation succeeds, so a failed attempt may have no raw response artifact.

Batch generation exits successfully if at least one requested sample was saved. Compare the printed `saved/total` count with the expected number of tasks times samples. Failed generations are absent from later evaluation aggregates, which can reduce the reported sample count. Report that loss when interpreting an experiment.

## No samples, stale files, or unexpected run contents

`evaluate.py` discovers `level*/<task>/sample_*` directories only when both of the run backend's artifacts exist: `custom_op.asc` and `model_new.py` for Ascend C, or `kernel.py` and `model_new.py` for pyasc. An empty run directory or incomplete sample produces no evaluable sample. Sample directory names must end in an integer, such as `sample_0`.

Run names are reusable, but the scripts do not provide automatic resume or experiment isolation. Regeneration overwrites matching sample files and the run configuration. Old results or extra sample directories can remain. Use a fresh run name for each experiment and preserve the original run when testing changes.

When moving between machines or incompatible Python environments, carry the candidate sources and generation metadata, and rebuild in a fresh sample directory on the target host. Existing `build/` contents and `libcustom_op.so` / `custom_op*.so` files belong to the environment that produced them. Do not install those libraries into site-packages or the CANN OPP vendors path; the evaluator loads the sample-local `.so` with `torch.ops.load_library`.

## Static-check rejection

`metadata.static_check_error` lists the violations, and the compiler is not invoked. The checker requires a real Ascend C kernel and a Python wrapper that calls `torch.ops.custom_op`. It rejects, among other things:

- PyTorch, tensor-method, or vendor prebuilt operator computation used in place of the custom kernel.
- CPU / NumPy fallback, exception-based fallback, and empty `pass` implementations.
- Timing manipulation, custom streams, result caching patterns, and host-side process or networking operations.

`ModelNew` may construct matching `nn` layers to hold initialized parameters, but it must pass their weights to the custom operator rather than call the layers for computation. Follow the [candidate contract](../task_authoring.md) when correcting a rejection.

The checks are heuristic and are not a security sandbox. If a valid wrapper is rejected, inspect the specific diagnostic and the checker implementation; do not interpret a static pass as proof of correctness.

## Compilation and module loading

### CANN or `find_package(ASC)` cannot be found

The build helper attempts to source `/usr/local/Ascend/cann-9.1.0/set_env.sh`. Override the location if your installation differs:

```bash
export CANN_SET_ENV='/absolute/path/to/cann/set_env.sh'
```

If that file does not exist, the helper silently uses the current environment. Confirm the path or start from a correctly initialized CANN shell. The helper caches the sourced environment per Python process, so restart after changing toolchain settings. Its environment setup applies to build subprocesses; the evaluation Python process must already be able to import and use the Ascend runtime.

The ACLNN CMake project requires CMake, the ASC toolchain, a suitable GCC toolchain, Python development files, PyTorch, and `torch_npu`. `Failed to locate libgcc.a` or `Failed to derive GCC toolchain root` points to compiler discovery; a missing header or library is usually clearer in `build/configure.log` or `build/build.log` than in the truncated result message.

### `configure failed`, `build failed`, or no built module

Read the matching build log and confirm that the selected hardware profile matches the device and compiler. The profile supplies `CMAKE_ASC_ARCHITECTURES`. The generated file must use the APIs expected by the project's build template and export `TORCH_LIBRARY(custom_op, ...)` plus `TORCH_LIBRARY_IMPL(custom_op, PrivateUse1, ...)`.

`Built shared library not found` means the build command returned successfully but `libcustom_op.so` was not found in the sample directory. Check the output location and preserve the build log for investigation.

### Build times: split layout versus legacy layout

New-style samples (with the `ASCEND_HOST_SECTION` marker) build in about
15-20 s on an Ascend 910B2 with CANN 9.1.0: roughly 3 s of CMake configure,
7 s for the device kernels, 1 s for the generated launch stubs, 4 s for the
torch host glue, and 2 s for the link, with the three translation units
compiled in parallel. Re-evaluating an unchanged sample reuses its build
directory and takes under a second.

The speedup comes from keeping torch headers out of the ASC front-end. A
single-unit `.asc` is parsed three times by `bisheng` (cube and vector device
passes plus the host pass), and `torch/extension.h` alone expands to roughly
5000 headers — about 110 s of parsing per sample. The split layout compiles
the device sections without torch headers, and compiles the host glue as
plain C++ against a shared precompiled header that covers
`torch/library.h`, `ATen/ATen.h`, and the `torch_npu` stream header.

The precompiled header lives in a per-environment cache keyed by compiler,
torch, torch_npu, and Python versions plus the exact flag set
(`ASCEND_KERNEL_BENCH_CACHE_DIR` overrides the location; the default follows
`XDG_CACHE_HOME`, i.e. `~/.cache/ascend-kernel-bench/pch/`). First use emits
it in about 20 s; every later sample reuses it. If emission or consumption
fails, the build falls back to a plain C++ host compile automatically.
`ASCEND_KERNEL_BENCH_DISABLE_PCH=1` forces the fallback.

Legacy samples without the marker still build as one ASC translation unit
(about 60-120 s depending on page-cache warmth), so older runs remain
re-evaluable. Regenerating a task with the current prompt produces the
split layout.

Notes on alternatives that were measured and rejected:

- A pybind11 registration TU (as in the Ascend devkit `00_pytorch/pybind`
  example) still includes `torch/extension.h` and measures the same ~120 s
  compile; the registration mechanism is not the cost driver.
- A precompiled header for the whole single-unit `.asc` is impossible: the
  three front-end passes target two different triples
  (`hiipu64-hisilicon-cce` and `aarch64-unknown-linux-gnu`), and one PCH
  cannot serve both.
- `ASCEND_KERNEL_BENCH_ENABLE_CCACHE=1` speeds up re-evaluation of unchanged
  samples only, because new sources cannot hit an existing cache entry.

Builds run inside each sample's own worker process, so samples cannot share a
compilation or configure step; the PCH cache is the shared artifact.

### `module load failed` or `candidate model init failed`

Compilation may have succeeded while the Python wrapper, reference task, or compiled extension failed to import. Read `metadata.runtime_error` for missing Python dependencies, undefined symbols, a module-name mismatch, or a missing `ModelNew` class. Constructor failures may also indicate that `ModelNew` does not match the reference's `get_init_inputs()` arguments.

Check runtime libraries in the evaluation process, the Python / PyTorch environment used to compile the extension, and the wrapper's exported function names. Use a fresh build when changing environments.

## Runtime failures, correctness, and timing

| Metadata or behavior | Interpretation and next step |
| --- | --- |
| `candidate runtime error` | The candidate failed during a seeded correctness trial. Inspect its launch arguments, shapes, data types, and device code. |
| `candidate mutated its inputs` | Inputs changed during candidate execution. Return results in separate output tensors. |
| `correctness_error` | Inspect the reported trial, output structure, difference, and tolerances. Match reference behavior before attempting performance optimization. |
| `reference: cpu` | The reference failed on NPU and fell back to CPU in float32. Read `reference_npu_error`; the candidate still ran on NPU. No NPU reference speedup is available. |
| `timing failed` | Correctness may remain true while timing is missing or incomplete. Check `runtime_error` before treating the result as a performance measurement. |
| `post-timing ... re-check failed` | The result failed a fresh-input check after timing. Investigate cached outputs, state changes, and numerical or memory errors. |
| `excessive_speedup: true` | The measured speedup exceeded the configured review threshold. Inspect the implementation and timing conditions before using the number. |

Missing reference timing is expected with a CPU reference. Excessive-speedup results remain eligible for the correctness rate (`fast_0`) but are excluded from positive speedup thresholds and the geometric mean.

### Timeouts and out-of-memory errors

Each CMake configure and build command has its own `build_timeout` budget. The parent worker budget is `eval_timeout + 2 * build_timeout`: 1,500 seconds with the default configuration. On that timeout, the parent terminates the worker process group. `worker produced no result.json`, a worker exit code, or invalid worker JSON indicates that the subprocess did not complete the expected result protocol.

Before increasing a timeout, distinguish a slow build from a kernel hang. Inspect build logs, memory demand, tiling and tail handling, UB usage, and synchronization. Some tasks allocate large tensors, and the evaluation protocol also creates input copies and timing buffers. Changing trial counts does not reduce a task's tensor dimensions.

Keep inference and evaluation on separate physical NPUs, and confirm that the benchmark card is otherwise idle. The scripts do not reserve devices or detect competing jobs. The `ascend950pr` profile is reserved; validate its specifications, toolchain mapping, and kernel behavior before relying on it.

### The evaluator exited with status zero, but samples failed

The batch evaluator writes per-sample results and completes even when candidates are incorrect. Its process exit status is not an all-samples-passed signal. Read `eval_results.json` and the summary printed at the end of `scripts/evaluate.py`. A correct result can still contain a timing error; inspect its metadata for performance claims.

## Reporting a reproducible issue

Include the repository commit, task ID, relevant command with credentials removed, hardware profile, and actual Python / CANN / PyTorch / `torch_npu` versions. For a kernel failure, include the candidate source, `eval_result.json`, and available build logs. For generation failures, include the requested model identifier and redacted service error. State whether the problem was observed on an Ascend device or during host-only work.
