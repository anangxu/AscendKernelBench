# Evaluation protocol

Evaluation builds a generated Ascend C extension, checks it against the reference task, and measures NPU latency. It requires a configured Ascend machine with CANN, PyTorch, and `torch_npu`. macOS can prepare samples and inspect existing results, but it cannot execute this protocol on an Ascend NPU.

This page describes the current implementation. No NPU evaluation was performed as part of the documentation update.

## Run evaluation

After generation has populated `runs/my_run/`, run:

```bash
python scripts/evaluate.py my_run
python scripts/evaluate.py my_run 1
```

The first command evaluates every complete sample in the run. The second evaluates only `level1/`. Hardware comes from the run's `generation_config.yaml` when present, otherwise from `configs/eval_default.yaml`. The worker always uses `npu:0`; pick a physical card with `ASCEND_RT_VISIBLE_DEVICES`. Evaluation always compiles, checks correctness, and measures latency.

The default protocol values are:

| Setting | Default | Meaning |
| --- | --- | --- |
| `hardware` | `ascend910b2` | Hardware profile unless the run recorded another name. |
| `seed` | `42` | Base seed for initialization, correctness seed generation, and timing. |
| `precision` | `fp32` | Floating-point input/model dtype. |
| `num_correct_trials` | `5` | All initial correctness trials must pass. |
| `num_warmup` | `10` | Warmup calls per timed model (SOL-ExecBench-style isolation). |
| `num_perf_trials` | `100` | Retained NPU-event measurements per timed model. |
| `excessive_speedup` | `10.0` | Speedups strictly above this value are flagged for review. |
| `build_timeout` | `600` seconds | Separate budget for CMake configure and CMake build. |
| `eval_timeout` | `300` seconds | Additional time included in the overall worker budget. |

The batch evaluator visits complete sample directories sequentially and re-evaluates them; it has no resume/skip flag. An optional level argument limits the visit to that level. Re-evaluation overwrites those `eval_result.json` files and rebuilds aggregate files from every stored result in the run. Preserve a copy of a run before measuring it on another device or with different settings.

## Public evaluation entry points

| Surface | Function | Use |
| --- | --- | --- |
| CLI | `scripts/evaluate.py <run> [level]` | Evaluate complete samples in `runs/<name>/`, optionally one level. |
| Library | `evaluate_run(run, level=None)` | Same batch as the CLI, including aggregate JSON writes. |

`scripts/analyze.py <run>` only reads `eval_results.json`. `scripts/_eval_worker.py`, `worker_main`, `eval_sample`, and `eval_sample_on_device` are internal.

## Worker and build lifecycle

For each sample, the host checks that the run backend's kernel artifact (`custom_op.asc` or `kernel.py`) and `model_new.py` exist and applies the [candidate static checks](../task_authoring.md#candidate-rules-and-checks). A static violation produces a failed result without launching the worker.

An accepted sample runs in a fresh subprocess using the same Python interpreter as the host. The host starts `scripts/_eval_worker.py`, which adds the checkout root to `sys.path` and calls `eval_device.eval_sample_on_device`. It builds `libcustom_op.so` with the fixed `build_template/CMakeLists.txt`, loads that library with `torch.ops.load_library` **in the worker process** (not via pybind import, `sys.path`, or a global install), then loads the task and `ModelNew`. Results travel through a temporary JSON file rather than standard output.

The host allows `eval_timeout + 2 * build_timeout` seconds for the whole worker: 1,500 seconds with the defaults. There is no separate 300-second timer started after compilation. On an overall timeout, the host attempts to kill the worker's entire process group and records a failure.

The build writes `custom_op.asc`, copies the fixed CMake template into the sample directory, and passes the profile's `cmake_arch` as `CMAKE_ASC_ARCHITECTURES`. Both files are written only when their content changes, and configure is skipped when the existing CMake cache already matches the request, so re-evaluating an unchanged sample is an incremental no-op build. The template is an `add_library(... SHARED)` project that writes `libcustom_op.so` next to the source, bakes RPATH, and sets `CMAKE_SKIP_INSTALL_RULES`. CMake locates PyTorch and `torch_npu` through the active Python interpreter. Configure and build logs are stored under `sample_*/build/`. Set `ASCEND_KERNEL_BENCH_ENABLE_CCACHE=1` to pass `-DENABLE_CCACHE=ON` when `ccache` is available; the launcher covers the ASC compiler, which dominates the build time.

Sources with an `ASCEND_HOST_SECTION` marker build in split mode: the device section compiles as `custom_op_kernel.asc` without torch headers, a generated `custom_op_launch.asc` holds the `extern "C"` launch stubs, and the host section compiles as plain C++17 against a shared precompiled header (see [troubleshooting](troubleshooting.md#build-times-split-layout-versus-legacy-layout) for measured numbers and the PCH cache). `eval_result.json` records `build_mode` (`split` or `legacy`) and `build_seconds` in its metadata.

The build environment helper sources `CANN_SET_ENV`, defaulting to `/usr/local/Ascend/cann-9.1.0/set_env.sh`, if that file exists; otherwise it uses the current environment. Prepare the runtime environment before launching evaluation, because the worker imports `torch_npu` before the build helper runs.

Subprocess separation prevents ordinary worker failures from directly sharing interpreter state with the host. It does not restrict the worker's filesystem, network, or operating-system privileges. Static checks and timeouts are not a security sandbox.

## Correctness

The worker seeds PyTorch and the NPU RNG, calls `get_init_inputs()`, and re-seeds before separately constructing `ModelNew` and `Model`. Both models enter `eval()` mode. Parameters are reproduced through identical seeded initialization, not copied from the reference.

It then derives `num_correct_trials` seeds from the base seed. For every trial:

1. Seed the generators and call `get_inputs()`.
2. Move top-level tensor arguments to the NPU, casting floating tensors to the configured precision while preserving integer/Boolean dtypes.
3. Reset the seed after input generation and synchronize the device.
4. Run the reference and synchronize.
5. Snapshot the current tensor inputs, run the candidate, and synchronize.
6. Reject candidate input mutation and compare the outputs.

A candidate exception or detected input mutation fails the sample immediately. Output mismatches are recorded while subsequent trials continue. Correctness requires every initial trial to pass.

### Hidden value distributions

Those seeded trials redraw `get_inputs()`. On this corpus that draw is usually uniform on `[0, 1)`, so extra seeds do not change the sign or the magnitude band. KernelBench-Verified showed that frontier models exploit exactly that narrow draw (a ReLU implemented as the identity reported a 374× speedup until inputs were negated).

After the seeded trials pass, the worker draws one more input set from the base seed and checks four value-only transforms of it. Shapes, integer and Boolean tensors, Python scalars, and exact zeros stay as they are. Nonzero floating and complex entries are scaled by:

| Name | Scale | What a failure usually means |
| --- | --- | --- |
| `d1` | `1` | The original draw. |
| `d2` | `3` | Large-magnitude overflow or an unstable reduction. |
| `d3` | `0.01` | Underflow, or a kernel that only works away from zero. |
| `d4` | `-1` | A sign shortcut, including an identity posing as ReLU. |

Any mismatch, candidate exception, or input mutation fails the sample. Timing is skipped, `metadata.hidden_failed` names the first failing transform, and `metadata.hidden_passed` lists the ones that matched. The factors are not written into the generation prompt, and there is no flag to skip the gate. Timing and the post-timing re-check still use unscaled `get_inputs()` draws, so speedup stays comparable to the eager reference on the distribution the model was shown.

### Comparison and tolerances

Default tolerances are:

| Precision | Absolute tolerance | Relative tolerance |
| --- | --- | --- |
| `fp32` | `1e-4` | `1e-4` |
| `fp16` | `1e-2` | `1e-2` |
| `bf16` | `1e-2` | `1e-2` |

Tensor shapes must match. Floating-point/complex tensors use `torch.allclose`; pairs of integer/Boolean tensors use `torch.equal`. The default matcher handles tuple/list outputs recursively. A nonempty task `TOLERANCE` overrides configured tolerances, and `custom_check(ref, out)` replaces the default comparison entirely. See [task authoring](../task_authoring.md#precision-and-output-checks) for contract limits.

### CPU reference fallback

The preferred reference is eager PyTorch with `torch_npu` on the same NPU and in the same precision as the candidate. If reference construction fails, or a reference forward/synchronization raises during any initial correctness trial, the worker switches to a CPU reference for that and subsequent trials. It does not rerun earlier successful trials.

CPU fallback uses the task's CPU model and float32 floating-point inputs; the candidate still runs on the NPU. Candidate outputs are transferred to CPU for comparison. Metadata records `reference: "cpu"` and the NPU exception in `reference_npu_error`.

This fallback catches any exception in those reference stages, including causes other than unsupported operators. Inspect the error before interpreting a result as expanded operator coverage. A later reference failure during timing or the post-timing check does not activate this fallback.

A correct CPU-reference sample contributes to correctness metrics. It can have a measured candidate latency, but has no NPU reference latency or speedup.

## Timing

Only samples that pass the seeded trials and all four hidden transforms enter timing. Candidate and NPU reference are measured separately in the same worker; the candidate is measured first. Evaluation always measures the reference live and does not read archived baseline JSON files.

### Input refresh

The evaluator measures the total size of top-level input tensors after transfer/casting:

- At most 256 MiB: draw new inputs before every warmup and measured call. Keep the previous input set alive to discourage pointer reuse. Candidate and reference timing begin with the same RNG seed.
- Above 256 MiB: reuse a fixed input set during timing to avoid repeatedly allocating very large tensors.

The result records this choice in `metadata.timing_fresh_inputs`. Input preparation runs outside the NPU-event measurement window.

### Event sequence

For each timed model, the timer performs ten warmups by default, synchronizing after each, then calls `torch.npu.empty_cache()` to release allocator caches. For every subsequent trial, it:

1. Refreshes inputs when enabled and synchronizes the device.
2. Creates a start/end NPU event pair.
3. Thrashes L2 with a buffer of `max(256 MiB, 2 × profile L2)` (384 MiB on the default 910B2 profile) before recording the start event.
4. Records the start event, invokes the model, and records the end event.
5. Synchronizes and reads elapsed event time in milliseconds.

When both candidate and NPU-reference means are available, the worker also records a roofline SOL bound (`bytes_moved / profile bandwidth`, optionally `max`ed with `FLOPS / peak_tflops` when the task declares a FLOP count) and a SOL-ExecBench-style `sol_score`. The bound is a documented profile estimate, not a NVIDIA SOLAR characterization. See [results and scoring](results.md).

Every scored sample also records a protocol snapshot (`seed`, trial counts, warmup, `l2_clear_size`) and the worker software stack (PyTorch, torch-npu, device name, and CANN version when the environment exposes it).

The first measured trial is discarded. With the defaults, there are 101 measured calls and 100 retained values per model, in addition to warmups. The L2-thrashing operation is outside the event window. The reported numbers are device-event latency for the invoked model, not end-to-end generation/build time or a host wall-clock benchmark.

Statistics contain mean, standard deviation, minimum, maximum, and retained trial count. Numeric statistics are rounded to three significant digits before persistence. The scoring code computes speedup from the persisted means.

### Post-timing check and flags

After the timing attempt, the evaluator draws another input set using `seed + 1` and compares candidate and reference outputs again. A mismatch or exception marks the result incorrect, even when timings have already been collected. This check can expose cached outputs or state drift; it does not guarantee detection of every prohibited behavior. It does not repeat the input-mutation check used by the initial correctness trials.

A timing exception is recorded in `metadata.runtime_error`. If the post-timing check succeeds, the result may remain correct with missing or partially populated timing fields.

When both mean runtimes are available, a speedup strictly above `excessive_speedup` is marked for manual review. The flag does not automatically make the sample incorrect. It excludes the sample from positive `fast_p` thresholds and geometric mean speedup while preserving its contribution to `fast_0` and pass@k. See [results and scoring](results.md).

## Archive a reference baseline

On the Ascend machine:

```bash
python scripts/baseline.py --task level1/19_ReLU --hardware ascend910b2 --device npu:0
```

The archive is `results/baseline/{hardware}/{task_name}.json`. Supported references include timing statistics, `supported_on_npu: true`, and `timing_fresh_inputs`. Reference failures inside the measurement block produce `supported_on_npu: false` with an error string; worker-level failures are printed and may leave no archive.

The baseline script uses the same timing helper and adaptive input refresh policy, but it does not evaluate candidates or provide CPU timing. Archives are useful records and are not the denominator used for evaluation speedups. Archive filenames omit the level, so equal task stems in different levels can collide; preserve such measurements separately.
