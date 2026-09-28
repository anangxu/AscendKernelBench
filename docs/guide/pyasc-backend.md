# pyasc backend

AscendKernelBench evaluates two authoring languages behind one protocol. One
run uses one backend; Ascend C stays the default so every existing run, CLI,
and result keeps its meaning.

| | Ascend C (default) | pyasc |
| --- | --- | --- |
| Kernel artifact | `custom_op.asc` | `kernel.py` |
| Wrapper artifact | `model_new.py` | `model_new.py` |
| Build | CMake + `find_package(ASC)` into a process-local library | pyasc JIT: trace, compile, launch on the first call |
| Selected by | `--backend ascendc`, or an absent `backend` field | `--backend pyasc` |

A run directory records its backend in `generation_config.yaml`. A run written
before the field existed, or copied without it, is read as Ascend C. An unknown
value is a hard error rather than a silent default.

## Install

pyasc is an official CANN package distributed on PyPI as `pyasc`, and the
import name is `asc`. Install the wheel that matches the host Python and the
CANN pairing; for the validated combination (CANN 9.1.0, Python 3.12, aarch64):

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python -m pip install pyasc==1.1.1
```

Generation and offline analysis never import `asc`, so a machine without an
NPU and without pyasc can still build prompts, parse responses, and score
results.

## Run

```bash
python scripts/generate.py \
  --task level1/19_ReLU \
  --model <model> \
  --backend pyasc \
  --hardware ascend910b2 \
  --run-name relu_pyasc

python scripts/evaluate.py relu_pyasc
python scripts/analyze.py relu_pyasc
```

`evaluate.py` takes no backend flag on purpose: it reads the run's recorded
backend, so a run cannot be evaluated as the wrong language by accident.

## What the model must produce

`kernel.py` holds one or more device kernels decorated `@asc.jit` plus a plain
Python host launcher. `model_new.py` defines `class ModelNew` with the
reference `Model`'s `__init__` and `forward` signatures and calls that
launcher. The prompt contract lives in `src/prompt.py` and the device-verified
examples in `src/prompts/examples/pyasc/` (element-wise add and row-wise sum).

Kernel semantics that the prompt, the checks, and the evaluator all rely on:

* Only `kernel_fn[core_num](args...)` compiles and launches. A bare
  `kernel_fn(args...)` call neither compiles nor launches, so a launcher that
  never subscripts is rejected and can never be scored as correct.
* Launch options are set with the subscript form only; `@asc.jit(core_num=...)`
  and `core_num=`/`stream=` call keywords are not supported by pyasc 1.1.1.
* Tensor arguments are annotated `asc.GlobalAddress` and devices see no shapes,
  so sizes travel as scalars or `asc.ConstExpr[int]` compile-time constants.
* Small `data_copy` write-backs are unreliable on the target tested here. On an
  Ascend 910B4 with pyasc 1.1.1, a GM write-back of 1 or 4 fp32 (4 / 16 bytes)
  did not take effect at all and left the destination at its previous value,
  while 8 and 16 fp32 (32 / 64 bytes) landed exactly, in both the TQue and the
  explicit `LocalTensor` styles. A tail that is not a whole 32-byte block is
  unsafe for the same reason. Compute an aligned tiling plan on the host, pad
  when the shape cannot be aligned, and keep every final write-back at 32 bytes
  or more. The scope of this measurement is that device, that pyasc version,
  and that call style; it is not a documented property of `data_copy`.
* `torch.bfloat16` and `torch.bool` are rejected by pyasc's dtype table.

## Verified device semantics

These were measured on an Ascend 910B4 (CANN 9.1.0, pyasc 1.1.1, torch_npu
2.10.0), not read off the docstrings. Versions, inputs, observed outputs, and
the boundary of each claim are in
[`docs/reference/pyasc-device-evidence.md`](../reference/pyasc-device-evidence.md),
and the runnable probes are in `experiments/pyasc_device_probes/`:

* `asc.whole_reduce_sum(dst, src, mask=..., repeat_time=..., dst_rep_stride=...,
  src_blk_stride=..., src_rep_stride=...)` reduces **per repeat**, so
  `repeat_time=R` yields `R` outputs rather than one scalar. In the contiguous
  mask mode `mask` is an **element count**, not a bit mask. `src_rep_stride`
  counts 32-byte data blocks of the source, while `dst_rep_stride` counts
  destination elements. A row-wise fp32 sum (64 columns per repeat) matched
  `torch.sum(dim=-1)` with `max_abs_diff = 0` on 4 and 8 cores; keep a real
  tolerance anyway, because pairwise tree accumulation need not agree with
  torch bit for bit at other sizes.
* `asc.LocalTensor(dtype, pos, addr, length)` only emits an address, it does
  not reserve UB. A static-tensor kernel has to reserve it through
  `asc.LocalMemAllocator().alloc(pos, dtype, tile_size)`, and that style must
  not be mixed with TPipe/TQue.
* In the explicit-address style, a GM-to-UB copy followed by a UB-to-GM copy
  needs an event that matches the consumer's pipe: `asc.HardEvent.MTE2_MTE3`.
  Holding the tile length and the addresses fixed and changing only that event
  from `MTE2_V` to `MTE2_MTE3` turned a write-back of uninitialised UB into an
  exact one, so do not synchronise a copy against the vector pipe. The TQue
  style handles this internally and is the recommended idiom for generated
  kernels.
* Match the event to the consumer of each data path; a queue does not supply
  arbitrary dependencies. Measured on the target device: a copy that goes
  straight from the inbound buffer to GM carries no dependency through one
  VECIN queue, and at a 64 KiB write-back the idiom alone was wrong in 5 of 5
  launches while the same kernel plus an `asc.HardEvent.MTE2_MTE3` pair from
  `pipe.alloc_event_id(...)` was exact in 10 of 10. The generated CCE source,
  dumped with `PYASC_DUMP_PATH`, contained no `SetFlag`/`WaitFlag` for the bare
  copy and did contain the pair for the explicit version. Paths that run a
  vector operation between the copies were not affected in these tests, but
  that is not a licence to insert an unrelated vector op: a vector-produced
  result must be waited on with the vector event (`MTE2_V` in, `V_MTE3` out),
  and a loop that reuses one buffer must also keep the previous reader finished
  before overwriting it. Scope, counts and the open questions are in
  `docs/reference/pyasc-device-evidence.md`.
* The bundled reduction example accepts only shapes it addresses exactly: cols
  must be whole 32-byte blocks (8..64 fp32) and rows must split into whole
  8-row blocks per core. It raises ValueError otherwise, because a column count
  that is not a whole block rounds `src_rep_stride` down and silently reduces
  the wrong elements (measured: cols=12 disagrees with torch.sum by order 4;
  4.41 unseeded historically, 3.9 with the probe's fixed seed). Representative shapes across the accepted range were checked
  against `torch.sum(dim=-1)` and matched exactly; the range is not exhaustively
  enumerated.

## Evaluation semantics

* **Import is not compilation.** Loading `kernel.py` only makes the module
  importable; `compiled` stays false until the first subscripted call
  succeeds. A first-call failure is reported with `failure_stage=jit`.
* **`first_call_seconds`** is the compound trace, compile, and launch cost of
  that first call, and is never presented as a pure compile time. The models
  are rebuilt afterwards so preparation cannot influence the measured trials.
* **Cache isolation.** `PYASC_CACHE_DIR` points inside the sample's `build/`
  directory and is exported before `asc` is imported, because pyasc resolves
  the cache location at import time. Samples therefore cannot share compiled
  kernels even when they reuse kernel function names.
* **Timing.** pyasc launches on its own runtime stream rather than torch's, and
  its launcher synchronizes at the end of every call. The evaluator therefore
  cross-checks the event-measured mean against wall-clock time and records
  `stream_mode`, `timing_event_ms`, `timing_wall_ms`, `timing_event_wall_ratio`
  and `timing_valid`. On the validated host the ratio was 0.30 to 0.35: the
  event captures device execution, while wall clock additionally pays pyasc's
  per-call host overhead (kernel re-registration plus a synchronize) that never
  reaches the device timeline. The cross-check exists to catch the failure mode
  where the events measure almost nothing (ratio near zero); such a sample is
  reported with `timing_valid=false`, no runtime, and no speedup.
* **Per-call overhead.** pyasc re-registers the kernel and synchronizes on
  every launch, so small operators look far slower than eager `torch_npu`.
  Compare pyasc speedups against other pyasc samples, and read the Ascend C
  numbers separately.

## Evaluation runs

The first end-to-end pyasc run, its configuration, its results, and the
limitations of those numbers are in
[`docs/reference/pyasc-evaluation-report.md`](../reference/pyasc-evaluation-report.md).

## Static checks

`check_sample_sources(backend, kernel_source, wrapper_source)` dispatches the
anti-cheat rules. For pyasc it checks three layers: the `model_new.py` wrapper,
every non-`@asc.jit` host function in `kernel.py` (launchers and their
helpers), and the `@asc.jit` device bodies. It also verifies the call graph, so
a wrapper that calls an aliased launcher is accepted while a launcher that
never launches is rejected. Host code may allocate, compute shapes, move data,
and launch; tensor compute outside the device kernels is rejected, including
when it is hidden inside a host helper.

## Checking one candidate without an NPU

pyasc ships a simulator. It runs the same `kernel.py` on CPU tensors, so a
candidate can be checked for tracing, compilation, and numeric correctness on
a machine with no device:

```bash
export ASCEND_KERNEL_BENCH_PYASC_BACKEND=model
export ASCEND_KERNEL_BENCH_PYASC_SOC=Ascend910B4
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.1.0/tools/simulator/Ascend910B4/lib:$LD_LIBRARY_PATH
```

The simulator libraries live in
`$ASCEND_HOME_PATH/tools/simulator/<soc>/lib`, and the directory exists for
every SOC pyasc supports. Verified on the validated host: the hand-written Add
kernel returned correct results for lengths 16384, 1024, 1031, 64, and 1 with
`Backend.Model`, no NPU involved. Two caveats: the simulator process printed a
`double free or corruption` message during teardown after every check had
already passed, so treat that as a simulator exit artifact; and this path
checks one candidate only. It is not a substitute for evaluation, whose
reference timing has to be measured live on the device.

## Limitations and how they are reported

Two kinds of limitation are kept apart, because they mean different things for
coverage:

* **Backend-wide.** pyasc's dtype table has no factory for `bfloat16` or
  `bool`, so no sample can marshal them at all. The evaluator checks the
  task's declared dtypes, including the raw inputs before the harness casts a
  floating tensor to the configured precision, and reports
  `failure_stage=unsupported_dtype` with `limitation_scope=backend` and the
  offending `unsupported_dtypes`. Unsupported inputs are rejected **before**
  any cast: the raw inputs are inspected first and nothing is converted while
  that verdict is unresolved, so converting them cannot mask the gap or push a
  refusal into `failure_stage=execution`. Converting them would also score a
  different task than the reference model defines. A backstop re-checks the
  tensors the kernel is actually handed. Supported floating types are still
  cast to the configured precision, exactly as before, so this is not a
  blanket "no conversion" rule.
* **Code generation.** `UnsupportedSyntaxError` means the traced body used
  something this codegen rejects. That is attributable to the generated
  source, and it may equally be a DSL syntax mistake in that source, so it is
  reported as `failure_stage=codegen_error` with `limitation_scope=code`
  rather than as an operator-capability gap.
* **Operator-level.** `failure_stage=unsupported_operator` with
  `limitation_scope=operator` is reserved for a limitation that is positively
  tied to one interface or operator, such as a dtype or shape combination an
  API does not implement. No exception class is mapped to it automatically;
  its counter stays zero until such a case is identified by hand.

None of these is a pass. All stay in the `fast_0` denominator, and
`summarize_eval_results` carries `unsupported_dtype`,
`unsupported_dtype_backend_wide`, and `unsupported_operator` counts so the
report shows the gap instead of letting a success rate hide it. The report
table prints both rows even when they are zero.
