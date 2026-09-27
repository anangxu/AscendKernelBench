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
launcher. The prompt contract and a verified one-shot example live in
`src/prompt.py` and `src/prompts/examples/pyasc/`.

Kernel semantics that the prompt, the checks, and the evaluator all rely on:

* Only `kernel_fn[core_num](args...)` compiles and launches. A bare
  `kernel_fn(args...)` call neither compiles nor launches, so a launcher that
  never subscripts is rejected and can never be scored as correct.
* Launch options are set with the subscript form only; `@asc.jit(core_num=...)`
  and `core_num=`/`stream=` call keywords are not supported by pyasc 1.1.1.
* Tensor arguments are annotated `asc.GlobalAddress` and devices see no shapes,
  so sizes travel as scalars or `asc.ConstExpr[int]` compile-time constants.
* `data_copy` counts must be 32-byte aligned; a 4-byte tail silently computes
  wrong values. Compute an aligned tiling plan on the host, and pad when the
  shape cannot be aligned.
* `torch.bfloat16` and `torch.bool` are rejected by pyasc's dtype table.

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
