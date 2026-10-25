# AscendKernelBench

```text
/======================================================================\
|                                                                      |
|            ___    _____   ______   ______    _   __    ____          |
|           /   |  / ___/  / ____/  / ____/   / | / /   / __ \         |
|          / /| |  \__ \  / /      / __/     /  |/ /   / / / /         |
|         / ___ | ___/ / / /___   / /___    / /|  /   / /_/ /          |
|        /_/  |_|/____/  \____/   \____/   /_/ |_/   /_____/           |
|            __ __   ______    ____     _   __   ______    __          |
|           / //_/  / ____/   / __ \   / | / /  / ____/   / /          |
|          / ,<    / __/     / /_/ /  /  |/ /  / __/     / /           |
|         / /| |  / /___    / _, _/  / /|  /  / /___    / /___         |
|        /_/ |_|  \____/   /_/ |_|  /_/ |_/   \____/   /_____/         |
|               ____    ______    _   __   ______    __  __            |
|              / __ )  / ____/   / | / /  / ____/   / / / /            |
|             / __  | / __/     /  |/ /  / /       / /_/ /             |
|            / /_/ / / /___    / /|  /  / /___    / __  /              |
|           /_____/  \____/   /_/ |_/   \____/   /_/ /_/               |
|                                                                      |
|                          AscendKernelBench                           |
|                                                                      |
|       [ LLM ]=======>{ Ascend C / pyasc }=======>{ Ascend NPU }       |
|                 fast_p   |   pass@k   |   SOL score                  |
|                                                                      |
\======================================================================/
```

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

A reproducible benchmark for **LLM-generated Ascend C and pyasc kernels**
on Huawei Ascend NPUs. A language model reads a PyTorch reference `Model`
and writes a kernel plus `model_new.py`. The harness builds or JIT-compiles
the kernel inside an isolated worker, checks outputs against the reference,
and times eligible candidates against live `torch_npu` eager execution.

[Documentation](docs/index.md) ·
[Getting started](docs/guide/getting-started.md) ·
[Evaluation protocol](docs/guide/evaluation.md) ·
[Results and metrics](docs/guide/results.md)

## Why this benchmark

AscendKernelBench is a KernelBench-style *generate, then evaluate* harness
for Ascend NPUs.

- **270 vendored KernelBench tasks** in four levels: 100 operators, 100
  fused operators, 50 networks or subgraphs, and 20 model tasks. This is
  corpus size, not a claim that every task is supported or device-validated.
- **Process-local operators.** The CMake build compiles `custom_op.asc` into
  `libcustom_op.so` next to the sample. The evaluator loads it inside the
  worker PyTorch process. Nothing is installed into site-packages or the
  global CANN OPP path, so parallel jobs do not lock a shared environment.
- **Kernels are actually checked.** Static anti-cheat on both files, five
  seeded correctness trials, input-mutation rejection, isolated workers,
  timing with L2 flush, and a post-timing fresh-input re-check.
- **Honest scores.** `fast_p` and `pass@k` follow KernelBench semantics.
  An optional roofline **SOL score** follows NVIDIA SOL-ExecBench's
  formula, using the hardware-profile bandwidth (not NVIDIA SOLAR). Each
  scored sample records the protocol snapshot and software stack used to
  produce it.
- **Two authoring backends.** Ascend C emits `custom_op.asc` and
  `model_new.py`; pyasc emits `kernel.py` and `model_new.py`. Each run records
  one authoring backend and one build configuration.

| Authoring backend | Build path | Timing for candidate and live reference |
| --- | --- | --- |
| `ascendc` (default) | `cmake` (default) | NPU events, device scope |
| `ascendc` | `deepjit` | NPU events, device scope |
| `pyasc` | pyasc JIT (`build_backend=cmake` is unused) | Synchronized wall clock, end-to-end scope |

`--build-backend deepjit` applies only to Ascend C; `pyasc` plus `deepjit` is
rejected. See the [DeepJIT guide](docs/guide/deepjit.md) and
[pyasc guide](docs/guide/pyasc-backend.md). pyasc timing includes host dispatch
and launch overhead. Compare latency and speedup only within the same timing
scope; pyasc latency is not device-only latency.

The current workflow supports generation and evaluation scripts. It does
not implement an automatic compile-error repair agent.

## Choose your environment

| Work | Machine without an NPU | Linux Ascend host |
| --- | --- | --- |
| Read tasks and build prompts | Yes | Yes |
| Generate candidates with a remote LLM | Yes | Yes |
| Lint (optional) | Yes | Yes |
| Analyze saved `eval_results.json` | Yes | Yes |
| Compile, check, or time a kernel | No | Required |

**CPU-reference fallback does not enable evaluation on a laptop.**
Candidates always run on an Ascend NPU. CPU fallback changes only the
reference used for correctness; those samples have no NPU speedup and
no SOL score.

## Quick start

Use Python 3.10 or later for the harness. Base dependencies live in
`requirements.txt`; optional backends need their own runtime. The earlier
Ascend host setup used the conda env **AscendKernelBench** (Python 3.12)
with CANN 9.1.0, **PyTorch 2.10.0**, and **torch-npu 2.10.0.post6**:

```bash
git clone https://github.com/wuzhenqing/AscendKernelBench.git
cd AscendKernelBench

# Host without an NPU: generation, analysis, and optional lint
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On a Linux Ascend host with CANN 9.1.0:

```bash
conda create -n AscendKernelBench python=3.12 -y
conda activate AscendKernelBench
source /usr/local/Ascend/cann-9.1.0/set_env.sh
python -m pip install -r requirements.txt
```

These dependencies do **not** include CANN, the NPU driver, pyasc, or a
DeepJIT checkout. Validate each chosen runtime pairing on the target host.
See [getting started](docs/guide/getting-started.md).

Generate (any machine with network access to your LLM):

```bash
export OPENAI_BASE_URL="https://your-provider.example/v1"
export OPENAI_API_KEY="your-api-key"

python scripts/generate.py \
  --task level1/19_ReLU \
  --model your-served-model \
  --hardware ascend910b2 \
  --run-name relu_demo
```

Evaluate and report (Ascend host, exclusive device):

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python scripts/evaluate.py relu_demo
python scripts/analyze.py relu_demo
```

For model-to-model or revision-to-revision comparisons, generate a fixed task
set instead of a single task. `configs/subsets/level1_20.txt` holds 20 level 1
tasks sampled systemically (every fifth task), and `--tasks-file` reads it:

```bash
python scripts/generate.py \
  --tasks-file configs/subsets/level1_20.txt \
  --model your-served-model \
  --hardware ascend910b2 \
  --run-name level1_20
```

## Command-line tools

| Script | Purpose |
| --- | --- |
| `scripts/generate.py` | Sample an OpenAI-compatible model; write sources under `runs/` |
| `scripts/evaluate.py` | Build, check, and time a run (`run` and optional `level`) |
| `scripts/analyze.py` | Print `fast_p`, `pass@k`, geomean speedup, mean SOL |
| `scripts/baseline.py` | Archive live `torch_npu` eager timings (not the eval denominator) |

## How kernels are scored

| Metric | Meaning |
| --- | --- |
| `fast_0` | Correctness rate over **collected** sample results |
| `fast_p` (`p > 0`) | Fraction of collected samples that are correct and strictly faster than `p`× the live NPU reference |
| `pass@k` | Unbiased chance of at least one correct sample in `k` draws |
| Geometric-mean speedup | Over correct, unflagged NPU-reference samples only |
| SOL score | `(T_b − T_sol) / ((T_k − T_sol) + (T_b − T_sol))`. `0.5` matches the baseline; `1.0` would match the roofline bound. Bound = estimated traffic / profile bandwidth |

Suspicious speedups (`> 10×` by default) stay in `fast_0` and `pass@k`
but are excluded from positive `fast_p` and the geometric mean. Generation
failures that never wrote a sample are **absent** from the denominator.
Compare runs only with matching hardware, CANN, precision, authoring/build
backends, timing method and scope, timing settings, and harness (`metadata.harness`: model, prompt mode, reasoning
effort). These scores are **not** comparable to CUDA KernelBench or to
NVIDIA SOL-ExecBench numbers.

## Evaluation environment

The default protocol is:

- 5 seeded correctness trials, then four hidden value transforms
  (original, ×3, ×0.01, ×−1); all must pass before timing
- 10 warmups (SOL-ExecBench-style isolation) and 100 retained trials
  (NPU events for Ascend C; synchronized wall clock for pyasc)
- L2 flush of `max(256 MiB, 2 × profile L2)` before each timed call
- Isolated worker process group; host kills the group on timeout
- Ascend C uses `TORCH_LIBRARY` / `TORCH_LIBRARY_IMPL`, loaded with
  `torch.ops.load_library`; pyasc uses `@asc.jit` device kernels

The default hardware profile is `ascend910b2`. `ascend950pr` is reserved
and must be validated before use.

## Repository layout

```text
KernelBench/                 Vendored reference tasks, level1–level4
src/                         Generation, build, evaluation, and scoring
scripts/                     Four user CLIs plus two internal worker scripts
configs/                     Eval defaults, hardware profiles, task subsets
build_template/              Default CMake project for libcustom_op.so
tests/                       Offline regressions and NPU backend smoke runner
docs/                        English Markdown guides and references
requirements.txt             Third-party dependencies for every machine
```

## Development

Run the offline regression suites without an NPU:

```bash
python tests/run_tests.py
```

On an Ascend host, validate the three execution paths and save the report:

```bash
export DEEPJIT_ROOT=/path/to/DeepJIT
python tests/npu_backend_smoke.py --output /tmp/backend-smoke.json
```

Use `--case` to select a case; run the command with `--help` for choices.
Offline tests cover generation, run identity, checks, dispatch, and build
recovery; they do not establish device correctness or full-corpus coverage.

Optional when you change Python under `src/` or `scripts/`:

```bash
pre-commit install && pre-commit run --all-files
```

## Acknowledgments and license

The task set is vendored from
[KernelBench](https://github.com/ScalingIntelligence/KernelBench).
Scoring ideas follow KernelBench (`fast_p`, `pass@k`) and NVIDIA
[SOL-ExecBench](https://github.com/NVIDIA/SOL-ExecBench) (isolated
timing and the SOL formula). The roofline bound used here is a
documented bandwidth estimate, not a SOLAR characterization.

Released under the [MIT License](LICENSE).

## pyasc backend

AscendKernelBench can generate and evaluate two authoring languages. One run
uses one backend; Ascend C remains the default.

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

A pyasc sample delivers `kernel.py` (an `@asc.jit` device kernel plus a host
launcher) and `model_new.py`, exactly as an Ascend C sample delivers
`custom_op.asc` and `model_new.py`. The run records its backend in
`generation_config.yaml`. Evaluation checks recorded backend identity against
the artifacts and refuses mixed runs.

Install pyasc with the wheel that matches the host CANN pairing
(`python -m pip install pyasc==1.1.1`); generation and offline analysis do not
import it. See [docs/guide/pyasc-backend.md](docs/guide/pyasc-backend.md) for
the contract, the JIT semantics, the static checks, and the timing caveats.
