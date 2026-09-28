# pyasc end-to-end evaluation report

First end-to-end pyasc run: a model writes `kernel.py` and `model_new.py` from a
KernelBench task, the harness statically checks the sample, and the device then
decides whether it compiled, whether it is correct, and how fast it is. This
page records the configuration, the results, the framework fixes the run
forced, and what the numbers do not cover.

## Configuration

| Item | Value |
| --- | --- |
| Repository commit | `cda07cc` (checker fix landed mid-run; the run itself is tracked below) |
| Backend | `pyasc` |
| Task subset | `configs/subsets/level1_20.txt`, 20 level1 tasks (every fifth task) |
| Samples per task | 3, template `--n-samples 3` |
| Planned candidates | 60 |
| Model | `deepseek-flash` |
| Endpoint | `https://api.deepseek.com/v1` (OpenAI-compatible) |
| Prompt mode | `one_shot` |
| Temperature | 0.0 |
| Max tokens | 131072 |
| Reasoning | `--reasoning-effort high` (thinking enabled) |
| Hardware profile | `ascend910b2` |
| Device | Ascend 910B4, one visible device |
| CANN | 9.1.0 |
| torch / torch_npu | 2.10.0+cpu / 2.10.0 |
| pyasc | 1.1.1 |
| Python | 3.12.13 |
| nproc / HBM | see device notes in `AGENTS.md`; 32 GB HBM on this instance |

Generation ran on the macOS host, evaluation on the Ascend host: samples are
written to `runs/{run}/level{L}/{task}/sample_{i}/` and can be evaluated later,
on another machine. Only the engine, the subset and the generated samples were
copied to the device, never the results back.

### Task list

The 20 tasks of `level1_20.txt`:

```text
level1/5_Matrix_scalar_multiplication
level1/10_3D_tensor_matrix_multiplication
level1/15_Matmul_for_lower_triangular_matrices
level1/20_LeakyReLU
level1/25_Swish
level1/30_Softsign
level1/35_GroupNorm_
level1/40_LayerNorm
level1/45_Average_Pooling_2D
level1/50_conv_standard_2D__square_input__square_kernel
level1/55_conv_standard_2D__asymmetric_input__square_kernel
level1/60_conv_standard_3D__square_input__asymmetric_kernel
level1/65_conv_transposed_2D__square_input__asymmetric_kernel
level1/70_conv_transposed_3D__asymmetric_input__square_kernel
level1/75_conv_transposed_2D_asymmetric_input_asymmetric_kernel_strided__grouped____padded____dilated__
level1/80_conv_standard_2D_square_input_asymmetric_kernel___dilated____padded__
level1/85_conv_depthwise_2D_asymmetric_input_asymmetric_kernel
level1/90_cumprod
level1/95_CrossEntropyLoss
level1/100_HingeLoss
```

## How a sample is scored

Per sample, in order: static check (no device needed), then in one worker
process the first subscripted call decides `compiled`, five correctness trials
with different seeds decide `correctness`, four hidden value transforms gate a
suspicious-looking result, and only then are NPU-event timings taken against a
live `torch_npu` eager reference. `pass@k` is the per-task estimator
`1 - C(n-c, k) / C(n, c)` averaged over tasks; `fast_0` is the fraction of
collected samples that are correct. Generation failures that never wrote a
sample are absent from the denominator rather than counted as failures.

## Results

<!-- filled in when the run finished -->

## Framework fixes the run forced

### The static checks rejected three legal pyasc shapes

The first task produced two candidates that the checks rejected as
`failure_stage=static_check`. Both compiled and matched the reference when run
directly on the device, so the checks were wrong:

| Shape | Why it was rejected | Evidence it is legal |
| --- | --- | --- |
| Kernel inside a module-level `if hasattr(asc, "mul"):` | kernels were collected from `tree.body` only | same file launched on the device, `max_abs_diff = 0` against `A * s` |
| Launcher delegating the launch to a helper | only the function holding the subscript counted as an entry point, so `model_new.py` looked like it never launched | same, `max_abs_diff = 0` |
| `int`-annotated sizes and constants defined beside the launcher | `module_scalars` were merged after bindings were resolved, and annotations were never consulted | same, `max_abs_diff = 0` |

The fix collects definitions anywhere in the file, grows the entry-point set to
host functions that reach a launch, merges `module_scalars` before resolving
bindings, and treats `int`, `float` and `bool` annotations as scalars.
Unannotated parameters stay unknown, so tensor arithmetic in a launcher is
still rejected, and a regression case covers that negative control.

Re-evaluating the same samples without regenerating them moved the first task
from one correct candidate to three. The samples were not regenerated, and no
failed generation was replaced.

## Known limitations

* The TQue copy dependency described in
  `docs/reference/pyasc-device-evidence.md` applies here: a kernel whose only
  work is a copy must order its inbound copy against its write-back itself. The
  contract tells the model so, and this run shows whether it does.
* Correctness is judged on one device, one CANN and one pyasc release. Nothing
  here is a cross-platform comparison; the GPU side of the same subset has not
  been run.
* Performance numbers are single-sample eager comparisons on a shared,
  ephemeral cloud instance. They are indicative, not a benchmark result.
* `one_shot` prompting was used, so the examples are part of the measured
  configuration. `zero_shot` was not measured.
* Generation failures are counted as failures. A sample the checks reject is
  not retried: the run keeps the first answer.

## Reproduction

```bash
# Generation (any machine; no NPU needed)
export OPENAI_API_KEY=... OPENAI_BASE_URL=https://api.deepseek.com/v1
python scripts/generate.py --tasks-file configs/subsets/level1_20.txt \
    --model deepseek-flash --hardware ascend910b2 --backend pyasc \
    --n-samples 3 --prompt-mode one_shot --temperature 0.0 \
    --max-tokens 131072 --reasoning-effort high --run-name pyasc_l1_20_n3

# Evaluation (Ascend host)
ASCEND_RT_VISIBLE_DEVICES=0 python scripts/evaluate.py pyasc_l1_20_n3 1
python scripts/analyze.py pyasc_l1_20_n3
```
