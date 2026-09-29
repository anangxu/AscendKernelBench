# Results and scoring

AscendKernelBench separates generation artifacts, per-sample evaluation results, and aggregate reports. You can inspect and analyze existing results on macOS without an NPU.

```bash
python scripts/analyze.py my_run
```

This command reads `runs/my_run/eval_results.json` and prints headline metrics and per-problem details. It does not run evaluation, regenerate the aggregate from sample directories, or write a new report file.

## Run artifacts

```text
runs/my_run/
├── generation_config.yaml
├── level1/
│   └── 19_ReLU/
│       └── sample_0/
│           ├── prompt.txt
│           ├── response_raw.txt
│           ├── custom_op.asc
│           ├── model_new.py
│           ├── CMakeLists.txt
│           ├── custom_op*.so
│           ├── build/
│           │   ├── configure.log
│           │   └── build.log
│           └── eval_result.json
├── eval_results.json
└── pass_at_k_results.json
```

This is a representative layout, not a promise that every file exists. `response_raw.txt` is written when a raw response is available. Build artifacts/logs appear only after their stages run. `scripts/evaluate.py` writes both aggregate JSON files.

`generation_config.yaml` records selected generation settings and tasks. It does not record the entire resolved evaluation configuration, driver/CANN versions, repository revision, or device state. Preserve those separately for reproducible comparisons. Using an existing run name can overwrite generated files and configuration; re-evaluation overwrites results.

## Per-sample result schema

`eval_result.json` contains the result for one sample. The aggregate `eval_results.json` maps each task ID to a list of those results with an added `sample_id`.

The following is an illustrative schema example, **not a measured result**:

```json
{
  "level1/19_ReLU": [
    {
      "sample_id": 0,
      "compiled": true,
      "correctness": true,
      "runtime": 0.02,
      "runtime_stats": {
        "mean": 0.02,
        "std": 0.001,
        "min": 0.018,
        "max": 0.023,
        "num_trials": 100
      },
      "ref_runtime": 0.03,
      "ref_runtime_stats": {
        "mean": 0.03,
        "std": 0.002,
        "min": 0.027,
        "max": 0.035,
        "num_trials": 100
      },
      "metadata": {
        "hardware": "ascend910b2",
        "precision": "fp32",
        "reference": "npu",
        "max_difference": 0.0,
        "correctness_trials": 5,
        "correctness_passed": 5,
        "atol": 0.0001,
        "rtol": 0.0001,
        "seed": 42,
        "num_warmup": 10,
        "num_perf_trials": 100,
        "l2_clear_size": 402653184,
        "timing_fresh_inputs": true,
        "speedup": 1.5,
        "excessive_speedup": false,
        "hidden_distributions": ["d1", "d2", "d3", "d4"],
        "hidden_passed": ["d1", "d2", "d3", "d4"],
        "hidden_failed": null,
        "harness": {
          "model": "example-model",
          "prompt_mode": "one_shot",
          "temperature": 0.0,
          "reasoning_effort": "high",
          "max_tokens": 131072
        },
        "bytes_moved": 8388608,
        "sol_bound_ms": 0.00524,
        "sol_bound_kind": "roofline_memory",
        "sol_score": 0.51,
        "torch_version": "2.10.0",
        "torch_npu_version": "2.10.0.post6",
        "cann_version": "9.1.0",
        "device": "npu:0",
        "device_name": "Ascend910B2"
      }
    }
  ]
}
```

| Field | Meaning |
| --- | --- |
| `sample_id` | Integer identifying a generated candidate within a task; added during aggregation. |
| `compiled` | Build-stage status in normal execution. Early failures and worker failures can report `false`; this is not a complete compiler audit trail. |
| `correctness` | Final correctness decision. This includes the seeded trials, the four hidden value transforms, and the post-timing fresh-input check. |
| `runtime` | Candidate mean NPU latency in milliseconds, or `null`. |
| `runtime_stats` | Candidate timing statistics, or `null`. |
| `ref_runtime` | Live NPU reference mean latency in milliseconds, or `null`. |
| `ref_runtime_stats` | NPU reference timing statistics, or `null`. |
| `metadata` | Stage-specific settings, diagnostics, and flags. Keys vary by execution path. |

Common diagnostic metadata includes `static_check_error` (a list), `compilation_error`, `runtime_error`, `correctness_error`, and `reference_npu_error`. Built samples also record `build_mode` (`split` for the two-section layout with the shared precompiled header, `legacy` for a single translation unit) and `build_seconds`. A successful correctness path also records a protocol snapshot (hardware, precision, seed, trial counts, warmup, L2 flush size, tolerances, operator mode), the reference mode, and the software stack (`torch_version`, `torch_npu_version`, `device`, and when available `device_name`, `cann_version`, `ascend_home`). Timed NPU-reference samples may also record `bytes_moved`, `sol_bound_ms`, `sol_bound_kind`, and `sol_score`. Early failures do not necessarily contain those contextual fields.

A sample that raised while running carries three classification fields beside `runtime_error`: `runtime_error_class` (the exception class name, for example `OutOfMemoryError`), `runtime_error_stage` (where it was raised: `input_preparation`, `device_preparation`, `first_call`, `model_init`, `hidden_inputs`, `hidden_reference`, `hidden_candidate`, `candidate_trial`, `timing`, `timing_inputs`, or `post_timing_recheck`), and `runtime_error_summary` (the message on one line, bounded). They name the symptom and the step; they never record a verdict on whether the model or the environment caused it, and they do not move a sample out of any rate. `summarize_eval_results` counts them into `runtime_error_classes` and `runtime_error_stages`, and `scripts/analyze.py` prints one row for each, even when both are empty. Samples evaluated before these fields existed count as `unclassified`.

`max_difference` is a limited diagnostic: it is updated for mismatched top-level tensor outputs of equal shape when a numeric difference can be calculated. It is not a complete maximum-error statistic across every output or successful trial. A value of zero does not prove bitwise equality.

`correctness_trials` and `correctness_passed` describe the initial seeded trials. A hidden-distribution failure or a post-timing failure can leave those counts fully passed while setting final `correctness` to `false`. `hidden_failed` is the first failing transform (`d1` through `d4`), or `null` when the gate passed. `harness` copies the generation settings that define the run (model, prompt mode, temperature, reasoning effort, max tokens), so a copied `eval_results.json` still names the model-harness pair that produced it.

### Missing timing and partial results

Missing latency is not zero latency. Typical cases include:

| Result state | Expected interpretation |
| --- | --- |
| Static/build failure | Incorrect; no timing. Inspect static violations or compiler logs. |
| Initial correctness failure, including a hidden-distribution mismatch | Incorrect; timing is skipped. |
| Correct with `reference: "cpu"` | Candidate timing may exist, but no NPU reference timing or speedup. |
| Timing exception | Correctness may remain true after a successful post-timing check; timing can be absent or partial. |
| Post-timing failure | Incorrect; already collected timing fields can still be present. |

Do not infer success from latency alone. Check `correctness`, diagnostics, reference mode, and the excessive-speedup flag together.

## Speedup and fast_p

For a correct sample with an available NPU baseline and without an excessive-speedup flag:

```text
speedup = ref_runtime / runtime
```

The scorer recomputes this ratio from the stored mean runtimes; it does not use the rounded `metadata.speedup` value.

`fast_p` is the fraction of collected sample results that are correct and have a speedup **strictly greater than** `p`. The built-in thresholds are `0`, `0.5`, `0.8`, `1`, `1.5`, and `2`. `fast_0` is a special case: it is the correctness rate, including correct CPU-reference, untimed, and flagged samples.

| Sample state | Counts in denominator | Counts toward `fast_0` | Eligible for positive thresholds |
| --- | --- | --- | --- |
| Compilation/static-check failure with a collected result | Yes | No | No |
| Incorrect candidate | Yes | No | No |
| Correct, untimed or CPU-reference candidate | Yes | Yes | No |
| Correct, excessive-speedup flag | Yes | Yes | No |
| Correct, valid unflagged NPU speedup | Yes | Yes | Yes, if speedup is strictly above the threshold |

For example, an illustrative aggregate with four samples—one compile failure, one incorrect sample, one correct CPU-reference sample, and one correct sample at exactly 2×—has `fast_0 = 0.5`, `fast_1 = 0.25`, and `fast_2 = 0`.

### What the denominator contains

Aggregation includes sample directories that have **both implementation files and an `eval_result.json`**. Compilation failures count when their results have been persisted. Tasks or samples with failed generation, missing implementation files, or missing results are absent; they are not automatically inserted as failures. The denominator is neither the requested generation count nor the entire vendored task set.

Consequently, report the requested task/sample counts alongside the evaluated counts and generation failures. A partial run can otherwise look better than its full planned workload. Runs with different sample counts per task also weight tasks differently in `fast_p`, because each sample receives equal weight.

## Geometric mean speedup

`geometric_mean_speedup_correct_only` is the geometric mean of available speedups over correct, unflagged samples. It excludes failures, untimed samples, and CPU-reference samples. It returns `0.0` when no eligible speedups exist.

Because this metric is conditioned on a selected subset of samples, always report it with correctness and coverage. A higher geometric mean on a smaller passing subset is not sufficient evidence of a stronger overall result.

## Roofline SOL score

When a sample has a candidate mean, an NPU-reference mean, and a positive profile `memory_bandwidth_gbps`, the evaluator records:

```text
T_sol = bytes_moved / bandwidth
S = (T_b - T_sol) / ((T_k - T_sol) + (T_b - T_sol))
```

`bytes_moved` is the sum of top-level input and last-trial output tensor sizes. It is not a full DRAM-traffic model. When a task declares a positive top-level `FLOPS` or `NUM_FLOPS` constant **and** the hardware profile has `peak_tflops` for the active precision, the bound becomes `max(memory, compute)` and `sol_bound_kind` is `roofline`. The vendored KernelBench corpus does not declare FLOPs, so published scores use the memory term only. `S = 0.5` matches the software baseline; `S` approaches `1` as the kernel approaches the bound. `analyze.py` reports `mean_sol_score` over correct samples whose timing was not rejected, the same eligibility the speedup numbers use. This is a reporting aid after NVIDIA SOL-ExecBench, not a claim that the machine was characterized with SOLAR. Do not compare these numbers to CUDA SOL-ExecBench leaderboards.

`summarize_eval_results` also counts `cpu_reference`, `npu_reference`, and `excessive_speedup` samples so a report can show coverage next to `fast_p`. The same summary carries `runtime_error_classes` and `runtime_error_stages`, the generic classification of runtime failures described above, and the report prints both as `runtime failures (class)` and `runtime failures (stage)`.

## pass@k

For a problem with `n` collected samples and `c` correct samples, the estimator for `k <= n` is:

```text
pass@k = 1 - C(n - c, k) / C(n, k)
```

When fewer than `k` incorrect samples exist, the estimate is `1.0`. Compilation failures count among the `n - c` unsuccessful samples. Correct CPU-reference, untimed, and flagged samples count in `c`, because pass@k measures correctness, not speed.

The default report requests `k = 1, 5, 10`. A problem contributes to `pass@5` or `pass@10` only when enough collected samples exist. The average for each `k` is an unweighted mean over the problems that have that key, so different `k` values may average different problem sets. `pass@1` is always included for a problem entry; normal aggregated problem entries contain at least one sample.

`pass_at_k_results.json` has this structure:

```json
{
  "per_problem": {
    "level1/19_ReLU": {
      "pass@1": 0.5
    }
  },
  "average": {
    "pass@1": 0.5
  }
}
```

The example is illustrative. Higher `k` values are omitted when the problem lacks enough samples. Generating one sample per task therefore does not produce `pass@5` or `pass@10`.

## Compare runs responsibly

Record the task source/revision, exact problem and sample counts, generation model/settings, hardware and runtime device, CANN/PyTorch/`torch_npu` versions, precision/tolerances, timing configuration, CPU fallback count, and flagged sample count.

The JSON format follows KernelBench-style result conventions. This format compatibility does not make scores directly comparable across NPU/GPU hardware, baseline implementations, task revisions, or timing protocols. Use the [evaluation protocol](evaluation.md) to identify those differences before interpreting a speedup.
