# Phase 4 pre-experiment: five frozen AKB baselines

- baseline commit: `4aecd30`
- baseline.json sha256: `6ddc84667e1f6ee6`
- model `deepseek-flash`, `zero_shot`, `reasoning_effort=low`, `max_tokens=32768`,
  `temperature=0`, `n_samples=1`, hardware `ascend910b4`, `ascendc` + `cmake`

## Generation

| Task | Result |
|---|---|
| swiglu | saved |
| softmax | saved |
| gather | saved |
| scatter | saved |
| linear_cross_entropy | truncated |

## Evaluation (one pinned baseline case each)

| Task | Cases | Passed | Error | Failure stage | Compiled |
|---|---|---|---|---|---|
| cann/nn/swiglu | 1 | 0 | 1 | static_check | False |
| cann/nn/softmax | 1 | 0 | 1 | static_check | False |
| cann/nn/gather | 1 | 0 | 1 | static_check | False |
| cann/auto/scatter | 1 | 0 | 1 | static_check | False |

## Tally

- generation saved: 4/5 (linear_cross_entropy truncated at max_tokens)
- static check passed: 0/4
- compile attempted: 0 (the static gate rejects before any build)
- NPU run: 0, correctness: 0, valid timings: 0, speedup: N/A

The candidates trip the anti-cheat rules (try/except, globals()) and, for
scatter, declare an rms_norm entry point. These are model-side results, not
framework failures; the evaluator classified each as failure_stage
static_check with a single pinned case per task.

## Pipeline note

The evaluator now scores a task on its pinned baseline case when one exists,
so a candidate is measured on one stable workload instead of the twenty
official cases. The official cases remain available as a separate suite.
