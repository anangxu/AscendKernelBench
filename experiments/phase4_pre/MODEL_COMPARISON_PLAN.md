# Model comparison experiment: configuration (not yet executed)

## Goal

Compare code-generation ability across models on the same five Ascend C domain
tasks, instead of tuning prompts for one model.

## Fixed variables

| Item | Value |
|---|---|
| Tasks | cann/nn/swiglu, cann/nn/softmax, cann/nn/gather, cann/auto/scatter, cann/auto/linear_cross_entropy |
| Baseline inputs | `domain_benchmarks/cann/baseline.json`, sha256 `6ddc84667e1f6ee6` |
| Prompt | V2 variant (anti-cheat rules stated), frozen at commit `1b738f0` |
| Device | Ascend 910B4, CANN 9.1.0, driver 25.5.1 |
| Pipeline | static check, CMake build, module load, NPU execution, comparator, timing |
| Correctness | AKB comparator with dtype tolerances, NaN mask, Inf sign (commit `e123cce`) |
| Samples | 1 per task per model |
| Generation | temperature 0, prompt_mode zero_shot, reasoning_effort low, max_tokens 65536 |

## Why V2 is the provisional baseline

V2 reached 4/5 static passes, the best of the five variants, but every variant
produced zero correct candidates, so V2 is a pragmatic choice, not a proven
optimum. This must be stated in the report.

## Models to compare

| Role | Candidate | Note |
|---|---|---|
| current baseline | `deepseek-flash` | already measured: 0/5 correct |
| second code model | to be chosen by the user | e.g. another OpenAI-compatible code endpoint |
| reasoning model | to be chosen by the user | only if the endpoint and budget allow |

The model identifiers must be confirmed by the user before any call, because the
provider list and pricing are not recorded in this repository.

## Metrics to record per model

generation success, static pass rate, compile rate, module load rate, execution
rate, numerical correctness, pass@1, Speedup for correct candidates, token usage
and cost.

Compile-repair rounds are *not* part of pass@1 and must be reported separately.

## Budget suggestion

| Item | Suggestion |
|---|---|
| tasks x models | 5 x 3 = 15 generations |
| tokens | at 32k-64k per generation, roughly 0.5-1M output tokens total |
| cost | must be priced by the user's provider; this repository holds no price table |
| device time | about 3-5 minutes per candidate build and correctness run |

## Not done in this round

No new model was called. No prompt was changed. No candidate was edited. No API
cost was incurred.
