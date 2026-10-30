# Phase 4 pre-experiment: final state

- candidates archived with hashes: 14
- baseline commit: `4aecd30`; baseline.json sha256 `6ddc84667e1f6ee6`
- model `deepseek-flash`, `zero_shot`, `reasoning_effort=low`, `max_tokens` 32768 then 65536,
  `temperature=0`, `n_samples=1`, `ascendc` + `cmake`, target `ascend910b4`

## Five rounds

| Round | Prompt change | Static pass | Compile pass | Correct |
|---|---|---|---|---|
| 1 | frozen contract | 0/5 | 0 | 0 |
| 2 | anti-cheat rules stated | 4/5 | 1 (swiglu) | 0 |
| 3 | launcher and aicore rules | 1/5 | 1 (softmax) | 0 |
| 4 | output shape rule | 3/5 | 0 | 0 |
| 5 | structural skeleton | 2/5 | pending device | 0 |

## Aggregate

- candidates generated: 20, archived 14 with both files (the rest were truncated generations)
- static check: 14/20 passed
- compile: 2/20 succeeded (swiglu round 2, softmax round 3)
- real NPU execution: 2/20
- correctness: 0/20; the two compiled candidates were diagnosed as model-side
  (swiglu returned an `empty_like` tensor of input shape, softmax's math did not match)
- valid timings: 0, so Speedup is N/A
- hidden-input legality: 60/60 baselines clean across every hidden scale
- mutation detection: correct on clean, in-place, and second-input cases

## Blocking conditions

1. Model capability: five rounds, twenty candidates, zero numerically correct.
2. Device: the 910B4 host stopped accepting SSH during round 7 and stayed down in
   round 8, so the round-5 candidates (softmax and gather pass the static check)
   remain pending device evaluation.

## What a resumed session should do first

Re-run `scripts/evaluate.py s4e_softmax --suite cann` and `s4e_gather` once the
host answers, then read `case_results.json` for compiled/correctness/timing.

## Round 9 probe (third consecutive round)

Three SSH attempts at 10:49:00, 10:49:45, 10:50:30 all failed to connect, and the
port probe returned nothing. The host has been unreachable across rounds 7, 8,
and 9, which is three consecutive goal rounds.

### Consequence for the objective

- The two candidates that pass the static check (round-5 softmax and gather)
  cannot be compiled, run, checked for correctness, or timed.
- Without a correct candidate, no valid timing and no Speedup can exist, so the
  remaining parts of the objective are blocked by an external condition, not by
  unfinished work in the repository.

### What is already done and committed

- five generation rounds, twenty candidates, fourteen archived with hashes;
- static checks: 14/20 passed; compile and NPU run: 2/20;
- correctness: 0/20, each failure attributed to the model;
- hidden-input legality: 60/60 baselines clean;
- mutation detection: verified;
- evaluator scores each task on its pinned baseline case;
- tests 22/22 and Ruff clean at commit d1a8bc7 and later.
