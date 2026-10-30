# Phase 4 pre-experiment: consolidated result

Three generation rounds were run with the same frozen protocol (deepseek-flash,
zero_shot, reasoning_effort low, temperature 0, one sample per task). Each round
tightened the shared case contract after reading the failures, and every
candidate stays in `runs/` unmodified.

| Task | Generation | Static | Compile | Correctness | Timing |
|---|---|---|---|---|---|
| swiglu | saved | PASS (r2) | PASS | 0/5 trials | none |
| softmax | saved | PASS (r3) | PASS | 0/1 case | none |
| gather | saved | PASS (r2) | FAIL: numBlocks redeclared | n/a | n/a |
| scatter | saved | PASS (r2) | FAIL: sqrtf not found | n/a | n/a |
| linear_cross_entropy | saved | PASS (r2/r3) | FAIL: float->unsigned cast | n/a | n/a |

## What the rounds show

- round 1: every candidate tripped the anti-cheat rules; nothing compiled.
- round 2: the contract named those rules, four of five passed the static check,
  and swiglu built and ran on the 910B4 (correctness 0/5).
- round 3: the contract also named the launcher and aicore pitfalls; softmax
  passed the static check and built, the other four each failed on a different
  model-side error (tensor comparison, numBlocks, sqrtf, unsigned cast).

Per-task best: static PASS 5/5, compile PASS 2/5, correctness PASS 0/5.

## Still open

- No candidate is numerically correct, so no valid timing, no Speedup, and the
  hidden-input legality check never runs (it sits behind correctness).
- `max_difference` reads 0.0 next to a failed comparison for one candidate; that
  field is only defined for equal-shaped top-level tensors, so it is not a
  verdict and the report must not read it as one.

## Framework fixes made along the way

- the evaluator scores a task on its pinned baseline case, not the twenty
  official cases;
- the case contract now states the anti-cheat rules, the launcher parameters a
  candidate must not redeclare, and the aicore restrictions (sqrt, casts).

## Round 4 (output-shape rule in the contract)

| Task | Static | Compile | Note |
|---|---|---|---|
| swiglu | PASS | FAIL | `template ...` error in the generated kernel |
| gather | PASS | FAIL | `reference ...` error |
| scatter | PASS | FAIL | cast error |
| softmax | generation truncated at max_tokens | n/a | no candidate |
| linear_cross_entropy | FAIL: tensor comparison in wrapper | n/a | no candidate |

### Cross-round tally

- candidates generated: 15 across four rounds (3.75 per task)
- static check passed: 12 of 15
- compile passed: 2 of 15 (swiglu round 2, softmax round 3)
- correctness passed: 0 of 15
- valid timings: 0, so Speedup is N/A and the hidden-input check is never reached

The two compiled candidates were diagnosed as model-side: swiglu returned an
`empty_like` tensor of the input shape, softmax's kernel math did not match the
reference. Contract clarifications raise the static and compile rate but have not
yet produced a numerically correct candidate.
