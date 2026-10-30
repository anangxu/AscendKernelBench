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
