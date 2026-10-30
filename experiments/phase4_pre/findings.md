# Phase 4 findings so far

## swiglu (round 2 candidate)

- static check: PASS
- compile: PASS (real CMake build on the 910B4 host)
- NPU run: PASS, reference computed on the NPU
- correctness: 0/5 trials, `trial 4: output mismatch`
- wrapper calls `torch.ops.custom_op.run(x, dim)`; the kernel declares its own
  signature, so the extra argument and the assumed split are the likely causes.
- `max_difference` reads 0.0 while the comparison fails, because that metric is
  only defined for equal-shaped top-level tensors; the field is not a verdict.

## The other three (round 2)

- gather: entry parameter `numBlocks` collides with the generated launcher.
- scatter: `sqrtf` does not exist on the aicore side.
- linear_cross_entropy: direct float to unsigned cast.
All three are model-side errors, and all three are now named in the entry
contract so the next round can avoid them.

## Hidden inputs

`_run_hidden_distributions` runs for case tasks, but this candidate never
reached it: correctness failed first. Hidden-input legality for constrained
tasks (indices, cross-input sums) is therefore still unverified.

## Why the two compiled candidates are wrong (round 3 diagnosis)

### swiglu (`runs/s4b_swiglu`)

The host allocates `at::empty_like(x)` and the kernel writes `L * N` elements with
`L = half * inner`, so the returned tensor keeps the input shape `[1024, 2048]`
instead of `[1024, 1024]`. The comparison fails on shape, so `max_difference`
reads 0.0 (that metric is only defined for equal shapes). Model-side bug, and the
case contract now says the output shape follows the operator definition.

### softmax (`runs/s4c_softmax`)

The host takes `dim` and ignores it (`(void)dim`), the baseline dim is -1 so that
is harmless, and the output shape is correct. The failure is numeric: the kernel
math does not reproduce the reference. Model-side.

### Consequence

Neither candidate can produce a valid timing, so Speedup stays N/A and the
hidden-input check never runs. Real NPU execution is proven (compile + run both
succeed); numerical correctness is not.
