# Phase 4 pre-experiment, round 2 (static-check rules stated)

| Task | Generation | Static | Compile | Correctness | Timing |
|---|---|---|---|---|---|
| cann/nn/swiglu | saved | PASS | PASS | 0/5 trials | none |
| cann/nn/gather | saved | PASS | FAIL: redefinition of parameter 'numBlocks' | n/a | n/a |
| cann/auto/scatter | saved | PASS | FAIL: no matching function for call to 'sqrtf' | n/a | n/a |
| cann/auto/linear_cross_entropy | saved | PASS | FAIL: cast between floating and unsigned integer | n/a | n/a |
| cann/nn/softmax | saved | FAIL: comparison on non-scalar values | not attempted | n/a | n/a |

Round 1 generated four candidates that all tripped the anti-cheat rules.
Round 2 states those rules in the case contract and every candidate passes
except softmax (one tensor comparison in the wrapper).

Compilation now actually runs: swiglu builds, and the other three fail with
real Ascend C errors that the entry contract now warns about (numBlocks,
sqrtf, float<->unsigned casts). swiglu runs on the NPU but fails correctness,
so no valid timing or speedup exists yet.
