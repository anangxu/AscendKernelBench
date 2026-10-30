# Corrected statistics (audit of every attempt)

## Attempt accounting

| Quantity | Count |
|---|---|
| generation attempts | 25 (5 variants x 5 tasks) |
| attempts that produced both files | 23 |
| attempts with no candidate | 2 (V1 linear_cross_entropy truncated, V4 softmax missing) |
| candidates passing the static check | 11 of 23 |
| candidates device-evaluated | 10 |
| candidates that compiled | 3 (V2 swiglu, V3 softmax, V5 gather) |
| candidates numerically correct | 0 of 3 compiled |
| valid timings / Speedup | 0 / N/A |

## Per variant

| Variant | Prompt change | Candidates | Static PASS | Device evaluated | Compiled | Correct |
|---|---|---|---|---|---|---|
| V1 frozen contract | — | 4/5 | 0 | 0 | 0 | 0 |
| V2 anti-cheat rules | — | 5/5 | 4 | 4 | 1 | 0 |
| V3 launcher+aicore | — | 5/5 | 2 | 1 | 1 | 0 |
| V4 output shape | — | 4/5 | 3 | 3 | 0 | 0 |
| V5 structural skeleton | — | 5/5 | 2 | 2 | 1 | 0 |

## Device results, per candidate

| Variant | Task | Result |
|---|---|---|
| s4b | swiglu | compiled: correctness FAIL (output mismatch) |
| s4c | softmax | compiled: correctness FAIL (output mismatch) |
| s4e | gather | compiled: correctness FAIL (module load: float on list) |
| s4b | gather | compile FAIL: numBlocks redeclared |
| s4b | scatter | compile FAIL: sqrtf not found |
| s4b | linear_cross_entropy | compile FAIL: float->unsigned cast |
| s4d | swiglu | compile FAIL: template error |
| s4d | gather | compile FAIL: reference error |
| s4d | scatter | compile FAIL: cast error |
| s4e | softmax | compile FAIL: see build log |

## Correction to earlier reports

Earlier notes said twenty candidates and fourteen static passes. The audit of
the five progress logs and every run directory gives 25 attempts, 23 candidates,
and 11 static passes. The earlier figures were wrong.

## Framework defect found and fixed this round

`compare.py` imported `compare_outputs` from `cases.py`, but that function did not
exist, so every case-task comparison raised ImportError. Correctness verdicts
recorded for case tasks before commit `e123cce` are therefore invalid; the three
compile results above were re-measured after the fix.
