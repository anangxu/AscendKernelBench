# Phase 4 pre-experiment: frozen statistics

## 1. Funnel (every step traced to a run record)

| Step | Count | Definition |
|---|---|---|
| generation attempts | 25 | 5 prompt variants x 5 tasks |
| candidates produced | 22 | both custom_op.asc and model_new.py written |
| static check passed | 11 | AKB static rules accepted the pair |
| sent to the device evaluator | 18 | an evaluate.py run covers the sample |
| compiler actually invoked | 11 | static gate passed, build started |
| compile succeeded | 3 | CMake/Ascend C produced libcustom_op.so |
| module loaded | 2 | torch.ops library loaded, wrapper importable |
| kernel executed | 2 | forward call ran on the NPU |
| numerically correct | 0 | comparator accepted every trial |
| valid timing / Speedup | 0 / N/A | timing needs correctness first |

## 2. Why 10 and 13 were both wrong

Neither figure came from the run records. The 10 was a hand-written dict in the
earlier report, and the 13 was the sum of a column built from that same dict.
Reading every sample directory and the evaluator logs gives 18 candidates sent to
the device evaluator. Of those, 11 reached the compiler (the rest were stopped by
the static gate) and 3 compiled.

## 3. Is the static check a gate for the device run?

Yes. `evaluate.py` runs the static check before any build: a sample that fails it
records `failure_stage=static_check` with `compiled=None`, and no compiler call
happens. So V1's four device-evaluated candidates were *sent* to the evaluator and
rejected at the gate; "device evaluated" is not "compiled", and the static pass
rate must not be read as a compile-attempt rate.

## 4. Per-variant table

| Variant | Prompt change | Candidates | Static PASS | Reached compiler | Compiled | Loaded | Correct |
|---|---|---|---|---|---|---|---|
| V1 | frozen contract | 4 | 0 | 0 | 0 | 0 | 0 |
| V2 | anti-cheat rules | 5 | 4 | 4 | 1 | 1 | 0 |
| V3 | launcher and aicore rules | 5 | 2 | 2 | 1 | 1 | 0 |
| V4 | output shape rule | 4 | 3 | 3 | 0 | 0 | 0 |
| V5 | structural skeleton | 4 | 2 | 2 | 1 | 0 | 0 |
| total | | 22 | 11 | 11 | 3 | 2 | 0 |

## 5. Attribution of the three compiled candidates

| Candidate | Compile | Load | Execute | Correctness | Attribution |
|---|---|---|---|---|---|
| V2 swiglu | PASS | PASS | PASS | FAIL | model: `trial 4: output mismatch`; the host returns `empty_like(x)` so the shape is half what it should be |
| V3 softmax | PASS | PASS | PASS | FAIL | model: `trial 4: output mismatch`, kernel math differs from the reference |
| V5 gather | PASS | FAIL | - | - | **undetermined**: `module load failed: TypeError(float() on list)`; the wrapper shown in the sample has no `float()` call, so the source is not yet isolated |

V5 gather is therefore *not* counted as a model correctness failure. It is a
module-load failure with undetermined origin.

## 6. Framework defect fixed during the audit

`compare.py` imported `compare_outputs` from `cases.py`, but the function did not
exist, so every case-task comparison raised ImportError. Fixed in `e123cce`; the
three compile/load results above were re-measured afterwards. Correctness verdicts
for case tasks recorded before that commit are void.

## 7. Evidence state

- local: `runs/s4*/.../{custom_op.asc, model_new.py}` (all candidates), `audit_final.json`,
  `progress_*.log`, `gen*.log`, `eval*.log`, `evidence/candidates.json` with hashes;
- device: only `/cache/akb/exp/p4f` and `/cache/akb/exp/p4g` survive (the host was
  recycled), so results for rounds 1-4 rest on the session log rather than on device
  files. That is a real evidence gap and is stated rather than papered over.

## 8. Frozen decisions

The five prompts, all 22 candidate sources, the baseline inputs, the tolerances, and
the scoring protocol are frozen as of this report. `0 correct candidates` stands as
the result.
