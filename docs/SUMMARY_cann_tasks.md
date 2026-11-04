# What this branch adds

## In one paragraph

Sixty industry operators from `hicann/cann-bench` (commit `2d7cd33`) become
first-class AscendKernelBench tasks. Each carries the official definition,
reference, and public cases, a reference that delegates to the official
`golden.py`, and one pinned baseline workload so a candidate is scored on a
stable shape. The case layer learns the structures those operators need:
optional inputs, tensor lists, semantic index ranges, per-case attributes,
multi-output trees, and a NaN/Inf-aware comparator. The original 270 KernelBench
tasks keep their old behaviour.

## Code, by area

### Task model and discovery
- `src/suite.py`: `kernelbench` and `cann` suites, path safety, task hashing,
  metadata including per-task `cases`.
- `src/dataset.py`: `Task.cases`, `load_task`, `discover_tasks`, and loading of a
  task's cases from `meta.yaml`.

### Case layer (`src/cases.py`)
- `CaseSpec` with optional (`None`) input slots, tensor-list shapes, per-input
  value ranges, and semantic ranges such as `[unique, min, max]`.
- `generate_inputs`: seeds from the case, keeps integer dtypes integer, draws
  distinct integers for `unique`, and raises on unsupported semantics.
- `compare_outputs`: NaN masks must agree, Inf signs must agree, then a finite
  tolerance applies; shape mismatch fails.
- `summarize`: per-case status counts and `task_correct` (true only when every
  required case passed).

### Baselines (`domain_benchmarks/cann/baseline.json`, `src/cann_baseline.py`)
- One pinned workload per task: shapes, dtypes, attributes, value range, origin.
- `build_inputs` regenerates data from a seed and runs the official `get_input`
  fixer when the golden ships one, so cross-input invariants hold.

### Reference binding (`domain_benchmarks/cann/**/task.py`)
- The reference delegates to the vendored official golden and binds parameters
  **by name**: tensors fill the parameters they belong to, attributes keep their
  own slot even when they sit between tensors (as `dim` does in scatter).

### Evaluation (`src/eval.py`, `src/eval_device.py`)
- A task with a pinned baseline is scored on that single case; the official
  twenty cases stay as a separate generalisation suite.
- Per-case attributes reach both the reference module and the candidate module as
  `CASE_ATTRS`, set before every forward call.
- Case tasks compare through `compare_outputs`; other tasks keep `allclose`.
- A failure is labelled by stage: `static_check`, `build_failed`, execution,
  correctness, or timing; one candidate that fails to build counts once, not once
  per case.
- Hidden distributions scale only nonzero floating values, so integer and index
  inputs stay byte-identical and shapes never change.

### Task data (`domain_benchmarks/cann/`)
- `nn/` (rmsnorm, swiglu, softmax, gather, apply_rotary_pos_emb) and `auto/` (the
  other fifty-five), each with the five official files under `official/`, a
  `SOURCE.md`, a `task.py`, and a `meta.yaml`.
- `baseline.json`, `task_inventory.json` (official path, level, case count, file
  hashes), and `NOTICE.md` (third-party license scope).

### Prompt contract (`src/prompt.py`)
- Case tasks get a case contract: dynamic shapes and dtypes, varying epsilon, the
  entry signature the builder accepts, the static rules that reject a wrapper,
  launcher and aicore restrictions, the output-shape rule, and a structural
  skeleton with no operator math.
- Tasks without cases keep the original prompt byte for byte.

## Verification in this branch

| Check | Result |
|---|---|
| `tests/run_tests.py` | 23/23 |
| `tests/test_cann_native.py` | comparison, binding, tensor list/None, multi-output, unique, hidden |
| `ruff check src/ scripts/` | clean |
| `ruff format --check` | clean |
| golden smoke, one official case per task | 60/60 |
| KernelBench discoverable | 270 |

## Not in this branch

Model-generated candidates, experiment logs, device measurements, and research
reports. Those live in the research branch and are not part of the adapter PR.
