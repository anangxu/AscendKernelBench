# CANN Bench domain tasks

## Provenance

All sixty tasks come from `hicann/cann-bench` at commit
`2d7cd332247f33a66ca67c77d9f09d446a780dfe`. Each task directory keeps the five
official files (`proto.yaml`, `golden.py`, `desc.md`, `cases.yaml`, `cases.csv`)
unchanged under `official/`, together with a `SOURCE.md` that records the path,
the commit, and the license.

`domain_benchmarks/cann/task_inventory.json` lists all sixty tasks with their
official path, level, case count, file list, and the SHA256 of every vendored
file as recorded in the upstream `suite_manifest.json`.

## License

The repository is MIT. The vendored CANN Bench files carry the **CANN Open
Software License Agreement Version 2.0** headers, which the upstream project
applies to those files. They are kept verbatim under `domain_benchmarks/cann/*/official/`
with their headers intact and are not relicensed. See `domain_benchmarks/cann/NOTICE.md`
for the scope statement. If a reviewer prefers not to redistribute them, the
alternative is to fetch the five files per task from the pinned commit at setup
time; that switch has not been made.

## How a task is defined

Each task directory holds a `task.py` with the AKB reference contract
(`Model`, `get_inputs`, `get_init_inputs`) and a `meta.yaml` that points at the
official cases. The reference delegates to `official/golden.py`, so the math is
always the upstream one.

## Baselines versus official cases

`domain_benchmarks/cann/baseline.json` pins one workload per task: shapes,
dtypes, attributes, and value range. It is what the main benchmark scores, so a
candidate is measured on one stable workload. The official twenty cases per task
stay available as a separate generalisation suite and are not part of the main
score.

`src/cann_baseline.py` loads that file, regenerates inputs from a seed, and runs
the official `get_input` fixer when the golden ships one, which keeps cross-input
invariants (distinct indices, expert token sums, sequence lengths) intact.

## Case attributes

Per-case attributes such as `dim`, `epsilon`, `layout`, or `rotaryMode` reach
both the reference and the candidate as the module global `CASE_ATTRS`, set by
the worker before every forward call.

## Comparison

Case tasks compare through `src.cases.compare_outputs`: NaN masks must agree,
Inf signs must agree, then a finite tolerance applies. Tasks without cases keep
the original `allclose` behaviour, so the KernelBench corpus is unaffected.

## Checks

```bash
python tests/run_tests.py           # 23 scripts including test_cann_native
python tests/test_cann_native.py    # comparison, binding, structure, unique, hidden
ruff check --select=E,F,W,I src/ scripts/
ruff format --check src/ scripts/
```

The golden smoke check runs one official case per task without touching the
shapes or dtypes recorded in `baseline.json`.
