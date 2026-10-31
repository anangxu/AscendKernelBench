# Third-party notice: CANN Bench

The files under `domain_benchmarks/cann/*/official/` are copied verbatim from
`hicann/cann-bench` at commit `2d7cd332247f33a66ca67c77d9f09d446a780dfe`.

- They retain their original CANN Open Software License Agreement Version 2.0
  headers; no copyright notice was changed and nothing is relicensed.
- The surrounding AKB code remains MIT. The two licenses apply to their own
  files.
- Purpose: the benchmark needs the official operator definition, reference, and
  public cases to define each task and to check a candidate.

Redistribution question for the reviewer: if vendoring these files is not
acceptable, the alternative is a setup step that downloads the five files per
task from the pinned commit and verifies them against
`task_inventory.json`. That change has not been made, because it would add a
network dependency to every run.
