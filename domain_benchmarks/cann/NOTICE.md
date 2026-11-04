# Third-party notice: CANN Bench

## What is vendored

`domain_benchmarks/cann/*/official/` holds files copied verbatim from
`hicann/cann-bench` at commit `2d7cd332247f33a66ca67c77d9f09d446a780dfe`:
`proto.yaml`, `golden.py`, `cases.yaml`, `cases.csv`, and `desc.md` per task.

## Copyright and license

Every one of those files keeps its original header:

```
Copyright (c) 2026 Huawei Technologies Co., Ltd.
CANN Open Software License Agreement Version 2.0
```

No copyright line was changed, no file was relicensed, and nothing was removed
from a header. The header itself says "See LICENSE in the root of the software
repository for the full text of the License", so the full text is vendored
alongside as `CANN-OSA-2.0-LICENSE.txt`, copied from the same commit.

Scope:

- the CANN Open Software License 2.0 covers the files under
  `domain_benchmarks/cann/*/official/`;
- the repository MIT license covers the AKB code (`src/`, `tests/`, task
  definitions, baselines, documentation);
- the two apply to their own files; nothing here re-licenses the other side.

## Purpose

The benchmark needs the official operator definition, reference, and public cases
to define each task and to check a candidate against the upstream math.

## Alternative the reviewer may prefer

If redistributing these files is not acceptable, they can instead be fetched at
setup time from the pinned commit and verified against
`domain_benchmarks/cann/task_inventory.json`, which records the SHA256 of every
file as published in the upstream `suite_manifest.json`. That change has not been
made, because it would add a network dependency to every run.
