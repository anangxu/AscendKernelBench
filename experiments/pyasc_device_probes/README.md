# pyasc device probes

Standalone scripts that re-measure the pyasc facts the engine relies on. They
exist so a reader can re-run the evidence behind
[`docs/reference/pyasc-device-evidence.md`](../../docs/reference/pyasc-device-evidence.md)
instead of trusting a report.

They are device-only: each needs an Ascend NPU, a sourced CANN environment, and
a Python with `torch`, `torch_npu` and `pyasc` installed. They are not part of
the offline test suite in `tests/`, and nothing in `src/` imports them.

## Running

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd experiments/pyasc_device_probes
python probe_r09_writeback_length.py          # one probe
for probe in probe_*.py; do python "$probe" || exit 1; done   # all of them
```

`_probe_common.py` holds the shared helpers; it is not a probe and prints
nothing when run. Each probe prints `PASS`/`FAIL` per expectation, then
`VERDICT: PASS` or `VERDICT: FAIL`, and exits non-zero on failure. No probe reads or writes any
path outside the repository: `probe_rowsum_shapes.py` imports the bundled example
from `src/prompts/examples/pyasc/003_rowsum/` by walking up from its own file.

Do not add `from __future__ import annotations` to a probe that defines an
`@asc.jit` kernel. pyasc inspects the kernel's annotations as objects and calls
`issubclass` on them, so stringified annotations fail with
`TypeError: issubclass() arg 1 must be a class`. This deviates from the
repository's style convention on purpose.

Expect a cold probe to spend a few seconds per compiled kernel: pyasc traces and
compiles on the first subscripted call.

## What each probe checks

| Probe | Expectation | Historical result |
| --- | --- | --- |
| `probe_s1_tque_copy_roundtrip.py` | TQue GM -> UB -> GM round trip is bit exact | passes |
| `probe_r01_rowsum_naive.py` | the naive explicit-address row sum does **not** produce a usable result: it returns wrong values, or it faults the vector core | fails both ways (cause unexplained) |
| `probe_r02_reduce_identity_input.py` | the R2 kernel does not reproduce the closed-form expectations | fails; its zero-initialised buffer cannot tell "wrote 0" from "wrote nothing" |
| `probe_r03_r02_plus_sync.py` | adding pipe sync alone still leaves the sentinel intact | sentinel survives |
| `probe_r04_explicit_copy.py` | with explicit addresses the copy round trip writes nothing | sentinel survives |
| `probe_r05_ub_reservation.py` | reserving UB alone still writes nothing | sentinel survives |
| `probe_r06_full_writeback.py` | enlarging the write-back produces a write, with uninitialised contents | writes garbage |
| `probe_r07_reduce_vs_expected.py` | with 64-byte write-backs `whole_reduce_sum` matches the closed-form sums | matches exactly |
| `probe_r08_multicore_vs_torch.py` | the multi-core row sum matches `torch.sum(dim=-1)` | matches exactly |
| `probe_r09_writeback_length.py` | a write-back below 32 bytes does not take effect | 4 B and 16 B silent, 32 B and 64 B exact |
| `probe_r10_pipe_event.py` | with the length fixed, `MTE2_MTE3` is exact where `MTE2_V` was not | exact |
| `probe_dtype_bypass.py` | pyasc rejects bfloat16/bool with no project pre-check in the way | rejected |
| `probe_rowsum_shapes.py` | representative accepted shapes match `torch.sum`, refused shapes raise | accepted exact, refused raise |

Probes that document a historical failure assert the **observation** (for
example "the sentinel survived"), not correctness, so a green verdict means the
finding reproduced. Their docstrings state which variable each one changes.

## Verdict vocabulary

`PASS`/`FAIL` is a correctness requirement. `EXPECTED_FAILURE_REPRODUCED` means a
known failure still happens, which is what the probes for R1 to R6, R9's
"does not take effect" cases and R10's `MTE2_V` case report. The summary counts
the two apart and prints "a reproduced failure is not a correctness pass", so a
row of green probes is not a row of correct operators.

## Known anomaly

A pure copy through a single VECIN queue carries **no MTE2 to MTE3 dependency**:
the IR handed to the compiler has no sync op in that function, and at 64 KiB the
idiom alone was wrong in 5 of 5 launches while the same kernel plus an allocated
`MTE2_MTE3` pair was exact in 10 of 10. The explicit-address path is exact at the
same sizes. A pre-launch `torch.npu.synchronize()` does not change it, so the
sentinel blocks are not a late PyTorch fill. A second, smaller manifestation
(stale or missing data at small write-backs) stays unexplained and parked, and a
stale value cannot be attributed to UB without a per-repetition marker. Details,
scope and what was excluded are in
[`docs/reference/pyasc-device-evidence.md`](../../docs/reference/pyasc-device-evidence.md).

## Bounded contrasts

`_ab_runner.sh` and `_ab_sync_runner.sh` are the A/B harnesses behind the
anomaly section; `_presync_contrast.py` tests the cross-stream ordering
explanation, `_tque_event_contrast.py` the missing in-kernel dependency, and
`_dump_ir.py` dumps the generated IR at both the codegen and compiler stages. They alternate two variants in fresh processes with the input,
shape, dtype, device, stream, synchronisation, sentinel, working directory and
cache pinned, and print one `METRICS` line per run. `_ab_entry_b.py` is the
second entry point, `_ab_sync_variant.py` the two synchronisation variants.
