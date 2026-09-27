# pyasc device evidence

The measured facts the pyasc backend relies on, with the versions, commands,
inputs, parameters, observed outputs, and boundaries behind each one. The
runnable form of every experiment lives in
[`experiments/pyasc_device_probes/`](../../experiments/pyasc_device_probes/).

This page exists because the backend refuses, times, and documents things a
reader cannot check without a device. It separates what was measured from what
is assumed, and it keeps one experiment explicitly unexplained.

## Device and software

| Item | Value |
| --- | --- |
| Device | Ascend 910B4, 32 GB HBM, one visible device |
| CANN | 9.1.0 |
| torch / torch_npu | 2.10.0 / 2.10.0 |
| pyasc | 1.1.1 (PyPI; import name `asc`) |
| Python | 3.12 |
| Platform selection | `asc.runtime.config.set_platform(asc.runtime.config.Backend.NPU, None, device_id=0)` |

Nothing here was measured on the simulator backend. The simulator path in
`docs/guide/pyasc-backend.md` is a separate, weaker check.

## Which code the records belong to

| Record | Code under test |
| --- | --- |
| Historical runs | pyasc 1.1.1 itself, exercised by scripts that lived outside the repository while the backend was being added. The engine side of those findings landed in commits `8b500bb`..`8a5792c`. |
| Re-run (this round) | The probes as committed under `experiments/pyasc_device_probes/`, on a checkout at or after `8e1d211`. `probe_rowsum_shapes.py` additionally tests the shipped example `src/prompts/examples/pyasc/003_rowsum/`, so its result tracks that file's revision. |

The pyasc facts themselves are properties of the installed package; a change in
this repository does not move them. Where a fact constrains repository code, the
section below says which file carries the constraint.

## Running a probe

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd experiments/pyasc_device_probes
python probe_r09_writeback_length.py
for probe in probe_*.py; do python "$probe" || exit 1; done
```

Each probe prints `PASS`/`FAIL` per expectation and a final `VERDICT`, exiting
non-zero when any expectation failed. A probe that documents a historical
failure asserts the observation, so a green verdict means the finding
reproduced, not that pyasc is correct.

Probe modules that define an `@asc.jit` kernel must not use
`from __future__ import annotations`: pyasc reads the kernel's annotations as
objects and calls `issubclass` on them, so stringified annotations fail with
`TypeError: issubclass() arg 1 must be a class`. Kernels are also defined at
module level, because pyasc resolves names such as `asc` from the defining
module's globals rather than from an enclosing function's locals.

## Verification status

The probes were re-run on a rebuilt container (the instance was reclaimed and
recreated between sessions; the venv was rebuilt with pyasc 1.1.1 and the
versions below were re-confirmed). Results:

| Probe | Historical | Re-run |
| --- | --- | --- |
| `probe_dtype_bypass.py` | bfloat16 and bool rejected | PASS |
| `probe_rowsum_shapes.py` | accepted shapes matched, refused shapes raised | PASS (15/15, twice) |
| `probe_r02_reduce_identity_input.py` | reproduced the failure | PASS |
| `probe_r03_r02_plus_sync.py` | sentinel survived | PASS |
| `probe_r04_explicit_copy.py` | sentinels survived | PASS |
| `probe_r05_ub_reservation.py` | sentinels survived | PASS |
| `probe_r06_full_writeback.py` | wrote uninitialised UB | PASS |
| `probe_r09_writeback_length.py` | 4/16 B silent, 32/64 B exact | PASS |
| `probe_r10_pipe_event.py` | `MTE2_MTE3` exact, `MTE2_V` not | PASS |
| `probe_r01_rowsum_naive.py` | wrong values | PASS both ways: returned wrong values once and faulted the vector core once |
| `probe_s1_tque_copy_roundtrip.py` | passed | **intermittent** (see below) |
| `probe_r07_reduce_vs_expected.py` | matched the closed form | **intermittent** control; the reduce matched in every run |
| `probe_r08_multicore_vs_torch.py` | matched `torch.sum` exactly | **intermittent** (see below) |

## Open anomaly: two TQue copy failure modes

The TQue copy path produced wrong output in several sessions while the same
sizes were exact through explicit addresses. Two manifestations were seen, and
they are recorded separately because they have different evidence.

### A. A write-back of roughly 16 KiB or more drops whole chunks

Provoked on demand, so this needs counts rather than rate studies. Each line is
five launches of a copy within one process, reporting how many output elements
were still holding the sentinel (never written):

| Write-back size | TQue (`enque`/`deque`) missing per launch | Explicit address + `MTE2_MTE3` |
| --- | --- | --- |
| 16 KiB | `[0, 0, 1032, 0, 1032]` | `[0, 0, 0, 0, 0]` |
| 32 KiB | `[256, 256, 0, 256, 0]` | not run |
| 64 KiB | `[128, 0, 128, 0, 128]` | `[0, 0, 0, 0, 0]` |
| 128 KiB | `[0, 0, 0, 0, 0]` once, `3072` missing in an earlier single launch | `[0, 0, 0, 0, 0]` |

The lost elements are contiguous chunks that never get written, and the loss is
not deterministic per launch. The explicit-address path was exact in all fifteen
launches at the same sizes, with the same input, dtype, device, stream and
sentinel, so the buffer-management style is the variable, not `data_copy` and
not the size on its own.

The bundled example `003_rowsum` was then run ten times at its own shape
(512x64, a 2 KiB write-back) through its own launcher: 10/10 exact, no sentinel
left. Its write-back sits below the size class that dropped chunks; that is a
statement about this shape, not a guarantee for other shapes.

### B. A small write-back occasionally reads stale or missing data

This is the original observation and it is still unexplained. Across the first
sessions the following were seen at small sizes (64 to 1024 elements):

* `probe_s1` failed 10 consecutive launches in one shell loop, with zeros, stale
  values, or garbage; the same kernel passed 9/9 as a standalone script with
  different inputs, each launch returning its own input.
* `probe_r08` failed 5/5 with a stable error in one session and passed 4/4 in
  one process in another.
* `probe_r07`'s copy control failed while its `whole_reduce_sum` matched the
  closed form every time.
* One entry-point run wrote 64 wrong elements with a maximum error of exactly
  the offset an earlier experiment had used, which is what stale UB content
  looks like.
* A priming probe (launch A, then launch B with different values) returned
  `arange(64)` on its first iteration, which was neither input but the residue
  of an earlier process. That run followed a killed runner, so it is recorded as
  a lead, not as clean evidence.

### What the bounded contrasts support

**Entry point.** `_ab_runner.sh` alternated the probe entry (A) and a direct
entry (B) that imports the same kernel, pins input, shape, dtype, device,
stream, synchronisation, sentinel, working directory and cache, and runs each in
a fresh process: A 20/20 exact, B 19/20. One failure in forty does not separate
the entries, so the entry point is not implicated. This also says nothing about
the device: two entries behaving alike would not establish the device as the
cause.

**Synchronisation.** `_ab_sync_runner.sh` contrasted the TQue idiom with the
same kernel plus an explicit `MTE2_MTE3` pair, and was stopped part way at
plain 27/27 and event 26/26. That contrast was underpowered by construction: a
rare event cannot be resolved by running more launches of it. It was abandoned
in favour of the size provocation above, which reproduces the defect in five
launches.

**Cache.** A fresh `PYASC_CACHE_DIR` reproduced both outcomes, so changing the
cache is not sufficient to remove the problem. That is weaker than clearing the
cache of involvement, which was not tested.

### Status and next variable

Unresolved and parked: manifestation B, and the mechanism behind A (whether the
queue's buffer accounting, its synchronisation, or its DMA split is at fault).
Neither is a claim that pyasc is defective, and no rate is quoted for either.

The next variable, if this is picked up again, is the TQue buffer accounting at
large tiles: hold the kernel and the sizes fixed and change only the queue depth
and the `init_buffer` byte count (`asc.TQue(pos, 1)` with `n * 4` versus depth 2
with the matching count), then re-run the five-launch size sweep above. The
explicit-address path is already exact at these sizes, so it is the reference
point for what correct looks like.

## The experiments

### S1: TQue copy round trip

Input `x = arange(64)` fp32, output prefilled with `-1`, single core. TQue
buffers, `alloc_tensor` -> `data_copy` -> `enque` -> `deque` -> `data_copy`.
Observed: the sentinel was overwritten and the round trip was bit exact.

Boundary: this proves the move path works in the TQue style. It says nothing
about the explicit-address style, because buffer management and synchronisation
differ at the same time.

### R1: naive explicit-address row sum

Input `x = torch.rand(128, 64)` fp32 (the historical run did not seed;
`probe_r01_rowsum_naive.py` seeds with 0), reference `torch.sum(x, dim=-1)`,
output `torch.empty(128)`. Four cores, `rows_per_core=32`, `cols=64`, `mask=64`,
`repeat_time=32`, `dst_rep_stride=1`, `src_blk_stride=1`, `src_rep_stride=8`,
`x_local` VECIN addr 0 len 2048, `y_local` VECOUT addr 8192 len 32.

Observed: `max_abs_diff = 49.2771`, outputs not matching the reference.

**R1 remains unexplained.** The write-back is 128 bytes per core, so the
sub-32-byte granule measured in R9 does not apply; the output buffer was
`torch.empty`, so the "four identical values" detail came from uninitialised
memory and is not a stable signal. R7 and R8 passing do not retroactively
explain R1, and this page does not claim a pyasc defect from it.

### R2: identifiable input, zero-initialised output

Input `x[r, c] = r * 1000 + c` (deterministic, 4x64). Same parameters as R1 with
`repeat_time=4`, single core. Contiguous-mask prediction `[2016, 66016, 130016,
194016]`; per-bit-mask prediction `[6, 1006, 2006, 3006]`. Observed: `[0, 0, 0, 0]`.

Boundary: the output buffer was `torch.zeros`, so the run **cannot** distinguish
"wrote 0" from "wrote nothing". Read R3 for the informative version of the same
kernel. This is a defect in the original record, not in pyasc.

### R3: R2 plus pipe synchronisation

Only change against R2: `set_flag`/`wait_flag` pairs on `MTE2_V` and `V_MTE3`;
the output becomes a `-1` sentinel. Observed: `[-1, -1, -1, -1]`, the sentinel
survived.

Boundary: adding sync is **not sufficient**. This does not show that sync is
irrelevant, and it does not establish a necessary condition.

### R4: explicit-address copy only

The reduce is removed. `R4` reads four elements back from `y_local` (VECOUT
addr 1024); `R4s` reads them from `x_local` (VECIN addr 0) so a wrong output
address cannot masquerade as a failed move. Observed: both sentinels survived.

Boundary: the move writes nothing in this style, at this length. The length
matters; see R9.

### R5: reserve UB explicitly

Only change against R4s and R3: `asc.LocalMemAllocator().alloc(...)` reserves UB.
A bare `asc.LocalTensor(dtype, pos, addr, len)` only emits an address
(`LocalTensorV2Op`) and reserves nothing. Observed: both sentinels survived.

### R6: enlarge the write-back

`R6a` = R5a with the write-back enlarged from 4 fp32 (16 B) to the whole 256 fp32
(1 KiB); `R6b` = the TQue reduce still writing 4 fp32 back. Observed: R6a moved
off the sentinel but held uninitialised UB (`head` not `[0..5]`); R6b left the
sentinel intact.

Boundary: R6a is the hint that the copy was synchronised against the wrong pipe.
It is not that result on its own, because R6a and R8 also differ in length; R10
settles it. R6b shows a 16-byte write-back hides the reduce entirely.

### R7: reduce with an adequate write-back

`repeats=16`, `cols=64` (64-byte write-backs), same input construction as R2.
The control copies the tile in and the first 16 elements back; the second call
adds `whole_reduce_sum`. Observed: the control returned `0..15`; the reduce
returned `[2016, 66016, ..., 962016]`, exactly the closed form.

Established semantics, all measured:

* the reduction is per repeat, so `repeat_time=R` produces `R` outputs;
* in the contiguous mask mode `mask` is an element count, not a bit mask;
* `src_rep_stride` counts 32-byte data blocks of the source, `dst_rep_stride`
  counts destination elements.

### R8: multi-core against torch

Rows 128/cores 4 and 256/cores 8, `torch.rand` seeded with 0, reference
`torch.sum(x, dim=-1)`. Observed: `max_abs_diff = 0` in both configurations. The
same file also copies a 1024-element tile through explicit addresses with
`MTE2_MTE3` and matches bit for bit.

Boundary: the engine keeps a 1e-4 tolerance because pairwise tree accumulation
need not agree with torch bit for bit at other sizes. Exactness here is a
property of these shapes.

### R9: write-back length sweep

Only the write-back length changes, from the same UB address into four sentinel
buffers: 1, 4, 8 and 16 fp32. Observed on this device, in both the TQue and the
explicit-address styles: 4 B and 16 B did not take effect; 32 B and 64 B were
exact.

Boundary: this is one device, one pyasc release, one call style. It is not a
documented property of `data_copy`. The engine treats it as a reason to keep
write-backs at 32 bytes or more (`docs/guide/pyasc-backend.md`, and the prompt
contract in `prompt.py`).

### R10: pipe event with the length held fixed

`n=256` for both variants, identical addresses and write-back length, only the
event differs. Observed: `MTE2_V` produced uninitialised UB; `MTE2_MTE3` was bit
exact. This is what turns R6a's hint into a single-variable result.

Boundary: explicit-address style with hand-written flags only. The TQue idiom
emits its own synchronisation.

### dtype bypass

No engine pre-check is involved: the probe imports `asc` and launches a kernel
whose dtype comes from the tensor. Observed: `float32` launched and was exact;
`bfloat16` and `bool` raised `CodegenError: ValueError: Unsupported DataType
name: <dtype>` from pyasc's own codegen. The bundled dtype factory table in
`asc/language/core/dtype.py` registers `int8/16/32/64`, `uint8/16/32/64` and
`float16/32/64` only, and raises for anything else.

This is the evidence that the limitation is backend-wide. Two different
operators rejected by the engine's pre-check prove only that the pre-check is
consistent, because the pre-check fires before any JIT call.

### rowsum example shapes

The probe imports the shipped example `src/prompts/examples/pyasc/003_rowsum/`
from the repository and exercises representative shapes across the range the
launcher accepts, samples outside it, and one unaligned column count called
directly. Observed: every accepted case matched `torch.sum(dim=-1)` with
`max_abs_diff = 0`; every out-of-range case raised `ValueError`; the direct
`cols=12` call did **not** match, by order 4, because `src_rep_stride = cols*4/32`
rounds down to one block and the reduction then reads the wrong elements.

Boundary: representative, not exhaustive. The accepted range is `cols` in
`{8..64}` that are multiples of 8 and per-core row blocks that are multiples of
8; the probe samples that range rather than enumerating it.

## What is not claimed

1. **R1's cause.** It stays unexplained, and no pyasc defect is asserted from it.
2. **Sync as a non-factor.** R3 shows only that adding sync is not sufficient.
   R7/R8 changed the write-back length and the event together, so they do not
   separate the two either.
3. **The 32-byte floor as a specification.** It is a measurement on one device,
   one release, one call style.
4. **Generation quality.** Every fact here comes from hand-written kernels. What
   an LLM writes with pyasc is a separate, unmeasured question.
5. **Cross-platform comparison.** Nothing on this page is a GPU-versus-NPU
   result.
