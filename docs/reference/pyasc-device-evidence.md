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

## Open anomaly: TQue copies that carry no MTE2 to MTE3 dependency

Two manifestations are recorded separately. The first now has a supported
mechanism and a workaround; the second stays unexplained.

### A. The TQue copy emits no dependency between the inbound and outbound copy

The pure-copy idiom used by several probes copies GM to UB through a single
VECIN queue and then copies UB to GM. The IR handed to the compiler for that
function contains no `set_flag`, `wait_flag` or other sync op at all:

```text
%4 = ascendc.que_bind.alloc_tensor %3
ascendc.data_copy_l2 %4, %0, %c4096_i32      // GM -> UB, MTE2
ascendc.que_bind.enque_tensor %3, %4
%5 = ascendc.que_bind.deque_tensor %3
ascendc.data_copy_l2 %1, %5, %c4096_i32      // UB -> GM, MTE3
ascendc.que_bind.free_tensor %3, %5
```

`_dump_ir.py` hooks both `_run_codegen` and `_run_compiler`; both stages show
zero sync ops for this function. The explicit-address kernel, by contrast, hands
the compiler `ascendc.set_flag mte2_mte3` and `ascendc.wait_flag mte2_mte3`
between its two copies. Scope: this is the IR at the compiler boundary. The
`que_bind` ops could still be lowered into synchronised code inside the
compiler, so this is strong evidence rather than proof about the final binary.

Behaviour agrees. `_tque_event_contrast.py` changes exactly one thing, an
`MTE2_MTE3` flag pair whose id comes from `pipe.alloc_event_id(...)` so it cannot
collide with a framework-owned event, and runs each arm at the sizes that
trigger the dropped chunks, one sentinel value per repetition:

| Arm | 16 KiB, 5 launches | 64 KiB, 5 launches |
| --- | --- | --- |
| TQue idiom alone | 0 wrong | **5/5 wrong** (wrong counts 1920, 1280, 256, 4224, 2560) |
| plus an allocated `MTE2_MTE3` pair | 0 wrong | **0 wrong in all 10 launches** |

So the corruption is not `data_copy`, not the size on its own and not
undetermined: a pure copy through one queue carries no MTE2 to MTE3 dependency,
and adding one removes it in every launch of the comparison.

#### Cross-stream ordering was tested and excluded for this phenomenon

A plausible alternative was that PyTorch fills the output and the kernel launches
without being ordered against that fill, so a sentinel block would mean "the
fill landed last", not "the kernel never wrote". `_presync_contrast.py` adds one
line, `torch.npu.synchronize()` before the launch, and reads the buffer back to
prove the fill completed (`fill_complete=True` in every run). The dropped chunks
persist under that control, with residue values matching **other repetitions'
sentinels**, which is recycled memory rather than the current fill. The
alternative is therefore not the explanation here. It had to be tested, and it
was.

### B. A small write-back occasionally reads stale or missing data

Still unexplained and parked. Observed at small sizes (64 to 1024 elements):
`probe_s1` failed 10 consecutive launches in one shell loop while the same
kernel passed 9/9 standalone with different inputs; `probe_r08` failed 5/5 in
one session and passed 4/4 inside one process in another; `probe_r07`'s copy
control failed while its `whole_reduce_sum` matched the closed form; one
entry-point run wrote 64 wrong elements whose maximum error was exactly the
offset an earlier experiment had used.

Caveat on reading that fingerprint: stale numbers could come from a previous GM
input or from a reused output buffer, not necessarily from stale UB. Matching
values against a per-repetition marker, as the new contrasts do, is what
distinguishes them; the earlier records used a single sentinel and cannot.

### What the bounded contrasts support

| Hypothesis | Test | Outcome |
| --- | --- | --- |
| PyTorch fill not ordered against the launch | `_presync_contrast.py` | excluded for A: drops persist with the fill proven complete |
| Missing MTE2 to MTE3 dependency inside the kernel | `_dump_ir.py`, `_tque_event_contrast.py` | supported for A: no sync op in the IR, and an allocated event pair removes the corruption |
| Entry point (probe wrapper versus direct) | `_ab_runner.sh` | no difference observed: 20/20 versus 19/20 |
| Cache state | fresh `PYASC_CACHE_DIR` | changing the cache is not sufficient to remove it; this does not clear the cache of involvement |
| Queue depth and buffer byte count | not run | deliberately deferred: changing depth and size together moves the UB layout, the resource use and the timing at once |

### Impact on the benchmark

The direction is not only false negatives. A false positive needs a wrong kernel
to pass, and that can happen when a fixed seed regenerates the same inputs, when
a warm-up leaves a correct result in the buffer, when the task output is
constant or zero, or when a race simply does not fire during the trials. A
kernel that is genuinely missing synchronisation being judged wrong is a correct
rejection, not a false negative; only a harness that fails to order its own
inputs could cause those.

The five correctness trials and the hidden gates catch a flaky kernel but cannot
repair a race, and a diagnostic device-wide synchronise belongs in the
correctness path, not in the timing loop where it would change the measured
protocol.

### Status

* **A**: mechanism supported, workaround known. Kernels whose only work is a copy
  need an explicit, allocated `MTE2_MTE3` wait before the write-back, or the
  copy has to keep a vector operation between the two copies as the reduction
  and element-wise examples do.
* **B**: parked, with the fingerprint caveat above.
* The compiler-boundary IR question stays open: do the `que_bind` ops lower into
  synchronised code inside the compiler, and if not, should they?

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
