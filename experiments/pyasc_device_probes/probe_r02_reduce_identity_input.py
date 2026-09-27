"""R2: the same explicit-address reduce with a value that identifies the mode.

Single variable against R1: the input becomes r * 1000 + c so that each
hypothesis predicts a different, checkable output.

Input construction: x[r, c] = r * 1000 + c, that is
row = arange(4).view(-1, 1) * 1000.0, col = arange(64).view(1, -1),
x = (row + col).to("npu"). Deterministic, no RNG.

Key parameters: repeats=4, cols=64, mask=64, repeat_time=4, dst_rep_stride=1,
src_blk_stride=1, src_rep_stride=8, x_local VECIN addr 0 len 256,
y_local VECOUT addr 1024 len 4, single core.

Expected if the contiguous mask honours the strides: [2016, 66016, 130016,
194016]. Expected for a per-bit mask (bit 6): [6, 1006, 2006, 3006].

The output buffer is torch.zeros, kept from the historical run for fidelity.
That choice makes the run uninformative about writing: an all-zero result
cannot distinguish "wrote 0" from "wrote nothing". probe_r03_r02_plus_sync.py
repeats this kernel with a sentinel buffer, which is the version that carries
information. The verdict here is green when neither expectation is met.
"""

import torch
import torch_npu  # noqa: F401

import asc
import asc.lib.runtime as rt

from _probe_common import expect, run, setup_device

COLS = 64
REPEATS = 4

@asc.jit
def probe_kernel(
    x: asc.GlobalAddress,
    y: asc.GlobalAddress,
    repeats: asc.ConstExpr[int],
    cols: asc.ConstExpr[int],
):
    x_gm = asc.GlobalTensor()
    y_gm = asc.GlobalTensor()
    x_gm.set_global_buffer(x, repeats * cols)
    y_gm.set_global_buffer(y, repeats)
    x_local = asc.LocalTensor(asc.float32, asc.TPosition.VECIN, 0, repeats * cols)
    y_local = asc.LocalTensor(asc.float32, asc.TPosition.VECOUT, repeats * cols * 4, repeats)

    asc.data_copy(x_local, x_gm, repeats * cols)
    asc.whole_reduce_sum(
        y_local,
        x_local,
        mask=cols,
        repeat_time=repeats,
        dst_rep_stride=1,
        src_blk_stride=1,
        src_rep_stride=cols // 8,
    )
    asc.data_copy(y_gm, y_local, repeats)

def body() -> None:
    setup_device()

    rows = torch.arange(REPEATS, dtype=torch.float32).view(-1, 1) * 1000.0
    cols = torch.arange(COLS, dtype=torch.float32).view(1, -1)
    x = (rows + cols).to("npu")
    y = torch.zeros(REPEATS, dtype=torch.float32, device="npu")
    probe_kernel[1, rt.current_stream()](x, y, REPEATS, COLS)
    rt.synchronize()
    torch.npu.synchronize()

    got = [round(v, 3) for v in y.cpu().tolist()]
    contiguous = [r * 64000 + 2016 for r in range(REPEATS)]
    bitmask = [r * 1000 + 6 for r in range(REPEATS)]
    print(f"     got        = {got}")
    print(f"     contiguous = {contiguous}")
    print(f"     bitmask    = {bitmask}")
    expect("contiguous-mask expectation is not met", got != [float(v) for v in contiguous])
    expect("per-bit-mask expectation is not met", got != [float(v) for v in bitmask])
    expect(
        "the zero-initialised buffer leaves the write ambiguous",
        all(v == 0.0 for v in got),
        "an all-zero result is consistent with both 'wrote 0' and 'wrote nothing'",
    )

if __name__ == "__main__":
    raise SystemExit(run("R2 identifiable-input reduce", body))
