"""R7: with 64-byte write-backs, does whole_reduce_sum produce the exact sums?

repeats=16 and cols=64 make every write-back 64 bytes, clear of the granule
that probe_r09_writeback_length.py pins down.

Input construction: x[r, c] = r * 1000 + c, no RNG. The closed form for a
per-repeat contiguous-mask reduction with src_rep_stride=8 blocks is
r * 64000 + 2016, which is also what torch.sum(x, dim=-1) gives for this input.

Key parameters: mask=64, repeat_time=16, dst_rep_stride=1, src_blk_stride=1,
src_rep_stride=cols // 8 = 8.

Two calls:
  r7_ctl copies the tile in and copies the first 16 elements of the same buffer
  back, so the write-back size itself is proven before the reduce is judged.
  r7_red adds whole_reduce_sum.

Expected: the control returns 0..15 exactly; the reduce returns the closed form
exactly. Failure criterion: any element differs.

This is what pins the three semantics the engine documents: the reduction is
per repeat, the contiguous mask counts elements rather than bits, and
src_rep_stride advances in 32-byte blocks.
"""

import torch
import torch_npu  # noqa: F401

import asc
import asc.lib.runtime as rt

from _probe_common import check, run, setup_device

COLS = 64
REPEATS = 16
OUT_N = REPEATS
N = REPEATS * COLS

@asc.jit
def control(
    x: asc.GlobalAddress,
    y: asc.GlobalAddress,
    n: asc.ConstExpr[int],
    out_n: asc.ConstExpr[int],
):
    x_gm = asc.GlobalTensor()
    y_gm = asc.GlobalTensor()
    x_gm.set_global_buffer(x, n)
    y_gm.set_global_buffer(y, out_n)

    pipe = asc.TPipe()
    qin = asc.TQue(asc.TPosition.VECIN, 1)
    pipe.init_buffer(qin, 1, n * 4)

    xl = qin.alloc_tensor(asc.float32)
    asc.data_copy(xl, x_gm, n)
    qin.enque(xl)
    xin = qin.deque(asc.float32)
    asc.data_copy(y_gm, xin, out_n)
    qin.free_tensor(xin)

@asc.jit
def reduce(
    x: asc.GlobalAddress,
    y: asc.GlobalAddress,
    repeats: asc.ConstExpr[int],
    cols: asc.ConstExpr[int],
):
    x_gm = asc.GlobalTensor()
    y_gm = asc.GlobalTensor()
    x_gm.set_global_buffer(x, repeats * cols)
    y_gm.set_global_buffer(y, repeats)

    pipe = asc.TPipe()
    qin = asc.TQue(asc.TPosition.VECIN, 1)
    qout = asc.TQue(asc.TPosition.VECOUT, 1)
    pipe.init_buffer(qin, 1, repeats * cols * 4)
    pipe.init_buffer(qout, 1, repeats * 4)

    xl = qin.alloc_tensor(asc.float32)
    asc.data_copy(xl, x_gm, repeats * cols)
    qin.enque(xl)
    xin = qin.deque(asc.float32)

    yl = qout.alloc_tensor(asc.float32)
    asc.whole_reduce_sum(
        yl,
        xin,
        mask=cols,
        repeat_time=repeats,
        dst_rep_stride=1,
        src_blk_stride=1,
        src_rep_stride=cols // 8,
    )
    qout.enque(yl)
    yout = qout.deque(asc.float32)

    asc.data_copy(y_gm, yout, repeats)
    qin.free_tensor(xin)
    qout.free_tensor(yout)

def body() -> None:
    setup_device()

    rows = torch.arange(REPEATS, dtype=torch.float32).view(-1, 1) * 1000.0
    cols = torch.arange(COLS, dtype=torch.float32).view(1, -1)
    x = (rows + cols).to("npu")
    stream = rt.current_stream()

    y = torch.full((OUT_N,), -1.0, dtype=torch.float32, device="npu")
    control[1, stream](x, y, N, OUT_N)
    rt.synchronize()
    torch.npu.synchronize()
    got = [round(v, 3) for v in y.cpu().tolist()]
    print(f"     control got={got}")
    check(
        "the 64-byte write-back is exact",
        got == [float(i) for i in range(REPEATS)],
        f"got={got}",
    )

    y2 = torch.full((REPEATS,), -1.0, dtype=torch.float32, device="npu")
    reduce[1, stream](x, y2, REPEATS, COLS)
    rt.synchronize()
    torch.npu.synchronize()
    got2 = [round(v, 3) for v in y2.cpu().tolist()]
    want = [float(r * 64000 + 2016) for r in range(REPEATS)]
    print(f"     reduce  got={got2}")
    print(f"     want       ={want}")
    check("whole_reduce_sum matches the closed form exactly", got2 == want)

if __name__ == "__main__":
    raise SystemExit(run("R7 reduce with adequate write-back", body))
