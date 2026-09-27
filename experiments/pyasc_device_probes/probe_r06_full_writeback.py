"""R6: enlarge the write-back, and keep the reduce under the TQue idiom.

Two changes, each against its own baseline:

  R6a = R5a with the write-back enlarged from 4 fp32 (16 B) to the whole 256
        fp32 (1 KiB) tile. One variable against R5a: the length.
  R6b = the TQue idiom with whole_reduce_sum added back, but still writing back
        4 fp32. It documents that a 16-byte write-back hides the reduce result
        and therefore says nothing about whole_reduce_sum.

Input construction: x[r, c] = r * 1000 + c, no RNG, sentinel outputs.

Expected:
  R6a writes something (the buffer moves off the sentinel) but its contents are
  uninitialised UB, i.e. the head is not [0, 1, 2, 3, 4, 5]. That is the hint
  that the copy was synchronised against the wrong pipe; R10 turns the hint into
  a single-variable result.
  R6b leaves the sentinel intact.
Failure criterion: R6a leaves the sentinel or looks exact, or R6b writes.
"""

import torch
import torch_npu  # noqa: F401

import asc
import asc.lib.runtime as rt

from _probe_common import check, reproduced, run, setup_device

COLS = 64
REPEATS = 4
N = REPEATS * COLS

@asc.jit
def raw_full_writeback(
    x: asc.GlobalAddress,
    y: asc.GlobalAddress,
    repeats: asc.ConstExpr[int],
    cols: asc.ConstExpr[int],
):
    x_gm = asc.GlobalTensor()
    y_gm = asc.GlobalTensor()
    x_gm.set_global_buffer(x, repeats * cols)
    y_gm.set_global_buffer(y, repeats * cols)
    x_local = asc.LocalTensor(asc.float32, asc.TPosition.VECIN, 0, repeats * cols)
    asc.data_copy(x_local, x_gm, repeats * cols)
    asc.set_flag(asc.HardEvent.MTE2_V, 0)
    asc.wait_flag(asc.HardEvent.MTE2_V, 0)
    asc.data_copy(y_gm, x_local, repeats * cols)

@asc.jit
def tque_reduce_small_writeback(
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

    y = torch.full((N,), -1.0, dtype=torch.float32, device="npu")
    raw_full_writeback[1, stream](x, y, REPEATS, COLS)
    rt.synchronize()
    torch.npu.synchronize()
    got = y.cpu()
    head = [round(v, 3) for v in got[:6].tolist()]
    print(f"     R6a head={head}")
    check("R6a writes something", not bool((got == -1.0).all()))
    reproduced(
        "R6a contents are not the input (uninitialised UB)",
        head != [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        f"head={head}",
    )

    y2 = torch.full((REPEATS,), -1.0, dtype=torch.float32, device="npu")
    tque_reduce_small_writeback[1, stream](x, y2, REPEATS, COLS)
    rt.synchronize()
    torch.npu.synchronize()
    got2 = [round(v, 3) for v in y2.cpu().tolist()]
    print(f"     R6b got={got2}")
    reproduced(
        "R6b leaves the sentinel intact (a 16-byte write-back hides the reduce)",
        got2 == [-1.0] * REPEATS,
        f"got={got2}",
    )

if __name__ == "__main__":
    raise SystemExit(run("R6 full write-back", body))
