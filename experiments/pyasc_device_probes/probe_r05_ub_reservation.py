"""R5: does reserving UB make the explicit-address path work?

One variable against the R4s and R3 baselines: an explicit UB reservation via
asc.LocalMemAllocator().alloc is added. A bare asc.LocalTensor(dtype, pos, addr,
len) only emits an address (LocalTensorV2Op) and does not reserve UB.

  R5a = R4s plus a reservation (copy round trip)
  R5b = R3  plus a reservation (copy plus whole_reduce_sum)

Input construction: x[r, c] = r * 1000 + c, no RNG, sentinel outputs.

Expected: both sentinels survive, so the reservation is not the missing piece
on its own. Failure criterion: any element differs from -1.
"""

import torch
import torch_npu  # noqa: F401

import asc
import asc.lib.runtime as rt

from _probe_common import expect, run, setup_device

COLS = 64
REPEATS = 4

@asc.jit
def copy_with_alloc(
    x: asc.GlobalAddress,
    y: asc.GlobalAddress,
    repeats: asc.ConstExpr[int],
    cols: asc.ConstExpr[int],
):
    x_gm = asc.GlobalTensor()
    y_gm = asc.GlobalTensor()
    x_gm.set_global_buffer(x, repeats * cols)
    y_gm.set_global_buffer(y, repeats)

    alloc = asc.LocalMemAllocator()
    alloc.alloc(asc.TPosition.VECIN, asc.float32, repeats * cols)
    alloc.alloc(asc.TPosition.VECOUT, asc.float32, repeats * cols)

    x_local = asc.LocalTensor(asc.float32, asc.TPosition.VECIN, 0, repeats * cols)
    asc.data_copy(x_local, x_gm, repeats * cols)
    asc.set_flag(asc.HardEvent.MTE2_V, 0)
    asc.wait_flag(asc.HardEvent.MTE2_V, 0)
    asc.data_copy(y_gm, x_local, repeats)

@asc.jit
def reduce_with_alloc(
    x: asc.GlobalAddress,
    y: asc.GlobalAddress,
    repeats: asc.ConstExpr[int],
    cols: asc.ConstExpr[int],
):
    x_gm = asc.GlobalTensor()
    y_gm = asc.GlobalTensor()
    x_gm.set_global_buffer(x, repeats * cols)
    y_gm.set_global_buffer(y, repeats)

    alloc = asc.LocalMemAllocator()
    alloc.alloc(asc.TPosition.VECIN, asc.float32, repeats * cols)
    alloc.alloc(asc.TPosition.VECOUT, asc.float32, repeats * cols)

    x_local = asc.LocalTensor(asc.float32, asc.TPosition.VECIN, 0, repeats * cols)
    y_local = asc.LocalTensor(asc.float32, asc.TPosition.VECOUT, repeats * cols * 4, repeats)

    asc.data_copy(x_local, x_gm, repeats * cols)
    asc.set_flag(asc.HardEvent.MTE2_V, 0)
    asc.wait_flag(asc.HardEvent.MTE2_V, 0)
    asc.whole_reduce_sum(
        y_local,
        x_local,
        mask=cols,
        repeat_time=repeats,
        dst_rep_stride=1,
        src_blk_stride=1,
        src_rep_stride=cols // 8,
    )
    asc.set_flag(asc.HardEvent.V_MTE3, 0)
    asc.wait_flag(asc.HardEvent.V_MTE3, 0)
    asc.data_copy(y_gm, y_local, repeats)

def body() -> None:
    setup_device()

    rows = torch.arange(REPEATS, dtype=torch.float32).view(-1, 1) * 1000.0
    cols = torch.arange(COLS, dtype=torch.float32).view(1, -1)
    x = (rows + cols).to("npu")
    stream = rt.current_stream()

    y = torch.full((REPEATS,), -1.0, dtype=torch.float32, device="npu")
    copy_with_alloc[1, stream](x, y, REPEATS, COLS)
    rt.synchronize()
    torch.npu.synchronize()
    got = [round(v, 3) for v in y.cpu().tolist()]
    print(f"     R5a copy+alloc   got={got}")
    expect("R5a sentinel survives", got == [-1.0] * REPEATS, f"got={got}")

    y2 = torch.full((REPEATS,), -1.0, dtype=torch.float32, device="npu")
    reduce_with_alloc[1, stream](x, y2, REPEATS, COLS)
    rt.synchronize()
    torch.npu.synchronize()
    got2 = [round(v, 3) for v in y2.cpu().tolist()]
    print(f"     R5b reduce+alloc got={got2}")
    expect("R5b sentinel survives", got2 == [-1.0] * REPEATS, f"got={got2}")

if __name__ == "__main__":
    raise SystemExit(run("R5 UB reservation", body))
