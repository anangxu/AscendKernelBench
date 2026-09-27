"""R4: does the explicit-address copy round trip write anything at all?

One variable against R3: whole_reduce_sum is removed, so the test is a pure
move. Two variants isolate the output address:

  R4  reads the 4 elements back from y_local (VECOUT addr 1024), the address
      R2/R3 used for the reduce output;
  R4s reads them from x_local (VECIN addr 0), the buffer the inbound copy
      filled, so a wrong output address cannot masquerade as a failed move.

Input construction: x[r, c] = r * 1000 + c over a 4x64 tile, no RNG. Both
outputs are prefilled with -1.

Expected: both sentinels survive. Failure criterion: any output element differs
from -1, which would mean the explicit-address move does work and the earlier
silence came from somewhere else.

Boundary: this and R3/R5 use a 4-element (16-byte) write-back, which
probe_r09_writeback_length.py later shows is below the write-back granule on
this device. The two findings are separate: R4 removes the reduce, R9 varies
only the length.
"""

import torch
import torch_npu  # noqa: F401

import asc
import asc.lib.runtime as rt

from _probe_common import expect, run, setup_device

COLS = 64
REPEATS = 4

@asc.jit
def copy_via_vecout(
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
    asc.set_flag(asc.HardEvent.MTE2_V, 0)
    asc.wait_flag(asc.HardEvent.MTE2_V, 0)
    asc.data_copy(y_gm, y_local, repeats)

@asc.jit
def copy_via_vecin(
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

    asc.data_copy(x_local, x_gm, repeats * cols)
    asc.set_flag(asc.HardEvent.MTE2_V, 0)
    asc.wait_flag(asc.HardEvent.MTE2_V, 0)
    asc.data_copy(y_gm, x_local, repeats)

def body() -> None:
    setup_device()

    rows = torch.arange(REPEATS, dtype=torch.float32).view(-1, 1) * 1000.0
    cols = torch.arange(COLS, dtype=torch.float32).view(1, -1)
    x = (rows + cols).to("npu")
    stream = rt.current_stream()

    y = torch.full((REPEATS,), -1.0, dtype=torch.float32, device="npu")
    copy_via_vecout[1, stream](x, y, REPEATS, COLS)
    rt.synchronize()
    torch.npu.synchronize()
    got = [round(v, 3) for v in y.cpu().tolist()]
    print(f"     R4  dst=VECOUT@1024 got={got}")
    expect("R4 sentinel survives", got == [-1.0] * REPEATS, f"got={got}")

    y2 = torch.full((REPEATS,), -1.0, dtype=torch.float32, device="npu")
    copy_via_vecin[1, stream](x, y2, REPEATS, COLS)
    rt.synchronize()
    torch.npu.synchronize()
    got2 = [round(v, 3) for v in y2.cpu().tolist()]
    print(f"     R4s src=VECIN@0     got={got2} (expected [0, 1, 2, 3] if the move worked)")
    expect("R4s sentinel survives", got2 == [-1.0] * REPEATS, f"got={got2}")

if __name__ == "__main__":
    raise SystemExit(run("R4 explicit-address copy", body))
