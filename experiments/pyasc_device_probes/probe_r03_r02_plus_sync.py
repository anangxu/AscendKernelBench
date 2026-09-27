"""R3: R2 verbatim plus the pipe synchronisation the explicit style lacks.

One variable against R2: two flag pairs are added between data_copy and
whole_reduce_sum (MTE2_V) and between whole_reduce_sum and the write-back
(V_MTE3). Addresses, layout, mask, strides, repeat and input are unchanged.

The output buffer becomes a sentinel rather than zeros, so the run can tell a
write from no write. That is a fidelity correction to R2, not an additional
variable under test.

Expected: the sentinel survives, i.e. adding sync alone does not make the
output appear. Failure criterion: any element differs from -1.

Boundary: this shows sync is not sufficient. It does NOT show that sync is
irrelevant, and it does not establish a necessary condition.
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
    y = torch.full((REPEATS,), -1.0, dtype=torch.float32, device="npu")
    probe_kernel[1, rt.current_stream()](x, y, REPEATS, COLS)
    rt.synchronize()
    torch.npu.synchronize()

    got = [round(v, 3) for v in y.cpu().tolist()]
    print(f"     got = {got}")
    expect("the sentinel survives", got == [-1.0] * REPEATS, f"got={got}")

if __name__ == "__main__":
    raise SystemExit(run("R3 R2 plus sync", body))
