"""R8: the multi-core row sum against torch.sum(dim=-1).

Takes the R7 kernel to multiple cores with asc.get_block_idx() and a GM offset
per core. Two shape configurations are exercised, plus a raw explicit-address
copy round trip at full length.

Input construction: torch.rand(rows, cols) fp32 on the NPU, seeded with 0.

Key parameters: cols=64, rows_per_core = rows // cores, mask=64,
repeat_time=rows_per_core, dst_rep_stride=1, src_blk_stride=1,
src_rep_stride=8. Write-back per core is rows_per_core fp32, which is 32 bytes
or more for every configuration here.

Expected: both configurations match torch.sum(dim=-1) within 1e-4, and the
explicit-address full-length copy round trip is bit exact. Failure criterion:
any mismatch.

Boundary: this is the shape family the bundled example accepts; the tolerance is
kept because pairwise tree accumulation need not agree with torch bit for bit at
other sizes. The measured max_abs_diff is reported so a reader sees how much
headroom there was.
"""

import torch
import torch_npu  # noqa: F401

import asc
import asc.lib.runtime as rt

from _probe_common import check, run, setup_device

COLS = 64

@asc.jit
def rowsum(
    x: asc.GlobalAddress,
    y: asc.GlobalAddress,
    rows_per_core: asc.ConstExpr[int],
    cols: asc.ConstExpr[int],
):
    core = asc.get_block_idx()
    x_gm = asc.GlobalTensor()
    y_gm = asc.GlobalTensor()
    x_gm.set_global_buffer(x + core * rows_per_core * cols, rows_per_core * cols)
    y_gm.set_global_buffer(y + core * rows_per_core, rows_per_core)

    pipe = asc.TPipe()
    qin = asc.TQue(asc.TPosition.VECIN, 1)
    qout = asc.TQue(asc.TPosition.VECOUT, 1)
    pipe.init_buffer(qin, 1, rows_per_core * cols * 4)
    pipe.init_buffer(qout, 1, rows_per_core * 4)

    xl = qin.alloc_tensor(asc.float32)
    asc.data_copy(xl, x_gm, rows_per_core * cols)
    qin.enque(xl)
    xin = qin.deque(asc.float32)

    yl = qout.alloc_tensor(asc.float32)
    asc.whole_reduce_sum(
        yl,
        xin,
        mask=cols,
        repeat_time=rows_per_core,
        dst_rep_stride=1,
        src_blk_stride=1,
        src_rep_stride=cols // 8,
    )
    qout.enque(yl)
    yout = qout.deque(asc.float32)

    asc.data_copy(y_gm, yout, rows_per_core)
    qin.free_tensor(xin)
    qout.free_tensor(yout)

@asc.jit
def raw_copy(x: asc.GlobalAddress, y: asc.GlobalAddress, n: asc.ConstExpr[int]):
    x_gm = asc.GlobalTensor()
    y_gm = asc.GlobalTensor()
    x_gm.set_global_buffer(x, n)
    y_gm.set_global_buffer(y, n)
    x_local = asc.LocalTensor(asc.float32, asc.TPosition.VECIN, 0, n)
    asc.data_copy(x_local, x_gm, n)
    asc.set_flag(asc.HardEvent.MTE2_MTE3, 0)
    asc.wait_flag(asc.HardEvent.MTE2_MTE3, 0)
    asc.data_copy(y_gm, x_local, n)

def body() -> None:
    setup_device()

    torch.manual_seed(0)
    for rows, cores in ((128, 4), (256, 8)):
        x = torch.rand(rows, COLS, dtype=torch.float32, device="npu")
        out = torch.full((rows,), -1.0, dtype=torch.float32, device="npu")
        rowsum[cores, rt.current_stream()](x, out, rows // cores, COLS)
        rt.synchronize()
        torch.npu.synchronize()
        want = torch.sum(x, dim=-1)
        diff = (out - want).abs().max().item()
        print(f"     rows={rows} cores={cores}: got[:3]={[round(v, 3) for v in out[:3].tolist()]}"
              f" want[:3]={[round(v, 3) for v in want[:3].tolist()]}")
        check(
            f"rows={rows} cores={cores} matches torch.sum(dim=-1)",
            bool(torch.allclose(out, want, atol=1e-4, rtol=1e-4)),
            f"max_abs_diff={diff:.3g}",
        )

    n = 1024
    flat = torch.arange(n, dtype=torch.float32).to("npu")
    y = torch.full((n,), -1.0, dtype=torch.float32, device="npu")
    raw_copy[1, rt.current_stream()](flat, y, n)
    rt.synchronize()
    torch.npu.synchronize()
    mismatches = int((y.cpu() != flat.cpu()).sum().item())
    check("explicit-address full-length copy is bit exact", mismatches == 0, f"mismatches={mismatches}")

if __name__ == "__main__":
    raise SystemExit(run("R8 multi-core row sum", body))
