"""R1: the naive row sum over explicit LocalTensor addresses.

Reproduces the first failing reduction attempt. Single variable under test:
the naive mapping of one row to one repeat with explicit addresses and no
reserved UB, launched on 4 cores with 32 rows each.

Input construction (this probe seeds torch with 0 for reproducibility; the
historical run did not seed and the phenomenon was not seed dependent):

    x = torch.rand(128, 64, dtype=torch.float32, device="npu")
    reference = torch.sum(x, dim=-1)
    out = torch.empty(128, dtype=torch.float32, device="npu")

Key parameters: rows_per_core=32, cols=64, mask=64, repeat_time=32,
dst_rep_stride=1, src_blk_stride=1, src_rep_stride=8, x_local VECIN addr 0
len 2048, y_local VECOUT addr 8192 len 32, launch on 4 cores.

Expected: this probe documents a FAILURE. The verdict is green when the kernel
does not reproduce the reference, which is what was observed. Two outcomes have
been seen for the same kernel: wrong values (max_abs_diff 49.2771, historical)
and a vector core device fault (error 507035, on a rebuilt container). Both are
graded as the expected failure. The cause of R1 is unexplained in either case:
the write-back is 128 bytes per core, so the sub-32-byte granule does not apply,
and no UB is reserved in this style. Do not read this probe as a pyasc defect
report.
"""

import torch
import torch_npu  # noqa: F401

import asc
import asc.lib.runtime as rt

from _probe_common import reproduced, run, setup_device

COLS = 64
CORES = 4
ROWS = 128
ROWS_PER_CORE = ROWS // CORES

@asc.jit
def rowsum_kernel(
    x: asc.GlobalAddress,
    y: asc.GlobalAddress,
    rows_per_core: asc.ConstExpr[int],
    cols: asc.ConstExpr[int],
):
    core = asc.get_block_idx()
    row0 = core * rows_per_core

    x_gm = asc.GlobalTensor()
    y_gm = asc.GlobalTensor()
    x_gm.set_global_buffer(x + row0 * cols, rows_per_core * cols)
    y_gm.set_global_buffer(y + row0, rows_per_core)

    x_local = asc.LocalTensor(asc.float32, asc.TPosition.VECIN, 0, rows_per_core * cols)
    y_local = asc.LocalTensor(
        asc.float32, asc.TPosition.VECOUT, rows_per_core * cols * 4, rows_per_core
    )

    asc.data_copy(x_local, x_gm, rows_per_core * cols)
    asc.whole_reduce_sum(
        y_local,
        x_local,
        mask=cols,
        repeat_time=rows_per_core,
        dst_rep_stride=1,
        src_blk_stride=1,
        src_rep_stride=cols // 8,
    )
    asc.data_copy(y_gm, y_local, rows_per_core)

def body() -> None:
    setup_device()

    torch.manual_seed(0)
    x = torch.rand(ROWS, COLS, dtype=torch.float32, device="npu")
    out = torch.empty(ROWS, dtype=torch.float32, device="npu")
    # The device fault surfaces inside pyasc's own launcher, so the launch has
    # to be guarded for the probe to report a verdict instead of crashing.
    fault: BaseException | None = None
    try:
        rowsum_kernel[CORES, rt.current_stream()](x, out, ROWS_PER_CORE, COLS)
        rt.synchronize()
        torch.npu.synchronize()
    except BaseException as exc:  # noqa: BLE001
        fault = exc

    if fault is not None:
        first_line = str(fault).splitlines()[0]
        print(f"     device fault = {type(fault).__name__}: {first_line}")
        reproduced(
            "the naive kernel faults instead of returning a result",
            True,
            "the same kernel returned wrong values historically; both are failures to compute",
        )
        return

    got = out.cpu()
    want = torch.sum(x, dim=-1).cpu()
    diff = (got - want).abs().max().item()
    head = [round(v, 6) for v in got[:4].tolist()]
    print(f"     got[:4]        = {head}")
    print(f"     want[:4]       = {[round(v, 4) for v in want[:4].tolist()]}")
    print(f"     identical head = {len(set(head)) == 1}")
    reproduced(
        "the naive explicit-address row sum does not match torch.sum(dim=-1)",
        not bool(torch.allclose(got, want, atol=1e-4, rtol=1e-4)),
        f"max_abs_diff={diff:.4g}",
    )

if __name__ == "__main__":
    raise SystemExit(run("R1 naive row sum", body))
