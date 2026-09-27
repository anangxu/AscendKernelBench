"""R9: sweep only the write-back length and find where it stops working.

UB already holds the right data (probe_r08_multicore_vs_torch.py proves the
explicit-address full-length copy). Here the same UB address is copied back as
1, 4, 8 and 16 fp32 into four separate sentinel buffers, so one compile answers
the question. The same sweep runs in the TQue style to see whether the buffer
style matters.

Input construction: x = arange(1024) fp32 on the NPU, deterministic.

Expected on the device this was written for: 1 and 4 fp32 (4 and 16 bytes) do
not take effect, 8 and 16 fp32 (32 and 64 bytes) are exact, in both styles.
Failure criterion: a 4-byte or 16-byte buffer moves, which would mean the
granule finding does not hold here; or a 32-byte or 64-byte buffer stays at the
sentinel.

Boundary: this measures one device, one pyasc release and this call style. It is
not a documented property of data_copy, and the engine treats it as a reason to
keep write-backs at 32 bytes or more rather than as a specification.
"""

import torch
import torch_npu  # noqa: F401

import asc
import asc.lib.runtime as rt

from _probe_common import expect, run, setup_device

N = 1024
SIZES = (1, 4, 8, 16)

@asc.jit
def raw_granule(
    x: asc.GlobalAddress,
    y1: asc.GlobalAddress,
    y4: asc.GlobalAddress,
    y8: asc.GlobalAddress,
    y16: asc.GlobalAddress,
    n: asc.ConstExpr[int],
):
    x_gm = asc.GlobalTensor()
    x_gm.set_global_buffer(x, n)
    g1 = asc.GlobalTensor()
    g1.set_global_buffer(y1, 1)
    g4 = asc.GlobalTensor()
    g4.set_global_buffer(y4, 4)
    g8 = asc.GlobalTensor()
    g8.set_global_buffer(y8, 8)
    g16 = asc.GlobalTensor()
    g16.set_global_buffer(y16, 16)

    x_local = asc.LocalTensor(asc.float32, asc.TPosition.VECIN, 0, n)
    asc.data_copy(x_local, x_gm, n)
    asc.set_flag(asc.HardEvent.MTE2_MTE3, 0)
    asc.wait_flag(asc.HardEvent.MTE2_MTE3, 0)

    asc.data_copy(g1, x_local, 1)
    asc.data_copy(g4, x_local, 4)
    asc.data_copy(g8, x_local, 8)
    asc.data_copy(g16, x_local, 16)

@asc.jit
def tque_granule(
    x: asc.GlobalAddress,
    y1: asc.GlobalAddress,
    y4: asc.GlobalAddress,
    y8: asc.GlobalAddress,
    y16: asc.GlobalAddress,
    n: asc.ConstExpr[int],
):
    x_gm = asc.GlobalTensor()
    x_gm.set_global_buffer(x, n)
    g1 = asc.GlobalTensor()
    g1.set_global_buffer(y1, 1)
    g4 = asc.GlobalTensor()
    g4.set_global_buffer(y4, 4)
    g8 = asc.GlobalTensor()
    g8.set_global_buffer(y8, 8)
    g16 = asc.GlobalTensor()
    g16.set_global_buffer(y16, 16)

    pipe = asc.TPipe()
    qin = asc.TQue(asc.TPosition.VECIN, 1)
    pipe.init_buffer(qin, 1, n * 4)
    xl = qin.alloc_tensor(asc.float32)
    asc.data_copy(xl, x_gm, n)
    qin.enque(xl)
    xin = qin.deque(asc.float32)

    asc.data_copy(g1, xin, 1)
    asc.data_copy(g4, xin, 4)
    asc.data_copy(g8, xin, 8)
    asc.data_copy(g16, xin, 16)
    qin.free_tensor(xin)

def body() -> None:
    setup_device()

    def fresh() -> dict[int, torch.Tensor]:
        return {k: torch.full((k,), -1.0, dtype=torch.float32, device="npu") for k in SIZES}

    def check(style: str, outs: dict[int, torch.Tensor]) -> None:
        for size in SIZES:
            got = [round(v, 3) for v in outs[size].cpu().tolist()]
            want = [float(i) for i in range(size)]
            moved = got != [-1.0] * size
            print(f"     {style} k={size:2d} ({size * 4:3d} B): moved={moved} got={got}")
            if size * 4 < 32:
                expect(f"{style}: a {size * 4}-byte write-back does not take effect", not moved)
            else:
                expect(f"{style}: a {size * 4}-byte write-back is exact", got == want, f"got={got}")

    x = torch.arange(N, dtype=torch.float32).to("npu")
    stream = rt.current_stream()

    outs = fresh()
    raw_granule[1, stream](x, outs[1], outs[4], outs[8], outs[16], N)
    rt.synchronize()
    torch.npu.synchronize()
    check("raw ", outs)

    outs2 = fresh()
    tque_granule[1, stream](x, outs2[1], outs2[4], outs2[8], outs2[16], N)
    rt.synchronize()
    torch.npu.synchronize()
    check("tque", outs2)

if __name__ == "__main__":
    raise SystemExit(run("R9 write-back length sweep", body))
