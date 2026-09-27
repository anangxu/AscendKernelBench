"""Step 1 of the reduction investigation: a TQue GM -> UB -> GM round trip.

One variable under test: whether the move path itself works when the buffers
come from a TQue. Single core, N=64, deterministic input, sentinel output.

Expected: the sentinel is overwritten and the round trip is bit exact.
Failure criterion: any element differs from arange(64), or the sentinel
survives, in which case only the TQue path is broken and nothing above it can
be diagnosed. This probe says nothing about the explicit-address path.
"""

import torch
import torch_npu  # noqa: F401

import asc
import asc.lib.runtime as rt

from _probe_common import expect, run, setup_device

N = 64

@asc.jit
def copy_kernel(x: asc.GlobalAddress, y: asc.GlobalAddress, n: asc.ConstExpr[int]):
    x_gm = asc.GlobalTensor()
    y_gm = asc.GlobalTensor()
    x_gm.set_global_buffer(x, n)
    y_gm.set_global_buffer(y, n)

    pipe = asc.TPipe()
    q = asc.TQue(asc.TPosition.VECIN, 1)
    pipe.init_buffer(q, 1, n * 4)

    t = q.alloc_tensor(asc.float32)
    asc.data_copy(t, x_gm, n)
    q.enque(t)
    out = q.deque(asc.float32)
    asc.data_copy(y_gm, out, n)
    q.free_tensor(out)

def body() -> None:
    setup_device()

    x_cpu = torch.arange(N, dtype=torch.float32)
    x = x_cpu.to("npu")
    y = torch.full((N,), -1.0, dtype=torch.float32, device="npu")
    copy_kernel[1, rt.current_stream()](x, y, N)
    rt.synchronize()
    torch.npu.synchronize()
    got = y.cpu()

    expect("round trip overwrites the sentinel", not bool((got == -1.0).all()))
    expect("round trip is bit exact", bool(torch.equal(got, x_cpu)), f"got[:6]={got[:6].tolist()}")

if __name__ == "__main__":
    raise SystemExit(run("TQue copy round trip", body))
