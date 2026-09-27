"""R10: hold the tile length and addresses fixed, change only the pipe event.

R6a wrote uninitialised UB with a 256-element write-back and MTE2_V, while R8's
full-length copy was exact with MTE2_MTE3 -- but those two differ in length
(256 vs 1024), so they do not isolate the event. This probe runs both events at
n=256 with identical addresses, which makes the event the only variable.

Input construction: x = arange(256) fp32 on the NPU, deterministic. Both outputs
are prefilled with -1.

Expected: the MTE2_MTE3 variant is bit exact; the MTE2_V variant is not, because
the write-back is synchronised against the vector pipe instead of the MTE3 pipe
that consumes the tile. Failure criterion: MTE2_MTE3 is not exact, or MTE2_V is
exact (which would mean the event choice is not what separates them here).

Boundary: this covers the explicit-address style with hand-written
set_flag/wait_flag calls. The TQue idiom emits its own synchronisation and is
not covered by this result.
"""

import torch
import torch_npu  # noqa: F401

import asc
import asc.lib.runtime as rt

from _probe_common import expect, run, setup_device

N = 256

@asc.jit
def copy_mte2_v(x: asc.GlobalAddress, y: asc.GlobalAddress, n: asc.ConstExpr[int]):
    x_gm = asc.GlobalTensor()
    y_gm = asc.GlobalTensor()
    x_gm.set_global_buffer(x, n)
    y_gm.set_global_buffer(y, n)
    x_local = asc.LocalTensor(asc.float32, asc.TPosition.VECIN, 0, n)
    asc.data_copy(x_local, x_gm, n)
    asc.set_flag(asc.HardEvent.MTE2_V, 0)
    asc.wait_flag(asc.HardEvent.MTE2_V, 0)
    asc.data_copy(y_gm, x_local, n)

@asc.jit
def copy_mte2_mte3(x: asc.GlobalAddress, y: asc.GlobalAddress, n: asc.ConstExpr[int]):
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

    x = torch.arange(N, dtype=torch.float32).to("npu")
    x_cpu = x.cpu()
    stream = rt.current_stream()

    for kernel, event, want_exact in (
        (copy_mte2_v, "MTE2_V", False),
        (copy_mte2_mte3, "MTE2_MTE3", True),
    ):
        y = torch.full((N,), -1.0, dtype=torch.float32, device="npu")
        kernel[1, stream](x, y, N)
        rt.synchronize()
        torch.npu.synchronize()
        got = y.cpu()
        exact = bool(torch.equal(got, x_cpu))
        head = [round(v, 3) for v in got[:6].tolist()]
        print(f"     event={event:9s} exact={exact} head={head}")
        if want_exact:
            expect(f"event {event} copies exactly", exact, f"mismatches={int((got != x_cpu).sum())}")
        else:
            expect(f"event {event} does not copy exactly", not exact, f"head={head}")

if __name__ == "__main__":
    raise SystemExit(run("R10 pipe event choice", body))
