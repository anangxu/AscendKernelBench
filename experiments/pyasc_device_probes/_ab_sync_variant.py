"""One-variable contrast on the copy-in / write-back synchronisation.

Both kernels are identical except for the wait between the inbound copy and the
write-back, so the synchronisation is the only variable:

  plain  -- the TQue idiom alone, exactly as probe_s1_tque_copy_roundtrip.py
  event  -- the same, plus an explicit MTE2_MTE3 flag pair after the inbound
            copy and before the write-back

The single B failure in the entry-point A/B left the sentinel overwritten but
all 64 elements wrong, and the error was exactly the offset used by an earlier
experiment on the same device, so the write-back read stale UB rather than
waiting for the copy. This contrast tests whether an explicit wait removes it.

Usage: python _ab_sync_variant.py {plain|event}
"""

import sys

import torch
import torch_npu  # noqa: F401

import asc
import asc.lib.runtime as rt

from _probe_common import report_metrics, setup_device

N = 64


@asc.jit
def copy_plain(x: asc.GlobalAddress, y: asc.GlobalAddress, n: asc.ConstExpr[int]):
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


@asc.jit
def copy_event(x: asc.GlobalAddress, y: asc.GlobalAddress, n: asc.ConstExpr[int]):
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
    asc.set_flag(asc.HardEvent.MTE2_MTE3, 0)
    asc.wait_flag(asc.HardEvent.MTE2_MTE3, 0)
    asc.data_copy(y_gm, out, n)
    q.free_tensor(out)


KERNELS = {"plain": copy_plain, "event": copy_event}


def main(argv: list[str]) -> int:
    variant = argv[1] if len(argv) > 1 else "plain"
    kernel = KERNELS[variant]
    setup_device()

    x_cpu = torch.arange(N, dtype=torch.float32)
    x = x_cpu.to("npu")
    y = torch.full((N,), -1.0, dtype=torch.float32, device="npu")
    kernel[1, rt.current_stream()](x, y, N)
    rt.synchronize()
    torch.npu.synchronize()
    got = y.cpu()

    report_metrics(variant, got, x_cpu)
    return 0 if bool(torch.equal(got, x_cpu)) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
