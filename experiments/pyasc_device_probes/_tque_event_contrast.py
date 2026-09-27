"""Single variable: does an allocated MTE2_MTE3 event fix the TQue write-back?

The IR handed to the compiler contains no set_flag/wait_flag in the TQue copy
function, while the explicit-address version contains the pair. This contrast
tests the behavioural consequence at the sizes that trigger the dropped chunks.

  base  -- TQue idiom alone
  event -- the same kernel plus a flag pair whose id comes from
           pipe.alloc_event_id(HardEvent.MTE2_MTE3), not a hard-coded 0, so it
           cannot collide with an event the framework already owns

Only the flag pair differs. Sizes and repetitions are fixed, each repetition
uses its own sentinel value, and the phenomenon is provoked rather than sampled,
so a handful of launches is enough.

Usage: python _tque_event_contrast.py {base|event}
"""

import sys

import torch
import torch_npu  # noqa: F401

import asc
import asc.lib.runtime as rt

from _probe_common import setup_device

SIZES = (4096, 16384)      # 16 KiB and 64 KiB write-backs
REPS = 5


@asc.jit
def tque_base(x: asc.GlobalAddress, y: asc.GlobalAddress, n: asc.ConstExpr[int]):
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
def tque_event(x: asc.GlobalAddress, y: asc.GlobalAddress, n: asc.ConstExpr[int]):
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

    event = pipe.alloc_event_id(asc.HardEvent.MTE2_MTE3)
    asc.set_flag(asc.HardEvent.MTE2_MTE3, event)
    asc.wait_flag(asc.HardEvent.MTE2_MTE3, event)

    asc.data_copy(y_gm, out, n)
    q.free_tensor(out)


KERNELS = {"base": tque_base, "event": tque_event}


def main(argv: list[str]) -> int:
    arm = argv[1] if len(argv) > 1 else "base"
    assert arm in KERNELS, arm
    kernel = KERNELS[arm]
    setup_device()

    for n in SIZES:
        want = torch.arange(n, dtype=torch.float32)
        for rep in range(REPS):
            sentinel = -float(rep + 1)
            x = want.to("npu")
            y = torch.full((n,), sentinel, dtype=torch.float32, device="npu")
            # Prove the fill completed before the launch; one extra line, kept
            # in both arms so it is not a variable.
            torch.npu.synchronize()
            fill_ok = bool((y.cpu() == sentinel).all())

            kernel[1, rt.current_stream()](x, y, n)
            rt.synchronize()
            torch.npu.synchronize()
            got = y.cpu()
            print(
                f"  arm={arm} n={n:6d} ({n * 4 // 1024:3d} KiB) rep={rep} "
                f"fill_ok={fill_ok} unwritten={int((got == sentinel).sum().item())} "
                f"wrong={int((got != want).sum().item())}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
