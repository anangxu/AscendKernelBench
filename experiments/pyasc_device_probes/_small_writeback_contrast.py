"""One budgeted contrast on the small-write-back manifestation.

Arms, alternating, one fresh process each:

  original -- the single VECIN TQue copy, as probe_s1 has it
  fixed    -- the same kernel plus an MTE2_MTE3 pair whose id comes from
              pipe.alloc_event_id(...)

Each run uses its own sentinel value, derived from the round and the arm, so the
statistics can separate three different outcomes instead of calling them all
"still the sentinel":

  current_sentinel  -- elements still holding THIS run's sentinel. The buffer is
                       read back before the launch, so this means the write did
                       not cover them.
  old_sentinel      -- elements holding the sentinel of an EARLIER run, which is
                       a content change, not an absence of one: stale data was
                       written into the output.
  wrong             -- elements differing from the expected values.

Budget: ROUNDS rounds of both arms, set by the runner. Read outcome per user
guidance: original fails and fixed passes supports a shared dependency problem;
fixed also fails means the event is not sufficient; both pass means this round
has no discriminating power and the item stays open.

Usage: python _small_writeback_contrast.py {original|fixed} ROUND
"""

import sys

import torch
import torch_npu  # noqa: F401

import asc
import asc.lib.runtime as rt

from _probe_common import setup_device

SIZES = (64, 256, 1024)


@asc.jit
def copy_original(x: asc.GlobalAddress, y: asc.GlobalAddress, n: asc.ConstExpr[int]):
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
def copy_fixed(x: asc.GlobalAddress, y: asc.GlobalAddress, n: asc.ConstExpr[int]):
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


KERNELS = {"original": copy_original, "fixed": copy_fixed}


def sentinel_for(round_index: int, arm_index: int) -> float:
    """A value unique to this run, so residues can be attributed."""
    return -float(round_index * 10 + arm_index + 1)


def main(argv: list[str]) -> int:
    arm = argv[1]
    round_index = int(argv[2])
    assert arm in KERNELS, arm
    arm_index = 0 if arm == "original" else 1
    setup_device()

    for n in SIZES:
        sentinel = sentinel_for(round_index, arm_index)
        want = torch.arange(n, dtype=torch.float32)
        x = want.to("npu")
        y = torch.full((n,), sentinel, dtype=torch.float32, device="npu")
        torch.npu.synchronize()
        fill_ok = bool((y.cpu() == sentinel).all())

        KERNELS[arm][1, rt.current_stream()](x, y, n)
        rt.synchronize()
        torch.npu.synchronize()
        got = y.cpu()

        current = int((got == sentinel).sum().item())
        old = int(
            sum(
                (got == sentinel_for(round_index, other)).sum().item()
                for other in (0, 1)
                if sentinel_for(round_index, other) != sentinel
            )
        )
        wrong = int((got != want).sum().item())
        print(
            f"  round={round_index:02d} arm={arm} n={n:5d} fill_ok={fill_ok} "
            f"current_sentinel={current} old_sentinel={old} wrong={wrong}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
