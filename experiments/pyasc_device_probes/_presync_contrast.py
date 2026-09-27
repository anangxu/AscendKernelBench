"""Does a pre-launch synchronise remove the dropped write-back chunks?

Alternative explanation for the sentinel blocks: PyTorch fills the output buffer
and the kernel launches on a stream that is not ordered against that fill, so
the fill can land after the kernel's write-back. "Still holding the sentinel"
then means "the fill came last", not "the kernel never wrote".

One variable between the two arms:

  nosync  -- fill, launch, sync            (the original order)
  presync -- fill, torch.npu.synchronize(), read the buffer back to prove the
             fill completed, launch, sync

Everything else is fixed: same kernel, sizes, dtype, device, stream, shapes and
input construction. Each repetition uses its own sentinel value -(rep + 1), so a
residue can be matched to the fill that left it instead of being read as a bare
count.

Usage: python _presync_contrast.py {nosync|presync}
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


def main(argv: list[str]) -> int:
    arm = argv[1] if len(argv) > 1 else "nosync"
    assert arm in ("nosync", "presync"), arm
    setup_device()

    for n in SIZES:
        want = torch.arange(n, dtype=torch.float32)
        for rep in range(REPS):
            sentinel = -float(rep + 1)
            x = want.to("npu")
            y = torch.full((n,), sentinel, dtype=torch.float32, device="npu")

            pre_ok = True
            if arm == "presync":
                # Order the fill against everything that follows, and prove it.
                torch.npu.synchronize()
                pre_ok = bool((y.cpu() == sentinel).all())

            copy_kernel[1, rt.current_stream()](x, y, n)
            rt.synchronize()
            torch.npu.synchronize()
            got = y.cpu()

            unwritten = int((got == sentinel).sum().item())
            stale_fills = sorted(
                {
                    -float(other + 1)
                    for other in range(REPS)
                    if other != rep and bool((got == -float(other + 1)).any().item())
                }
            )
            wrong = int((got != want).sum().item())
            print(
                f"  arm={arm} n={n:6d} ({n * 4 // 1024:3d} KiB) rep={rep} "
                f"fill_complete={pre_ok} unwritten={unwritten} wrong={wrong} "
                f"stale_fills={stale_fills} got[:3]={got[:3].tolist()}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
