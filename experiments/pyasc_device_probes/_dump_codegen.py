"""Dump the official compile artifacts for one kernel and look for synchronisation.

pyasc's compiler already writes the whole pipeline to PYASC_DUMP_PATH:

  codegen.mlir  -- IR before the pass pipeline
  ascir.mlir    -- IR after the pipeline (this is where auto-sync would insert)
  ascendc.cpp   -- the CCE source handed to the compiler
  binary.o      -- the compiled object

Reading those files is stronger evidence than hooking _run_codegen: the question
is whether the sync dependency exists after the passes, and `ascendc.cpp` shows
what the backend actually compiles. Each kernel gets its own dump directory so
the three files belong to one kernel and one stage.

Usage: python _dump_codegen.py {tque|explicit|tque_verify}
"""

import os
import re
import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401

import asc
import asc.lib.runtime as rt

from _probe_common import setup_device

N = 4096
SYNC_PATTERN = re.compile(r"SetFlag|WaitFlag|PipeBarrier|SyncAll|CrossCoreSetFlag|SyncBlock")


@asc.jit
def tque_copy(x: asc.GlobalAddress, y: asc.GlobalAddress, n: asc.ConstExpr[int]):
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
def explicit_copy(x: asc.GlobalAddress, y: asc.GlobalAddress, n: asc.ConstExpr[int]):
    x_gm = asc.GlobalTensor()
    y_gm = asc.GlobalTensor()
    x_gm.set_global_buffer(x, n)
    y_gm.set_global_buffer(y, n)
    x_local = asc.LocalTensor(asc.float32, asc.TPosition.VECIN, 0, n)
    asc.data_copy(x_local, x_gm, n)
    asc.set_flag(asc.HardEvent.MTE2_MTE3, 0)
    asc.wait_flag(asc.HardEvent.MTE2_MTE3, 0)
    asc.data_copy(y_gm, x_local, n)


@asc.jit
def tquebind_copy(x: asc.GlobalAddress, y: asc.GlobalAddress, n: asc.ConstExpr[int]):
    """The binding path the docs name for a direct VECIN to VECOUT copy."""
    x_gm = asc.GlobalTensor()
    y_gm = asc.GlobalTensor()
    x_gm.set_global_buffer(x, n)
    y_gm.set_global_buffer(y, n)

    pipe = asc.TPipe()
    que = asc.TQueBind(asc.TPosition.VECIN, asc.TPosition.VECOUT, 1)
    pipe.init_buffer(que=que, num=1, len=n * 4)

    t = que.alloc_tensor(asc.float32)
    asc.data_copy(t, x_gm, n)
    que.enque(t)
    out = que.deque(asc.float32)
    asc.data_copy(y_gm, out, n)
    que.free_tensor(out)


ARMS = {
    "tque": (tque_copy, {}),
    "explicit": (explicit_copy, {}),
    "tque_verify": (tque_copy, {"verify_sync": True}),
    "tquebind": (tquebind_copy, {}),
}


def sync_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if SYNC_PATTERN.search(line)]


def main(argv: list[str]) -> int:
    arm = argv[1] if len(argv) > 1 else "tque"
    kernel, call_options = ARMS[arm]
    setup_device()

    x = torch.arange(N, dtype=torch.float32).to("npu")
    y = torch.full((N,), -1.0, dtype=torch.float32, device="npu")
    kernel[1, rt.current_stream()](x, y, N, **call_options)
    torch.npu.synchronize()

    dump = Path(os.environ["PYASC_DUMP_PATH"])
    print(f"\n===== arm={arm} dump={dump} =====")
    for name in ("codegen.mlir", "ascir.mlir", "ascendc.cpp"):
        path = dump / name
        if not path.is_file():
            print(f"  {name}: MISSING")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        hits = sync_lines(text)
        print(f"  {name}: {len(text.splitlines())} lines, sync ops = {len(hits)}")
        for line in hits[:6]:
            print(f"      {line[:120]}")

    # Behaviour of the same arm at the size that drops chunks.
    size = 16384
    wrong = []
    want = torch.arange(size, dtype=torch.float32)
    for rep in range(5):
        xa = want.to("npu")
        ya = torch.full((size,), -float(rep + 1), dtype=torch.float32, device="npu")
        torch.npu.synchronize()
        kernel[1, rt.current_stream()](xa, ya, size, **call_options)
        torch.npu.synchronize()
        wrong.append(int((ya.cpu() != want).sum().item()))
    print(f"  64 KiB behaviour: wrong per launch = {wrong}")

    cce = dump / "ascendc.cpp"
    if cce.is_file():
        text = cce.read_text(encoding="utf-8", errors="replace")
        body = re.search(r"void\s+\w+\(.*?\n\}", text, re.S)
        if body:
            print("  --- generated kernel body ---")
            for line in body.group(0).splitlines()[:40]:
                print("      " + line.strip()[:120])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
