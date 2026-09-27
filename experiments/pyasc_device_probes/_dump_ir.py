"""Dump the generated IR for the TQue copy and for the explicit-address copy.

Answers the question the counts cannot: does the TQue idiom actually emit a
dependency between the inbound copy (MTE2) and the write-back (MTE3), and where
does the explicit-address version place its own event?

Usage: python _dump_ir.py
"""

import io
import sys
from contextlib import redirect_stdout

import torch
import torch_npu  # noqa: F401

import asc
import asc.lib.runtime as rt

from asc.runtime import jit as asc_jit

from _probe_common import setup_device

N = 4096


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


CAPTURED: list[tuple[str, object]] = []
_CODEGEN = asc_jit.JITFunction._run_codegen
_COMPILER = asc_jit.JITFunction._run_compiler


def _capture_codegen(self, spec, options):
    module = _CODEGEN(self, spec, options)
    CAPTURED.append(("after codegen", module))
    return module


def _capture_compiler(self, module, options):
    # This is the IR the compiler is actually handed, after any passes.
    CAPTURED.append(("handed to the compiler", module))
    return _COMPILER(self, module, options)


def dump(kernel, label: str) -> None:
    asc_jit.JITFunction._run_codegen = _capture_codegen
    asc_jit.JITFunction._run_compiler = _capture_compiler
    try:
        x = torch.arange(N, dtype=torch.float32).to("npu")
        y = torch.full((N,), -1.0, dtype=torch.float32, device="npu")
        kernel[1, rt.current_stream()](x, y, N)
        torch.npu.synchronize()
    finally:
        asc_jit.JITFunction._run_codegen = _CODEGEN
        asc_jit.JITFunction._run_compiler = _COMPILER

    stage, module = CAPTURED[-1]
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        module.dump()
    text = buffer.getvalue()
    print(f"\n===== {label} [{stage}] =====")
    for line in text.splitlines():
        low = line.lower()
        if any(key in low for key in ("copy", "flag", "wait", "queue", "pipe", "sync")):
            print("   " + line.strip()[:150])


def main() -> None:
    setup_device()
    dump(tque_copy, "TQue copy (enque/deque only)")
    dump(explicit_copy, "explicit address + MTE2_MTE3")


if __name__ == "__main__":
    main()
