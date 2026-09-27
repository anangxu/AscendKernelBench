"""Does pyasc itself reject bfloat16, with the engine's pre-check out of the way?

The engine refuses bfloat16 before it JITs anything, which by itself only shows
the pre-check is consistent: two different operators rejected by the same check
proves the check, not the backend. This probe imports asc directly and launches
a kernel whose dtype comes from the tensor, so any rejection here is pyasc's.

Input construction: torch.rand on the NPU cast to the dtype under test. The
tensors are created through a float32 cast because torch.rand does not build
every dtype directly.

Expected: the float32 control launches and is exact; bfloat16 and bool raise with
"Unsupported DataType name" naming the dtype. Failure criterion: bfloat16 or
bool launches, which would mean the engine is refusing something pyasc could
actually run.
"""

import torch
import torch_npu  # noqa: F401

import asc
import asc.lib.runtime as rt

from _probe_common import expect, run, setup_device

N = 64

@asc.jit
def add_kernel(
    x: asc.GlobalAddress,
    y: asc.GlobalAddress,
    z: asc.GlobalAddress,
    n: asc.ConstExpr[int],
):
    x_gm = asc.GlobalTensor()
    y_gm = asc.GlobalTensor()
    z_gm = asc.GlobalTensor()
    x_gm.set_global_buffer(x, n)
    y_gm.set_global_buffer(y, n)
    z_gm.set_global_buffer(z, n)

    pipe = asc.TPipe()
    qx = asc.TQue(asc.TPosition.VECIN, 1)
    qy = asc.TQue(asc.TPosition.VECIN, 1)
    qz = asc.TQue(asc.TPosition.VECOUT, 1)
    pipe.init_buffer(qx, 1, n * x.dtype.sizeof())
    pipe.init_buffer(qy, 1, n * y.dtype.sizeof())
    pipe.init_buffer(qz, 1, n * z.dtype.sizeof())

    xl = qx.alloc_tensor(x.dtype)
    asc.data_copy(xl, x_gm, n)
    qx.enque(xl)
    yl = qy.alloc_tensor(y.dtype)
    asc.data_copy(yl, y_gm, n)
    qy.enque(yl)
    xi = qx.deque(x.dtype)
    yi = qy.deque(y.dtype)
    zl = qz.alloc_tensor(z.dtype)
    asc.add(zl, xi, yi, n)
    qz.enque(zl)
    qx.free_tensor(xi)
    qy.free_tensor(yi)
    zo = qz.deque(z.dtype)
    asc.data_copy(z_gm, zo, n)
    qz.free_tensor(zo)

def body() -> None:
    setup_device()

    def attempt(dtype: torch.dtype) -> BaseException | None:
        a = torch.rand(N, device="npu").to(dtype)
        b = torch.rand(N, device="npu").to(dtype)
        c = torch.empty(N, dtype=dtype, device="npu")
        try:
            add_kernel[1, rt.current_stream()](a, b, c, N)
            rt.synchronize()
            torch.npu.synchronize()
            diff = (c.float() - (a.float() + b.float())).abs().max().item()
            print(f"     {str(dtype):16s} launched, max_abs_diff={diff:.3g}")
            return None
        except BaseException as exc:  # noqa: BLE001
            print(f"     {str(dtype):16s} {type(exc).__name__}: {str(exc)[:90]}")
            return exc

    expect("float32 launches with pyasc alone", attempt(torch.float32) is None)

    for dtype in (torch.bfloat16, torch.bool):
        exc = attempt(dtype)
        name = str(dtype).split(".")[-1]
        message = "" if exc is None else str(exc)
        expect(
            f"pyasc rejects {name} on its own",
            exc is not None and "Unsupported DataType name" in message and name in message,
            f"{type(exc).__name__ if exc else 'no error'}: {message[:80]}",
        )

if __name__ == "__main__":
    raise SystemExit(run("dtype bypass", body))
