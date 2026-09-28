"""Row-wise fp32 sum built on whole_reduce_sum, verified on an Ascend 910B4.

Facts this encodes, all measured with pyasc 1.1.1 rather than read off the
docstring:
  - whole_reduce_sum reduces per repeat, so repeat_time=R produces R outputs;
  - in the contiguous mask mode mask is an element count, not a bit mask;
  - src_rep_stride counts 32-byte data blocks of the source while
    dst_rep_stride counts destination elements;
  - a data_copy write-back shorter than 32 bytes silently writes nothing, so a
    core's write-back must cover whole 32-byte blocks, which is why the
    launcher only accepts cols and rows_per_core that are multiples of 8 fp32.

The TQue style is used on purpose: it reserves UB and emits the pipe
synchronisation itself, which a bare LocalTensor address does not.
"""

import asc
import asc.lib.runtime as rt
import torch
import torch_npu  # noqa: F401

MAX_COLS = 64  # contiguous mask limit for fp32
BYTE_ALIGN = 32
BLOCK_ELEMS = BYTE_ALIGN // 4  # fp32 elements in one 32-byte block
MIN_ROWS_PER_CORE = BLOCK_ELEMS  # a write-back is a whole number of blocks
DEFAULT_CORES = 8


@asc.jit
def rowsum_kernel(
    x: asc.GlobalAddress,
    y: asc.GlobalAddress,
    rows_per_core: asc.ConstExpr[int],
    cols: asc.ConstExpr[int],
):
    core = asc.get_block_idx()

    x_gm = asc.GlobalTensor()
    y_gm = asc.GlobalTensor()
    x_gm.set_global_buffer(x + core * rows_per_core * cols, rows_per_core * cols)
    y_gm.set_global_buffer(y + core * rows_per_core, rows_per_core)

    pipe = asc.TPipe()
    in_q = asc.TQue(asc.TPosition.VECIN, 1)
    out_q = asc.TQue(asc.TPosition.VECOUT, 1)
    pipe.init_buffer(in_q, 1, rows_per_core * cols * x.dtype.sizeof())
    pipe.init_buffer(out_q, 1, rows_per_core * y.dtype.sizeof())

    x_local = in_q.alloc_tensor(x.dtype)
    asc.data_copy(x_local, x_gm, rows_per_core * cols)
    in_q.enque(x_local)
    x_in = in_q.deque(x.dtype)

    y_local = out_q.alloc_tensor(y.dtype)
    asc.whole_reduce_sum(
        y_local,
        x_in,
        mask=cols,
        repeat_time=rows_per_core,
        dst_rep_stride=1,
        src_blk_stride=1,
        src_rep_stride=cols * x.dtype.sizeof() // 32,
    )
    out_q.enque(y_local)
    y_out = out_q.deque(y.dtype)

    asc.data_copy(y_gm, y_out, rows_per_core)
    in_q.free_tensor(x_in)
    out_q.free_tensor(y_out)


def pick_cores(rows: int) -> int:
    """Return the largest core count that splits rows into whole blocks."""
    cores = DEFAULT_CORES
    while cores > 1 and (rows % cores != 0 or (rows // cores) % MIN_ROWS_PER_CORE != 0):
        cores //= 2
    return cores


def rowsum_launch(x: torch.Tensor) -> torch.Tensor:
    """Reduce the last axis of a contiguous fp32 tensor shaped (rows, cols).

    Only shapes the kernel addresses exactly are accepted. cols must be whole
    32-byte blocks (a multiple of 8 fp32, within the 64-element contiguous
    mask), because src_rep_stride is derived from it in blocks and every row
    has to start on a block boundary. rows_per_core is also the write-back
    length, so each core must own a whole number of 32-byte row blocks.
    Anything else is rejected rather than computed with a rounded stride.
    """
    rows, cols = int(x.shape[0]), int(x.shape[1])
    if cols < BLOCK_ELEMS or cols > MAX_COLS or cols % BLOCK_ELEMS != 0:
        raise ValueError("cols must be 8..64 fp32 in whole 32-byte blocks")
    cores = pick_cores(rows)
    if rows % cores != 0 or (rows // cores) % MIN_ROWS_PER_CORE != 0:
        raise ValueError("rows must split into whole 32-byte row blocks per core")
    y = torch.empty(rows, dtype=x.dtype, device=x.device)
    rowsum_kernel[cores, rt.current_stream()](x, y, rows // cores, cols)
    rt.synchronize()
    return y
