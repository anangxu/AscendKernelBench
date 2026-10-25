"""pyasc Add, alignment-aware host plan.

Verified facts this encodes:
  - `range(tiles)` accepts a ConstExpr[int] kernel argument.
  - `data_copy` needs 32-byte-aligned element counts; a 4-byte tail fails.
  - the traced body cannot use `while`/`if` for static branching, so the
    host computes the tiling plan and passes it as ConstExpr values.
"""

import asc
import asc.lib.runtime as rt
import torch
import torch_npu  # noqa: F401

DEFAULT_CORES = 8
DEFAULT_TILES = 8
BYTE_ALIGN = 32


@asc.jit
def add_kernel(
    x: asc.GlobalAddress,
    y: asc.GlobalAddress,
    z: asc.GlobalAddress,
    block_len: asc.ConstExpr[int],
    tiles: asc.ConstExpr[int],
    tile_len: asc.ConstExpr[int],
):
    offset = asc.get_block_idx() * block_len

    x_gm = asc.GlobalTensor()
    y_gm = asc.GlobalTensor()
    z_gm = asc.GlobalTensor()
    x_gm.set_global_buffer(x + offset, block_len)
    y_gm.set_global_buffer(y + offset, block_len)
    z_gm.set_global_buffer(z + offset, block_len)

    pipe = asc.TPipe()
    in_qx = asc.TQue(asc.TPosition.VECIN, 1)
    in_qy = asc.TQue(asc.TPosition.VECIN, 1)
    out_qz = asc.TQue(asc.TPosition.VECOUT, 1)
    pipe.init_buffer(in_qx, 1, tile_len * x.dtype.sizeof())
    pipe.init_buffer(in_qy, 1, tile_len * y.dtype.sizeof())
    pipe.init_buffer(out_qz, 1, tile_len * z.dtype.sizeof())

    for i in range(tiles):
        x_local = in_qx.alloc_tensor(x_gm.dtype)
        asc.data_copy(x_local, x_gm[i * tile_len :], tile_len)
        in_qx.enque(x_local)
        y_local = in_qy.alloc_tensor(y_gm.dtype)
        asc.data_copy(y_local, y_gm[i * tile_len :], tile_len)
        in_qy.enque(y_local)

        x_in = in_qx.deque(x_gm.dtype)
        y_in = in_qy.deque(y_gm.dtype)
        z_local = out_qz.alloc_tensor(z_gm.dtype)
        asc.add(z_local, x_in, y_in, tile_len)
        out_qz.enque(z_local)
        in_qx.free_tensor(x_in)
        in_qy.free_tensor(y_in)

        z_out = out_qz.deque(z_gm.dtype)
        asc.data_copy(z_gm[i * tile_len :], z_out, tile_len)
        out_qz.free_tensor(z_out)


def elem_align(element_size: int) -> int:
    """Return the element count that makes one aligned block."""
    return max(1, BYTE_ALIGN // element_size)


def plan_tiling(total: int, align: int) -> tuple[int, int, int, int]:
    """Pick (cores, block_len, tiles, tile_len), all 32-byte aligned.

    total must already be a multiple of align; the caller pads first.
    """
    cores = DEFAULT_CORES
    while cores > 1 and (total % cores != 0 or (total // cores) % align != 0):
        cores //= 2
    block_len = total // cores
    tiles = DEFAULT_TILES
    while tiles > 1 and (block_len % tiles != 0 or (block_len // tiles) % align != 0):
        tiles -= 1
    return cores, block_len, tiles, block_len // tiles


def add_launch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Launch Add on a shape-aligned plan, padding only when required."""
    align = elem_align(x.element_size())
    total = x.numel()
    padded = ((total + align - 1) // align) * align
    cores, block_len, tiles, tile_len = plan_tiling(padded, align)
    assert cores * block_len == padded and tile_len % align == 0
    if padded == total:
        z = torch.empty_like(x)
        add_kernel[cores, rt.current_stream()](x, y, z, block_len, tiles, tile_len)
        return z
    x_pad = torch.zeros(padded, dtype=x.dtype, device=x.device)
    y_pad = torch.zeros(padded, dtype=y.dtype, device=y.device)
    x_pad[:total] = x.reshape(-1)
    y_pad[:total] = y.reshape(-1)
    z_pad = torch.empty_like(x_pad)
    add_kernel[cores, rt.current_stream()](
        x_pad, y_pad, z_pad, block_len, tiles, tile_len
    )
    return z_pad[:total].reshape(x.shape)
