"""Exercise the bundled reduction example: representative shapes and refusals.

This probe imports the shipped example from src/prompts/examples/pyasc/003_rowsum
by walking up from its own location, so it measures the file the prompt hands
out rather than a copy.

Shapes here are representative, not an exhaustive enumeration. The launcher
accepts cols in {8..64} that are multiples of 8 and per-core row blocks that are
multiples of 8, the range its addressing is exact for; the cases below sample
that range at both ends and in between.

Input construction: torch.rand(rows, cols) fp32 on the NPU, seeded with 0.

Expected:
  accepted shapes match torch.sum(dim=-1) within 1e-4;
  shapes outside the accepted range raise ValueError;
  cols=12, which the earlier launcher accepted, is called directly and does NOT
  match, because src_rep_stride = cols*4/32 rounds down to one block and the
  reduction then reads the wrong elements. That is why the guard exists.
Failure criterion: an accepted shape mismatches, an out-of-range shape is
accepted, or the cols=12 direct call happens to match (which would mean the
guard is not protecting against a real error).
"""

import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401

import asc
import asc.lib.runtime as rt

from _probe_common import check, run, setup_device

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = REPO_ROOT / "src" / "prompts" / "examples" / "pyasc" / "003_rowsum"
sys.path.insert(0, str(EXAMPLE_DIR))

import kernel  # noqa: E402  the bundled example, unmodified

ACCEPTED = ((64, 8), (64, 24), (64, 48), (64, 64), (8, 64), (128, 64), (512, 64), (1024, 32))
REFUSED = ((64, 4), (64, 10), (64, 12), (64, 72), (100, 64), (7, 64))

def body() -> None:
    _, rt = setup_device()
    torch.manual_seed(0)

    for rows, cols in ACCEPTED:
        x = torch.rand(rows, cols, dtype=torch.float32, device="npu")
        out = kernel.rowsum_launch(x)
        want = torch.sum(x, dim=-1)
        diff = (out - want).abs().max().item()
        check(
            f"rows={rows} cols={cols} matches torch.sum(dim=-1)",
            bool(torch.allclose(out, want, atol=1e-4, rtol=1e-4)),
            f"cores={kernel.pick_cores(rows)} max_abs_diff={diff:.3g}",
        )

    for rows, cols in REFUSED:
        try:
            kernel.rowsum_launch(torch.rand(rows, cols, dtype=torch.float32, device="npu"))
            check(f"rows={rows} cols={cols} is refused", False, "accepted instead")
        except ValueError as exc:
            check(f"rows={rows} cols={cols} is refused", True, str(exc))

    rows, cols = 64, 12
    x = torch.rand(rows, cols, dtype=torch.float32, device="npu")
    out = torch.empty(rows, dtype=torch.float32, device="npu")
    kernel.rowsum_kernel[8, rt.current_stream()](x, out, 8, cols)
    torch.npu.synchronize()
    want = torch.sum(x, dim=-1)
    diff = (out - want).abs().max().item()
    check(
        "cols=12 called directly does not match (the guard is not cosmetic)",
        not bool(torch.allclose(out, want, atol=1e-4, rtol=1e-4)),
        f"max_abs_diff={diff:.3g}",
    )

if __name__ == "__main__":
    raise SystemExit(run("rowsum example shapes", body))
