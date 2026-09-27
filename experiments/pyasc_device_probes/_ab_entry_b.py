"""Entry B for the entry-point A/B: same kernel, direct invocation.

Entry A is probe_s1_tque_copy_roundtrip.py, which runs its body through
_probe_common.run(). Entry B imports the same @asc.jit kernel and the same
constant from that module and launches it directly from main(). Input
construction, dtype, shape, device, stream, synchronisation, sentinel value and
the metrics helper are identical, so the invocation path is the only variable.

Run both with _ab_runner.sh, which alternates them in fresh processes.
"""

import torch
import torch_npu  # noqa: F401

import asc
import asc.lib.runtime as rt

from _probe_common import check, report, report_metrics, setup_device
from probe_s1_tque_copy_roundtrip import N, copy_kernel


def main() -> int:
    setup_device()

    x_cpu = torch.arange(N, dtype=torch.float32)
    x = x_cpu.to("npu")
    y = torch.full((N,), -1.0, dtype=torch.float32, device="npu")
    copy_kernel[1, rt.current_stream()](x, y, N)
    rt.synchronize()
    torch.npu.synchronize()
    got = y.cpu()

    report_metrics("B", got, x_cpu)
    check("B round trip overwrites the sentinel", not bool((got == -1.0).all()))
    check("B round trip is bit exact", bool(torch.equal(got, x_cpu)), f"got[:6]={got[:6].tolist()}")
    return report("TQue copy round trip (entry B)")


if __name__ == "__main__":
    raise SystemExit(main())
