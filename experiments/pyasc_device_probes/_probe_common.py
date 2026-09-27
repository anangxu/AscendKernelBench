"""Shared helpers for the pyasc device probes.

Every probe in this directory is a standalone script. Run it on an Ascend host
with the CANN environment sourced and a Python that has torch, torch_npu and
pyasc installed:

    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    for probe in probe_*.py; do python "$probe" || exit 1; done

Two kinds of expectation are counted apart. check() is a correctness
requirement; reproduced() records that a known failure still happens. A probe
whose expectations are mostly reproduced() is measuring a limitation, not
proving an operator correct, and its green verdict does not mean the kernels
under test are right.

Probe modules that define an @asc.jit kernel must not use "from __future__
import annotations": pyasc calls issubclass on the annotation objects, and
kernels must be defined at module level so pyasc can resolve names such as asc.
"""

from __future__ import annotations

from collections.abc import Callable

_CHECKS: list[tuple[str, bool, str]] = []
_REPRODUCTIONS: list[tuple[str, bool, str]] = []


def setup_device(device_id: int = 0):
    """Select the NPU platform and return the asc module and its runtime."""
    import asc
    import asc.lib.runtime as rt
    import asc.runtime.config as asc_config

    asc_config.set_platform(asc_config.Backend.NPU, None, device_id=device_id)
    return asc, rt


def check(name: str, ok: bool, detail: str = "") -> bool:
    """Record a correctness requirement."""
    ok = bool(ok)
    _CHECKS.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'} - {name}" + (f": {detail}" if detail else ""))
    return ok


def reproduced(name: str, ok: bool, detail: str = "") -> bool:
    """Record that a known failure still happens; ok means it was reproduced."""
    ok = bool(ok)
    _REPRODUCTIONS.append((name, ok, detail))
    tag = "EXPECTED_FAILURE_REPRODUCED" if ok else "EXPECTED_FAILURE_NOT_REPRODUCED"
    print(f"{tag} - {name}" + (f": {detail}" if detail else ""))
    return ok


def report_metrics(label: str, got, want, sentinel: float = -1.0) -> None:
    """Print one machine-readable line about an output buffer.

    sentinel_uncovered counts elements still holding the sentinel, which is a
    write-back that did not happen; wrong_elems counts mismatches against the
    expected values; max_abs_err is the largest absolute difference.
    """
    got_cpu = got.detach().to("cpu").reshape(-1) if hasattr(got, "detach") else got
    want_cpu = want.detach().to("cpu").reshape(-1) if hasattr(want, "detach") else want
    uncovered = int((got_cpu == sentinel).sum().item())
    wrong = int((got_cpu != want_cpu).sum().item())
    max_err = float((got_cpu - want_cpu).abs().max().item()) if got_cpu.numel() else 0.0
    print(
        f"METRICS label={label} sentinel_uncovered={uncovered} "
        f"wrong_elems={wrong} max_abs_err={max_err:.6g}"
    )


def report(title: str) -> int:
    """Print the summary and return the process exit code."""
    failed = [name for name, ok, _ in _CHECKS if not ok]
    missing = [name for name, ok, _ in _REPRODUCTIONS if not ok]
    print(
        f"\n{title}: checks {len(_CHECKS) - len(failed)}/{len(_CHECKS)} held, "
        f"known failures reproduced {len(_REPRODUCTIONS) - len(missing)}/{len(_REPRODUCTIONS)}"
    )
    if _REPRODUCTIONS:
        print("(a reproduced failure is not a correctness pass)")
    if failed or missing:
        if failed:
            print("FAILED CHECKS: " + "; ".join(failed))
        if missing:
            print("NOT REPRODUCED: " + "; ".join(missing))
        print("VERDICT: FAIL")
        return 1
    print("VERDICT: PASS")
    return 0


def run(title: str, body: Callable[[], None]) -> int:
    """Run one probe body and turn its expectations into an exit code."""
    body()
    return report(title)
