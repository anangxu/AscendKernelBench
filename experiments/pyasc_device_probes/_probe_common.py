"""Shared helpers for the pyasc device probes.

Every probe in this directory is a standalone script. Run it on an Ascend host
with the CANN environment sourced and a Python that has torch, torch_npu and
pyasc installed:

    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python probe_r09_writeback_length.py

A probe prints one line per expectation, then a verdict, and exits non-zero
when any expectation failed, so a shell loop can gate on it. Probe modules
that define an @asc.jit kernel must not use "from __future__ import
annotations": pyasc calls issubclass on the annotation objects.

    for probe in probe_*.py; do python "$probe" || exit 1; done
"""

from __future__ import annotations

from collections.abc import Callable

_RESULTS: list[tuple[str, bool, str]] = []


def setup_device(device_id: int = 0):
    """Select the NPU platform and return the asc module and its runtime."""
    import asc
    import asc.lib.runtime as rt
    import asc.runtime.config as asc_config

    asc_config.set_platform(asc_config.Backend.NPU, None, device_id=device_id)
    return asc, rt


def expect(name: str, ok: bool, detail: str = "") -> bool:
    """Record and print one expectation; return the verdict for chaining."""
    ok = bool(ok)
    _RESULTS.append((name, ok, detail))
    suffix = f": {detail}" if detail else ""
    print(f"{'PASS' if ok else 'FAIL'} - {name}{suffix}")
    return ok


def report(title: str) -> int:
    """Print the summary and return the process exit code."""
    failed = [name for name, ok, _ in _RESULTS if not ok]
    total = len(_RESULTS)
    print(f"\n{title}: {total - len(failed)}/{total} expectations held")
    if failed:
        print("FAILED: " + "; ".join(failed))
        print("VERDICT: FAIL")
        return 1
    print("VERDICT: PASS")
    return 0


def run(title: str, body: Callable[[], None]) -> int:
    """Run one probe body and turn its expectations into an exit code."""
    body()
    return report(title)
