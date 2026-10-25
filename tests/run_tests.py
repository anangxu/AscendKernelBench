"""Run the offline regression scripts without importing device runtimes."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    """Run each script in its own process and report any failures."""
    root = Path(__file__).resolve().parents[1]
    scripts = sorted((root / "tests").glob("test_*.py"))
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    failed = []
    for script in scripts:
        try:
            result = subprocess.run(
                [sys.executable, str(script)],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            print(f"FAIL {script.name}: exceeded 120 seconds", flush=True)
            failed.append(script.name)
            continue
        if result.returncode:
            print(f"FAIL {script.name}\n{result.stdout}\n{result.stderr}", flush=True)
            failed.append(script.name)
        else:
            print(f"PASS {script.name}", flush=True)
    print(f"{len(scripts) - len(failed)}/{len(scripts)} offline scripts passed")
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
