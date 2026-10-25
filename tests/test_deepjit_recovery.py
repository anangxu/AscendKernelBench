"""Device-free checks for dependency changes and interrupted builds."""

from __future__ import annotations

import os
import select
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DeepJitConfig
from src.deepjit_build import _dependency_digest, _signature


class DeepJitRecoveryTests(unittest.TestCase):
    def test_successive_dependency_edits_invalidate_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            include = root / "include"
            include.mkdir()
            external = root / "external"
            external.mkdir()
            header = include / "compiler.hpp"
            helper = external / "helper.h"
            helper.write_text("constexpr int scale = 1;\n")
            config = DeepJitConfig(root=str(root), include_dirs=[str(external)])
            hashes = []
            with patch("src.build.cann_env", return_value={}):
                for value in (1, 2, 3):
                    header.write_text(f"constexpr int version = {value};\n")
                    hashes.append(_dependency_digest(root, config))
                self.assertEqual(len(set(hashes)), 3)
                self.assertEqual(hashes[-1], _dependency_digest(root, config))
                helper.write_text("constexpr int scale = 2;\n")
                self.assertNotEqual(hashes[-1], _dependency_digest(root, config))
            with patch("src.deepjit_build._bisheng_version", return_value="compiler"):
                self.assertNotEqual(
                    _signature("commit", "soc", "dirty", hashes[0]),
                    _signature("commit", "soc", "dirty", hashes[1]),
                )

    def test_killed_builder_releases_guard_and_stale_torch_lock(self) -> None:
        script = """
import sys
from pathlib import Path
from src.deepjit_build import _build_guard
root = Path(sys.argv[1])
with _build_guard(root):
    lock = root / "deepjit/build/lock"
    if sys.argv[2] == "owner":
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.touch()
        print("ready", flush=True)
        sys.stdin.read()
    else:
        assert not lock.exists()
        print("recovered", flush=True)
"""
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ)
            env["PYTHONPATH"] = os.pathsep.join(
                [str(Path(__file__).resolve().parents[1]), env.get("PYTHONPATH", "")]
            )
            owner = subprocess.Popen(
                [sys.executable, "-c", script, tmp, "owner"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
            waiter = None
            try:
                self.assertTrue(select.select([owner.stdout], [], [], 10)[0])
                self.assertEqual(owner.stdout.readline().strip(), "ready")
                waiter = subprocess.Popen(
                    [sys.executable, "-c", script, tmp, "waiter"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env,
                )
                self.assertFalse(select.select([waiter.stdout], [], [], 0.3)[0])
                self.assertTrue((Path(tmp) / "deepjit/build/lock").exists())
                owner.kill()
                owner.communicate(timeout=10)
                output, errors = waiter.communicate(timeout=10)
                self.assertEqual(waiter.returncode, 0, errors)
                self.assertEqual(output.strip(), "recovered")
            finally:
                if owner.poll() is None:
                    owner.kill()
                owner.communicate(timeout=10)
                if waiter is not None:
                    if waiter.poll() is None:
                        waiter.kill()
                    waiter.communicate(timeout=10)


if __name__ == "__main__":
    unittest.main()
