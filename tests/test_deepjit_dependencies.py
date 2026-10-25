"""Offline checks for DeepJIT host dependency diagnostics and search paths."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import deepjit_build
from src.config import DeepJitConfig
from src.deepjit_build import DeepJitBuildError, DeepJitPaths


class DeepJitDependencyTests(unittest.TestCase):
    def test_missing_elfutils_headers_explain_include_configuration(self):
        for detail in (
            "exception.hpp:3:10: fatal error: elfutils/libdwfl.h: No such file or directory",
            "exception.hpp:3:10: fatal error: 'elfutils/libdwfl.h' file not found",
            "libdwfl.h:30:10: fatal error: elfutils/libdw.h: No such file or directory",
            "libdw.h:32:10: fatal error: gelf.h: No such file or directory",
            "gelf.h:32:10: fatal error: libelf.h: No such file or directory",
        ):
            with self.subTest(detail=detail):
                message = deepjit_build._host_build_error(RuntimeError(detail))
                self.assertIn("deepjit.include_dirs", message)
                self.assertIn("DEEPJIT_INCLUDE_DIRS", message)
                self.assertIn("not include/elfutils", message)
                self.assertIn("docs/guide/deepjit.md", message)
                self.assertTrue(message.endswith(detail))

    def test_unrelated_compiler_errors_are_not_reclassified(self):
        for detail in (
            "custom_op_host.cpp:10: error: expected ';' before '}' token",
            "In file included from elfutils/libdwfl.h:3: error: invalid declaration",
            "fatal error: kernel_operator.h: No such file or directory",
        ):
            with self.subTest(detail=detail):
                self.assertEqual(
                    deepjit_build._host_build_error(RuntimeError(detail)),
                    f"deepjit host build failed: {detail}",
                )

    def test_real_host_build_failure_keeps_the_diagnostic_and_original_output(self):
        detail = "fatal error: elfutils/libdwfl.h: No such file or directory"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = self.paths(root)
            compiler = Mock(side_effect=RuntimeError(detail))
            with (
                patch.object(deepjit_build, "_prepare_layout", return_value=paths),
                patch.object(deepjit_build, "_extension_include_dirs", return_value=[]),
                patch.object(deepjit_build, "_extension_link_flags", return_value=[]),
                patch.dict(
                    sys.modules,
                    {
                        "torch.utils.cpp_extension": SimpleNamespace(load=compiler),
                    },
                ),
                patch.dict(deepjit_build.os.environ, {}, clear=False),
                self.assertRaises(DeepJitBuildError) as raised,
            ):
                deepjit_build._build_deepjit_op_locked(
                    "source",
                    root,
                    settings=DeepJitConfig(root=str(root)),
                )
            compiler.assert_called_once()
            self.assertIn("elfutils development headers", str(raised.exception))
            self.assertIn(detail, str(raised.exception))

    def test_extra_paths_work_without_forcing_libdw_linkage(self):
        root = Path("/configured")
        settings = DeepJitConfig(
            include_dirs=["/private-elfutils/include"],
            library_dirs=["/private-elfutils/lib"],
        )
        with (
            patch.object(
                deepjit_build, "_cann_paths", return_value=("/cann", "/cann/lib64")
            ),
            patch.object(deepjit_build, "_torch_npu_dir", return_value="/torch_npu"),
            patch.dict(
                sys.modules,
                {
                    "torch.utils.cpp_extension": SimpleNamespace(
                        include_paths=lambda: ["/torch/include"],
                        library_paths=lambda: ["/torch/lib"],
                    ),
                },
            ),
        ):
            includes = deepjit_build._extension_include_dirs(self.paths(root), settings)
            links = deepjit_build._extension_link_flags(settings)
            explicit = deepjit_build._extension_link_flags(
                settings.model_copy(update={"extra_ldflags": ["-ldw", "-lelf"]})
            )
        self.assertIn("/private-elfutils/include", includes)
        self.assertIn("-L/private-elfutils/lib", links)
        self.assertIn("-Wl,-rpath,/private-elfutils/lib", links)
        self.assertIn("-ldl", links)
        self.assertNotIn("-ldw", links)
        self.assertNotIn("-lelf", links)
        self.assertEqual(explicit[-2:], ["-ldw", "-lelf"])

    @staticmethod
    def paths(root: Path) -> DeepJitPaths:
        return DeepJitPaths(
            root=root,
            library_root=root / "deepjit",
            commit="a" * 40,
            cache_dir=root / "deepjit/cache",
            build_dir=root / "deepjit/build",
            kernel_source=root / "deepjit/kernel.hpp",
            signature="signature",
        )


if __name__ == "__main__":
    unittest.main()
