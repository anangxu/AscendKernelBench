"""Regression tests for backend selection and safe generation continuation."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import generate

from src import rundir
from src.backend import Backend, validate_backend_build
from src.cli_util import EvalRuntime, generation_run_config, resolve_generation_settings
from src.config import (
    BuildBackend,
    DeepJitConfig,
    load_eval_config,
    load_hardware_profile,
)
from src.dataset import load_task
from src.llm import (
    AscendCGeneration,
    GenerationResult,
    PyAscGeneration,
    extract_generation,
)
from src.prompt import build_prompt, load_examples, system_prompt


class BackendGenerationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root_patch = patch.object(rundir, "RUNS_DIR", Path(self.temp.name))
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.config = load_eval_config()
        self.hardware = load_hardware_profile("ascend910b2")
        self.task = "level1/19_ReLU"

    def record(self, name="resume", **kwargs):
        settings = resolve_generation_settings(self.config, **kwargs)
        return rundir.create_run(
            name,
            generation_run_config(
                settings, hardware_name=self.hardware.name, task_ids=[self.task]
            ),
        )

    def invoke(self, args, config=None, result=None):
        client = MagicMock()
        if result is None:
            result = GenerationResult(
                generation=AscendCGeneration(
                    custom_op_asc="source", model_new_py="wrapper"
                ),
                raw_text="response",
            )
        client.return_value.generate.return_value = result
        runtime = EvalRuntime(config=config or self.config, hardware=self.hardware)
        with (
            patch.object(generate, "LLMClient", client),
            patch.object(generate, "load_eval_runtime", return_value=runtime),
            patch.object(sys, "argv", ["generate.py", *args]),
        ):
            with self.assertRaises(SystemExit) as stopped:
                generate.main()
        return stopped.exception.code, client

    def test_supported_pairs_and_prompt_pools(self):
        task = load_task(self.task)
        for backend, build in (
            ("ascendc", "cmake"),
            ("ascendc", "deepjit"),
            ("pyasc", "cmake"),
        ):
            with self.subTest(backend=backend, build=build):
                pair = validate_backend_build(backend, build)
                self.assertEqual(pair, (Backend(backend), BuildBackend(build)))
                prompt = build_prompt(
                    task, self.hardware, backend=backend, build_backend=build
                )
                examples = load_examples(build, backend=backend)
                self.assertEqual(len(examples), 2)
                self.assertIn(
                    "```kernel.py" if backend == "pyasc" else "```custom_op.asc", prompt
                )
                if build == "deepjit":
                    self.assertIn("DeepJIT", system_prompt(build, backend=backend))
                if backend == "pyasc":
                    self.assertNotIn("```custom_op.asc", prompt)
        for call in (
            lambda: validate_backend_build("pyasc", "deepjit"),
            lambda: resolve_generation_settings(
                self.config, backend="pyasc", build_backend="deepjit"
            ),
            lambda: build_prompt(
                task, self.hardware, backend="pyasc", build_backend="deepjit"
            ),
        ):
            with self.assertRaisesRegex(ValueError, "requires backend=ascendc"):
                call()

    def test_resume_preserves_null_and_entire_deepjit_block(self):
        settings = {
            "root": "/old/root",
            "commit": "a" * 40,
            "cxx": "/old/cxx",
            "include_dirs": ["/old/include"],
            "library_dirs": ["/old/lib"],
            "env": {"SETTING": "old"},
            "extra_cflags": ["-O2"],
            "extra_ldflags": ["-lcustom"],
            "cache_dir": "/old/cache",
        }
        run = self.record(
            reasoning_effort=None, build_backend="deepjit", deepjit=settings
        )
        before = (run / "generation_config.yaml").read_bytes()
        config = self.config.model_copy(
            update={"deepjit": DeepJitConfig(root="/new/root")}
        )
        code, client = self.invoke(
            ["--run-name", "resume", "--task", self.task], config
        )
        self.assertEqual(code, 0)
        self.assertIsNone(client.call_args.kwargs["reasoning_effort"])
        self.assertEqual(before, (run / "generation_config.yaml").read_bytes())
        self.assertEqual(rundir.generation_deepjit(run), settings)

    def test_completed_resume_never_constructs_client(self):
        run = self.record()
        rundir.save_sample(
            run,
            self.task,
            0,
            prompt="original",
            generation=AscendCGeneration(
                custom_op_asc="original kernel", model_new_py="original wrapper"
            ),
        )
        result_file = rundir.sample_dir(run, self.task, 0) / "eval_result.json"
        result_file.write_text('{"correctness": true}')
        code, client = self.invoke(["--run-name", "resume"])
        self.assertEqual(code, 0)
        client.assert_not_called()
        self.assertEqual(result_file.read_text(), '{"correctness": true}')

    def test_custom_hardware_file_is_inherited(self):
        profile = Path(self.temp.name) / "hardware.yaml"
        profile.write_text((ROOT / "configs/hardware/ascend910b2.yaml").read_text())
        payload = generation_run_config(
            resolve_generation_settings(self.config),
            hardware_name=self.hardware.name,
            task_ids=[self.task],
            hardware_file=str(profile),
        )
        run = rundir.create_run("hardware", payload)
        rundir.save_sample(
            run,
            self.task,
            0,
            prompt="p",
            generation=AscendCGeneration(
                custom_op_asc="source", model_new_py="wrapper"
            ),
        )
        with patch.object(
            generate, "load_eval_runtime", wraps=generate.load_eval_runtime
        ) as load:
            with patch.object(generate, "LLMClient") as client:
                with patch.object(
                    sys, "argv", ["generate.py", "--run-name", "hardware"]
                ):
                    with self.assertRaises(SystemExit) as stopped:
                        generate.main()
        self.assertEqual(stopped.exception.code, 0)
        self.assertEqual(load.call_args.kwargs["hardware"], str(profile))
        client.assert_not_called()
        self.assertEqual(rundir.generation_hardware_file(run), str(profile))

    def test_partial_sample_refused_before_api_or_config_write(self):
        run = self.record()
        before = (run / "generation_config.yaml").read_bytes()
        sample = rundir.sample_dir(run, self.task, 0)
        sample.mkdir(parents=True)
        (sample / "custom_op.asc").write_text("partial")
        code, client = self.invoke(["--run-name", "resume"])
        self.assertEqual(code, 1)
        client.assert_not_called()
        self.assertEqual(before, (run / "generation_config.yaml").read_bytes())

    def test_pyasc_cli_uses_python_artifacts(self):
        result = GenerationResult(
            generation=PyAscGeneration(
                kernel_py="import asc\n@asc.jit\ndef add(): pass\n",
                model_new_py="class ModelNew: pass\n",
            ),
            raw_text="pyasc response",
        )
        code, client = self.invoke(
            ["--backend", "pyasc", "--run-name", "pyasc", "--task", self.task],
            result=result,
        )
        self.assertEqual(code, 0)
        self.assertEqual(client.call_args.kwargs["backend"], Backend.PYASC)
        run = Path(self.temp.name) / "pyasc"
        self.assertEqual(rundir.resolve_run_backend(run), Backend.PYASC)
        sample = rundir.sample_dir(run, self.task, 0)
        self.assertTrue((sample / "kernel.py").is_file())
        self.assertFalse((sample / "custom_op.asc").exists())

    def test_mixed_artifacts_rejected_even_with_matching_sample(self):
        run = self.record()
        sample = rundir.sample_dir(run, self.task, 0)
        sample.mkdir(parents=True)
        for name in ("custom_op.asc", "kernel.py", "model_new.py"):
            (sample / name).write_text("source")
        with self.assertRaisesRegex(ValueError, "incompatible backend artifacts"):
            list(rundir.iter_sample_dirs(run))

    def test_json_and_filename_blocks_remain_backend_specific(self):
        for backend, field, filename in (
            ("ascendc", "custom_op_asc", "custom_op.asc"),
            ("pyasc", "kernel_py", "kernel.py"),
        ):
            payload = {field: "kernel source", "model_new_py": "class ModelNew: pass"}
            parsed = extract_generation(json.dumps(payload), backend=backend)
            self.assertEqual(getattr(parsed, field), "kernel source")
            fenced = f"```{filename}\nkernel source\n```\n```model_new.py\nclass ModelNew: pass\n```"
            parsed = extract_generation(fenced, backend=backend)
            self.assertEqual(getattr(parsed, field).strip(), "kernel source")


if __name__ == "__main__":
    unittest.main()
