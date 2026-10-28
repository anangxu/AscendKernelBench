"""Offline checks for the task-suite layer: KernelBench plus a CANN fixture."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import _stubs  # noqa: E402

_stubs.install_engine_stubs()

from src import rundir, suite  # noqa: E402
from src.cli_util import (  # noqa: E402
    generation_run_config,
    resolve_generation_settings,
    select_tasks,
)
from src.config import load_eval_config, load_hardware_profile  # noqa: E402
from src.dataset import discover_tasks, load_task, task_meta  # noqa: E402
from src.prompt import build_prompt  # noqa: E402

FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "domain_suite"
CANN_ROOTS = {"cann": FIXTURE_ROOT / "cann"}
FIXTURE_ID = "cann/fixture_add"


def write_task(directory: Path, body: str) -> Path:
    """Write one task source and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "task.py"
    path.write_text(body, encoding="utf-8")
    return path


VALID_TASK = (
    "import torch\n"
    "import torch.nn as nn\n"
    "class Model(nn.Module):\n"
    "    def forward(self, x):\n"
    "        return x + 1\n"
    "def get_inputs():\n"
    "    return [torch.rand(4)]\n"
    "def get_init_inputs():\n"
    "    return []\n"
)


class KernelBenchCompatibilityTests(unittest.TestCase):
    """The historical corpus and its ids must keep working unchanged."""

    def test_default_discovery_is_still_kernelbench(self):
        tasks = discover_tasks()
        self.assertEqual(len(tasks), 270)
        self.assertTrue(all(item.suite == "kernelbench" for item in tasks))
        self.assertEqual(tasks[0].task_id, "level1/1_Square_matrix_multiplication_")
        self.assertEqual(tasks[0].level, 1)

    def test_level_filter_and_ids_stay_identical(self):
        level_one = discover_tasks(level=1)
        self.assertEqual(len(level_one), 100)
        task = load_task("level1/19_ReLU")
        self.assertEqual(task.level, 1)
        self.assertEqual(task.name, "19_ReLU")
        self.assertEqual(task.suite, "kernelbench")
        self.assertEqual(task.problem_id, "level1/19_ReLU")
        self.assertTrue(task.task_hash)
        self.assertEqual(task.provenance()["task_hash"], task.task_hash)

    def test_selection_precedence_is_unchanged(self):
        explicit = select_tasks(task_ids=["level1/19_ReLU"], level=2)
        self.assertEqual([item.task_id for item in explicit], ["level1/19_ReLU"])
        manifest = select_tasks(level=2)
        self.assertEqual(len(manifest), 100)

    def test_legacy_run_layout_still_scans(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "legacy"
            sample = run_dir / "level1" / "19_ReLU" / "sample_0"
            sample.mkdir(parents=True)
            (sample / "custom_op.asc").write_text("// kernel\n", encoding="utf-8")
            (sample / "model_new.py").write_text("import torch\n", encoding="utf-8")
            (sample / "eval_result.json").write_text(
                json.dumps({"correctness": True}), encoding="utf-8"
            )
            found = list(rundir.iter_sample_dirs(run_dir))
            self.assertEqual([item[0] for item in found], ["level1/19_ReLU"])
            self.assertEqual(rundir.resolve_run_backend(run_dir).value, "ascendc")
            results = rundir.collect_eval_results(run_dir)
            self.assertEqual(list(results), ["level1/19_ReLU"])
            self.assertEqual(len(list(rundir.iter_sample_dirs(run_dir, level=1))), 1)
            self.assertEqual(len(list(rundir.iter_sample_dirs(run_dir, level=2))), 0)

    def test_kernelbench_corpus_is_untouched(self):
        files = sorted((ROOT / "KernelBench").rglob("*.py"))
        self.assertEqual(len(files), 270)
        self.assertEqual(
            [item.task_id for item in discover_tasks(suite="cann")],
            ["cann/nn/rmsnorm", "cann/nn/swiglu"],
        )  # noqa: E501


class CannSuiteTests(unittest.TestCase):
    """A namespaced suite loads, prompts, and validates without a level."""

    def test_fixture_loads_without_a_level(self):
        task = load_task(FIXTURE_ID, roots=CANN_ROOTS)
        self.assertEqual(task.suite, "cann")
        self.assertIsNone(task.level)
        self.assertEqual(task.name, "fixture_add")
        self.assertEqual(task.source, "test-fixture")
        self.assertEqual(task.source_revision, "phase-1")
        self.assertEqual(task.category, "elementwise")
        self.assertEqual(task.provenance()["suite"], "cann")
        self.assertEqual(len(task.task_hash), 64)

    def test_suite_discovery_finds_the_fixture(self):
        tasks = discover_tasks(suite="cann", roots=CANN_ROOTS)
        self.assertEqual([item.task_id for item in tasks], [FIXTURE_ID])
        selected = select_tasks(task_ids=[FIXTURE_ID], roots=CANN_ROOTS)
        self.assertEqual([item.task_id for item in selected], [FIXTURE_ID])

    def test_tasks_file_can_select_a_cann_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "tasks.txt"
            manifest.write_text(f"{FIXTURE_ID}\n", encoding="utf-8")
            tasks = select_tasks(tasks_file=manifest, roots=CANN_ROOTS)
            self.assertEqual([item.task_id for item in tasks], [FIXTURE_ID])

    def test_prompt_contains_the_reference_source(self):
        task = load_task(FIXTURE_ID, roots=CANN_ROOTS)
        hardware = load_hardware_profile("ascend910b2")
        prompt = build_prompt(task, hardware, mode="zero_shot")
        self.assertIn("class Model(nn.Module)", prompt)
        self.assertIn("def get_inputs", prompt)
        self.assertNotIn("Level 1", prompt)

    def test_invalid_ids_are_refused(self):
        for bad in ["", "cann", "cann/", "/cann/x", "cann/../escape", "cann/./x"]:
            with self.assertRaises(ValueError, msg=bad):
                load_task(bad, roots=CANN_ROOTS)
        with self.assertRaises(FileNotFoundError):
            load_task("cann/missing_task", roots=CANN_ROOTS)

    def test_path_escape_is_refused(self):
        with self.assertRaises(ValueError):
            suite.task_dir("cann/../outside", CANN_ROOTS)
        with self.assertRaises(ValueError):
            suite.suite_of("cann//double")

    def test_missing_contract_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_task(root / "broken", "def unrelated():\n    return 1\n")
            with self.assertRaises(ValueError) as caught:
                load_task("cann/broken", roots={"cann": root})
            self.assertIn("missing contract symbols", str(caught.exception))


class SuitePathAndLegacyTests(unittest.TestCase):
    """Nested ids, symlink escapes, and records written before provenance."""

    def test_nested_cann_id_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "cann"
            nested = root / "nn" / "swiglu"
            nested.mkdir(parents=True)
            nested.joinpath("task.py").write_text(VALID_TASK, encoding="utf-8")
            nested.joinpath("meta.yaml").write_text(
                "source: ops-nn\nsource_revision: r1\ncategory: nn\n", encoding="utf-8"
            )
            task = load_task("cann/nn/swiglu", roots={"cann": root})
            self.assertEqual(task.task_id, "cann/nn/swiglu")
            self.assertEqual(task.name, "swiglu")
            self.assertEqual(task.source, "ops-nn")
            self.assertIsNone(task.level)
            self.assertEqual(
                [item.task_id for item in discover_tasks(suite="cann", roots={"cann": root})],
                ["cann/nn/swiglu"],
            )

    def test_symlink_escape_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "cann"
            root.mkdir()
            outside = Path(tmp) / "outside"
            outside.mkdir()
            (outside / "task.py").write_text(VALID_TASK, encoding="utf-8")
            (root / "link").symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ValueError):
                load_task("cann/link", roots={"cann": root})

    def test_legacy_record_without_provenance_still_works(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "legacy"
            sample = run_dir / "level1" / "19_ReLU" / "sample_0"
            sample.mkdir(parents=True)
            (sample / "custom_op.asc").write_text("// kernel\n", encoding="utf-8")
            (sample / "model_new.py").write_text("import torch\n", encoding="utf-8")
            (run_dir / "generation_config.yaml").write_text(
                "model: offline\nbackend: ascendc\ntasks:\n- level1/19_ReLU\n",
                encoding="utf-8",
            )
            self.assertEqual(rundir.recorded_task_meta(run_dir), {})
            rundir.verify_task_meta(run_dir, [load_task("level1/19_ReLU")])
            self.assertEqual(
                [item[0] for item in rundir.iter_sample_dirs(run_dir)],
                ["level1/19_ReLU"],
            )


class CannRunDirectoryTests(unittest.TestCase):
    """Runs, scanning, aggregation, and provenance live under the full id."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.runs = Path(self.temp.name)
        self.root_patch = patch.object(rundir, "RUNS_DIR", self.runs)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.task = load_task(FIXTURE_ID, roots=CANN_ROOTS)
        self.config = load_eval_config()
        self.hardware = load_hardware_profile("ascend910b2")

    def _settings(self, **overrides):
        return resolve_generation_settings(
            self.config, model="offline", num_samples=1, **overrides
        )

    def _run(self, name: str = "cann_run", settings=None) -> Path:
        record = generation_run_config(
            settings or self._settings(),
            hardware_name=self.hardware.name,
            task_ids=[self.task.task_id],
            task_meta=task_meta([self.task]),
        )
        return rundir.create_run(name, record, resume=True)

    def test_sample_path_keeps_the_full_id(self):
        run_dir = self._run()
        sample = rundir.sample_dir(run_dir, FIXTURE_ID, 0)
        self.assertEqual(sample, run_dir / "cann" / "fixture_add" / "sample_0")
        self.assertEqual(
            rundir.sample_dir(run_dir, FIXTURE_ID, 0).relative_to(run_dir).as_posix(),
            "cann/fixture_add/sample_0",
        )

    def test_generated_cann_sample_is_scanned_and_aggregated(self):
        run_dir = self._run()
        sample = rundir.sample_dir(run_dir, FIXTURE_ID, 0)
        sample.mkdir(parents=True)
        (sample / "custom_op.asc").write_text("// kernel\n", encoding="utf-8")
        (sample / "model_new.py").write_text("import torch\n", encoding="utf-8")
        (sample / "eval_result.json").write_text(
            json.dumps({"correctness": True}), encoding="utf-8"
        )
        found = list(rundir.iter_sample_dirs(run_dir))
        self.assertEqual([item[0] for item in found], [FIXTURE_ID])
        self.assertEqual(rundir.resolve_run_backend(run_dir).value, "ascendc")
        self.assertEqual(list(rundir.iter_sample_dirs(run_dir, suite="cann")), found)
        self.assertEqual(
            list(rundir.iter_sample_dirs(run_dir, suite="kernelbench")), []
        )
        results = rundir.collect_eval_results(run_dir)
        self.assertEqual(list(results), [FIXTURE_ID])
        self.assertEqual(results[FIXTURE_ID][0]["sample_id"], 0)

    def test_run_record_keeps_provenance(self):
        run_dir = self._run()
        recorded = rundir._read_generation_config(run_dir)["task_meta"][FIXTURE_ID]
        self.assertEqual(recorded["suite"], "cann")
        self.assertEqual(recorded["source"], "test-fixture")
        self.assertEqual(recorded["task_hash"], self.task.task_hash)

    def test_resume_accepts_the_same_source(self):
        self._run()
        self._run()

    def test_resume_refuses_a_changed_source(self):
        run_dir = self._run()
        payload = dict(self.task.__dict__)
        payload["task_py"] = self.task.task_py + "\n"
        changed = self.task.__class__(**payload)
        with self.assertRaises(ValueError):
            rundir.verify_task_meta(run_dir, [changed])
        rundir.verify_task_meta(run_dir, [self.task])

    def test_provenance_change_is_refused_even_with_the_same_source(self):
        run_dir = self._run()
        changed = self.task.__class__(**{**self.task.__dict__, "source_revision": "phase-2"})
        with self.assertRaises(ValueError) as caught:
            rundir.verify_task_meta(run_dir, [changed])
        self.assertIn("source_revision", str(caught.exception))
        stale = task_meta([changed])
        record = generation_run_config(
            self._settings(),
            hardware_name=self.hardware.name,
            task_ids=[FIXTURE_ID],
            task_meta=stale,
        )
        with self.assertRaises(ValueError):
            rundir.create_run("cann_run", record, resume=True)

    def test_generation_refuses_a_changed_source(self):
        self._run()
        stale = task_meta([self.task])
        stale[FIXTURE_ID]["task_hash"] = "0" * 64
        record = generation_run_config(
            self._settings(),
            hardware_name=self.hardware.name,
            task_ids=[FIXTURE_ID],
            task_meta=stale,
        )
        with self.assertRaises(ValueError):
            rundir.create_run("cann_run", record, resume=True)

    def test_mixed_run_scans_filters_and_aggregates_both_suites(self):
        run_dir = self._run("mixed_run")
        cann = rundir.sample_dir(run_dir, FIXTURE_ID, 0)
        cann.mkdir(parents=True)
        (cann / "custom_op.asc").write_text("// kernel\n", encoding="utf-8")
        (cann / "model_new.py").write_text("import torch\n", encoding="utf-8")
        (cann / "eval_result.json").write_text(
            json.dumps({"correctness": True}), encoding="utf-8"
        )
        legacy = run_dir / "level1" / "19_ReLU" / "sample_0"
        legacy.mkdir(parents=True)
        (legacy / "custom_op.asc").write_text("// kernel\n", encoding="utf-8")
        (legacy / "model_new.py").write_text("import torch\n", encoding="utf-8")
        (legacy / "eval_result.json").write_text(
            json.dumps({"correctness": True}), encoding="utf-8"
        )
        both = [item[0] for item in rundir.iter_sample_dirs(run_dir)]
        self.assertEqual(both, [FIXTURE_ID, "level1/19_ReLU"])
        self.assertEqual(
            [item[0] for item in rundir.iter_sample_dirs(run_dir, suite="cann")],
            [FIXTURE_ID],
        )
        self.assertEqual(
            [item[0] for item in rundir.iter_sample_dirs(run_dir, suite="kernelbench")],
            ["level1/19_ReLU"],
        )
        self.assertEqual(
            [item[0] for item in rundir.iter_sample_dirs(run_dir, level=1)],
            ["level1/19_ReLU"],
        )
        results = rundir.collect_eval_results(run_dir)
        self.assertEqual(sorted(results), sorted(["level1/19_ReLU", FIXTURE_ID]))

    def test_old_kernelbench_run_is_unaffected(self):
        legacy = self.runs / "legacy_kb"
        sample = legacy / "level1" / "19_ReLU" / "sample_0"
        sample.mkdir(parents=True)
        (sample / "custom_op.asc").write_text("// kernel\n", encoding="utf-8")
        (sample / "model_new.py").write_text("import torch\n", encoding="utf-8")
        found = list(rundir.iter_sample_dirs(legacy))
        self.assertEqual([item[0] for item in found], ["level1/19_ReLU"])
        self.assertEqual(list(rundir.iter_sample_dirs(legacy, suite="cann")), [])
        self.assertEqual(list(rundir.iter_sample_dirs(legacy, level=1)), found)


if __name__ == "__main__":
    unittest.main(verbosity=2)
