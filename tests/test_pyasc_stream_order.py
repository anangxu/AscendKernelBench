"""Check cross-stream input and mutation-snapshot ordering without an NPU."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import eval_device
from src.backend import Backend
from src.eval_device import DeviceEvalRequest, SampleEvaluator


class QueuedDevice:
    """Model ordered Torch work plus immediate independent-stream work."""

    def __init__(self):
        self.pending = []
        self.events = []
        self.snapshots = []

    def synchronize(self, *, device=None):
        self.events.append("sync")
        while self.pending:
            self.pending.pop(0)()


class QueuedTensor:
    """Delay a snapshot's read until its stream is synchronized."""

    def __init__(self, device, value):
        self.device = device
        self.value = value

    def clone(self):
        result = QueuedTensor(self.device, None)
        self.device.snapshots.append(result)
        self.device.events.append("snapshot_queued")

        def finish():
            result.value = self.value
            self.device.events.append("snapshot_ready")

        self.device.pending.append(finish)
        return result


def make_evaluator(backend):
    request = DeviceEvalRequest(
        task_py="",
        sample_dir="/unused",
        cmake_arch="dav-2201",
        hardware_name="ascend910b2",
        device="npu:0",
        seed=42,
        num_correct_trials=1,
        num_perf_trials=1,
        num_warmup=0,
        precision="fp32",
        tolerances={},
        excessive_speedup=10.0,
        build_timeout=1,
        backend=backend.value,
    )
    evaluator = SampleEvaluator(request)
    device = QueuedDevice()
    torch = SimpleNamespace(
        npu=device,
        Tensor=QueuedTensor,
        equal=lambda left, right: left.value == right.value,
    )
    evaluator.torch = torch
    evaluator._process_input = lambda value: value
    return evaluator, device, torch


class PyascStreamOrderTests(unittest.TestCase):
    def test_first_call_finishes_pending_input_work_before_launch(self):
        evaluator, device, _ = make_evaluator(Backend.PYASC)
        value = QueuedTensor(device, None)

        def prepare():
            device.events.append("input_queued")

            def finish():
                value.value = 5
                device.events.append("input_ready")

            device.pending.append(finish)
            return [value]

        def candidate(item):
            device.events.append("candidate")
            self.assertEqual(item.value, 5)
            return item.value

        evaluator.get_inputs = prepare
        evaluator.new_model = candidate
        evaluator._unsupported_pyasc_dtypes = lambda values: []
        evaluator._record_first_call_correctness = Mock()
        evaluator._construct_models = Mock(return_value=None)
        with (
            patch.object(eval_device, "seed_torch"),
            patch.object(eval_device, "runtime_facts", return_value={}),
        ):
            self.assertIsNone(evaluator._warm_up_pyasc())
        self.assertTrue(evaluator.compiled)
        self.assertEqual(
            device.events,
            ["input_queued", "sync", "input_ready", "candidate", "sync"],
        )

    def assert_mutation_detected(self, backend, hidden):
        evaluator, device, torch = make_evaluator(backend)
        value = QueuedTensor(device, 1)
        evaluator.get_inputs = lambda: [value]
        evaluator._run_reference = lambda *args, **kwargs: 1
        evaluator._outputs_ok = lambda reference, candidate: reference == candidate

        def candidate(item):
            device.events.append("candidate")

            def mutate():
                item.value = 7
                device.events.append("mutated")

            if backend is Backend.PYASC:
                mutate()
            else:
                device.pending.append(mutate)
            return 1

        evaluator.new_model = candidate
        with (
            patch.dict(sys.modules, {"torch": torch}),
            patch.object(eval_device, "seed_torch"),
            patch.object(
                eval_device,
                "perturb_floating_inputs",
                side_effect=lambda values, scale: values,
            ),
        ):
            if hidden:
                result = evaluator._one_hidden_distribution("d1", 1.0, [value])
            else:
                result = evaluator._one_correctness_trial(0, 42)
        self.assertIsNotNone(result)
        metadata = result if hidden else result["metadata"]
        self.assertIn("mutated its inputs", metadata["runtime_error"])
        self.assertEqual(device.snapshots[0].value, 1)
        self.assertEqual(value.value, 7)
        if backend is Backend.PYASC:
            self.assertEqual(device.events.count("sync"), 3)
            self.assertLess(
                device.events.index("snapshot_ready"),
                device.events.index("candidate"),
            )
        else:
            self.assertEqual(device.events.count("sync"), 2)
            self.assertLess(
                device.events.index("snapshot_ready"),
                device.events.index("mutated"),
            )

    def test_pyasc_regular_trial_cannot_race_its_snapshot(self):
        self.assert_mutation_detected(Backend.PYASC, hidden=False)

    def test_pyasc_hidden_trial_cannot_race_its_snapshot(self):
        self.assert_mutation_detected(Backend.PYASC, hidden=True)

    def test_ascendc_regular_trial_keeps_its_existing_stream_order(self):
        self.assert_mutation_detected(Backend.ASCENDC, hidden=False)

    def test_ascendc_hidden_trial_keeps_its_existing_stream_order(self):
        self.assert_mutation_detected(Backend.ASCENDC, hidden=True)


if __name__ == "__main__":
    unittest.main()
