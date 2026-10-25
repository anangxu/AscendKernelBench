"""Offline regression tests for synchronized timing and NPU-only pyasc runs."""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import pyasc_runtime, timing


class FakeDevice:
    """Clock with queued device work and observable synchronization."""

    def __init__(self):
        self.now = 0.0
        self.pending = 0.0
        self.device = 7
        self.events = []
        self.calls = 0

    def current_device(self):
        return self.device

    def set_device(self, device):
        self.device = device
        self.events.append(("device", device))

    def synchronize(self, *, device):
        self.events.append(("sync", device))
        self.now += self.pending
        self.pending = 0.0

    def empty_cache(self):
        self.events.append(("empty",))

    def prepare(self):
        self.events.append(("prepare",))
        self.now += 0.1
        self.pending += 0.02

    def flush(self, *, device, size_bytes):
        self.events.append(("flush", device, size_bytes))
        self.now += 0.2
        self.pending += 0.03

    def call(self, value):
        self.events.append(("call", value))
        self.calls += 1
        self.now += 0.002
        self.pending += 0.008 * self.calls

    def clock(self):
        self.events.append(("clock",))
        return self.now


class TimingScopeTests(unittest.TestCase):
    def test_wall_timer_covers_dispatch_and_device_completion(self):
        device = FakeDevice()
        with (
            patch.dict(sys.modules, {"torch": SimpleNamespace(npu=device)}),
            patch.object(timing, "clear_l2_cache", device.flush),
            patch.object(timing.time, "perf_counter", device.clock),
        ):
            elapsed = timing.time_execution_with_synchronized_wall(
                device.call,
                ["input"],
                num_warmup=1,
                num_trials=2,
                device="npu:1",
                l2_clear_size=4096,
                setup=device.prepare,
            )
        # Call one warms up, call two is discarded, and calls three/four remain.
        self.assertEqual(len(elapsed), 2)
        self.assertAlmostEqual(elapsed[0], 26.0)
        self.assertAlmostEqual(elapsed[1], 34.0)
        self.assertEqual(device.device, 7)
        self.assertEqual(device.pending, 0.0)
        self.assertEqual(
            device.events[:5],
            [
                ("device", "npu:1"),
                ("prepare",),
                ("call", "input"),
                ("sync", "npu:1"),
                ("empty",),
            ],
        )
        expected_trial = [
            ("prepare",),
            ("flush", "npu:1", 4096),
            ("sync", "npu:1"),
            ("clock",),
            ("call", "input"),
            ("sync", "npu:1"),
            ("clock",),
        ]
        self.assertEqual(device.events[5:-1], expected_trial * 3)

    def test_wall_timer_restores_device_on_failure(self):
        device = FakeDevice()
        for stage in ("prepare", "flush", "call"):
            with self.subTest(stage=stage):
                fail = Mock(side_effect=RuntimeError(stage))
                with (
                    patch.dict(sys.modules, {"torch": SimpleNamespace(npu=device)}),
                    patch.object(
                        timing,
                        "clear_l2_cache",
                        fail if stage == "flush" else device.flush,
                    ),
                    patch.object(timing.time, "perf_counter", device.clock),
                    self.assertRaisesRegex(RuntimeError, stage),
                ):
                    timing.time_execution_with_synchronized_wall(
                        fail if stage == "call" else device.call,
                        ["input"],
                        num_warmup=0,
                        num_trials=1,
                        device="npu:1",
                        l2_clear_size=4096,
                        setup=fail if stage == "prepare" else device.prepare,
                    )
                self.assertEqual(device.device, 7)

    def test_adaptive_wall_timer_refreshes_equal_seeded_inputs(self):
        device = FakeDevice()
        rng = random.Random()
        candidate_inputs = []
        reference_inputs = []

        def draw():
            device.prepare()
            return [rng.random()]

        def candidate(value):
            candidate_inputs.append(value)
            device.call(value)

        def reference(value):
            reference_inputs.append(value)
            device.call(value)

        with (
            patch.dict(sys.modules, {"torch": SimpleNamespace(npu=device)}),
            patch.object(timing, "clear_l2_cache", device.flush),
            patch.object(timing.time, "perf_counter", device.clock),
            patch.object(timing, "seed_torch", rng.seed),
            patch.object(timing, "tensor_nbytes", return_value=128),
            patch.object(
                timing,
                "time_execution_with_npu_event",
                side_effect=AssertionError("event timer used"),
            ),
        ):
            timer = timing.AdaptiveInputTimer(
                draw,
                seed=42,
                num_warmup=1,
                num_trials=2,
                device="npu:1",
                l2_clear_size=4096,
                timing_method="synchronized_wall",
            )
            self.assertTrue(timer.fresh_per_trial)
            self.assertEqual(len(timer.time(candidate)), 2)
            self.assertEqual(len(timer.time(reference)), 2)
        self.assertEqual(len(candidate_inputs), 4)
        self.assertEqual(candidate_inputs, reference_inputs)
        self.assertEqual(len(set(candidate_inputs)), 4)
        self.assertEqual(device.events.count(("prepare",)), 9)

    def test_large_inputs_keep_one_allocation_and_default_to_events(self):
        draw = Mock(return_value=[object()])
        recorded = []

        def measure(fn, args, **kwargs):
            self.assertIsNone(kwargs["setup"])
            recorded.append(kwargs)
            fn(*args)
            return [1.0]

        candidate, reference = Mock(), Mock()
        with (
            patch.object(timing, "seed_torch"),
            patch.object(
                timing,
                "tensor_nbytes",
                return_value=timing.REFRESH_INPUT_BYTES_LIMIT + 1,
            ),
            patch.object(timing, "time_execution_with_npu_event", measure),
            patch.object(
                timing,
                "time_execution_with_synchronized_wall",
                side_effect=AssertionError("wall timer used"),
            ),
        ):
            timer = timing.AdaptiveInputTimer(
                draw,
                seed=42,
                num_warmup=1,
                num_trials=2,
                device="npu:0",
                l2_clear_size=4096,
            )
            timer.time(candidate)
            timer.time(reference)
        draw.assert_called_once()
        self.assertFalse(timer.fresh_per_trial)
        self.assertEqual(candidate.call_args, reference.call_args)
        self.assertEqual(recorded[0], recorded[1])

    def test_invalid_timing_samples_are_rejected(self):
        for values in (
            [],
            [0.0],
            [-1.0],
            [float("nan")],
            [1.0, float("inf")],
            [float("-inf")],
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                timing.get_timing_stats(values)
        self.assertEqual(
            timing.get_timing_stats([1.0, 3.0]),
            {
                "mean": 2.0,
                "std": 1.0,
                "min": 1.0,
                "max": 3.0,
                "num_trials": 2,
            },
        )

    def test_pyasc_simulator_is_rejected_before_runtime_import(self):
        for requested in ("model", "invalid", ""):
            with (
                self.subTest(requested=requested),
                patch.dict(
                    pyasc_runtime.os.environ,
                    {"ASCEND_KERNEL_BENCH_PYASC_BACKEND": requested},
                ),
                patch.object(pyasc_runtime.importlib, "import_module") as importer,
                self.assertRaisesRegex(
                    pyasc_runtime.PyAscError, "requires the live NPU"
                ),
            ):
                pyasc_runtime.configure_platform(SimpleNamespace(), 0)
            importer.assert_not_called()

    def test_pyasc_selects_requested_npu_and_live_platform(self):
        config = SimpleNamespace(
            Backend=SimpleNamespace(NPU="NPU"), set_platform=Mock()
        )
        runtime = SimpleNamespace(current_platform=lambda: "Ascend910B4")
        with (
            patch.dict(
                pyasc_runtime.os.environ, {"ASCEND_KERNEL_BENCH_PYASC_BACKEND": "npu"}
            ),
            patch.object(
                pyasc_runtime.importlib, "import_module", side_effect=[config, runtime]
            ),
        ):
            soc = pyasc_runtime.configure_platform(SimpleNamespace(), 3)
        config.set_platform.assert_called_once_with("NPU", None, device_id=3)
        self.assertEqual(soc, "Ascend910B4")


if __name__ == "__main__":
    unittest.main()
