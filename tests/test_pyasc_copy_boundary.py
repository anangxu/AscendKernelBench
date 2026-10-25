"""Check pyasc host-copy synchronization and its narrow checker allowance."""

from __future__ import annotations

import ast
import sys
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.checks.pyasc import check_pyasc_sources
from src.checks.python_source import check_model_new

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "src/prompts/examples/pyasc/001_elementwise_add/kernel.py"
WRAPPER = """import torch
from kernel import launch

class ModelNew(torch.nn.Module):
    def forward(self, x):
        return launch(x)
"""


def kernel_source(statement):
    """Put a host statement before a valid pyasc launch."""
    return (
        "import torch\nimport asc\n\n"
        "@asc.jit\ndef device(x: asc.GlobalAddress):\n    pass\n\n"
        "def launch(x):\n"
        + textwrap.indent(statement, "    ")
        + "\n    device[1](x)\n    return x\n"
    )


class SynchronizeAllowanceTests(unittest.TestCase):
    def test_pyasc_allows_torch_synchronize_in_launcher_and_wrapper(self):
        self.assertEqual(
            check_pyasc_sources(kernel_source("torch.npu.synchronize()"), WRAPPER),
            [],
        )
        wrapper = WRAPPER.replace(
            "        return launch(x)",
            "        torch.npu.synchronize()\n        return launch(x)",
        )
        self.assertEqual(check_pyasc_sources(kernel_source("x.numel()"), wrapper), [])

    def test_ascendc_still_rejects_torch_synchronize(self):
        source = WRAPPER.replace("from kernel import launch\n", "").replace(
            "return launch(x)",
            "torch.npu.synchronize()\n        return torch.ops.custom_op.run(x)",
        )
        found = check_model_new(source)
        self.assertTrue(any("torch.npu.synchronize()" in item for item in found), found)

    def test_other_npu_methods_and_similar_names_remain_rejected(self):
        for call in (
            "torch.npu.current_stream()",
            "torch.npu.set_device(0)",
            "torch.npu.Stream()",
            "torch.npu.Event()",
            "torch.npu.synchronize_other()",
            "torch.npu.synchronize.__call__()",
            "torch.cuda.synchronize()",
        ):
            with self.subTest(call=call):
                self.assertTrue(check_pyasc_sources(kernel_source(call), WRAPPER))

    def test_synchronize_monkeypatch_remains_rejected(self):
        statement = "torch.npu.synchronize = lambda: None"
        found = check_pyasc_sources(kernel_source(statement), WRAPPER)
        self.assertTrue(any("monkey patch" in item for item in found), found)
        wrapper = WRAPPER.replace(
            "        return launch(x)", f"        {statement}\n        return launch(x)"
        )
        found = check_pyasc_sources(kernel_source("x.numel()"), wrapper)
        self.assertTrue(any("monkey patch" in item for item in found), found)

    def test_sync_alias_and_attribute_mutations_remain_rejected(self):
        for statement in (
            "import torch as t\nt.npu.synchronize = lambda: None\nt.npu.synchronize()",
            'setattr(torch.npu, "synchronize", lambda: None)\ntorch.npu.synchronize()',
            'delattr(torch.npu, "synchronize")',
            "del torch.npu.synchronize",
        ):
            with self.subTest(statement=statement):
                found = check_pyasc_sources(kernel_source(statement), WRAPPER)
                self.assertTrue(any("monkey patch" in item for item in found), found)

    def test_shipped_add_passes_static_checks(self):
        wrapper = EXAMPLE.with_name("model_new.py").read_text()
        self.assertEqual(check_pyasc_sources(EXAMPLE.read_text(), wrapper), [])


class QueuedTensor:
    """Queue Torch writes instead of making them immediately visible."""

    def __init__(self, queues, values):
        self.queues = queues
        self.values = list(values)
        self.dtype = "float32"
        self.device = "npu:0"
        self.shape = (len(values),)

    def element_size(self):
        return 4

    def numel(self):
        return len(self.values)

    def reshape(self, *shape):
        return self

    def __setitem__(self, index, source):
        self.queues.torch_pending.append(
            lambda: self.values.__setitem__(index, list(source.values))
        )

    def __getitem__(self, index):
        return QueuedTensor(self.queues, self.values[index])


class QueuedStreams:
    """Keep Torch initialization and asc launches on independent queues."""

    def __init__(self):
        self.torch_pending = []
        self.asc_pending = []
        self.events = []
        self.stream = object()
        self.torch = SimpleNamespace(
            zeros=self.zeros,
            empty_like=self.empty_like,
            npu=SimpleNamespace(synchronize=self.torch_synchronize),
        )
        self.rt = SimpleNamespace(
            current_stream=lambda: self.stream,
            synchronize=self.asc_synchronize,
        )

    def zeros(self, size, **kwargs):
        tensor = QueuedTensor(self, [None] * size)
        self.torch_pending.append(
            lambda: tensor.values.__setitem__(slice(None), [0] * size)
        )
        return tensor

    def empty_like(self, other):
        return QueuedTensor(self, [None] * other.numel())

    def torch_synchronize(self):
        self.events.append("torch_sync")
        while self.torch_pending:
            self.torch_pending.pop(0)()

    def asc_synchronize(self):
        self.events.append("asc_sync")
        while self.asc_pending:
            self.asc_pending.pop(0)()

    def __getitem__(self, launch_options):
        assert launch_options[1] is self.stream

        def launch(x, y, output, *plan):
            assert not self.torch_pending, "Torch writes are still pending at launch"
            self.events.append("launch")
            self.asc_pending.append(
                lambda: output.values.__setitem__(
                    slice(None),
                    [left + right for left, right in zip(x.values, y.values)],
                )
            )

        return launch


def load_add_launcher(queues):
    """Execute the actual host example with two fake asynchronous streams."""
    tree = ast.parse(EXAMPLE.read_text())
    body = ast.parse("from __future__ import annotations").body
    body.extend(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        or isinstance(node, ast.FunctionDef)
        and not node.decorator_list
    )
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    namespace = {"torch": queues.torch, "rt": queues.rt, "add_kernel": queues}
    exec(compile(module, str(EXAMPLE), "exec"), namespace)
    return namespace["add_launch"]


class AddCopyBoundaryTests(unittest.TestCase):
    def test_unaligned_add_completes_torch_writes_before_asc_launch(self):
        for size in (1, 7, 9, 31):
            with self.subTest(size=size):
                queues = QueuedStreams()
                launch = load_add_launcher(queues)
                x = QueuedTensor(queues, range(size))
                y = QueuedTensor(queues, [3] * size)
                output = launch(x, y)
                self.assertEqual(output.values, [value + 3 for value in range(size)])
                self.assertEqual(queues.events, ["torch_sync", "launch", "asc_sync"])
                self.assertFalse(queues.asc_pending)

    def test_aligned_add_waits_for_asc_without_an_extra_torch_boundary(self):
        queues = QueuedStreams()
        launch = load_add_launcher(queues)
        output = launch(QueuedTensor(queues, range(8)), QueuedTensor(queues, [3] * 8))
        self.assertEqual(output.values, list(range(3, 11)))
        self.assertEqual(queues.events, ["launch", "asc_sync"])
        self.assertFalse(queues.asc_pending)

    def test_asc_synchronize_cannot_flush_torch_padding_copies(self):
        queues = QueuedStreams()
        queues.torch.npu.synchronize = queues.asc_synchronize
        launch = load_add_launcher(queues)
        with self.assertRaisesRegex(AssertionError, "Torch writes are still pending"):
            launch(QueuedTensor(queues, [2]), QueuedTensor(queues, [3]))


if __name__ == "__main__":
    unittest.main()
