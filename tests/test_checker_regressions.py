"""Regression coverage for scalar inference in both authoring backends."""

from __future__ import annotations

import sys
import textwrap
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.checks.pyasc import check_pyasc_sources  # noqa: E402
from src.checks.python_source import check_model_new  # noqa: E402

DEVICE = """import torch
import asc

@asc.jit
def dummy(x: asc.GlobalAddress):
    pass

"""
WRAPPER = """import torch
from kernel import launch

class ModelNew(torch.nn.Module):
    def forward(self, x):
        return launch(x)
"""


def wrapper(body: str, *, parameter: str = "x", preamble: str = "") -> str:
    """Build a wrapper with one trusted custom-op call."""
    return (
        "import torch\n" + textwrap.dedent(preamble) + "\n"
        "class ModelNew(torch.nn.Module):\n"
        f"    def forward(self, {parameter}):\n"
        "        torch.ops.custom_op.noop(x)\n"
        + textwrap.indent(textwrap.dedent(body), "        ")
    )


class ScalarInferenceTests(unittest.TestCase):
    def assert_rejected(self, source: str) -> None:
        self.assertTrue(check_model_new(source), source)

    def assert_pyasc(self, host: str, *, rejected: bool) -> None:
        found = check_pyasc_sources(DEVICE + textwrap.dedent(host), WRAPPER)
        self.assertEqual(bool(found), rejected, "\n".join(found) or host)

    def test_annotations_do_not_make_tensor_inputs_scalar(self) -> None:
        for annotation in ("int", "float", "bool"):
            for parameter in (
                f"x: {annotation}",
                f"x: {annotation}, /",
                f"*, x: {annotation}",
            ):
                with self.subTest(parameter=parameter):
                    self.assert_rejected(wrapper("return x + 1\n", parameter=parameter))

    def test_scalar_names_do_not_leak_across_functions(self) -> None:
        source = wrapper(
            "return x + 1\n", preamble="def unrelated():\n    x = 1\n    return x\n"
        )
        self.assert_rejected(source)
        self.assert_rejected(wrapper("return x + 1\n", preamble="x = 1\n"))

    def test_mixed_assignments_do_not_make_tensor_values_scalar(self) -> None:
        for expression in ("x", "torch.empty_like(x)"):
            with self.subTest(expression=expression):
                self.assert_rejected(
                    wrapper(f"n = 1\nn = {expression}\nreturn n + 1\n")
                )

    def test_shape_arithmetic_remains_valid(self) -> None:
        source = wrapper(
            "n = x.numel()\nblocks = max(1, (n + 31) // 32)\nreturn torch.ops.custom_op.run(x, blocks)\n"
        )
        self.assertEqual(check_model_new(source), [])

    def test_wrapper_helper_requires_all_calls_to_be_scalar(self) -> None:
        helper = "def plan(n: int):\n    return (n + 31) // 32\n"
        clean = wrapper(
            "blocks = plan(x.numel())\nreturn torch.ops.custom_op.run(x, blocks)\n",
            preamble=helper,
        )
        self.assertEqual(check_model_new(clean), [])
        bad = wrapper("plan(1)\nreturn plan(x)\n", preamble=helper)
        self.assert_rejected(bad)

    def test_tensor_builtins_remain_rejected(self) -> None:
        for builtin in ("abs", "min", "max", "sum", "round"):
            with self.subTest(builtin=builtin):
                self.assert_rejected(wrapper(f"return {builtin}(x)\n"))

    def test_mixed_pyasc_call_sites_reject_tensor_math(self) -> None:
        for first, second in (
            ("helper(1)", "helper(x)"),
            ("helper(x)", "helper(1)"),
            ("helper(n=1)", "helper(n=x)"),
        ):
            with self.subTest(first=first, second=second):
                self.assert_pyasc(
                    f"""
                    def helper(n: int):
                        return n + 1

                    def launch(x):
                        {first}
                        y = {second}
                        dummy[1](x)
                        return y
                """,
                    rejected=True,
                )

    def test_pyasc_alias_calls_are_included(self) -> None:
        self.assert_pyasc(
            """
            def helper(n):
                return n + 1

            def launch(x):
                alias = helper
                helper(1)
                y = alias(x)
                dummy[1](x)
                return y
        """,
            rejected=True,
        )

    def test_module_alias_calls_are_included(self) -> None:
        self.assert_pyasc(
            """
            def helper(n):
                return n + 1

            alias = helper

            def launch(x):
                helper(1)
                y = alias(x)
                dummy[1](x)
                return y
        """,
            rejected=True,
        )

    def test_multiple_callers_must_agree(self) -> None:
        self.assert_pyasc(
            """
            def helper(n):
                return n + 1

            def first(x):
                return helper(x.numel())

            def second(x):
                return helper(x)

            def launch(x):
                n = first(x)
                y = second(x)
                dummy[n](x)
                return y
        """,
            rejected=True,
        )

    def test_keyword_and_default_scalar_arguments_are_valid(self) -> None:
        self.assert_pyasc(
            """
            def helper(n, /, *, alignment=32):
                return (n + alignment - 1) // alignment

            def launch(x):
                blocks = helper(x.numel(), alignment=64)
                more = helper(x.numel())
                dummy[blocks + more](x)
                return x
        """,
            rejected=False,
        )

    def test_unknown_unpacking_cannot_override_scalar_evidence(self) -> None:
        self.assert_pyasc(
            """
            def helper(n):
                return n + 1

            def launch(x):
                helper(1)
                values = [x]
                y = helper(*values)
                dummy[1](x)
                return y
        """,
            rejected=True,
        )

    def test_scalar_default_cannot_hide_tensor_override(self) -> None:
        source = (
            DEVICE
            + """def launch(x, *, n: int = 1):
    dummy[1](x)
    return n + 1
"""
        )
        model = WRAPPER.replace("launch(x)", "launch(x, n=x)")
        self.assertTrue(check_pyasc_sources(source, model))

    def test_shipped_samples_remain_accepted(self) -> None:
        examples = Path(__file__).resolve().parent.parent / "src/prompts/examples"
        for model in examples.rglob("model_new.py"):
            with self.subTest(sample=str(model.parent.relative_to(examples))):
                kernel = model.parent / "kernel.py"
                found = (
                    check_pyasc_sources(kernel.read_text(), model.read_text())
                    if kernel.exists()
                    else check_model_new(model.read_text())
                )
                self.assertEqual(found, [])


if __name__ == "__main__":
    unittest.main()
