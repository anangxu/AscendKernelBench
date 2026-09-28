#!/usr/bin/env python
"""Self-contained tests for the generation backend layer.

Run from the checkout root as

    python tests/test_generation_backend.py

Needs no NPU, no network, and no API key. The generation modules are
imported with tiny stand-ins for their third-party dependencies (loguru,
openai, torch) so the test also runs on a machine where the benchmark
dependencies are not installed; real packages are reused when present.
Exits non-zero on the first failed assertion.
"""

from __future__ import annotations

import contextlib
import json
import sys
import tempfile
import types
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CHECKS: list[str] = []


###########################################################################
# SHARED DEPENDENCY STAND-INS
###########################################################################

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _stubs  # noqa: E402

###########################################################################
# FAKE LLM ENDPOINT
###########################################################################


class _Message:
    def __init__(self, content: str, parsed: Any = None) -> None:
        self.content = content
        self.parsed = parsed


class _Choice:
    def __init__(self, content: str, parsed: Any = None, finish: str = "stop") -> None:
        self.message = _Message(content, parsed)
        self.finish_reason = finish


class _Usage:
    def model_dump(self) -> dict:
        return {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}


class _Response:
    def __init__(self, content: str, parsed: Any = None, finish: str = "stop") -> None:
        self.choices = [_Choice(content, parsed, finish)]
        self.usage = _Usage()


class _ScriptedCompletions:
    """Serve the fake endpoint's scripted responses."""

    def __init__(self, parent: "_FakeOpenAI") -> None:
        self._parent = parent

    def create(self, **kwargs: Any) -> _Response:
        return self._parent.next_response(kwargs)

    def parse(self, **kwargs: Any) -> _Response:
        return self._parent.next_response(kwargs)


class _ScriptedChat:
    def __init__(self, parent: "_FakeOpenAI") -> None:
        self.completions = _ScriptedCompletions(parent)


class _FakeOpenAI:
    """Minimal OpenAI client whose responses are scripted in order."""

    script: list[Any] = []
    last_kwargs: dict = {}
    last_messages: list = []

    def __init__(self, **kwargs: Any) -> None:
        completions = _ScriptedCompletions(self)
        self.chat = types.SimpleNamespace(completions=completions)
        self.beta = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=completions)
        )

    def next_response(self, kwargs: dict) -> _Response:
        _FakeOpenAI.last_kwargs = kwargs
        _FakeOpenAI.last_messages = kwargs.get("messages", [])
        if not self.script:
            raise AssertionError("fake endpoint ran out of scripted responses")
        scripted = self.script.pop(0)
        if isinstance(scripted, _Response):
            return scripted
        return _Response(scripted.get("content", ""), parsed=scripted.get("parsed"))


###########################################################################
# FIXTURES
###########################################################################

ASC_KERNEL = (
    "// device\n"
    "__global__ __vector__ void add_custom() {}\n"
    "// ==================== ASCEND_HOST_SECTION ====================\n"
    "#include <torch/library.h>\n"
    "TORCH_LIBRARY(custom_op, m) {}\n"
    "TORCH_LIBRARY_IMPL(custom_op, PrivateUse1, m) {}\n"
)
ASC_WRAPPER = (
    "import torch\n"
    "class ModelNew(torch.nn.Module):\n"
    "    def forward(self, x):\n"
    "        return torch.ops.custom_op.run(x)\n"
)
PYASC_KERNEL = (
    "import asc\n"
    "@asc.jit\n"
    "def kernel_add(x: asc.GlobalAddress, z: asc.GlobalAddress,\n"
    "               total: int) -> None:\n"
    "    return None\n"
)
PYASC_WRAPPER = (
    "import torch\n"
    "from kernel import run\n"
    "class ModelNew(torch.nn.Module):\n"
    "    def forward(self, x):\n"
    "        return run(x)\n"
)


def _make_generations():
    """Return one valid generation object per backend."""
    from src.llm import AscendCGeneration, PyAscGeneration

    return (
        AscendCGeneration(custom_op_asc=ASC_KERNEL, model_new_py=ASC_WRAPPER),
        PyAscGeneration(kernel_py=PYASC_KERNEL, model_new_py=PYASC_WRAPPER),
    )


def _make_client(backend: str, script: list[Any]):
    """Return an LLMClient wired to the scripted fake endpoint."""
    from src.llm import LLMClient

    _FakeOpenAI.script = list(script)
    return LLMClient(
        "fake-model", base_url="http://localhost", api_key="x", backend=backend
    )


def _block(tag: str, body: str) -> str:
    """Return one fenced markdown block."""
    return f"```{tag}\n{body}```\n"


def _asc_fenced_response() -> str:
    return (
        "Here you go.\n\n"
        + _block("custom_op.asc", ASC_KERNEL)
        + _block("model_new.py", ASC_WRAPPER)
    )


def _pyasc_fenced_response() -> str:
    return (
        "Here you go.\n\n"
        + _block("kernel.py", PYASC_KERNEL)
        + _block("model_new.py", PYASC_WRAPPER)
    )


@contextlib.contextmanager
def _temp_runs_dir():
    """Import rundir and point RUNS_DIR at a temporary directory."""
    import src.rundir as rundir

    original = rundir.RUNS_DIR
    with tempfile.TemporaryDirectory() as temp:
        rundir.RUNS_DIR = Path(temp)
        try:
            yield rundir, Path(temp)
        finally:
            rundir.RUNS_DIR = original


###########################################################################
# TESTS
###########################################################################


def test_backend_defaults_and_unknown() -> None:
    """parse_backend defaults, accepts both names, and rejects junk."""
    import src.backend as backend
    from src.cli_util import resolve_backend

    assert backend.DEFAULT_BACKEND is backend.Backend.ASCENDC
    for value in (None, "", "   "):
        assert backend.parse_backend(value) is backend.Backend.ASCENDC
    assert backend.parse_backend("ascendc") is backend.Backend.ASCENDC
    assert backend.parse_backend("ASCENDC") is backend.Backend.ASCENDC
    assert backend.parse_backend("pyasc") is backend.Backend.PYASC
    assert backend.kernel_file(backend.Backend.PYASC) == "kernel.py"
    # parse_backend is a string API: str(Backend.X) is "Backend.X" for this
    # str-Enum, so every call site normalizes to .value first.
    for enum_value, name in (
        (backend.Backend.ASCENDC, "ascendc"),
        (backend.Backend.PYASC, "pyasc"),
    ):
        assert backend.parse_backend(enum_value.value) is enum_value
        assert backend.kernel_field(enum_value) in {"custom_op_asc", "kernel_py"}
        assert name in {item.value for item in backend.Backend}
    assert backend.sample_files(backend.Backend.ASCENDC) == (
        "custom_op.asc",
        "model_new.py",
    )
    assert backend.sample_files(backend.Backend.PYASC) == ("kernel.py", "model_new.py")
    try:
        backend.parse_backend("triton")
    except ValueError as exc:
        message = str(exc)
        assert "triton" in message, message
        assert "ascendc" in message and "pyasc" in message, message
    else:
        raise AssertionError("unknown backend was accepted")
    assert resolve_backend(None, None) is backend.Backend.ASCENDC
    assert resolve_backend(None, "pyasc") is backend.Backend.PYASC
    assert resolve_backend("ascendc", "pyasc") is backend.Backend.ASCENDC
    try:
        resolve_backend("cuda", None)
    except ValueError as exc:
        assert "cuda" in str(exc)
    else:
        raise AssertionError("unknown CLI backend was accepted")
    CHECKS.append("backend default + unknown rejection")


def test_generation_config_round_trip() -> None:
    """generation_config.yaml carries the backend, and old runs default."""
    from src.cli_util import generation_run_config, resolve_generation_settings
    from src.config import EvalConfig, load_eval_config

    config = load_eval_config()
    assert config.generation.get("backend") == "ascendc"
    default_settings = resolve_generation_settings(config)
    assert default_settings.backend.value == "ascendc"
    assert resolve_generation_settings(config, backend="pyasc").backend.value == "pyasc"
    assert (
        resolve_generation_settings(config, backend="ascendc").backend.value
        == "ascendc"
    )
    pyasc_config = EvalConfig(generation={"backend": "pyasc"})
    assert resolve_generation_settings(pyasc_config).backend.value == "pyasc"
    try:
        resolve_generation_settings(EvalConfig(generation={"backend": "triton"}))
    except ValueError as exc:
        assert "triton" in str(exc)
    else:
        raise AssertionError("unknown config backend was accepted")

    with _temp_runs_dir() as (rundir, _):
        for backend_name in ("ascendc", "pyasc"):
            settings = resolve_generation_settings(config, backend=backend_name)
            payload = generation_run_config(
                settings,
                hardware_name="ascend910b2",
                task_ids=["level1/19_ReLU"],
            )
            assert payload["backend"] == backend_name, payload
            run_dir = rundir.create_run(f"roundtrip_{backend_name}", payload)
            assert rundir.generation_backend(run_dir).value == backend_name
            assert rundir.generation_hardware_name(run_dir) == "ascend910b2"
            assert rundir.generation_harness(run_dir)["backend"] == backend_name
            assert (run_dir / "generation_config.yaml").is_file()
        old_run = rundir.create_run(
            "old_run", {"model": "deepseek-flash", "prompt_mode": "one_shot"}
        )
        assert rundir.generation_backend(old_run) is None
        settings = resolve_generation_settings(config)
        fallback = rundir.generation_backend(old_run)
        assert (fallback or settings.backend) is not None
        assert (fallback or settings.backend).value == "ascendc"
    CHECKS.append("generation_config.yaml round-trip + old run default")


def test_sample_save_and_discovery_per_backend() -> None:
    """Both backends save, discover, read back, and check completeness."""
    from src.backend import Backend

    asc_gen, pyasc_gen = _make_generations()
    with _temp_runs_dir() as (rundir, _):
        for backend_name, kernel_file, generation in (
            ("ascendc", "custom_op.asc", asc_gen),
            ("pyasc", "kernel.py", pyasc_gen),
        ):
            run_dir = rundir.create_run(
                f"run_{backend_name}",
                {"backend": backend_name, "tasks": ["level1/19_ReLU"]},
            )
            out = rundir.save_sample(
                run_dir,
                "level1/19_ReLU",
                0,
                prompt="prompt text",
                generation=generation,
                raw_response="raw text",
                backend=backend_name,
            )
            assert (out / kernel_file).is_file(), kernel_file
            assert (out / "model_new.py").is_file()
            assert (out / "prompt.txt").read_text(encoding="utf-8") == "prompt text"
            assert (out / "response_raw.txt").read_text(encoding="utf-8") == "raw text"
            other_kernel = "kernel.py" if backend_name == "ascendc" else "custom_op.asc"
            assert not (out / other_kernel).exists()
            assert rundir.sample_complete(out, backend_name)
            other_backend = "pyasc" if backend_name == "ascendc" else "ascendc"
            assert not rundir.sample_complete(out, other_backend)
            kernel_src, wrapper_src = rundir.load_sample(out, backend_name)
            assert kernel_src and wrapper_src
            discovered = list(rundir.iter_sample_dirs(run_dir))
            assert discovered == [("level1/19_ReLU", 0, out)], discovered
            assert list(rundir.iter_sample_dirs(run_dir, backend=backend_name))
            assert list(rundir.iter_sample_dirs(run_dir, level=1))
            assert not list(rundir.iter_sample_dirs(run_dir, level=2))

        # A run with no backend field but a pyasc sample still discovers it.
        orphan = rundir.create_run("orphan", {"model": "deepseek-flash"})
        rundir.save_sample(
            orphan,
            "level1/19_ReLU",
            1,
            prompt="p",
            generation=pyasc_gen,
            backend=Backend.PYASC,
        )
        assert rundir.generation_backend(orphan) is None
        found = list(rundir.iter_sample_dirs(orphan))
        assert found and found[0][1] == 1, found

        # An incomplete sample is never discovered.
        incomplete = rundir.create_run("incomplete", {"backend": "pyasc"})
        sample = incomplete / "level1" / "19_ReLU" / "sample_0"
        sample.mkdir(parents=True)
        (sample / "kernel.py").write_text("x = 1\n", encoding="utf-8")
        assert not list(rundir.iter_sample_dirs(incomplete))
        assert not rundir.sample_complete(sample, "pyasc")

        # A generation object of the wrong backend is rejected, not written.
        try:
            rundir.save_sample(
                incomplete,
                "level1/19_ReLU",
                2,
                prompt="p",
                generation=asc_gen,
                backend="pyasc",
            )
        except TypeError as exc:
            assert "PyAscGeneration" in str(exc)
        else:
            raise AssertionError("mismatched generation was accepted")
        assert not (incomplete / "level1" / "19_ReLU" / "sample_2").exists()
    CHECKS.append("per-backend save + discovery + completeness")


def test_ascendc_parsing_unchanged() -> None:
    """An Ascend C response parses exactly as before."""
    from src.llm import AscendCGeneration, extract_generation, validate_generation

    fenced = extract_generation(_asc_fenced_response())
    assert isinstance(fenced, AscendCGeneration)
    assert fenced.custom_op_asc.strip() == ASC_KERNEL.strip()
    assert fenced.model_new_py.strip() == ASC_WRAPPER.strip()
    assert validate_generation(fenced) == []

    payload = {
        "custom_op_asc": ASC_KERNEL,
        "model_new_py": ASC_WRAPPER,
    }
    from_json = extract_generation(json.dumps(payload))
    assert isinstance(from_json, AscendCGeneration)
    assert from_json.custom_op_asc == ASC_KERNEL
    assert from_json.model_new_py == ASC_WRAPPER

    # The default path still accepts language-tagged blocks with no filename.
    language_tagged = _block("cpp", ASC_KERNEL) + _block("python", ASC_WRAPPER)
    tagged = extract_generation(language_tagged)
    assert tagged.custom_op_asc.strip() == ASC_KERNEL.strip()
    assert tagged.model_new_py.strip() == ASC_WRAPPER.strip()

    # Content-based Ascend C validation is untouched.
    broken = AscendCGeneration(custom_op_asc="int main() {}", model_new_py="x = 1")
    problems = validate_generation(broken)
    assert any("__global__" in item for item in problems), problems
    assert any("TORCH_LIBRARY" in item for item in problems), problems
    assert any("ASCEND_HOST_SECTION" in item for item in problems), problems
    CHECKS.append("ascendc parsing + validation unchanged")


def test_pyasc_parsing() -> None:
    """A two-Python-block pyasc response parses by tag, ambiguous raises."""
    from src.llm import PyAscGeneration, extract_generation, validate_generation

    fenced = extract_generation(_pyasc_fenced_response(), backend="pyasc")
    assert isinstance(fenced, PyAscGeneration)
    assert fenced.kernel_py.strip() == PYASC_KERNEL.strip()
    assert fenced.model_new_py.strip() == PYASC_WRAPPER.strip()
    assert validate_generation(fenced) == []

    payload = {"kernel_py": PYASC_KERNEL, "model_new_py": PYASC_WRAPPER}
    from_json = extract_generation(json.dumps(payload), backend="pyasc")
    assert isinstance(from_json, PyAscGeneration)
    assert from_json.kernel_py == PYASC_KERNEL

    # Field-name tags also work.
    field_tagged = _block("kernel_py", PYASC_KERNEL) + _block(
        "model_new_py", PYASC_WRAPPER
    )
    assert extract_generation(field_tagged, backend="pyasc").kernel_py == PYASC_KERNEL

    # Two untagged Python blocks resolve only because exactly one is a
    # ModelNew definition.
    untagged = _block("python", PYASC_KERNEL) + _block("python", PYASC_WRAPPER)
    resolved = extract_generation(untagged, backend="pyasc")
    assert resolved.kernel_py.strip() == PYASC_KERNEL.strip()
    assert resolved.model_new_py.strip() == PYASC_WRAPPER.strip()

    # Two identical Python blocks are ambiguous.
    ambiguous = _block("python", PYASC_KERNEL) + _block("python", PYASC_KERNEL)
    try:
        extract_generation(ambiguous, backend="pyasc")
    except ValueError as exc:
        assert "ambiguous" in str(exc), exc
    else:
        raise AssertionError("ambiguous pyasc response was accepted")

    # Two ModelNew blocks are ambiguous too.
    both_wrappers = _block("python", PYASC_WRAPPER) + _block("python", PYASC_WRAPPER)
    try:
        extract_generation(both_wrappers, backend="pyasc")
    except ValueError as exc:
        assert "ambiguous" in str(exc), exc
    else:
        raise AssertionError("two wrapper blocks were accepted")

    # One tagged, one untagged, is ambiguous rather than guessed.
    half_tagged = _block("kernel.py", PYASC_KERNEL) + _block("python", PYASC_WRAPPER)
    try:
        extract_generation(half_tagged, backend="pyasc")
    except ValueError as exc:
        assert "ambiguous" in str(exc), exc
    else:
        raise AssertionError("half-tagged pyasc response was accepted")

    # A pyasc response that only carries the Ascend C fields is rejected.
    try:
        extract_generation(
            json.dumps(
                {
                    "custom_op_asc": ASC_KERNEL,
                    "model_new_py": ASC_WRAPPER,
                }
            ),
            backend="pyasc",
        )
    except ValueError as exc:
        assert "fenced" in str(exc) or "identify" in str(exc), exc
    else:
        raise AssertionError("ascendc JSON fields were accepted for pyasc")

    # pyasc validation uses pyasc markers, never the Ascend C ones.
    problems = validate_generation(
        PyAscGeneration(kernel_py="print('hi')\n", model_new_py="x = 1\n")
    )
    assert any("'asc.jit'" in item for item in problems), problems
    assert any("class ModelNew" in item for item in problems), problems
    assert not any("custom_op_asc" in item for item in problems), problems
    CHECKS.append("pyasc parsing (tag, JSON, ambiguity) + pyasc markers")


def _attempt(backend: str, response: str) -> list[dict]:
    """Return one scripted response per parser for one LLM attempt."""
    from src.llm import extract_generation

    return [
        {"content": response, "parsed": extract_generation(response, backend=backend)},
        {"content": response},
    ]


def _wrong_attempt(backend: str) -> list[dict]:
    """Return one scripted response per parser that fails validation."""
    from src.llm import AscendCGeneration, PyAscGeneration

    if backend == "pyasc":
        parsed: Any = PyAscGeneration(kernel_py="x = 1\n", model_new_py="y = 2\n")
        content = _block("kernel.py", "x = 1\n") + _block("model_new.py", "y = 2\n")
    else:
        parsed = AscendCGeneration(custom_op_asc="int x;\n", model_new_py="y = 1\n")
        content = _block("custom_op.asc", "int x;\n")
        content += _block("model_new.py", "y = 1\n")
    return [{"content": content, "parsed": parsed}, {"content": content}]


def test_client_selects_schema_and_retries() -> None:
    """The client validates per backend and retries once on bad output."""
    from src.llm import AscendCGeneration, PyAscGeneration, _generation_schema

    pyasc_gen = PyAscGeneration(kernel_py=PYASC_KERNEL, model_new_py=PYASC_WRAPPER)
    client = _make_client(
        "pyasc",
        [{"content": "", "parsed": pyasc_gen}, {"content": _pyasc_fenced_response()}],
    )
    result = client.generate("prompt", system="system", max_retries=0)
    assert result.is_pyasc is True
    assert isinstance(result.generation, PyAscGeneration)
    assert result.generation.kernel_py == PYASC_KERNEL
    assert _generation_schema(client.backend) is PyAscGeneration
    assert _FakeOpenAI.last_kwargs["response_format"] is PyAscGeneration

    # A truncated/invalid first answer is retried with the pyasc reminder.
    client = _make_client(
        "pyasc",
        _wrong_attempt("pyasc") + _attempt("pyasc", _pyasc_fenced_response()),
    )
    retried = client.generate("prompt", system="system", max_retries=1)
    assert isinstance(retried.generation, PyAscGeneration)
    assert "kernel_py" in _FakeOpenAI.last_messages[-1]["content"]
    assert "custom_op_asc" not in _FakeOpenAI.last_messages[-1]["content"]

    # Ascend C keeps its own schema, field names, and reminder.
    client = _make_client(
        "ascendc",
        _wrong_attempt("ascendc") + _attempt("ascendc", _asc_fenced_response()),
    )
    asc_result = client.generate("prompt", system="system", max_retries=1)
    assert asc_result.is_pyasc is False
    assert asc_result.generation.custom_op_asc.strip() == ASC_KERNEL.strip()
    assert "custom_op_asc" in _FakeOpenAI.last_messages[-1]["content"]
    assert "kernel_py" not in _FakeOpenAI.last_messages[-1]["content"]
    assert _generation_schema(client.backend) is AscendCGeneration
    CHECKS.append("client schema + validation + retry per backend")


def _read_config_scalar(path: Path, section: str, key: str) -> Any:
    """Read one scalar out of a nested YAML block without a YAML library.

    This covers the flat generation block of configs/eval_default.yaml; the
    full PyYAML dependency is exercised by the round-trip test instead.
    """
    in_section = section == ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" "):
            in_section = line.split(":", 1)[0].strip() == section
            if in_section and section == "":
                head, _, rest = line.partition(":")
                if head.strip() == key and rest.strip():
                    return _stubs.yaml_value(rest.strip())
            continue
        if not in_section:
            continue
        head, _, rest = line.strip().partition(":")
        if head.strip() == key and rest.strip():
            return _stubs.yaml_value(rest.strip())
    return None


def _prompt_test_config():
    """Return an EvalConfig carrying the repo's recorded backend."""
    from src.config import EvalConfig

    backend = _read_config_scalar(
        REPO_ROOT / "configs" / "eval_default.yaml", "generation", "backend"
    )
    return EvalConfig(generation={"backend": backend})


def _prompt_test_hardware():
    """Return a hardware profile built without reading any YAML."""
    from src.config import HardwareProfile

    return HardwareProfile(
        name="ascend910b2",
        soc_version="Ascend910B2",
        cmake_arch="dav-2201",
        ai_core_num=48,
        ub_size_kb=192,
        l2_cache_mb=192,
        hbm_gb=64,
        memory_bandwidth_gbps=1600.0,
        supported_dtypes=["fp16", "fp32", "bf16", "int32", "int64"],
        api_style="- Kernel functions are annotated __global__ __vector__.",
        cube_core_num=24,
        vector_core_num=48,
        peak_tflops={"fp32": 47.0},
    )


def test_import_without_torch() -> None:
    """The generation modules import without torch or an NPU."""
    for name in list(sys.modules):
        if name.startswith("src."):
            del sys.modules[name]
    import src.cli_util  # noqa: F401
    import src.llm  # noqa: F401
    import src.prompt  # noqa: F401
    import src.rundir  # noqa: F401

    CHECKS.append("import src.prompt/llm/rundir/cli_util without torch")


def test_prompts_and_cli() -> None:
    """Both prompts build for a real task; the CLI help lists --backend."""
    from src.dataset import load_task
    from src.prompt import build_prompt, load_examples, load_pyasc_examples

    config = _prompt_test_config()
    assert config.generation.get("backend") == "ascendc", config.generation
    hardware = _prompt_test_hardware()
    task_id = "level1/19_ReLU"
    task = load_task(task_id)
    asc_prompt = build_prompt(task, hardware, mode="one_shot")
    pyasc_prompt = build_prompt(task, hardware, mode="one_shot", backend="pyasc")
    assert "```custom_op.asc" in asc_prompt
    assert "```kernel.py" in pyasc_prompt
    assert "```custom_op.asc" not in pyasc_prompt
    assert "pyasc" in pyasc_prompt
    assert "asc.jit" in pyasc_prompt
    assert "torch.ops.custom_op" not in pyasc_prompt
    for prompt in (asc_prompt, pyasc_prompt):
        order = [
            prompt.index("## Problem Statement"),
            prompt.index("## Target Hardware Contract"),
            prompt.index("## Example"),
            prompt.index("## Output Contract"),
            prompt.index("## Instruction"),
        ]
        assert order == sorted(order), order
    assert len(asc_prompt) > len(pyasc_prompt)
    asc_examples = load_examples()
    pyasc_examples = load_pyasc_examples()
    assert [item.name for item in asc_examples] == [
        "001_elementwise_add",
        "002_leaky_relu",
    ], [item.name for item in asc_examples]
    assert all(item.kernel_tag == "custom_op.asc" for item in asc_examples)
    assert [item.name for item in pyasc_examples] == [
        "001_elementwise_add",
        "003_rowsum",
    ], [item.name for item in pyasc_examples]
    assert pyasc_examples[0].kernel_tag == "kernel.py"
    assert "asc.jit" in pyasc_examples[0].kernel_src
    assert "class ModelNew" in pyasc_examples[0].model_new_py
    # The example must launch with the subscript form and must not select the
    # platform: the evaluator owns platform and device selection.
    assert "add_kernel[cores, rt.current_stream()]" in pyasc_examples[0].kernel_src
    assert "set_platform" not in pyasc_examples[0].kernel_src
    # The reduction example must teach the device-verified call shape.
    rowsum = pyasc_examples[1]
    assert rowsum.kernel_tag == "kernel.py"
    assert "whole_reduce_sum" in rowsum.kernel_src
    assert "rowsum_kernel[cores, rt.current_stream()]" in rowsum.kernel_src
    assert "set_platform" not in rowsum.kernel_src
    # The prompt mentions set_platform only to forbid it.
    assert "must not call" in pyasc_prompt
    assert "set_platform" in pyasc_prompt
    print(f"[prompt] {task_id} ascendc={len(asc_prompt)} chars")
    print(f"[prompt] {task_id} pyasc={len(pyasc_prompt)} chars")

    import subprocess

    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "generate.py"), "--help"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=_cli_env(),
    )
    assert proc.returncode == 0, proc.stderr
    assert "--backend" in proc.stdout, proc.stdout
    assert "ascendc" in proc.stdout and "pyasc" in proc.stdout, proc.stdout
    print("[cli] generate.py --help lists --backend {ascendc,pyasc}")
    CHECKS.append("prompt dry run for both backends + CLI --help")


_CLI_BOOTSTRAP = """import pathlib, sys

_tests_dir = pathlib.Path({tests_dir!r})
if _tests_dir.is_dir():
    sys.path.insert(0, str(_tests_dir))
    import _stubs

    _stubs.install_engine_stubs()
"""


def _cli_env() -> dict:
    """Return an env that stubs missing imports inside the CLI process."""
    import os
    import tempfile

    stub_dir = Path(tempfile.gettempdir()) / "akb_cli_stubs"
    stub_dir.mkdir(parents=True, exist_ok=True)
    (stub_dir / "sitecustomize.py").write_text(
        _CLI_BOOTSTRAP.format(tests_dir=str(Path(__file__).resolve().parent)),
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(stub_dir)] + env.get("PYTHONPATH", "").split(os.pathsep)
    )
    return env


def main() -> int:
    """Run every check in order and report."""
    _stubs.install_engine_stubs()
    _stubs.install_openai(_FakeOpenAI)

    tests = [
        test_backend_defaults_and_unknown,
        test_generation_config_round_trip,
        test_sample_save_and_discovery_per_backend,
        test_ascendc_parsing_unchanged,
        test_pyasc_parsing,
        test_client_selects_schema_and_retries,
        test_import_without_torch,
        test_prompts_and_cli,
    ]
    for test in tests:
        test()
    for line in CHECKS:
        print(f"ok - {line}")
    print(f"{len(CHECKS)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
