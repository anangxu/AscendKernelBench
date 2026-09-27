"""OpenAI-compatible LLM client for Ascend C and pyasc generation.

Structured output uses the OpenAI parse API with a pydantic model; endpoints
without it fall back to JSON field or fenced-block extraction.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Protocol

from loguru import logger
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .backend import DEFAULT_BACKEND, Backend, parse_backend
from .checks.text import HOST_SECTION_MARKER

_FENCE_EDGE_RE = re.compile(
    r"^\s*```[A-Za-z0-9_+.-]*\s*\n(?P<body>.*?)\n?\s*```\s*$", re.DOTALL
)


def _strip_fence(value: str) -> str:
    """Remove one outer fenced-code-block wrapper if the model added it."""
    match = _FENCE_EDGE_RE.match(value)
    body = match.group("body") if match else value
    # Some models emit a bare filename line ("custom_op.asc") as the first line.
    lines = body.split("\n")
    if lines and re.fullmatch(
        r"\s*(custom_op\.asc|kernel\.py|model_new\.py)\s*", lines[0]
    ):
        body = "\n".join(lines[1:])
    return body


class AscendCGeneration(BaseModel):
    """The two code deliverables of one generation (docs/task_authoring.md)."""

    custom_op_asc: str = Field(
        description=(
            "Complete Ascend C source file custom_op.asc in two sections "
            "separated by one // ==================== ASCEND_HOST_SECTION "
            "==================== comment line: the device section holds "
            "the kernel class and __global__ __vector__ kernels (no torch "
            "headers); the host section holds the at::Tensor wrapper, which "
            "launches kernels via the generated <kernel>_launch stubs, and "
            "a process-local torch.library binding "
            "(TORCH_LIBRARY(custom_op, ...) and "
            "TORCH_LIBRARY_IMPL(custom_op, PrivateUse1, ...)). "
            "Raw file content only, no markdown fences. Do not use pybind11."
        )
    )
    model_new_py: str = Field(
        description=(
            "Python source of model_new.py defining class ModelNew with the "
            "same __init__ and forward signatures as the reference Model, "
            "calling the evaluator-loaded operator via torch.ops.custom_op. "
            "Raw file content only, no markdown fences."
        )
    )

    @field_validator("custom_op_asc", "model_new_py", mode="before")
    @classmethod
    def _strip_markdown_fence(cls, value: str) -> str:
        """Drop an outer markdown fence or leading filename line."""
        return _strip_fence(value) if isinstance(value, str) else value


class PyAscGeneration(BaseModel):
    """The two pyasc deliverables: the kernel module and its wrapper."""

    kernel_py: str = Field(
        description=(
            "Complete Python source of kernel.py: one or more kernels "
            "decorated with @asc.jit (pyasc, the official CANN Python DSL, "
            "import name asc), plus a plain-Python host launcher that "
            "allocates outputs and launches the kernel as "
            "kernel[core_num, rt.current_stream()](...). All tensor compute "
            "happens inside the @asc.jit kernels. Raw file content only, no "
            "markdown fences."
        )
    )
    model_new_py: str = Field(
        description=(
            "Python source of model_new.py defining class ModelNew with the "
            "same __init__ and forward signatures as the reference Model, "
            "importing the launcher from kernel.py. Raw file content only, "
            "no markdown fences."
        )
    )

    @field_validator("kernel_py", "model_new_py", mode="before")
    @classmethod
    def _strip_markdown_fence(cls, value: str) -> str:
        """Drop an outer markdown fence or leading filename line."""
        return _strip_fence(value) if isinstance(value, str) else value


class GenerationResult(BaseModel):
    """One LLM response after structured or fenced-block extraction."""

    model_config = ConfigDict(frozen=True)

    generation: AscendCGeneration | PyAscGeneration
    raw_text: str
    model: str
    usage: dict
    is_pyasc: bool = False


class ResponseParser(Protocol):
    """Strategy for turning a chat transcript into deliverables."""

    name: str

    def parse(self, messages: list[dict]) -> GenerationResult:
        """Parse messages into a GenerationResult."""
        ...


STRUCTURED_OUTPUT_NOTE = (
    "\n\n## Response Format\n\n"
    "Respond through the structured JSON schema: put the FULL raw content of "
    "the Ascend C file into the `custom_op_asc` field and the FULL raw content "
    "of the Python file into the `model_new_py` field. The field values are "
    "the files themselves (every line of code), NOT filenames, NOT summaries, "
    "and without markdown fences."
)

PYASC_STRUCTURED_OUTPUT_NOTE = (
    "\n\n## Response Format\n\n"
    "Respond through the structured JSON schema: put the FULL raw content of "
    "kernel.py into the `kernel_py` field and the FULL raw content of "
    "model_new.py into the `model_new_py` field. The field values are the "
    "files themselves (every line of code), NOT filenames, NOT summaries, and "
    "without markdown fences. Every value is Python source; the two fields "
    "are distinguished by their field names, never by their content."
)


def _backend_name(value: Backend | str) -> str:
    """Return the name parse_backend expects for a Backend or its value."""
    return value.value if isinstance(value, Backend) else value


def structured_output_note(backend: Backend | str = DEFAULT_BACKEND) -> str:
    """Return the response-format note that matches a backend."""
    if parse_backend(_backend_name(backend)) is Backend.PYASC:
        return PYASC_STRUCTURED_OUTPUT_NOTE
    return STRUCTURED_OUTPUT_NOTE


def _usage_dict(response: object) -> dict:
    """Return response.usage as a dict, or {} when absent."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    dump = getattr(usage, "model_dump", None)
    return dump() if callable(dump) else {}


_MIN_ASC_KERNEL_MARKERS = ("__global__", "__vector__")
_MIN_ASC_BINDING_MARKERS = ("TORCH_LIBRARY", "TORCH_LIBRARY_IMPL")
_MIN_ASC_LAYOUT_MARKERS = (HOST_SECTION_MARKER,)
_MIN_PY_MARKERS = ("class ModelNew", "torch.ops.custom_op")

# pyasc markers are real pyasc API markers, never the Ascend C ones.
_MIN_PYASC_KERNEL_MARKERS = ("asc.jit",)


def validate_generation(gen: AscendCGeneration | PyAscGeneration) -> list[str]:
    """Sanity-check that the fields carry real file content.

    Returns:
        Human-readable problems; empty means the fields look complete.
    """
    if isinstance(gen, PyAscGeneration):
        return _validate_pyasc_generation(gen)
    return _validate_ascend_c_generation(gen)


def _validate_pyasc_generation(gen: PyAscGeneration) -> list[str]:
    """Check the pyasc deliverables for their real API markers."""
    problems = []
    for marker in _MIN_PYASC_KERNEL_MARKERS:
        if marker not in gen.kernel_py:
            problems.append(f"kernel_py missing {marker!r}")
    if "class ModelNew" not in gen.model_new_py:
        problems.append("model_new_py missing 'class ModelNew'")
    return problems


def _validate_ascend_c_generation(gen: AscendCGeneration) -> list[str]:
    """Check the Ascend C deliverables for their required markers."""
    problems = []
    for marker in _MIN_ASC_KERNEL_MARKERS:
        if marker not in gen.custom_op_asc:
            problems.append(f"custom_op_asc missing {marker!r}")
    for marker in _MIN_ASC_BINDING_MARKERS:
        if marker not in gen.custom_op_asc:
            problems.append(f"custom_op_asc missing {marker!r}")
    for marker in _MIN_ASC_LAYOUT_MARKERS:
        if marker not in gen.custom_op_asc:
            problems.append(
                f"custom_op_asc missing the {marker!r} separator comment "
                "between the device and host sections"
            )
    if "PYBIND11_MODULE" in gen.custom_op_asc:
        problems.append("custom_op_asc must not use PYBIND11_MODULE")
    for marker in _MIN_PY_MARKERS:
        if marker not in gen.model_new_py:
            problems.append(f"model_new_py missing {marker!r}")
    return problems


class LLMClient:
    """Thin OpenAI-compatible client for kernel generation."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 131072,
        timeout: float = 1800.0,
        reasoning_effort: str | None = None,
        backend: Backend | str = DEFAULT_BACKEND,
    ) -> None:
        """Create a client for one generation model and backend."""
        self.model = model
        self.backend = parse_backend(_backend_name(backend))
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.extra_body = (
            {"reasoning_effort": reasoning_effort} if reasoning_effort else None
        )
        self.client = OpenAI(
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            timeout=timeout,
        )

    ################################ PROTOCOL ################################
    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_retries: int = 1,
    ) -> GenerationResult:
        """Generate one sample via structured parse, else fenced blocks.

        Raises:
            ValueError: If every attempt fails validation or the
                endpoint returns no usable fenced blocks.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append(
            {
                "role": "user",
                "content": prompt + structured_output_note(self.backend),
            }
        )

        last_error: Exception | None = None
        last_raw = ""
        for attempt in range(max_retries + 1):
            if attempt > 0:
                _append_retry_turn(messages, last_raw, self.backend)
            try:
                result = self._generate_once(messages)
                problems = validate_generation(result.generation)
                if not problems:
                    return result
                last_error = ValueError("; ".join(problems))
                last_raw = result.raw_text
            except Exception as exc:
                last_error = exc
                last_raw = ""
        raise ValueError(f"generation failed validation: {last_error}")

    ################################ PROTOCOL ################################

    def _parsers(self) -> tuple[ResponseParser, ...]:
        """Return structured parse first, then fenced-block fallback."""
        return (
            StructuredOutputParser(self),
            FencedBlockParser(self),
        )

    def _generate_once(self, messages: list[dict]) -> GenerationResult:
        """Request one completion; fall back to fenced-block extraction."""
        last_error: Exception | None = None
        for parser in self._parsers():
            try:
                return parser.parse(messages)
            except Exception as exc:
                last_error = exc
                logger.debug("LLM parser {} failed: {}", parser.name, exc)
        raise last_error or ValueError("no LLM response parser succeeded")


class StructuredOutputParser:
    """Parse the OpenAI structured-output schema into deliverables."""

    name = "structured"

    def __init__(self, llm: LLMClient) -> None:
        """Bind the parent client (model, backend, temperature, tokens)."""
        self._llm = llm

    def parse(self, messages: list[dict]) -> GenerationResult:
        """Parse a structured-output response into the two deliverables."""
        response = self._llm.client.beta.chat.completions.parse(
            model=self._llm.model,
            messages=messages,
            response_format=_generation_schema(self._llm.backend),
            temperature=self._llm.temperature,
            max_tokens=self._llm.max_tokens,
            extra_body=self._llm.extra_body,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise ValueError("structured parse returned None")
        return GenerationResult(
            generation=parsed,
            raw_text=response.choices[0].message.content or "",
            model=self._llm.model,
            usage=_usage_dict(response),
            is_pyasc=self._llm.backend is Backend.PYASC,
        )


class FencedBlockParser:
    """Complete without a schema and extract fenced code blocks."""

    name = "fenced"

    def __init__(self, llm: LLMClient) -> None:
        """Bind the parent client (model, backend, temperature, tokens)."""
        self._llm = llm

    def parse(self, messages: list[dict]) -> GenerationResult:
        """Complete without a schema and extract fenced code blocks."""
        response = self._llm.client.chat.completions.create(
            model=self._llm.model,
            messages=messages,
            temperature=self._llm.temperature,
            max_tokens=self._llm.max_tokens,
            extra_body=self._llm.extra_body,
        )
        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise ValueError(
                "response truncated at max_tokens; raise "
                "generation.max_tokens or lower reasoning_effort"
            )
        raw = choice.message.content or ""
        return GenerationResult(
            generation=extract_generation(raw, backend=self._llm.backend),
            raw_text=raw,
            model=self._llm.model,
            usage=_usage_dict(response),
            is_pyasc=self._llm.backend is Backend.PYASC,
        )


_RETRY_REMINDER = (
    "Your previous answer did not contain the "
    "required file contents. Return the COMPLETE "
    "custom_op.asc source in `custom_op_asc` and "
    "the COMPLETE model_new.py source in "
    "`model_new_py` — full code, no placeholders."
)

_PYASC_RETRY_REMINDER = (
    "Your previous answer did not contain the "
    "required file contents. Return the COMPLETE "
    "kernel.py source in `kernel_py` and the "
    "COMPLETE model_new.py source in "
    "`model_new_py` — full code, no placeholders. "
    "Both fields hold Python source; use the field "
    "names to tell them apart."
)


def retry_reminder(backend: Backend | str = DEFAULT_BACKEND) -> str:
    """Return the correction reminder that matches a backend."""
    if parse_backend(_backend_name(backend)) is Backend.PYASC:
        return _PYASC_RETRY_REMINDER
    return _RETRY_REMINDER


def _append_retry_turn(
    messages: list[dict], last_raw: str, backend: Backend | str = DEFAULT_BACKEND
) -> None:
    """Append the previous answer and a correction reminder."""
    if last_raw:
        messages.append({"role": "assistant", "content": last_raw})
    messages.append({"role": "user", "content": retry_reminder(backend)})


_FENCE_RE = re.compile(r"```(?P<tag>[A-Za-z0-9_+.-]*)\s*\n(?P<body>.*?)```", re.DOTALL)


# One entry per backend: the pydantic schema, the deliverable type, the
# structured-output field names, and the filename tags of the fenced-block
# fallback. Both pyasc files are Python, so its tags are the only reliable
# way to tell the two blocks apart.
@dataclass(frozen=True)
class _GenerationProfile:
    """Backend-specific generation schema, fields, and block tags."""

    schema: type[BaseModel]
    generation_type: type[BaseModel]
    kernel_field: str
    wrapper_field: str
    kernel_tag: str
    wrapper_tag: str


_PROFILES = {
    Backend.ASCENDC: _GenerationProfile(
        schema=AscendCGeneration,
        generation_type=AscendCGeneration,
        kernel_field="custom_op_asc",
        wrapper_field="model_new_py",
        kernel_tag="custom_op.asc",
        wrapper_tag="model_new.py",
    ),
    Backend.PYASC: _GenerationProfile(
        schema=PyAscGeneration,
        generation_type=PyAscGeneration,
        kernel_field="kernel_py",
        wrapper_field="model_new_py",
        kernel_tag="kernel.py",
        wrapper_tag="model_new.py",
    ),
}


def _generation_schema(backend: Backend) -> type[BaseModel]:
    """Return the structured-output schema for a backend."""
    return _PROFILES[backend].schema


def extract_generation(
    text: str, backend: Backend | str = DEFAULT_BACKEND
) -> AscendCGeneration | PyAscGeneration:
    """Extract the two deliverables from JSON fields or fenced code blocks.

    Args:
        text: Raw model response.
        backend: Authoring language that decides the field names and the
            filename tags, and therefore which block is which.

    Raises:
        ValueError: If neither layout carries a usable pair of sources, or
            if the two blocks cannot be told apart unambiguously.
    """
    resolved = parse_backend(_backend_name(backend))
    profile = _PROFILES[resolved]
    payload = _json_payload(text, profile)
    if payload is not None:
        return payload
    blocks = list(_FENCE_RE.finditer(text))
    if not blocks:
        raise ValueError("no JSON fields and no fenced code blocks in model response")
    kernel_src, wrapper_src = _pair_fenced_blocks(blocks, profile)
    if kernel_src is None or wrapper_src is None:
        raise ValueError(
            f"could not identify the {profile.kernel_tag} and "
            f"{profile.wrapper_tag} blocks; tag each fenced block with its "
            "filename"
        )
    return profile.generation_type(
        **{profile.kernel_field: kernel_src, profile.wrapper_field: wrapper_src}
    )


def _json_payload(text: str, profile: _GenerationProfile) -> BaseModel | None:
    """Return the deliverables from a bare JSON body, or None.

    Endpoints without structured-output support answer the structured
    request with a JSON object instead of fenced blocks.
    """
    candidate = text.strip()
    match = _FENCE_EDGE_RE.match(candidate)
    if match:
        candidate = match.group("body").strip()
    if not candidate.startswith("{"):
        return None
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    kernel_src = data.get(profile.kernel_field)
    wrapper_src = data.get(profile.wrapper_field)
    if not isinstance(kernel_src, str) or not isinstance(wrapper_src, str):
        return None
    return profile.generation_type(
        **{profile.kernel_field: kernel_src, profile.wrapper_field: wrapper_src}
    )


def _pair_fenced_blocks(
    blocks: list[re.Match[str]],
    profile: _GenerationProfile,
) -> tuple[str | None, str | None]:
    """Pick the kernel and wrapper bodies for a backend."""
    if profile is _PROFILES[Backend.PYASC]:
        return _pair_pyasc_blocks(blocks)
    return _pair_ascend_c_blocks(blocks)


def _pair_ascend_c_blocks(
    blocks: list[re.Match[str]],
) -> tuple[str | None, str | None]:
    """Pick Ascend C and Python bodies by filename tag, language, or order."""
    asc_src: str | None = None
    py_src: str | None = None
    for block in blocks:
        tag = block.group("tag").lower()
        body = block.group("body")
        if "custom_op.asc" in tag or "custom_op_asc" in tag:
            asc_src = body
        elif "model_new.py" in tag or "model_new_py" in tag:
            py_src = body
    if asc_src is None or py_src is None:
        for block in blocks:
            tag = block.group("tag").lower()
            body = block.group("body")
            if asc_src is None and tag in {"cpp", "c++", "asc", "c"}:
                asc_src = body
            elif py_src is None and tag in {"python", "py"}:
                py_src = body
    if asc_src is None and blocks:
        asc_src = blocks[0].group("body")
    if py_src is None and len(blocks) >= 2:
        py_src = blocks[1].group("body")
    return asc_src, py_src


def _pair_pyasc_blocks(
    blocks: list[re.Match[str]],
) -> tuple[str | None, str | None]:
    """Pick the pyasc kernel and wrapper bodies; both files are Python.

    Only the filename tags identify a block. With exactly two blocks the
    one holding a ModelNew class definition is the wrapper; every other
    layout is ambiguous and raises instead of guessing.

    Raises:
        ValueError: If the two blocks cannot be told apart.
    """
    kernel_src: str | None = None
    wrapper_src: str | None = None
    for block in blocks:
        tag = block.group("tag").lower()
        body = block.group("body")
        if "kernel.py" in tag or "kernel_py" in tag:
            kernel_src = body
        elif "model_new.py" in tag or "model_new_py" in tag:
            wrapper_src = body
    if (kernel_src is None) != (wrapper_src is None):
        raise ValueError(
            "ambiguous pyasc response: only one of the two blocks carries a "
            "filename tag; tag both blocks with kernel.py and model_new.py"
        )
    if kernel_src is not None and wrapper_src is not None:
        if kernel_src.strip() == wrapper_src.strip():
            raise ValueError(
                "ambiguous pyasc response: kernel.py and model_new.py carry "
                "the same content; return both full files"
            )
        return kernel_src, wrapper_src
    if len(blocks) == 2:
        bodies = [block.group("body") for block in blocks]
        model_new = [body for body in bodies if "class ModelNew" in body]
        if len(model_new) == 1:
            kernel_src = next(body for body in bodies if body != model_new[0])
            return kernel_src, model_new[0]
        raise ValueError(
            "ambiguous pyasc response: the kernel block and the wrapper block "
            "are indistinguishable; tag each fenced block with its filename"
        )
    raise ValueError(
        "could not identify the kernel.py and model_new.py blocks; tag each "
        "fenced block with its filename"
    )
