"""OpenAI-compatible LLM client for Ascend C generation.

Structured output uses the OpenAI parse API with a pydantic model; endpoints
without it fall back to JSON field or fenced-block extraction.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable

from loguru import logger
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .checks.text import (
    HOST_SECTION_MARKER,
    HOST_SECTION_MARKER_RE,
    strip_cpp_comments,
)

_FENCE_EDGE_RE = re.compile(
    r"^\s*```[A-Za-z0-9_+.-]*\s*\n(?P<body>.*?)\n?\s*```\s*$", re.DOTALL
)


def _strip_fence(value: str) -> str:
    """Remove one outer fenced-code-block wrapper if the model added it."""
    match = _FENCE_EDGE_RE.match(value)
    body = match.group("body") if match else value
    # Some models emit a bare filename line ("custom_op.asc") as the first line.
    lines = body.split("\n")
    if lines and re.fullmatch(r"\s*(custom_op\.asc|model_new\.py)\s*", lines[0]):
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


class GenerationResult(BaseModel):
    """One LLM response after structured or fenced-block extraction."""

    model_config = ConfigDict(frozen=True)

    generation: AscendCGeneration
    raw_text: str


STRUCTURED_OUTPUT_NOTE = (
    "\n\n## Response Format\n\n"
    "Respond through the structured JSON schema: put the FULL raw content of "
    "the Ascend C file into the `custom_op_asc` field and the FULL raw content "
    "of the Python file into the `model_new_py` field. The field values are "
    "the files themselves (every line of code), NOT filenames, NOT summaries, "
    "and without markdown fences."
)


_MIN_ASC_KERNEL_MARKERS = ("__global__", "__vector__")
_ASC_BINDING_MARKERS = (
    (r"\bTORCH_LIBRARY\s*\(", "TORCH_LIBRARY(...)"),
    (r"\bTORCH_LIBRARY_IMPL\s*\(", "TORCH_LIBRARY_IMPL(...)"),
)
_ASC_PYBIND_RE = re.compile(r'#\s*include\s*[<"]pybind11/')
_MIN_PY_MARKERS = ("class ModelNew", "torch.ops.custom_op")


def validate_generation(gen: AscendCGeneration) -> list[str]:
    """Sanity-check that the fields carry real file content.

    Returns:
        Human-readable problems; empty means the fields look complete.
    """
    problems = []
    code = strip_cpp_comments(gen.custom_op_asc)
    masked = strip_cpp_comments(gen.custom_op_asc, mask_literals=True)
    for marker in _MIN_ASC_KERNEL_MARKERS:
        if marker not in masked:
            problems.append(f"custom_op_asc missing {marker!r}")
    for pattern, label in _ASC_BINDING_MARKERS:
        if not re.search(pattern, masked):
            problems.append(f"custom_op_asc missing {label}")
    marker_lines = len(HOST_SECTION_MARKER_RE.findall(gen.custom_op_asc))
    if marker_lines != 1:
        problems.append(
            f"custom_op_asc must carry exactly one {HOST_SECTION_MARKER!r} "
            f"separator comment line ({marker_lines} found)"
        )
    if "PYBIND11_MODULE" in masked or _ASC_PYBIND_RE.search(code):
        problems.append("custom_op_asc must not use pybind11")
    for marker in _MIN_PY_MARKERS:
        if marker not in gen.model_new_py:
            problems.append(f"model_new_py missing {marker!r}")
    return problems


class LLMClient:
    """Thin OpenAI-compatible client for Ascend C operator generation."""

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
    ) -> None:
        """Create a client for one generation model."""
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
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
        messages.append({"role": "user", "content": prompt + STRUCTURED_OUTPUT_NOTE})

        last_error: Exception | None = None
        last_raw = ""
        for attempt in range(max_retries + 1):
            if attempt > 0:
                _append_retry_turn(messages, last_raw)
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

    def _attempt_parsers(
        self,
    ) -> tuple[tuple[str, Callable[[list[dict]], GenerationResult]], ...]:
        """Return structured parsing first, then the plain fallback."""
        return (
            ("structured", self._parse_structured),
            ("fenced", self._parse_fenced),
        )

    def _generate_once(self, messages: list[dict]) -> GenerationResult:
        """Request one completion; fall back to plain extraction."""
        last_error: Exception | None = None
        for name, parser in self._attempt_parsers():
            try:
                return parser(messages)
            except Exception as exc:
                last_error = exc
                logger.debug("LLM parser {} failed: {}", name, exc)
        raise last_error or ValueError("no LLM response parser succeeded")

    def _parse_structured(self, messages: list[dict]) -> GenerationResult:
        """Parse the OpenAI structured-output schema into deliverables."""
        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=messages,
            response_format=AscendCGeneration,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            extra_body=self.extra_body,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise ValueError("structured parse returned None")
        return GenerationResult(
            generation=parsed,
            raw_text=response.choices[0].message.content or "",
        )

    def _parse_fenced(self, messages: list[dict]) -> GenerationResult:
        """Complete without a schema and extract the two deliverables."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            extra_body=self.extra_body,
        )
        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise ValueError(
                "response truncated at max_tokens; raise "
                "generation.max_tokens or lower reasoning_effort"
            )
        raw = choice.message.content or ""
        return GenerationResult(
            generation=extract_generation(raw),
            raw_text=raw,
        )


_RETRY_REMINDER = (
    "Your previous answer did not contain the "
    "required file contents. Return the COMPLETE "
    "custom_op.asc source in `custom_op_asc` and "
    "the COMPLETE model_new.py source in "
    "`model_new_py` — full code, no placeholders."
)


def _append_retry_turn(messages: list[dict], last_raw: str) -> None:
    """Append the previous answer and a correction reminder."""
    if last_raw:
        messages.append({"role": "assistant", "content": last_raw})
    messages.append({"role": "user", "content": _RETRY_REMINDER})


_FENCE_RE = re.compile(r"```(?P<tag>[A-Za-z0-9_+.-]*)\s*\n(?P<body>.*?)```", re.DOTALL)


def extract_generation(text: str) -> AscendCGeneration:
    """Extract the two deliverables from JSON fields or fenced code blocks.

    Raises:
        ValueError: If neither layout carries a usable pair of sources.
    """
    payload = _json_payload(text)
    if payload is not None:
        return payload
    blocks = list(_FENCE_RE.finditer(text))
    if not blocks:
        raise ValueError("no JSON fields and no fenced code blocks in model response")
    asc_src, py_src = _pair_fenced_blocks(blocks)
    if asc_src is None or py_src is None:
        raise ValueError("could not identify custom_op.asc and model_new.py blocks")
    return AscendCGeneration(custom_op_asc=asc_src, model_new_py=py_src)


def _json_payload(text: str) -> AscendCGeneration | None:
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
    asc_src = data.get("custom_op_asc")
    py_src = data.get("model_new_py")
    if not isinstance(asc_src, str) or not isinstance(py_src, str):
        return None
    return AscendCGeneration(custom_op_asc=asc_src, model_new_py=py_src)


def _pair_fenced_blocks(
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
