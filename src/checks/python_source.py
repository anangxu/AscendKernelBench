"""Regex catalogs and the public model_new.py static-check entry."""

from __future__ import annotations

import ast

from .python_ast import Sink, WrapperSemantics, _collect_module_scalars
from .rules import PatternRule, run_rules
from .text import dedupe, prepare_python_source

TRY_EXCEPT_PATTERNS = (r"\btry\s*:", r"\bexcept\s*:", r"\bexcept\s+\w+")
PASS_PATTERN = r"\bpass\b"

CPU_FALLBACK_PATTERNS = (
    r"\.cpu\s*\(",
    r"\.numpy\s*\(",
    r"\bimport\s+numpy\b",
    r"\bfrom\s+numpy\b",
    r"\bnumpy\s*\.\s*\w+",
)

NPU_NATIVE_PATTERNS = (
    r"torch_npu\.npu_\w+",
    r"\baclnn\w*",
    r"torch\.ops\.op_plugin",
    r"torch\.ops\.atlas",
)

STREAM_PATTERNS = (
    r"torch\.cuda\.Stream\s*\(",
    r"cuda\.Stream\s*\(",
    r"torch\.npu\.Stream\s*\(",
    r"npu\.Stream\s*\(",
    r"with\s+torch\.(cuda|npu)\.stream",
    r"\.wait_stream\s*\(",
    r"\.record_stream\s*\(",
)

THREAD_PATTERNS = (
    r"threading\.Thread\s*\(",
    r"\bimport\s+threading\b",
    r"\bfrom\s+threading\s+import\b",
    r"multiprocessing\.(Process|Pool|Manager|Queue|Pipe)",
    r"\bimport\s+multiprocessing\b",
    r"concurrent\.futures",
    r"ThreadPoolExecutor",
    r"ProcessPoolExecutor",
)

LAZY_TENSOR_PATTERNS = (
    r"_make_subclass",
    r"class\s+\w+.*\(torch\.Tensor\)",
    r"class\s+\w+.*\(Tensor\)",
    r"torch\.Tensor\.__new__",
)

TIMING_EVENT_PATCH_PATTERNS = (
    r"torch\.(cuda|npu)\.Event\.record\s*=",
    r"torch\.(cuda|npu)\.Event\.elapsed_time\s*=",
    r"torch\.(cuda|npu)\.synchronize\s*=",
    r"torch\.(cuda|npu)\.Event\s*=",
    r"time\.perf_counter\s*=",
    r"time\.time\s*=",
)

RESULT_CACHE_PATTERNS = (
    r"_cached_output",
    r"_result_cache",
    r"functools\.(lru_cache|cache)",
)

############################# REGEX RULE CATALOG #############################
PYTHON_PATTERN_RULES: tuple[PatternRule, ...] = (
    PatternRule(
        TRY_EXCEPT_PATTERNS,
        "Contains try-except block (potential fallback bypass)",
    ),
    PatternRule(
        (PASS_PATTERN,),
        "Contains 'pass' statement (inheritance bypass)",
    ),
    PatternRule(CPU_FALLBACK_PATTERNS, "Contains CPU/NumPy fallback pattern"),
    PatternRule(
        NPU_NATIVE_PATTERNS,
        "Uses vendor native op shortcut",
        include_match=True,
    ),
    PatternRule(
        STREAM_PATTERNS,
        "Uses stream primitives (potential timing manipulation)",
    ),
    PatternRule(
        THREAD_PATTERNS,
        "Uses threading/multiprocessing (potential timing manipulation)",
    ),
    PatternRule(
        LAZY_TENSOR_PATTERNS,
        "Uses lazy tensor pattern (potential correctness hack)",
    ),
    PatternRule(
        TIMING_EVENT_PATCH_PATTERNS,
        "Reassigns timing function (monkey patch detected)",
    ),
    PatternRule(
        RESULT_CACHE_PATTERNS,
        "Caches results across calls (outputs must depend on current inputs)",
    ),
)
############################# REGEX RULE CATALOG #############################


def check_model_new(source: str, *, sink: Sink | None = None) -> list[str]:
    """Return static-check violations for generated model_new.py source.

    Args:
        source: Raw model_new.py text.
        sink: Allowed compute sink; None means torch.ops.custom_op, the
            Ascend C contract. pyasc passes its kernel-module sink.

    Returns:
        Deduplicated human-readable violations; empty means pass.
    """
    code = prepare_python_source(source)
    violations = run_rules(code, PYTHON_PATTERN_RULES)
    try:
        # Module-level value constants are shape bookkeeping; function names are
        # not, so a helper call stays visible to the checker.
        module_scalars = _collect_module_scalars(
            ast.parse(code), include_functions=False
        )
        violations.extend(
            WrapperSemantics(
                source, sink=sink, module_scalars=module_scalars
            ).violations
        )
    except SyntaxError as exc:
        violations.append(f"model_new.py does not parse as Python: {exc}")
    return dedupe(violations)
