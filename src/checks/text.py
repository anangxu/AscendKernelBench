"""Comment stripping and string masking for static checks.

Comments are blanked in place rather than re-rendered from tokens, since
patterns such as import numpy depend on the original inter-token whitespace.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize

# Separator between the device and host sections of a split custom_op.asc.
# Only a standalone comment line counts, so prose mentions of the marker do
# not split a file.
HOST_SECTION_MARKER = "ASCEND_HOST_SECTION"
HOST_SECTION_MARKER_RE = re.compile(
    r"^\s*//[\s=]*" + HOST_SECTION_MARKER + r"[\s=]*$", re.MULTILINE
)

# Optional split of the device section into one compiled unit per kernel entry
# for the deepjit build backend; see docs/guide/deepjit.md.
KERNEL_SECTION_MARKER = "ASCEND_KERNEL_SECTION"
KERNEL_SECTION_MARKER_RE = re.compile(
    r"^\s*//[\s=]*" + KERNEL_SECTION_MARKER + r"[\s=]*$"
)


def mask_string_constants(source: str) -> str:
    """Blank string constants so embedded text cannot trip regex rules.

    Returns:
        Source with string spans blanked; unparsable input unchanged.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))

    ranges: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and getattr(node, "end_lineno", None) is not None
            and getattr(node, "end_col_offset", None) is not None
        ):
            continue
        start = offsets[node.lineno - 1] + node.col_offset
        end = offsets[node.end_lineno - 1] + node.end_col_offset
        ranges.append((start, end))

    if not ranges:
        return source
    parts: list[str] = []
    cursor = 0
    for start, end in sorted(ranges):
        parts.append(source[cursor:start])
        parts.append(" " * (end - start))
        cursor = end
    parts.append(source[cursor:])
    return "".join(parts)


def strip_python_comments(code: str) -> str:
    """Blank Python # comments in place, preserving all other bytes.

    Args:
        code: Python source, usually after string masking.
    """
    lines = code.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    chars = list(code)
    try:
        tokens = tokenize.generate_tokens(io.StringIO(code).readline)
        for tok in tokens:
            if tok.type != tokenize.COMMENT:
                continue
            (srow, scol), (erow, ecol) = tok.start, tok.end
            start = offsets[srow - 1] + scol
            end = offsets[erow - 1] + ecol
            for i in range(start, end):
                chars[i] = " "
    except (tokenize.TokenError, IndentationError):
        return "\n".join(line.split("#", 1)[0].rstrip() for line in code.splitlines())
    return "".join(chars)


def strip_cpp_comments(code: str) -> str:
    """Blank C++ comments so markers inside comments do not count."""
    code = re.sub(
        r"/\*.*?\*/",
        lambda match: " " * (match.end() - match.start()),
        code,
        flags=re.DOTALL,
    )
    return re.sub(
        r"//[^\n]*",
        lambda match: " " * (match.end() - match.start()),
        code,
    )


def prepare_python_source(source: str) -> str:
    """Return comment-stripped, string-masked Python source."""
    return strip_python_comments(mask_string_constants(source))


def dedupe(violations: list[str]) -> list[str]:
    """Return violations with first-occurrence order preserved."""
    return list(dict.fromkeys(violations))
