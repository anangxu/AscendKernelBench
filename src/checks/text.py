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


def _byte_col_to_char_col(line: str, byte_col: int) -> int:
    """Convert an AST UTF-8 byte column to a Python code-point column."""
    prefix = line.encode("utf-8")[:byte_col]
    return len(prefix.decode("utf-8", errors="ignore"))


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
        start = offsets[node.lineno - 1] + _byte_col_to_char_col(
            lines[node.lineno - 1], node.col_offset
        )
        end = offsets[node.end_lineno - 1] + _byte_col_to_char_col(
            lines[node.end_lineno - 1], node.end_col_offset
        )
        ranges.append((start, end))

    if not ranges:
        return source
    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    parts: list[str] = []
    cursor = 0
    for start, end in merged:
        parts.append(source[cursor:start])
        parts.append(
            "".join("\n" if char == "\n" else " " for char in source[start:end])
        )
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


def _blank_span(chars: list[str], start: int, end: int) -> None:
    """Replace non-newline characters in chars[start:end] with spaces."""
    for index in range(start, min(end, len(chars))):
        if chars[index] != "\n":
            chars[index] = " "


def _raw_string_end(code: str, index: int) -> int | None:
    """Return the exclusive end of a C++ raw string, or None."""
    for prefix in ('u8R"', 'uR"', 'UR"', 'LR"', 'R"'):
        if not code.startswith(prefix, index):
            continue
        start = index + len(prefix)
        open_paren = code.find("(", start)
        if open_paren == -1:
            return None
        delimiter = code[start:open_paren]
        close = ")" + delimiter + '"'
        end = code.find(close, open_paren + 1)
        return None if end == -1 else end + len(close)
    return None


def _is_digit_separator(code: str, index: int) -> bool:
    """Return True for a C++14 digit separator such as 1'000."""
    if index <= 0 or index + 1 >= len(code):
        return False
    hex_digits = "0123456789abcdefABCDEF"
    return code[index - 1] in hex_digits and code[index + 1] in hex_digits


def strip_cpp_comments(code: str, *, mask_literals: bool = False) -> str:
    """Blank C++ comments in place, keeping literal content visible.

    Comment delimiters inside ordinary string and character literals do not
    start a comment, so a literal such as "/*" cannot hide the code after
    it. With mask_literals set, literal contents are blanked as well; the
    default keeps quoted include paths visible to include checks. Newlines
    are preserved so line-oriented patterns keep working.
    """
    chars = list(code)
    index = 0
    length = len(chars)
    while index < length:
        char = code[index]
        next_char = code[index + 1] if index + 1 < length else ""
        raw_end = _raw_string_end(code, index)
        if raw_end is not None:
            if mask_literals:
                _blank_span(chars, index, raw_end)
            index = raw_end
        elif char == "'" and _is_digit_separator(code, index):
            index += 1
        elif char == "/" and next_char == "/":
            end = code.find("\n", index)
            if end == -1:
                end = length
            _blank_span(chars, index, end)
            index = end
        elif char == "/" and next_char == "*":
            end = code.find("*/", index + 2)
            end = length if end == -1 else end + 2
            _blank_span(chars, index, end)
            index = end
        elif char in {'"', "'"}:
            quote = char
            cursor = index + 1
            while cursor < length:
                if code[cursor] == "\\":
                    cursor += 2
                    continue
                if code[cursor] == quote:
                    cursor += 1
                    break
                cursor += 1
            if mask_literals:
                _blank_span(chars, index, cursor)
            index = cursor
        else:
            index += 1
    return "".join(chars)


def prepare_python_source(source: str) -> str:
    """Return comment-stripped, string-masked Python source."""
    return strip_python_comments(mask_string_constants(source))


def dedupe(violations: list[str]) -> list[str]:
    """Return violations with first-occurrence order preserved."""
    return list(dict.fromkeys(violations))
