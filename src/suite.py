"""Task suites: which corpus a task id belongs to.

KernelBench stays the default suite and keeps its historical ``levelN/stem``
ids. The CANN suite is opt-in, namespaced under its own root, and every id
carries its suite so the two corpora cannot collide.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Mapping

from ._paths import DOMAIN_ROOT, KB_ROOT

DEFAULT_SUITE = "kernelbench"
SUITE_IDS = ("kernelbench", "cann")
SUITE_ROOTS: Mapping[str, Path] = {
    "kernelbench": KB_ROOT,
    "cann": DOMAIN_ROOT / "cann",
}

# KernelBench keeps its historical id shape; other suites are namespaced by
# their suite id and may nest (cann/nn/rmsnorm).
KB_TASK_ID_RE = re.compile(r"^level(?P<level>\d+)/(?P<stem>[^/]+)$")


def suite_roots(roots: Mapping[str, Path] | None = None) -> dict[str, Path]:
    """Return the root directory of every known suite.

    Args:
        roots: Optional overrides for tests; missing keys fall back to the
            checkout layout.

    Returns:
        Mapping of suite id to its task root.
    """
    import os

    resolved = dict(SUITE_ROOTS)
    # Test fixtures may relocate the CANN root without touching the real corpus.
    override = os.environ.get("AKB_CANN_ROOT", "")
    if override:
        resolved["cann"] = Path(override)
    if roots:
        resolved.update({name: Path(path) for name, path in roots.items()})
    return resolved


def suite_of(task_id: str) -> str:
    """Return the suite id a full task id belongs to.

    A bare ``levelN/stem`` id is KernelBench, so every pre-existing run and
    command keeps its meaning. A ``cann/...`` id names the CANN suite.

    Args:
        task_id: Full task id, such as ``level1/19_ReLU`` or
            ``cann/nn/rmsnorm``.

    Returns:
        The suite id, defaulting to ``kernelbench``.

    Raises:
        ValueError: If the id is empty, absolute, or carries no task name.
    """
    cleaned = _clean_id(task_id)
    head, _, tail = cleaned.partition("/")
    if head in SUITE_IDS and tail:
        return head
    return DEFAULT_SUITE


def relative_id(task_id: str) -> str:
    """Return a task id without its suite prefix.

    Args:
        task_id: Full task id.

    Returns:
        ``level1/19_ReLU`` for KernelBench ids, the stripped id for namespaced
        suites.

    Raises:
        ValueError: If the id is malformed for its suite.
    """
    cleaned = _clean_id(task_id)
    suite = suite_of(cleaned)
    if suite == DEFAULT_SUITE:
        if not KB_TASK_ID_RE.match(cleaned):
            raise ValueError(
                f"invalid KernelBench task id {task_id!r}; "
                "expected 'level{L}/{stem}'"
            )
        return cleaned
    stripped = cleaned[len(suite) + 1 :]
    if not stripped:
        raise ValueError(f"task id {task_id!r} carries no task name after {suite!r}")
    return stripped


def task_dir(task_id: str, roots: Mapping[str, Path] | None = None) -> Path:
    """Resolve the directory that holds a task, refusing unsafe ids.

    Args:
        task_id: Full task id.
        roots: Optional suite root overrides.

    Returns:
        The task directory inside the suite root.

    Raises:
        ValueError: If the id would escape the suite root.
    """
    suite = suite_of(task_id)
    rel = relative_id(task_id)
    root = suite_roots(roots)[suite].resolve()
    target = (root / rel).resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"task id {task_id!r} escapes the {suite} task root")
    return target


def _clean_id(task_id: str) -> str:
    """Validate the shared shape of a task id before any suite logic."""
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError("task id must be a nonempty string")
    cleaned = task_id.strip()
    if cleaned.startswith("/") or "\\" in cleaned or "\x00" in cleaned:
        raise ValueError(f"task id {task_id!r} must be a relative POSIX path")
    segments = cleaned.split("/")
    if any(segment in {"", ".", ".."} for segment in segments):
        raise ValueError(f"task id {task_id!r} contains an empty, '.' or '..' segment")
    return cleaned


def task_hash(task_py: str) -> str:
    """Return the content hash recorded for one task source."""
    return hashlib.sha256(task_py.encode("utf-8")).hexdigest()


def task_metadata(directory: Path) -> dict[str, str]:
    """Read the optional task metadata file beside a task source.

    Args:
        directory: Task directory that may hold ``meta.yaml``.

    Returns:
        Recorded provenance fields; an absent file yields empty defaults.
    """
    path = Path(directory) / "meta.yaml"
    if not path.is_file():
        return {"source": "", "source_revision": "", "category": "", "cases": ""}
    import yaml

    try:
        loaded: Any = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"{path}: unreadable task metadata: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"{path}: task metadata must be a mapping")
    return {
        "source": str(loaded.get("source", "")),
        "source_revision": str(loaded.get("source_revision", "")),
        "category": str(loaded.get("category", "")),
        "cases": str(loaded.get("cases", "")),
    }
