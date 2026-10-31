"""Task discovery, loading and contract validation.

KernelBench stays the default corpus and keeps its historical ids. A second
suite (the CANN domain tasks) is namespaced, loaded through the same
``Task`` contract, and never mixed into a KernelBench run.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from . import suite as suite_module

TASK_ID_RE = suite_module.KB_TASK_ID_RE
STEM_NUM_RE = __import__("re").compile(r"^(?P<num>\d+)_")
# Contract metadata recorded with a run so a changed task source is refused
# instead of silently re-evaluated.
META_FIELDS = ("suite", "source", "source_revision", "category", "task_hash")


@dataclass(frozen=True)
class Task:
    """A single benchmark task backed by one task source file."""

    task_id: str  # e.g. level1/19_ReLU or cann/nn/rmsnorm
    level: int | None  # None for suites that have no KernelBench level
    name: str
    path: Path
    task_py: str
    suite: str = suite_module.DEFAULT_SUITE
    source: str = ""
    source_revision: str = ""
    task_hash: str = ""
    category: str = ""
    cases: tuple = ()

    @property
    def problem_id(self) -> str:
        """Task id stored in eval_results.json."""
        return self.task_id

    def provenance(self) -> dict[str, str]:
        """Return the provenance fields recorded for a run."""
        recorded = {
            "suite": self.suite,
            "source": self.source,
            "source_revision": self.source_revision,
            "category": self.category,
            "task_hash": suite_module.task_hash(self.task_py),
        }
        return {key: recorded[key] for key in META_FIELDS}


def _validate_contract(task_py: str, task_id: str) -> None:
    """Statically verify the task contract without executing task code."""
    try:
        tree = ast.parse(task_py)
    except SyntaxError as exc:
        raise ValueError(f"{task_id}: task source has a syntax error: {exc}") from exc
    top_level = {getattr(n, "name", None) for n in tree.body}
    top_level |= {
        n.targets[0].id
        for n in tree.body
        if isinstance(n, ast.Assign)
        and len(n.targets) == 1
        and isinstance(n.targets[0], ast.Name)
    }
    missing = {"Model", "get_inputs", "get_init_inputs"} - {
        name for name in top_level if name
    }
    if missing:
        raise ValueError(f"{task_id}: missing contract symbols: {missing}")


def _task_file(directory: Path, rel_id: str, root: Path) -> Path:
    """Return the source file of a task, tolerating both layouts."""
    if directory.is_dir():
        candidate = directory / "task.py"
        if candidate.is_file():
            return candidate
    elif directory.is_file():
        return directory
    flat = root / f"{rel_id}.py"
    if flat.is_file():
        return flat
    raise FileNotFoundError(f"Task not found: {rel_id} under {root}")


def load_task(
    task_id: str,
    kb_root: Path | None = None,
    roots: Mapping[str, Path] | None = None,
) -> Task:
    """Load one task by its full id.

    KernelBench ids stay ``level{L}/{file_stem}``; namespaced ids such as
    ``cann/nn/rmsnorm`` are resolved under that suite's root.

    Args:
        task_id: Full task id.
        kb_root: Optional KernelBench root override for tests.
        roots: Optional per-suite root overrides for tests.

    Raises:
        ValueError: If the id is malformed, escapes its suite root, or the
            contract is missing.
        FileNotFoundError: If the task source does not exist.
    """
    resolved_roots = suite_module.suite_roots(roots)
    if kb_root is not None:
        resolved_roots[suite_module.DEFAULT_SUITE] = Path(kb_root)
    suite_id = suite_module.suite_of(task_id)
    rel_id = suite_module.relative_id(task_id)
    root = resolved_roots[suite_id]
    directory = suite_module.task_dir(task_id, resolved_roots)
    task_file = _task_file(directory, rel_id, root)
    task_py = task_file.read_text(encoding="utf-8")
    _validate_contract(task_py, task_id)
    metadata = suite_module.task_metadata(task_file.parent)
    cases = _load_task_cases(task_file.parent, metadata)
    match = TASK_ID_RE.match(rel_id) if suite_id == suite_module.DEFAULT_SUITE else None
    return Task(
        task_id=task_id,
        level=int(match.group("level")) if match else None,
        name=match.group("stem") if match else Path(rel_id).name,
        path=task_file,
        task_py=task_py,
        suite=suite_id,
        source=metadata["source"],
        source_revision=metadata["source_revision"],
        category=metadata["category"],
        task_hash=suite_module.task_hash(task_py),
        cases=cases,
    )


def _load_task_cases(directory: Path, metadata: dict[str, str]) -> tuple:
    """Load the optional case file a task declares in its metadata."""
    from . import cases as cases_module

    declared = metadata.get("cases", "")
    if not declared:
        return ()
    path = (directory / declared).resolve()
    if directory.resolve() not in path.parents and path.parent != directory.resolve():
        raise ValueError(f"{declared}: case file escapes the task directory")
    return cases_module.load_cases(
        path,
        source_revision=metadata.get("source_revision", ""),
        seed_base=0,
    )


def _sort_key(path: Path) -> tuple[int, str]:
    """Return a numeric-then-name sort key for numbered task stems."""
    match = STEM_NUM_RE.match(path.stem)
    return (int(match.group("num")) if match else 0, path.stem)


def discover_tasks(
    level: int | None = None,
    kb_root: Path | None = None,
    suite: str | None = None,
    roots: Mapping[str, Path] | None = None,
) -> list[Task]:
    """Discover tasks of one suite, sorted by suite order and name.

    Args:
        level: KernelBench level filter; ignored by suites without levels.
        kb_root: Optional KernelBench root override for tests.
        suite: Suite id; defaults to KernelBench so old behavior is unchanged.
        roots: Optional per-suite root overrides for tests.

    Returns:
        Loaded tasks. A namespaced suite is discovered by its task
        directories, each holding ``task.py``.
    """
    resolved_roots = suite_module.suite_roots(roots)
    if kb_root is not None:
        resolved_roots[suite_module.DEFAULT_SUITE] = Path(kb_root)
    suite_id = suite or suite_module.DEFAULT_SUITE
    if suite_id == suite_module.DEFAULT_SUITE:
        return _discover_kernelbench(level, resolved_roots[suite_id])
    if suite_id not in resolved_roots:
        raise ValueError(f"unknown task suite {suite_id!r}")
    root = resolved_roots[suite_id]
    if not root.is_dir():
        raise FileNotFoundError(f"task suite root not found: {root}")
    tasks: list[Task] = []
    for task_file in sorted(root.rglob("task.py")):
        rel_dir = task_file.parent.relative_to(root).as_posix()
        if rel_dir in {"", "."}:
            continue
        tasks.append(load_task(f"{suite_id}/{rel_dir}", roots=resolved_roots))
    return tasks


def _discover_kernelbench(level: int | None, root: Path) -> list[Task]:
    """Discover KernelBench tasks exactly as before, one level or all."""
    tasks: list[Task] = []
    level_dirs = [root / f"level{level}"] if level else sorted(root.glob("level*"))
    for level_dir in level_dirs:
        if not level_dir.is_dir():
            continue
        for task_file in sorted(level_dir.glob("*.py"), key=_sort_key):
            tasks.append(load_task(f"{level_dir.name}/{task_file.stem}", kb_root=root))
    return tasks


def task_meta(tasks: list[Task]) -> dict[str, dict[str, str]]:
    """Return the per-task provenance mapping recorded with a run."""
    return {task.task_id: task.provenance() for task in tasks}
