"""Task discovery, loading and contract validation.

The task set is the vendored KernelBench copy under KernelBench/, used in
place in its original single-file format, checked by an AST contract.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

from ._paths import KB_ROOT

TASK_ID_RE = re.compile(r"^level(?P<level>\d+)/(?P<stem>[^/]+)$")
STEM_NUM_RE = re.compile(r"^(?P<num>\d+)_")


@dataclass(frozen=True)
class Task:
    """A single benchmark task backed by one KernelBench source file."""

    task_id: str  # e.g. level1/19_ReLU
    level: int
    name: str
    path: Path
    task_py: str

    @property
    def problem_id(self) -> str:
        """KernelBench problem id stored in eval_results.json."""
        return self.task_id


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


def load_task(task_id: str, kb_root: Path | None = None) -> Task:
    """Load one task by id (level{L}/{file_stem}).

    Raises:
        ValueError: If task_id is malformed or the contract is missing.
        FileNotFoundError: If the task file does not exist.
    """
    root = Path(kb_root) if kb_root else KB_ROOT
    match = TASK_ID_RE.match(task_id)
    if not match:
        raise ValueError(f"Invalid task id {task_id!r}; expected 'level{{L}}/{{stem}}'")
    task_file = root / task_id
    if task_file.is_dir():
        task_file = task_file / "task.py"  # tolerate legacy migrated layout
    elif not task_file.is_file():
        task_file = root / f"{task_id}.py"
    if not task_file.is_file():
        raise FileNotFoundError(f"Task not found: {task_id} under {root}")
    task_py = task_file.read_text(encoding="utf-8")
    _validate_contract(task_py, task_id)
    return Task(
        task_id=task_id,
        level=int(match.group("level")),
        name=match.group("stem"),
        path=task_file,
        task_py=task_py,
    )


def _sort_key(path: Path) -> tuple[int, str]:
    """Return a numeric-then-name sort key for numbered task stems."""
    match = STEM_NUM_RE.match(path.stem)
    return (int(match.group("num")) if match else 0, path.stem)


def discover_tasks(level: int | None = None, kb_root: Path | None = None) -> list[Task]:
    """Discover tasks under the KernelBench root, sorted by level and stem.

    With level set, only that levelN directory is scanned.
    """
    root = Path(kb_root) if kb_root else KB_ROOT
    tasks: list[Task] = []
    level_dirs = [root / f"level{level}"] if level else sorted(root.glob("level*"))
    for level_dir in level_dirs:
        if not level_dir.is_dir():
            continue
        for task_file in sorted(level_dir.glob("*.py"), key=_sort_key):
            tasks.append(load_task(f"{level_dir.name}/{task_file.stem}", root))
    return tasks
