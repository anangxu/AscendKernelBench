"""No-device tests for DeepJIT version resolution and pin verification.

Runs with plain asserts and no test framework:

    python tests/test_deepjit_version.py

Every layout is built with real Git commands in a temporary directory, so a
packed-refs clone and a linked worktree are reproduced rather than mocked. No
torch, no NPU, and no DeepJIT build is needed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DeepJitConfig  # noqa: E402
from src.deepjit_build import (  # noqa: E402
    DeepJitVersionError,
    _prepare_layout,
    deepjit_commit,
)

# A host may need an explicit binary, as when a launcher shim refuses to run
# before its licence is accepted; the device and CI just use git on PATH.
GIT = os.environ.get("AKB_TEST_GIT") or shutil.which("git") or "git"

SOURCE = (
    '#include "kernel_operator.h"\n'
    "__global__ __vector__ void probe(__gm__ uint8_t* x, uint32_t n) {}\n"
    "// ==================== ASCEND_HOST_SECTION ====================\n"
    "#include <torch/library.h>\n"
    "void host_probe() {}\n"
)
PINNED = "3732a3b9a0a2e5126396908a362ba5abdf043b2f"


def check(name: str, condition: bool, detail: str = "") -> None:
    """Record one check result."""
    print(f"  {'ok  ' if condition else 'FAIL'} - {name}{'' if condition else ' :: ' + detail}")
    if not condition:
        raise AssertionError(f"{name}: {detail}")


def git(repo: Path, *args: str) -> str:
    """Run one Git command in a repository and return its stdout."""
    result = subprocess.run(
        [GIT, "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def make_checkout(tmp: Path, name: str = "deepjit") -> tuple[Path, str]:
    """Create a real Git checkout that also looks like a DeepJIT tree."""
    repo = tmp / name
    repo.mkdir()
    git(repo, "init", "--quiet", "--initial-branch=main")
    git(repo, "config", "user.email", "agent@example.invalid")
    git(repo, "config", "user.name", "AscendKernelBench agent")
    (repo / "include" / "deep_jit" / "utils").mkdir(parents=True)
    (repo / "include" / "deep_jit" / "utils" / "command.hpp").write_text(
        "// deepjit header\n", encoding="utf-8"
    )
    git(repo, "add", "-A")
    git(repo, "commit", "--quiet", "-m", "initial")
    return repo, git(repo, "rev-parse", "HEAD")


def test_normal_ref(tmp: Path) -> None:
    """A plain checkout resolves through its loose reference."""
    repo, head = make_checkout(tmp, "plain")
    check("loose ref exists", (repo / ".git" / "refs" / "heads" / "main").is_file())
    check("plain checkout resolves", deepjit_commit(repo) == head, deepjit_commit(repo))


def test_packed_refs(tmp: Path) -> None:
    """A packed-refs checkout resolves even without the loose reference."""
    repo, head = make_checkout(tmp, "packed")
    git(repo, "pack-refs", "--all")
    check(
        "loose ref is gone after pack-refs",
        not (repo / ".git" / "refs" / "heads" / "main").exists(),
    )
    check("packed refs resolve", deepjit_commit(repo) == head, deepjit_commit(repo))


def test_detached_head(tmp: Path) -> None:
    """A detached HEAD resolves to the commit it points at."""
    repo, head = make_checkout(tmp, "detached")
    git(repo, "checkout", "--quiet", "--detach", head)
    check("detached head resolves", deepjit_commit(repo) == head, deepjit_commit(repo))


def test_linked_worktree(tmp: Path) -> None:
    """A linked worktree resolves even though .git is a file."""
    repo, head = make_checkout(tmp, "origin_repo")
    worktree = tmp / "linked"
    git(repo, "worktree", "add", "--quiet", "-b", "linked", str(worktree))
    check("worktree .git is a file", (worktree / ".git").is_file())
    check("worktree resolves", deepjit_commit(worktree) == head, deepjit_commit(worktree))


def test_unreadable_version(tmp: Path) -> None:
    """A source tree without readable Git metadata reports no version."""
    plain = tmp / "not_a_repo"
    (plain / "include" / "deep_jit").mkdir(parents=True)
    check("a non-repository yields no commit", deepjit_commit(plain) == "")
    try:
        _prepare_layout(
            tmp / "work_nonrepo", SOURCE, DeepJitConfig(root=str(plain)), "Ascend910B4"
        )
    except DeepJitVersionError as exc:
        check("the layout refuses an unreadable version",
              "cannot read the DeepJIT version" in str(exc), str(exc))
    else:
        check("the layout refuses an unreadable version", False, "accepted")


def test_pin_match(tmp: Path) -> None:
    """A matching pin is verified against the real checkout and recorded."""
    repo, head = make_checkout(tmp, "pinned")
    paths = _prepare_layout(
        tmp / "work_match", SOURCE, DeepJitConfig(root=str(repo), commit=head), "Ascend910B4"
    )
    check("the actual commit is recorded", paths.commit == head, paths.commit)
    check("the signature carries the actual commit",
          paths.signature.startswith(f"deepjit-{head[:12]}-"), paths.signature)


def test_pin_mismatch(tmp: Path) -> None:
    """A pin that differs from the checkout is refused, naming both."""
    repo, head = make_checkout(tmp, "mismatch")
    other = "0" * 40
    try:
        _prepare_layout(
            tmp / "work_mismatch",
            SOURCE,
            DeepJitConfig(root=str(repo), commit=other),
            "Ascend910B4",
        )
    except DeepJitVersionError as exc:
        message = str(exc)
        check("a pin mismatch is refused", "version mismatch" in message, message)
        check("the message names the checkout", head in message, message)
        check("the message names the pin", other in message, message)
    else:
        check("a pin mismatch is refused", False, "accepted")


def test_recorded_commit_is_never_the_constant(tmp: Path) -> None:
    """Without a pin the record carries the real commit, not the constant."""
    repo, head = make_checkout(tmp, "unpinned")
    check("the checkout is not the verified pin", head != PINNED)
    paths = _prepare_layout(
        tmp / "work_unpinned", SOURCE, DeepJitConfig(root=str(repo)), "Ascend910B4"
    )
    check("the record is the real commit", paths.commit == head, paths.commit)
    check("the record is not the verified constant", paths.commit != PINNED)


def main() -> int:
    """Run every version check."""
    print("deepjit version checks")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        test_normal_ref(root)
        test_packed_refs(root)
        test_detached_head(root)
        test_linked_worktree(root)
        test_unreadable_version(root)
        test_pin_match(root)
        test_pin_mismatch(root)
        test_recorded_commit_is_never_the_constant(root)
    print("\nall deepjit version checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
