"""Optional DeepJIT build path for the Ascend C backend.

A candidate keeps its shape: custom_op.asc with the host-section marker and a
model_new.py that calls torch.ops.custom_op. Only the build path changes: a
PyTorch extension embeds the DeepJIT runtime, so the device kernel is compiled
at prepare time and launched through DeepJIT instead of the generated launch
stub. The framework owns the runtime, the kernel source, and the cache; the
candidate never runs a build command.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sysconfig
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from loguru import logger

from .config import DeepJitConfig

# The DeepJIT commit this path was verified against (Ascend 910B4, CANN
# 9.1.0, torch 2.10.0). A checkout at another commit is reported, not trusted.
DEEPJIT_VERIFIED_COMMIT = "3732a3b9a0a2e5126396908a362ba5abdf043b2f"
_COMMIT_RE = re.compile(r"[0-9a-f]{40,64}")

# Generated layout inside the sample directory.
DEEPJIT_DIR_NAME = "deepjit"
DEEPJIT_KERNEL_DIR_NAME = "akb_deepjit_kernel"
DEEPJIT_KERNEL_PREFIX = DEEPJIT_KERNEL_DIR_NAME + "/"
DEEPJIT_RUNTIME_HEADER_NAME = "akb_deepjit_runtime.hpp"
DEEPJIT_LAUNCHER_NAME = "akb_deepjit_launcher.cpp"
DEEPJIT_DECLS_NAME = "akb_deepjit_launch_decls.h"
DEEPJIT_HOST_NAME = "custom_op_host.cpp"
DEEPJIT_CACHE_DIR_NAME = "cache"
DEEPJIT_FACTS_NAME = "build_facts.json"
DEEPJIT_ENV_PREFIX = "AKB_DEEPJIT"
DEEPJIT_PREPARE_ENTRY = "akb_deepjit_prepare"
DEEPJIT_ERROR_BUFFER_SIZE = 8192
DEEPJIT_EXTENSION_NAME = "akb_deepjit_host"

# DeepJIT passes compiler, include, and flag strings through a shell command,
# so every path it sees must stay free of whitespace and shell metacharacters.
_UNSAFE_PATH_RE = re.compile(r"""[\s"'`$\\;&|<>()*?!#~]""")
_INCLUDE_RE = re.compile(r"^\s*#\s*include\s*([<\"])([^\">]+)[\">]")
_CONVERTED_INCLUDE = "kernel_operator.h"

# Device parameter types this path marshals; anything else is rejected rather
# than passed through on a guess.
DEEPJIT_POINTER_TYPES = frozenset({"uint8_t", "int8_t", "float", "int32_t", "uint32_t"})
DEEPJIT_SCALAR_TYPES = frozenset({"int", "int32_t", "uint32_t", "int64_t", "float"})
_POINTER_PARAM_RE = re.compile(r"^__gm__\s+(\w+)\s*\*\s*\w+$")
_SCALAR_PARAM_RE = re.compile(r"^(?:const\s+)?(\w+)\s+\w+$")


class DeepJitBuildError(RuntimeError):
    """Raised when the DeepJIT path rejects a source or fails to build."""


class DeepJitVersionError(DeepJitBuildError):
    """Raised when a checkout's version is unreadable or does not match."""


@dataclass(frozen=True)
class DeepJitPaths:
    """Resolved filesystem facts of one DeepJIT build."""

    root: Path
    library_root: Path
    commit: str
    cache_dir: Path
    build_dir: Path
    kernel_source: Path
    signature: str


def resolve_deepjit_settings(settings: DeepJitConfig) -> DeepJitConfig:
    """Return DeepJIT settings with empty fields filled from the environment."""
    updates: dict[str, object] = {}
    for name, variable in (
        ("root", "DEEPJIT_ROOT"),
        ("commit", "DEEPJIT_COMMIT"),
        ("cxx", "DEEPJIT_CXX"),
        ("cache_dir", "DEEPJIT_CACHE_DIR"),
    ):
        if not getattr(settings, name):
            value = os.environ.get(variable, "").strip()
            if value:
                updates[name] = value
    if not settings.include_dirs:
        value = os.environ.get("DEEPJIT_INCLUDE_DIRS", "").strip()
        if value:
            updates["include_dirs"] = [item for item in value.split(":") if item]
    if not settings.library_dirs:
        value = os.environ.get("DEEPJIT_LIBRARY_DIRS", "").strip()
        if value:
            updates["library_dirs"] = [item for item in value.split(":") if item]
    if not updates:
        return settings
    return settings.model_copy(update=updates)


def deepjit_commit(root: Path) -> str:
    """Return the HEAD commit of a DeepJIT checkout, or an empty string.

    Git resolves the revision, so a packed-refs clone, a detached HEAD, and a
    worktree (where .git is a file) all resolve the same way. An empty string
    means the version could not be read, never that it matched.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--verify", "HEAD"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if result.returncode != 0:
        return ""
    value = result.stdout.strip()
    return value if _COMMIT_RE.fullmatch(value) else ""


def _write_if_changed(path: Path, content: str) -> None:
    """Write content only when it differs, keeping the mtime stable.

    The host extension is incremental, so rewriting an identical generated
    file would force a needless relink on every re-evaluation.
    """
    if path.is_file() and path.read_text(encoding="utf-8") == content:
        return
    path.write_text(content, encoding="utf-8")


_UNSAFE_ARGUMENT_RE = re.compile(r"""[\s;|&$`<>()"'\\]""")
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _require_safe_argument(value: str, what: str) -> str:
    """Return a compiler flag or environment value that a shell can carry.

    DeepJIT passes flags through a shell command and exports none of them
    itself, so whitespace and shell metacharacters are refused instead of
    being escaped silently.
    """
    text = str(value)
    if _UNSAFE_ARGUMENT_RE.search(text):
        raise DeepJitBuildError(
            f"{what} must not contain whitespace or shell metacharacters: {text!r}"
        )
    return text


def _require_safe_environment(env: dict[str, str]) -> dict[str, str]:
    """Validate the process environment a build is allowed to apply."""
    checked: dict[str, str] = {}
    for key, value in env.items():
        name = str(key)
        if not _ENV_NAME_RE.match(name):
            raise DeepJitBuildError(
                f"deepjit env name must be a plain identifier: {name!r}"
            )
        checked[name] = _require_safe_argument(str(value), f"deepjit env {name}")
    return checked


def _require_safe_path(value: str, what: str) -> str:
    """Return an absolute, shell-safe path string or raise."""
    text = str(value).strip()
    if not text:
        raise DeepJitBuildError(f"{what} must not be empty in the deepjit settings")
    if not Path(text).is_absolute():
        raise DeepJitBuildError(f"{what} must be an absolute path: {text}")
    if _UNSAFE_PATH_RE.search(text):
        raise DeepJitBuildError(
            f"{what} must not contain whitespace or shell metacharacters: {text}"
        )
    return text


@dataclass(frozen=True)
class DeviceUnit:
    """One compiled device unit: a shared preamble plus exactly one entry."""

    source: str
    signature: object


def _marker_lines(lines: list[str]) -> list[int]:
    """Return the indices of kernel-section marker lines in code position.

    The scan is textual and line based: it does not preprocess the source, so
    a marker inside an ``#if 0`` region still separates units. It does track
    block comments, because a marker line inside one is almost always
    documentation and splitting there would truncate a unit silently.

    Raises:
        DeepJitBuildError: If a marker line sits inside a block comment.
    """
    from .checks.text import KERNEL_SECTION_MARKER_RE

    markers: list[int] = []
    in_block_comment = False
    for index, line in enumerate(lines):
        body = line.rstrip("\n")
        is_marker = KERNEL_SECTION_MARKER_RE.match(body) is not None
        if in_block_comment:
            if is_marker:
                raise DeepJitBuildError(
                    f"kernel-section marker inside a block comment at line "
                    f"{index + 1}; a marker only separates units on a code line"
                )
            if "*/" in body:
                in_block_comment = False
            continue
        if is_marker:
            markers.append(index)
        code = body.split("//", 1)[0]
        if "/*" in code and "*/" not in code.partition("/*")[2]:
            in_block_comment = True
    return markers


def split_device_units(device_source: str) -> list[DeviceUnit]:
    """Split the device section into one compilation unit per kernel entry.

    Without the kernel-section marker the whole device section is one unit,
    which keeps every single-kernel sample unchanged. With markers, the text
    before the first marker is a shared preamble that is prepended to each
    following segment, and the marker lines themselves are dropped. The
    split happens on whole marker lines only, so function bodies, templates,
    and conditional blocks are never cut.

    Raises:
        DeepJitBuildError: If a unit does not hold exactly one kernel entry,
            or if two entries share a name.
    """
    from .build import BuildError, extract_kernel_signatures

    lines = device_source.splitlines(keepends=True)
    markers = _marker_lines(lines)
    if not markers:
        shared, segments = "", [device_source]
    else:
        shared = "".join(lines[: markers[0]])
        segments = []
        for position, start in enumerate(markers):
            end = markers[position + 1] if position + 1 < len(markers) else len(lines)
            segments.append("".join(lines[start + 1 : end]))

    units: list[DeviceUnit] = []
    for index, segment in enumerate(segments):
        source = shared + segment
        where = "device section" if not markers else f"device unit {index}"
        try:
            signatures = extract_kernel_signatures(source)
        except BuildError as exc:
            raise DeepJitBuildError(f"{where} holds no kernel entry: {exc}") from None
        if len(signatures) != 1:
            names = sorted(item.name for item in signatures)
            raise DeepJitBuildError(
                f"deepjit build backend needs exactly one kernel entry per "
                f"compiled unit; {where} holds {len(signatures)}: {names}"
            )
        units.append(DeviceUnit(source=source, signature=signatures[0]))
    names = [str(unit.signature.name) for unit in units]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise DeepJitBuildError(
            f"deepjit build backend needs distinct kernel names, repeated: {duplicates}"
        )
    return units


def convert_device_source(device_source: str) -> str:
    """Return the device source DeepJIT compiles, converting one include.

    The standard kernel_operator.h include may be written with quotes, which
    the DeepJIT include parser rejects; it is rewritten as an angle include
    and the result is what actually gets compiled. Every other quoted include
    is refused instead of being rewritten on a guess.

    Raises:
        DeepJitBuildError: On an unsupported quoted include.
    """
    converted: list[str] = []
    for number, line in enumerate(device_source.splitlines(keepends=True), start=1):
        match = _INCLUDE_RE.match(line)
        if match is None:
            converted.append(line)
            continue
        delimiter, target = match.group(1), match.group(2).strip()
        if delimiter == "<":
            converted.append(line)
            continue
        if target == _CONVERTED_INCLUDE:
            converted.append(
                line[: match.start(1)] + f"<{target}>" + line[match.end(0) :]
            )
            continue
        raise DeepJitBuildError(
            f"deepjit build backend does not support the quoted include "
            f"{target!r} in the device section (line {number}); only "
            f"{_CONVERTED_INCLUDE} is converted"
        )
    return "".join(converted)


def validate_kernel_parameters(signature: object) -> None:
    """Reject device parameters this path does not marshal.

    Args:
        signature: A KernelSignature parsed from the device section.

    Raises:
        DeepJitBuildError: On a parameter type outside the supported sets.
    """
    for raw in str(signature.decl_params).split(","):
        parameter = raw.strip()
        if not parameter:
            continue
        pointer = _POINTER_PARAM_RE.match(parameter)
        if pointer is not None:
            if pointer.group(1) not in DEEPJIT_POINTER_TYPES:
                raise DeepJitBuildError(
                    f"deepjit build backend cannot marshal device pointer type "
                    f"{pointer.group(1)!r} ({parameter!r}); supported pointer "
                    f"types: {sorted(DEEPJIT_POINTER_TYPES)}"
                )
            continue
        scalar = _SCALAR_PARAM_RE.match(parameter)
        if scalar is not None and scalar.group(1) in DEEPJIT_SCALAR_TYPES:
            continue
        raise DeepJitBuildError(
            f"deepjit build backend cannot marshal device parameter "
            f"{parameter!r}; supported scalars: {sorted(DEEPJIT_SCALAR_TYPES)}"
        )


def _runtime_header(paths: DeepJitPaths, include_dirs: list[str]) -> str:
    """Render the framework-owned DeepJIT runtime header."""
    includes = ", ".join(f'"{item}"' for item in include_dirs)
    return f"""// Generated by AscendKernelBench for the deepjit build backend.
#pragma once

#include <cstdint>
#include <cstdio>

// The canonical CANN ACL headers must be expanded before torch_npu includes
// its trimmed copies, whose identical include guards would hide the symbols
// DeepJIT loads. This include order is deliberate; see docs/guide/deepjit.md.
#include <acl/acl.h>
#include <deep_jit/backend/ascend/backend.hpp>

namespace akb_deepjit {{

inline deep_jit::Runtime<deep_jit::Ascend>& runtime() {{
    static deep_jit::LazyInit<deep_jit::Runtime<deep_jit::Ascend>> jit(nullptr);
    static bool initialized = false;
    if (not initialized) {{
        jit = deep_jit::create_lazy_jit<deep_jit::Ascend>(deep_jit::Config(
            "{paths.library_root}", "{DEEPJIT_ENV_PREFIX}", "{paths.signature}",
            {{"{paths.library_root}"{", " + includes if includes else ""}}},
            {{"{DEEPJIT_KERNEL_PREFIX}"}}));
        initialized = true;
    }}
    return *jit.get();
}}

inline void write_error(char* buffer, int size, const char* message) {{
    if (buffer == nullptr or size <= 0) {{
        return;
    }}
    const char* text = message == nullptr ? "" : message;
    std::snprintf(buffer, static_cast<size_t>(size), "%s", text);
}}

}}  // namespace akb_deepjit
"""


def _launcher_source(units: list[DeviceUnit], kernel_paths: list[Path]) -> str:
    """Render the launcher: per-kernel prepare helpers, one ABI entry, stubs.

    Every kernel gets its own handle, its own prepare helper that names the
    kernel in the error message, and its own launch stub. The framework
    prepare entry prepares them in order and stops at the first failure, so
    compiled is only ever true when every kernel is ready.
    """
    parts = [
        "// Generated by AscendKernelBench for the deepjit build backend.",
        f'#include "{DEEPJIT_RUNTIME_HEADER_NAME}"',
        "",
        "#include <exception>",
        "#include <stdexcept>",
        "#include <string>",
        "",
        "namespace {",
    ]
    for index, (unit, path) in enumerate(zip(units, kernel_paths, strict=True)):
        include = f"#include <{DEEPJIT_KERNEL_PREFIX}{path.name}>\\n"
        parts += [
            f'constexpr const char* kKernelName{index} = "{unit.signature.name}";',
            f'constexpr const char* kKernelSource{index} = "{include}";',
            "",
        ]
    parts.append("// Filled by prepare, read by the launch stubs. A launch never")
    parts.append("// compiles and never touches the include parser.")
    for index in range(len(units)):
        parts += [
            f"std::shared_ptr<deep_jit::ascend::Kernel> kHandle{index};",
        ]
    parts.append("}  // namespace")
    parts.append("")
    for index in range(len(units)):
        parts += [
            "// Called across a C ABI: a C++ exception becomes a status code plus a",
            "// message that names the kernel it came from.",
            f"int prepare_kernel{index}(char* error_message, int error_size) {{",
            "    try {",
            f"        kHandle{index} = akb_deepjit::runtime().compile(",
            f"            kKernelName{index}, kKernelSource{index});",
            "        return 0;",
            "    } catch (const std::exception& error) {",
            "        const std::string detail =",
            '            std::string("kernel ") + kKernelName' + str(index),
            '            + ": " + error.what();',
            "        akb_deepjit::write_error(",
            "            error_message, error_size, detail.c_str());",
            "        return 1;",
            "    } catch (...) {",
            "        const std::string detail =",
            '            std::string("kernel ") + kKernelName' + str(index),
            '            + ": unknown C++ exception";',
            "        akb_deepjit::write_error(",
            "            error_message, error_size, detail.c_str());",
            "        return 2;",
            "    }",
            "}",
            "",
        ]
    signature = "char* error_message, int error_size"
    parts.append(f'extern "C" int {DEEPJIT_PREPARE_ENTRY}({signature}) {{')
    for index in range(len(units)):
        parts += [
            f"    const int status{index} = "
            f"prepare_kernel{index}(error_message, error_size);",
            f"    if (status{index} != 0) {{",
            f"        return status{index};",
            "    }",
        ]
    parts += ["    return 0;", "}", ""]
    for unit in units:
        launch_params = ", ".join(
            part
            for part in (
                "uint32_t numBlocks",
                "void* stream",
                str(unit.signature.host_params),
            )
            if part
        )
        index = units.index(unit)
        parts += [
            f'extern "C" void {unit.signature.name}_launch({launch_params})',
            "{",
            "    if (kHandle" + str(index) + " == nullptr) {",
            "        throw std::runtime_error(",
            f'            std::string("kernel ") + kKernelName{index} +',
            '            " was not prepared; every kernel is prepared '
            'before it runs");',
            "    }",
            "    akb_deepjit::runtime().launch(",
            f"        kHandle{index},",
            "        deep_jit::ascend::LaunchOptions{",
            "            .stream = static_cast<aclrtStream>(stream),",
            "            .num_blocks = static_cast<int>(numBlocks)},",
            f"        {unit.signature.arg_names});",
            "}",
            "",
        ]
    return "\n".join(parts) + "\n"


def _decls_source(signature: object) -> str:
    """Render the declarations force-included into the candidate host TU."""
    name = str(signature.name)
    launch_params = ", ".join(
        part
        for part in ("uint32_t numBlocks", "void* stream", str(signature.host_params))
        if part
    )
    return f"""// Generated by AscendKernelBench for the deepjit build backend.
#pragma once
#include <cstdint>

extern "C" void {name}_launch({launch_params});
"""


def _host_source(host_source: str) -> str:
    """Render the candidate host TU with the framework include order."""
    prologue = (
        "// Generated by AscendKernelBench for the deepjit build backend.\n"
        "// The canonical ACL header is expanded first on purpose; see\n"
        "// docs/guide/deepjit.md.\n"
        "#include <acl/acl.h>\n"
        "#include <cstdint>\n\n"
    )
    return prologue + host_source


def deepjit_worktree_state(root: Path) -> str:
    """Return clean, dirty, or unknown for a DeepJIT checkout.

    A dirty checkout still reports its HEAD commit, so the state is recorded
    and folded into the dependency signature: artifacts built from modified
    headers must not be reused as if the commit described them.
    """
    if not (Path(root) / ".git").exists():
        return "unknown"
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    return "dirty" if result.stdout.strip() else "clean"


def _bisheng_version() -> str:
    """Return the CANN bisheng version string, or an empty string."""
    from .build import cann_env

    env = cann_env()
    bisheng = shutil.which("bisheng", path=env.get("PATH", ""))
    toolkit = env.get("ASCEND_TOOLKIT_HOME") or env.get("ASCEND_HOME_PATH")
    if bisheng is None and toolkit:
        candidate = Path(toolkit) / "bin" / "bisheng"
        bisheng = str(candidate) if candidate.is_file() else None
    if bisheng is None:
        return ""
    try:
        result = subprocess.run(
            [bisheng, "--version"], capture_output=True, text=True, timeout=60
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.splitlines()[0].strip() if result.stdout else ""


def _signature(commit: str, soc_version: str, worktree: str = "clean") -> str:
    """Return the dependency signature recorded for one DeepJIT build.

    The include parser tracks the kernel source and its includes; the commit,
    the checkout state, the compiler, and the SoC are not tracked, so they are
    folded in here. A dirty checkout gets its own signature so it can never
    reuse artifacts that a clean checkout of the same commit produced.
    """
    payload = "|".join((commit, worktree, _bisheng_version(), soc_version))
    return f"deepjit-{commit[:12]}-{hashlib.sha256(payload.encode()).hexdigest()[:12]}"


def _prepare_layout(
    work_dir: Path,
    device_source: str,
    settings: DeepJitConfig,
    soc_version: str,
) -> DeepJitPaths:
    """Create the generated directory layout and return its paths."""
    from .build import split_asc_source

    work_dir = Path(work_dir).resolve()
    sections = split_asc_source(device_source)
    if sections is None:
        raise DeepJitBuildError(
            "deepjit build backend requires the host-section marker in custom_op.asc"
        )
    device_src, host_src = sections
    units = split_device_units(device_src)
    for unit in units:
        validate_kernel_parameters(unit.signature)

    root = Path(_require_safe_path(settings.root, "deepjit root"))
    if not (root / "include" / "deep_jit").is_dir():
        raise DeepJitBuildError(
            f"deepjit root does not look like a DeepJIT checkout: {root}"
        )
    found_commit = deepjit_commit(root)
    if not found_commit:
        raise DeepJitVersionError(
            f"cannot read the DeepJIT version at {root}: the checkout has no "
            "readable Git metadata, so neither a pinned commit nor a cache "
            "signature can be trusted"
        )
    if settings.commit and settings.commit != found_commit:
        raise DeepJitVersionError(
            f"DeepJIT version mismatch: the checkout at {root} is "
            f"{found_commit}, but the configured commit is {settings.commit}"
        )
    worktree = deepjit_worktree_state(root)
    if worktree == "dirty":
        logger.warning(
            "DeepJIT checkout {} has uncommitted changes; artifacts get "
            "their own signature and the state is recorded",
            root,
        )
    effective_commit = found_commit
    if not settings.commit and found_commit != DEEPJIT_VERIFIED_COMMIT:
        logger.warning(
            "DeepJIT checkout {} is {}; this path was verified against {}",
            root,
            found_commit,
            DEEPJIT_VERIFIED_COMMIT,
        )

    generated = work_dir / DEEPJIT_DIR_NAME
    _require_safe_path(str(generated), "deepjit generated directory")
    kernel_dir = generated / DEEPJIT_KERNEL_DIR_NAME
    kernel_dir.mkdir(parents=True, exist_ok=True)
    kernel_paths = []
    for index, unit in enumerate(units):
        path = kernel_dir / f"kernel_{index}.hpp"
        _write_if_changed(path, convert_device_source(unit.source))
        kernel_paths.append(path)
    for stale in kernel_dir.glob("kernel_*.hpp"):
        if stale not in kernel_paths:
            stale.unlink()

    cache_dir = Path(
        _require_safe_path(
            settings.cache_dir or str(generated / DEEPJIT_CACHE_DIR_NAME),
            "deepjit cache dir",
        )
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    facts = DeepJitPaths(
        root=root,
        library_root=generated,
        commit=effective_commit,
        cache_dir=cache_dir,
        build_dir=generated / "build",
        kernel_source=kernel_paths[0],
        signature=_signature(effective_commit, soc_version, worktree),
    )
    _write_if_changed(
        generated / DEEPJIT_RUNTIME_HEADER_NAME,
        _runtime_header(
            facts,
            [
                _require_safe_path(item, "deepjit include dir")
                for item in settings.include_dirs
            ],
        ),
    )
    _write_if_changed(
        generated / DEEPJIT_LAUNCHER_NAME, _launcher_source(units, kernel_paths)
    )
    _write_if_changed(
        generated / DEEPJIT_DECLS_NAME,
        "\n".join(_decls_source(unit.signature) for unit in units),
    )
    _write_if_changed(generated / DEEPJIT_HOST_NAME, _host_source(host_src))

    _write_if_changed(
        generated / DEEPJIT_FACTS_NAME,
        json.dumps(
            build_facts(facts, [str(unit.signature.name) for unit in units], worktree),
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    return facts


def build_facts(
    paths: DeepJitPaths,
    kernel_names: list[str] | None = None,
    worktree: str = "unknown",
) -> dict[str, object]:
    """Return the recorded facts of one DeepJIT build for result metadata."""
    return {
        "deepjit_worktree": worktree,
        "build_backend": "deepjit",
        "deepjit_kernels": list(kernel_names or []),
        "deepjit_root": str(paths.root),
        "deepjit_library_root": str(paths.library_root),
        "deepjit_commit": paths.commit,
        "deepjit_verified_commit": DEEPJIT_VERIFIED_COMMIT,
        "deepjit_signature": paths.signature,
        "deepjit_cache_dir": str(paths.cache_dir),
        "deepjit_kernel_source": str(paths.kernel_source),
        "deepjit_kernel_sha256": hashlib.sha256(
            paths.kernel_source.read_bytes()
        ).hexdigest(),
    }


def read_build_facts(work_dir: Path) -> dict[str, object]:
    """Return the DeepJIT facts recorded in work_dir, or an empty mapping."""
    manifest = Path(work_dir) / DEEPJIT_DIR_NAME / DEEPJIT_FACTS_NAME
    try:
        loaded = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


@lru_cache(maxsize=1)
def _cann_paths() -> tuple[str, str]:
    """Return (CANN include dirs, CANN library dir) from the CANN env."""
    from .build import cann_env

    env = cann_env()
    home = env.get("ASCEND_TOOLKIT_HOME") or env.get("ASCEND_HOME_PATH") or ""
    if not home:
        raise DeepJitBuildError(
            "ASCEND_TOOLKIT_HOME or ASCEND_HOME_PATH must be set for deepjit"
        )
    return home, str(Path(home) / "lib64")


def _extension_include_dirs(paths: DeepJitPaths, settings: DeepJitConfig) -> list[str]:
    """Return the host build include path list, CANN before torch_npu."""
    from torch.utils.cpp_extension import include_paths

    cann_home, _ = _cann_paths()
    torch_npu_dir = Path(_torch_npu_dir())
    ordered = [
        str(paths.root / "include"),
        str(Path(cann_home) / "include"),
        str(Path(cann_home) / "aarch64-linux" / "include"),
        str(torch_npu_dir / "include"),
        str(torch_npu_dir / "include" / "third_party" / "acl" / "inc"),
        sysconfig.get_paths()["include"],
        *[str(item) for item in include_paths()],
        *[
            _require_safe_path(item, "deepjit include dir")
            for item in settings.include_dirs
        ],
    ]
    return ordered


def _torch_npu_dir() -> str:
    """Return the installed torch_npu package directory."""
    import torch_npu

    return str(Path(torch_npu.__file__).resolve().parent)


def _extension_link_flags(settings: DeepJitConfig) -> list[str]:
    """Return the link flags that bind torch_npu, ACL, and the toolchain."""
    from torch.utils.cpp_extension import library_paths

    _, cann_lib = _cann_paths()
    npu_dir = Path(_torch_npu_dir())
    flags = [f"-L{item}" for item in library_paths()]
    flags += [
        f"-L{cann_lib}",
        f"-L{npu_dir / 'lib'}",
        "-lascendcl",
        "-ltorch_npu",
        "-ldl",
        f"-Wl,-rpath,{cann_lib}",
        f"-Wl,-rpath,{npu_dir / 'lib'}",
    ]
    for item in settings.library_dirs:
        safe = _require_safe_path(item, "deepjit library dir")
        flags += [f"-L{safe}", f"-Wl,-rpath,{safe}"]
    flags += [
        _require_safe_argument(flag, "deepjit extra ldflag")
        for flag in settings.extra_ldflags
    ]
    return flags


def build_deepjit_op(
    source: str,
    work_dir: Path,
    *,
    settings: DeepJitConfig,
    soc_version: str = "",
) -> tuple[Path, DeepJitPaths]:
    """Build the sample's host library with the embedded DeepJIT runtime.

    Returns:
        The shared library path and the resolved DeepJIT layout.

    Raises:
        DeepJitBuildError: If the source is unsupported or the build fails.
    """
    settings = resolve_deepjit_settings(settings)
    paths = _prepare_layout(work_dir, source, settings, soc_version)
    # torch's extension loader opens build_directory/lock before compiling.
    paths.build_dir.mkdir(parents=True, exist_ok=True)

    from torch.utils.cpp_extension import load

    if settings.cxx:
        os.environ["CXX"] = _require_safe_path(settings.cxx, "deepjit cxx")
    for key, value in _require_safe_environment(settings.env).items():
        os.environ[key] = value
    # The DeepJIT runtime reads its cache directory when it is constructed.
    os.environ["DJ_JIT_CACHE_DIR"] = str(paths.cache_dir)

    generated = paths.build_dir.parent
    cflags = [
        "-std=c++20",
        "-O1",
        "-fPIC",
        "-DDJ_DISABLE_GIL=1",
        "-Wno-attributes",
        "-Wno-deprecated-declarations",
        "-Wno-unused-function",
        "-Wno-missing-field-initializers",
        "-include",
        str(generated / DEEPJIT_DECLS_NAME),
        *[
            _require_safe_argument(flag, "deepjit extra cflag")
            for flag in settings.extra_cflags
        ],
    ]
    try:
        built = load(
            name=DEEPJIT_EXTENSION_NAME,
            sources=[
                str(generated / DEEPJIT_LAUNCHER_NAME),
                str(generated / DEEPJIT_HOST_NAME),
            ],
            extra_cflags=cflags,
            extra_include_paths=_extension_include_dirs(paths, settings),
            extra_ldflags=_extension_link_flags(settings),
            build_directory=str(paths.build_dir),
            with_cuda=False,
            is_python_module=False,
            verbose=False,
        )
    except Exception as exc:  # torch raises a variety of build errors
        raise DeepJitBuildError(f"deepjit host build failed: {exc}") from None
    library = Path(str(built))
    if not library.is_file():
        raise DeepJitBuildError(f"deepjit host build produced no library: {library}")
    return library, paths


def prepare_deepjit_module(library: Path) -> None:
    """Compile the device kernel through the framework prepare entry.

    The entry is called on the already-loaded library, so the device compile
    is a distinct step that runs before any candidate code, and it never
    launches a kernel.

    Raises:
        DeepJitBuildError: If the entry is missing or the device compile fails.
    """
    import ctypes

    try:
        handle = ctypes.CDLL(str(library))
        entry = handle[DEEPJIT_PREPARE_ENTRY]
    except (OSError, KeyError) as exc:
        raise DeepJitBuildError(
            f"deepjit prepare entry {DEEPJIT_PREPARE_ENTRY} is unavailable: {exc}"
        ) from None
    entry.restype = ctypes.c_int
    entry.argtypes = [ctypes.c_char_p, ctypes.c_int]
    message = ctypes.create_string_buffer(DEEPJIT_ERROR_BUFFER_SIZE)
    status = int(entry(message, DEEPJIT_ERROR_BUFFER_SIZE))
    if status != 0:
        detail = message.value.decode("utf-8", errors="replace")
        raise DeepJitBuildError(
            f"deepjit device compile failed (status {status}): {detail}"
        )
