"""Build and load a process-local libcustom_op.so, installing nothing.

Marked sources build as split device/launcher/host units with a shared
PCH; legacy unmarked sources build as one ASC translation unit.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import sysconfig
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from loguru import logger

from ._paths import BUILD_TEMPLATE_DIR
from .backend import DEFAULT_BUILD_BACKEND, BuildBackend
from .checks.text import (
    HOST_SECTION_MARKER,
    HOST_SECTION_MARKER_RE,
    strip_cpp_comments,
)
from .config import DeepJitConfig

SHARED_LIBRARY_NAME = "libcustom_op.so"
DEFAULT_CANN_SET_ENV = "/usr/local/Ascend/cann-9.1.0/set_env.sh"

# Split-mode file names inside the sample directory.
KERNEL_ASC_NAME = "custom_op_kernel.asc"
LAUNCH_ASC_NAME = "custom_op_launch.asc"
HOST_CPP_NAME = "custom_op_host.cpp"
LAUNCH_DECLS_NAME = "custom_op_launch_decls.h"

# The host TU and its PCH include exactly these headers. Keep in sync with
# the prompt contract; the PCH preparses them once per environment.
HOST_TU_INCLUDES = (
    "#include <torch/library.h>",
    "#include <ATen/ATen.h>",
    '#include "torch_npu/csrc/core/npu/NPUStream.h"',
)

# Host-TU optimization level. The wrapper is thin glue, so -O1 costs no
# measurable runtime; the PCH emit must use the same level.
HOST_TU_OPT_LEVEL = "-O1"

# Flags shared by PCH emission and the CMake host-TU compile. Clang rejects
# a PCH when these differ; _pch_key folds them into the cache key.
HOST_TU_FLAGS = (
    HOST_TU_OPT_LEVEL,
    "-DNDEBUG",
    "-fPIC",
    "-fvisibility=hidden",
    "-fvisibility-inlines-hidden",
)

PCH_INPUT_NAME = "akb_host_pch_input.h"
PCH_FILE_NAME = "libcustom_op_host.pch"


class BuildError(RuntimeError):
    """Raised when CMake configure or build fails, with compiler output."""


class LoadError(RuntimeError):
    """Raised when the sample-local shared library cannot be loaded."""


@lru_cache(maxsize=1)
def cann_env() -> dict[str, str]:
    """Return the environment produced by sourcing CANN set_env.sh.

    Cached per process. CANN_SET_ENV overrides the script path; a missing
    script falls back to the current environment.
    """
    set_env = os.environ.get("CANN_SET_ENV", DEFAULT_CANN_SET_ENV)
    if not Path(set_env).is_file():
        return dict(os.environ)
    result = subprocess.run(
        ["bash", "-c", f'source "{set_env}" >/dev/null 2>&1 && env'],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise BuildError(f"Failed to source CANN env {set_env}: {result.stderr}")
    env = dict(os.environ)
    for line in result.stdout.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            env[key] = value
    return env


def split_asc_source(source: str) -> tuple[str, str] | None:
    """Split a new-style source into (device, host) sections.

    Returns None for legacy sources without the marker. The marker line
    itself is dropped from both sections.

    Raises:
        BuildError: If more than one marker line is present.
    """
    lines = source.splitlines(keepends=True)
    markers = [
        i
        for i, line in enumerate(lines)
        if HOST_SECTION_MARKER_RE.match(line.rstrip("\n"))
    ]
    if not markers:
        return None
    if len(markers) > 1:
        raise BuildError(
            f"custom_op.asc carries {len(markers)} lines containing "
            f"{HOST_SECTION_MARKER!r}; exactly one is required"
        )
    idx = markers[0]
    return "".join(lines[:idx]), "".join(lines[idx + 1 :])


@dataclass(frozen=True)
class KernelSignature:
    """One __global__ __vector__ kernel parsed from the device section."""

    name: str
    decl_params: str
    host_params: str
    arg_names: str


_KERNEL_DEF_RE = re.compile(r"__global__\s+__vector__\s+void\s+(\w+)\s*\(([^()]*)\)")
_PARAM_RE = re.compile(r"^(.+?[\s*])([A-Za-z_]\w*)$")


def _parse_kernel_params(kernel_name: str, raw_params: str) -> KernelSignature:
    """Split one kernel parameter list into declaration fragments."""
    decl_parts: list[str] = []
    host_parts: list[str] = []
    names: list[str] = []
    for param in raw_params.split(","):
        param = param.strip()
        if not param:
            continue
        if "[" in param or "]" in param:
            raise BuildError(
                f"kernel {kernel_name}: array parameters are not supported "
                f"in the launch stub ({param!r})"
            )
        host_param = " ".join(param.replace("__gm__", " ").split())
        match = _PARAM_RE.match(host_param)
        if match is None:
            raise BuildError(f"kernel {kernel_name}: cannot parse parameter {param!r}")
        decl_parts.append(param)
        host_parts.append(f"{match.group(1).strip()} {match.group(2)}")
        names.append(match.group(2))
    return KernelSignature(
        name=kernel_name,
        decl_params=", ".join(decl_parts),
        host_params=", ".join(host_parts),
        arg_names=", ".join(names),
    )


def extract_kernel_signatures(device_source: str) -> list[KernelSignature]:
    """Parse all kernel signatures from the device section.

    Raises:
        BuildError: If no kernel is present or a parameter list uses
            constructs the stub generator does not support.
    """
    code = strip_cpp_comments(device_source)
    sigs: dict[str, KernelSignature] = {}
    for match in _KERNEL_DEF_RE.finditer(code):
        sig = _parse_kernel_params(match.group(1), match.group(2))
        previous = sigs.get(sig.name)
        if previous is not None and previous.decl_params != sig.decl_params:
            raise BuildError(
                f"kernel {sig.name} appears with conflicting parameter lists"
            )
        sigs.setdefault(sig.name, sig)
    if not sigs:
        raise BuildError("no __global__ __vector__ kernel found in the device section")
    return list(sigs.values())


def generate_launcher_source(sigs: list[KernelSignature]) -> str:
    """Render the launcher TU: kernel declarations plus extern C stubs."""
    parts = [
        "// Auto-generated by the AscendKernelBench build; edit custom_op.asc.",
        "#include <cstdint>",
        "",
    ]
    for sig in sigs:
        parts.append(f"__global__ __vector__ void {sig.name}({sig.decl_params});")
    parts.append("")
    for sig in sigs:
        params = ", ".join(
            part
            for part in ("uint32_t numBlocks", "void* stream", sig.host_params)
            if part
        )
        parts.append(
            f'extern "C" void {sig.name}_launch({params})\n'
            "{\n"
            f"    {sig.name}<<<numBlocks, 0, stream>>>({sig.arg_names});\n"
            "}"
        )
    return "\n".join(parts) + "\n"


def generate_launcher_decls(sigs: list[KernelSignature]) -> str:
    """Render the force-included header declaring the launch stubs."""
    lines = [
        "// Auto-generated by the AscendKernelBench build; edit custom_op.asc.",
        "#pragma once",
        "#include <cstdint>",
        "",
    ]
    for sig in sigs:
        params = ", ".join(
            part
            for part in ("uint32_t numBlocks", "void* stream", sig.host_params)
            if part
        )
        lines.append(f'extern "C" void {sig.name}_launch({params});')
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class _TorchFacts:
    """Compiler and library facts that determine PCH validity."""

    bisheng: str
    bisheng_version: str
    torch_version: str
    torch_npu_version: str
    python_version: str
    include_dirs: tuple[str, ...]
    defines: tuple[str, ...]
    gcc_toolchain: str


def _find_bisheng(env: dict[str, str]) -> str | None:
    """Resolve the bisheng driver from the CANN environment."""
    found = shutil.which("bisheng", path=env.get("PATH", ""))
    if found:
        return str(Path(found).resolve())
    toolkit = env.get("ASCEND_TOOLKIT_HOME")
    if toolkit:
        candidate = Path(toolkit) / "bin" / "bisheng"
        if candidate.is_file():
            return str(candidate.resolve())
    return None


def _gcc_toolchain_root() -> str | None:
    """Derive the GCC toolchain root from libgcc.a, like the template."""
    try:
        result = subprocess.run(
            ["gcc", "-print-file-name=libgcc.a"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    path = result.stdout.strip()
    if result.returncode != 0 or not Path(path).is_file():
        return None
    text = str(Path(path).resolve())
    marker = "/lib/gcc/"
    if marker not in text:
        return None
    return text.split(marker, 1)[0]


def _torch_interface_defines(torch_dir: Path) -> tuple[str, ...] | None:
    """Read INTERFACE_COMPILE_DEFINITIONS of the imported torch target."""
    targets = torch_dir / "share" / "cmake" / "Caffe2" / "Caffe2Targets.cmake"
    if not targets.is_file():
        return None
    text = targets.read_text(encoding="utf-8", errors="replace")
    match = re.search(r'INTERFACE_COMPILE_DEFINITIONS "([^"]+)"', text)
    if match is None:
        return None
    return tuple(sorted(item for item in match.group(1).split(";") if item))


@lru_cache(maxsize=1)
def _torch_facts() -> _TorchFacts | None:
    """Collect PCH-relevant facts from the worker's own torch install."""
    try:
        import torch
        import torch_npu
    except ImportError:
        return None
    env = cann_env()
    bisheng = _find_bisheng(env)
    if bisheng is None:
        return None
    try:
        version_out = subprocess.run(
            [bisheng, "--version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    torch_dir = Path(torch.__file__).resolve().parent
    npu_dir = Path(torch_npu.__file__).resolve().parent
    defines = _torch_interface_defines(torch_dir)
    gcc_root = _gcc_toolchain_root()
    if defines is None or gcc_root is None:
        return None
    return _TorchFacts(
        bisheng=bisheng,
        bisheng_version=(
            version_out.stdout.splitlines()[0] if version_out.stdout else ""
        ),
        torch_version=str(torch.__version__),
        torch_npu_version=str(getattr(torch_npu, "__version__", "")),
        python_version=platform_python_version(),
        include_dirs=(
            str(torch_dir / "include"),
            str(torch_dir / "include" / "torch" / "csrc" / "api" / "include"),
            str(npu_dir / "include"),
            sysconfig.get_paths()["include"],
        ),
        defines=defines,
        gcc_toolchain=gcc_root,
    )


def platform_python_version() -> str:
    """Return the running Python version string."""
    return ".".join(str(part) for part in sys.version_info[:3])


def _pch_cache_root() -> Path:
    override = os.environ.get("ASCEND_KERNEL_BENCH_CACHE_DIR")
    if override:
        return Path(override).expanduser() / "pch"
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "ascend-kernel-bench" / "pch"


def _pch_key(facts: _TorchFacts) -> str:
    payload = {
        "bisheng_version": facts.bisheng_version,
        "torch": facts.torch_version,
        "torch_npu": facts.torch_npu_version,
        "python": facts.python_version,
        "include_dirs": list(facts.include_dirs),
        "defines": list(facts.defines),
        "gcc_toolchain": facts.gcc_toolchain,
        "flags": list(HOST_TU_FLAGS),
        "pch_input": list(HOST_TU_INCLUDES),
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def ensure_host_pch() -> Path | None:
    """Return the shared host-TU precompiled header, emitting it on a miss.

    Emission costs about 20 s once per environment; consumers then skip
    re-parsing the torch headers. Returns None when PCH use is disabled or
    unavailable, in which case the host TU compiles without one.
    """
    disabled = os.environ.get("ASCEND_KERNEL_BENCH_DISABLE_PCH", "").strip().lower()
    if disabled in {"1", "true", "yes", "on"}:
        return None
    facts = _torch_facts()
    if facts is None:
        return None
    cache_dir = _pch_cache_root() / _pch_key(facts)
    pch = cache_dir / PCH_FILE_NAME
    if pch.is_file():
        return pch
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with (cache_dir / "emit.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                if pch.is_file():
                    return pch
                _emit_host_pch(facts, cache_dir)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
    except OSError:
        return None
    return pch if pch.is_file() else None


def _emit_host_pch(facts: _TorchFacts, cache_dir: Path) -> None:
    """Emit the host PCH via a temporary file and an atomic rename."""
    input_header = cache_dir / PCH_INPUT_NAME
    input_header.write_text("\n".join(HOST_TU_INCLUDES) + "\n", encoding="utf-8")
    tmp_pch = cache_dir / f"{PCH_FILE_NAME}.tmp.{os.getpid()}"
    cmd = [
        facts.bisheng,
        *(f"-D{name}" for name in facts.defines),
        *(f"-I{path}" for path in facts.include_dirs),
        "-std=c++17",
        *HOST_TU_FLAGS,
        f"--gcc-toolchain={facts.gcc_toolchain}",
        "-x",
        "c++-header",
        str(input_header),
        "-o",
        str(tmp_pch),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            env=cann_env(),
        )
    except (OSError, subprocess.TimeoutExpired):
        result = None
    if result is not None and result.returncode == 0 and tmp_pch.is_file():
        os.replace(tmp_pch, cache_dir / PCH_FILE_NAME)
        logger.info("AscendKernelBench host PCH cached at {}", cache_dir)
        return
    tmp_pch.unlink(missing_ok=True)
    tail = ""
    if result is not None:
        tail = "\n".join(result.stderr.strip().splitlines()[-5:])
    logger.warning("host PCH emission failed, building without PCH: {}", tail)


def _torch_discovery() -> tuple[str, str] | None:
    """Return (cmake prefix path, torch_npu path) from the live torch."""
    try:
        import torch
        import torch_npu
    except ImportError:
        return None
    prefix = str(torch.utils.cmake_prefix_path)
    npu_path = str(Path(torch_npu.__file__).resolve().parent)
    return prefix, npu_path


def build_custom_op(
    asc_source: str,
    work_dir: Path,
    *,
    cmake_arch: str,
    timeout_s: int = 600,
    build_backend: BuildBackend = DEFAULT_BUILD_BACKEND,
    deepjit: DeepJitConfig | None = None,
    soc_version: str = "",
) -> Path:
    """Write custom_op.asc into work_dir and build libcustom_op.so.

    The sample directory doubles as the build cache, so an unchanged sample
    is an incremental no-op build. cmake_arch is passed as
    CMAKE_ASC_ARCHITECTURES, and timeout_s covers configure and build
    separately. New-style sources (with the host-section marker) build as
    split translation units; legacy sources build as one ASC unit. The
    deepjit build backend compiles the same marked source through DeepJIT
    instead of the CMake template and returns its own shared library.

    Returns:
        Path to the process-local shared library.

    Raises:
        BuildError: If the source cannot be split, CMake fails, the DeepJIT
            path rejects the source, or no .so is produced.
    """
    if build_backend is BuildBackend.DEEPJIT:
        return _build_via_deepjit(
            asc_source,
            Path(work_dir),
            settings=deepjit or DeepJitConfig(),
            timeout_s=timeout_s,
            soc_version=soc_version,
        )
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    # Write only on content change, so cmake/make can skip the Ascend C
    # recompile when a sample is re-evaluated unchanged.
    _write_if_changed(work_dir / "custom_op.asc", asc_source)
    _write_if_changed(
        work_dir / "CMakeLists.txt",
        (BUILD_TEMPLATE_DIR / "CMakeLists.txt").read_text(encoding="utf-8"),
    )

    sections = split_asc_source(asc_source)
    if sections is None:
        _build_via_cmake(work_dir, cmake_arch, timeout_s, split=False, pch=None)
        return find_built_library(work_dir)

    device_src, host_src = sections
    sigs = extract_kernel_signatures(device_src)
    _write_if_changed(work_dir / KERNEL_ASC_NAME, device_src)
    _write_if_changed(work_dir / LAUNCH_ASC_NAME, generate_launcher_source(sigs))
    _write_if_changed(work_dir / HOST_CPP_NAME, host_src)
    _write_if_changed(work_dir / LAUNCH_DECLS_NAME, generate_launcher_decls(sigs))

    pch = ensure_host_pch()
    try:
        _build_via_cmake(work_dir, cmake_arch, timeout_s, split=True, pch=pch)
    except BuildError as first_error:
        if pch is None:
            raise
        logger.warning("split build with PCH failed; retrying without PCH")
        try:
            _build_via_cmake(work_dir, cmake_arch, timeout_s, split=True, pch=None)
        except BuildError:
            # A genuine source error, not a PCH problem; keep the cache.
            raise first_error from None
        # The PCH was the problem; drop it so the next build re-emits.
        with contextlib.suppress(OSError):
            pch.unlink()
    return find_built_library(work_dir)


def _build_via_deepjit(
    asc_source: str,
    work_dir: Path,
    *,
    settings: DeepJitConfig,
    timeout_s: int,
    soc_version: str,
) -> Path:
    """Build the sample through the DeepJIT path, raising BuildError."""
    from .deepjit_build import DeepJitBuildError, build_deepjit_op

    work_dir.mkdir(parents=True, exist_ok=True)
    _write_if_changed(work_dir / "custom_op.asc", asc_source)
    try:
        library, _ = build_deepjit_op(
            asc_source,
            work_dir,
            settings=settings,
            soc_version=soc_version,
            timeout_s=timeout_s,
        )
    except DeepJitBuildError as exc:
        raise BuildError(str(exc)) from None
    return library


def _build_via_cmake(
    work_dir: Path,
    cmake_arch: str,
    timeout_s: int,
    *,
    split: bool,
    pch: Path | None,
) -> None:
    """Configure when the cache is stale, then build in work_dir/build."""
    build_dir = work_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    env = cann_env()
    env.setdefault("ASCEND_SLOG_PRINT_TO_STDOUT", "0")
    enable_ccache = os.environ.get(
        "ASCEND_KERNEL_BENCH_ENABLE_CCACHE", ""
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    expected = _configure_defines(
        cmake_arch, split=split, pch=pch, enable_ccache=enable_ccache
    )
    cached = _read_cmake_cache(build_dir)
    if cached is not None and "CMAKE_CXX_COMPILER" in expected:
        have = cached.get("CMAKE_CXX_COMPILER")
        if have is not None and have != expected["CMAKE_CXX_COMPILER"]:
            # CMake refuses a compiler swap on a configured tree.
            shutil.rmtree(build_dir)
            build_dir.mkdir(parents=True, exist_ok=True)
            cached = None

    ##################### BUILD PATH #####################
    if not _cmake_cache_matches(cached, expected):
        _run_cmake(
            "configure",
            [
                "cmake",
                "-S",
                str(work_dir),
                "-B",
                str(build_dir),
                *(f"-D{key}={value}" for key, value in expected.items()),
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={work_dir}",
            ],
            cwd=work_dir,
            env=env,
            timeout_s=timeout_s,
        )
    _run_cmake(
        "build",
        ["cmake", "--build", str(build_dir), "-j"],
        cwd=work_dir,
        env=env,
        timeout_s=timeout_s,
    )
    ##################### BUILD PATH #####################


def _configure_defines(
    cmake_arch: str,
    *,
    split: bool,
    pch: Path | None,
    enable_ccache: bool,
) -> dict[str, str]:
    """Assemble the -D defines that identify one configure request."""
    defines = {
        "CMAKE_ASC_ARCHITECTURES": cmake_arch,
        "Python3_EXECUTABLE": sys.executable,
        "ENABLE_CCACHE": "ON" if enable_ccache else "OFF",
        "AKB_SPLIT_BUILD": "ON" if split else "OFF",
    }
    discovery = _torch_discovery()
    if discovery is not None:
        defines["TORCH_CMAKE_PREFIX_PATH"] = discovery[0]
        defines["TORCH_NPU_PATH"] = discovery[1]
    if split:
        defines["AKB_PCH_PATH"] = str(pch) if pch is not None else ""
        facts = _torch_facts()
        if facts is not None:
            defines["CMAKE_CXX_COMPILER"] = facts.bisheng
    return defines


def _read_cmake_cache(build_dir: Path) -> dict[str, str] | None:
    """Return the cached variable mapping, or None when unconfigured."""
    cache = build_dir / "CMakeCache.txt"
    if not cache.is_file():
        return None
    values: dict[str, str] = {}
    text = cache.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            values[key.split(":", 1)[0]] = value
    return values


def _cmake_cache_matches(
    cached: dict[str, str] | None, expected: dict[str, str]
) -> bool:
    """Return True when the configure cache covers every expected value."""
    if cached is None:
        return False
    return all(cached.get(key) == value for key, value in expected.items())


def _write_if_changed(path: Path, content: str) -> None:
    """Write content to path only when it differs, preserving the mtime."""
    if path.is_file() and path.read_text(encoding="utf-8") == content:
        return
    path.write_text(content, encoding="utf-8")


def find_built_library(work_dir: Path) -> Path:
    """Return libcustom_op.so from work_dir or work_dir/build."""
    work_dir = Path(work_dir)
    for candidate in (
        work_dir / SHARED_LIBRARY_NAME,
        work_dir / "build" / SHARED_LIBRARY_NAME,
    ):
        if candidate.is_file():
            return candidate
    raise BuildError(
        f"Built shared library not found in {work_dir} (expected {SHARED_LIBRARY_NAME})"
    )


def load_custom_op(so_path: Path, source: str = "") -> None:
    """Load so_path so torch.ops.custom_op.* becomes callable.

    source, when given, is custom_op.asc text checked for a TORCH_LIBRARY
    marker.

    Raises:
        LoadError: If the file is missing, lacks TORCH_LIBRARY, or fails
            to load.
    """
    so_path = Path(so_path).resolve()
    if not so_path.is_file():
        raise LoadError(f"shared library not found: {so_path}")
    if source and "TORCH_LIBRARY" not in source:
        raise LoadError(
            f"{so_path.name} is missing TORCH_LIBRARY; the evaluator loads "
            "operators with torch.ops.load_library only"
        )
    try:
        import torch
    except ImportError as exc:
        raise LoadError("torch is required to load a TORCH_LIBRARY operator") from exc
    try:
        torch.ops.load_library(str(so_path))
    except Exception as exc:
        raise LoadError(f"torch.ops.load_library({so_path}) failed: {exc!r}") from exc


def _run_cmake(
    stage: str,
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout_s: int,
) -> None:
    """Run one CMake stage, persist the log, and raise on failure."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise BuildError(f"{stage} timed out after {timeout_s}s") from exc
    log = (result.stdout or "") + "\n" + (result.stderr or "")
    log_path = cwd / "build" / f"{stage}.log"
    with contextlib.suppress(OSError):
        log_path.write_text(log, encoding="utf-8")
    if result.returncode != 0:
        tail = "\n".join(log.strip().splitlines()[-60:])
        raise BuildError(f"{stage} failed (see {log_path}):\n{tail}")
