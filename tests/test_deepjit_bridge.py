"""No-device tests for the deepjit build bridge.

Runs with plain asserts and no test framework:

    python tests/test_deepjit_bridge.py

Nothing here imports torch or touches an NPU. It covers the device-source
conversion, the parameter whitelist, the kernel-section split into one
compilation unit per entry, the generated launcher and declaration text, the
recorded build facts, and the evaluator's failure handling for a device
compile that fails.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import eval_device  # noqa: E402
from src.backend import BuildBackend  # noqa: E402
from src.build import KernelSignature  # noqa: E402
from src.deepjit_build import (  # noqa: E402
    DeepJitBuildError,
    _decls_source,
    _launcher_source,
    _require_safe_argument,
    _require_safe_environment,
    _signature,
    build_facts,
    convert_device_source,
    split_device_units,
    validate_kernel_parameters,
)
from src.eval_device import DeviceEvalRequest, SampleEvaluator  # noqa: E402

KERNEL_SECTION = "// ==================== ASCEND_KERNEL_SECTION ===================="

PASSED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    """Record one check result."""
    status = "ok  " if condition else "FAIL"
    print(f"  {status} - {name}{'' if condition else ' :: ' + detail}")
    if not condition:
        raise AssertionError(f"{name}: {detail}")


def signature(name: str, params: str) -> KernelSignature:
    """Return a KernelSignature with host params mirroring the device ones."""
    host = ", ".join(
        " ".join(part.replace("__gm__", " ").split())
        for part in params.split(",")
    )
    args = ", ".join(part.split()[-1].rstrip("*") for part in params.split(","))
    args = ", ".join(part.split()[-1].replace("*", "") for part in params.split(","))
    return KernelSignature(
        name=name, decl_params=params, host_params=host, arg_names=args
    )


def refused(target: str) -> bool:
    """Return True when the converter refuses one quoted include."""
    try:
        convert_device_source(f'#include "{target}"\n')
    except DeepJitBuildError:
        return True
    return False


def test_include_conversion() -> None:
    """The standard include is converted exactly and nothing else is."""
    converted = convert_device_source(
        '#include "kernel_operator.h"\nint x;\n#include <cstdint>\n'
    )
    lines = converted.splitlines()
    check("quoted kernel_operator.h becomes an angle include",
          lines[0] == "#include <kernel_operator.h>", repr(lines[0]))
    check("angle include untouched", lines[2] == "#include <cstdint>")
    check("another quoted include is refused", refused("kernel_operator_vec.h"))
    check("angle-only source unchanged",
          convert_device_source("#include <kernel_operator.h>\n")
          == "#include <kernel_operator.h>\n")


def test_marker_scan_ignores_code_only() -> None:
    """A marker inside a block comment is refused, not silently split on."""
    documented = (
        "#include <kernel_operator.h>\n"
        "/* the convention is:\n"
        f"{KERNEL_SECTION}\n"
        "*/\n"
        f"{KERNEL_SECTION}\n"
        "__global__ __vector__ void one(__gm__ uint8_t* x, uint32_t n) {}\n"
    )
    try:
        split_device_units(documented)
    except DeepJitBuildError as exc:
        check("marker inside a block comment is refused",
              "block comment" in str(exc), str(exc))
    else:
        check("marker inside a block comment is refused", False)
    plain = (
        "#include <kernel_operator.h>\n"
        f"{KERNEL_SECTION}\n"
        "__global__ __vector__ void one(__gm__ uint8_t* x, uint32_t n) {}\n"
    )
    check("a code marker still splits", len(split_device_units(plain)) == 1)


def test_unsafe_arguments_are_refused() -> None:
    """Flags and environment values cannot smuggle shell metacharacters."""
    check("a plain flag passes",
          _require_safe_argument("-DTILE=2048", "flag") == "-DTILE=2048")
    for label, value in (
        ("space", "-DNAME=a b"),
        ("semicolon", "-DX;rm -rf /"),
        ("dollar", "-DX=$HOME"),
        ("backtick", "-DX=`id`"),
    ):
        try:
            _require_safe_argument(value, "flag")
        except DeepJitBuildError:
            check(f"unsafe {label} refused", True)
        else:
            check(f"unsafe {label} refused", False, value)
    try:
        _require_safe_environment({"NOT A NAME": "1"})
    except DeepJitBuildError:
        check("an invalid env name is refused", True)
    else:
        check("an invalid env name is refused", False)
    check("a safe environment passes",
          _require_safe_environment({"CPLUS_INCLUDE_PATH": "/opt/x/include"})
          == {"CPLUS_INCLUDE_PATH": "/opt/x/include"})


def test_signature_tracks_checkout_state() -> None:
    """A dirty checkout gets its own signature and the facts record it."""
    clean = _signature("deadbeef", "Ascend910B4", "clean")
    dirty = _signature("deadbeef", "Ascend910B4", "dirty")
    check("signature differs when dirty", clean != dirty)
    check("signature is stable when clean",
          clean == _signature("deadbeef", "Ascend910B4", "clean"))
    paths = SimpleNamespace(
        root=Path("/tmp/deepjit"),
        library_root=Path("/tmp/lib"),
        commit="deadbeef",
        cache_dir=Path("/tmp/cache"),
        kernel_source=Path(__file__),
        signature=clean,
    )
    facts = build_facts(paths, ["a"], "dirty")
    check("facts record the checkout state", facts["deepjit_worktree"] == "dirty")


def test_parameter_whitelist() -> None:
    """Supported pointer and scalar types pass; others are refused."""
    good = signature(
        "k", "__gm__ uint8_t* x, __gm__ float* y, uint32_t n, int32_t m, float s"
    )
    validate_kernel_parameters(good)
    check("ascend c pointer and scalar parameters accepted", True)
    for label, params in (
        ("unsupported pointer type", "__gm__ half* p"),
        ("unsupported scalar type", "__gm__ float* p, double s"),
        ("array parameter", "__gm__ float* p[4]"),
    ):
        try:
            validate_kernel_parameters(signature("k", params))
        except DeepJitBuildError:
            check(f"{label} refused", True)
        else:
            check(f"{label} refused", False, params)


def test_single_kernel_units() -> None:
    """A source without markers stays exactly one unit."""
    source = "#include <kernel_operator.h>\nvoid helper() {}\n"
    source += "__global__ __vector__ void only(__gm__ uint8_t* x, uint32_t n) {}\n"
    units = split_device_units(source)
    check("one unit without markers", len(units) == 1)
    check("unit keeps the whole device section", units[0].source == source)
    check("entry name parsed", units[0].signature.name == "only")


def test_multi_kernel_units() -> None:
    """Markers split the device section into per-entry compilation units."""
    source = (
        "#include <kernel_operator.h>\n"
        "constexpr uint32_t kTile = 2048;\n"
        f"{KERNEL_SECTION}\n"
        "__global__ __vector__ void first(__gm__ uint8_t* x, uint32_t n) {}\n"
        f"{KERNEL_SECTION}\n"
        "__global__ __vector__ void second(__gm__ uint8_t* x, uint32_t n) {}\n"
    )
    units = split_device_units(source)
    check("two units with two markers", len(units) == 2, str(len(units)))
    check("entry names in order",
          [unit.signature.name for unit in units] == ["first", "second"])
    check("shared preamble prepended to every unit",
          all("kTile = 2048" in unit.source for unit in units))
    check("first unit holds only its entry", "second" not in units[0].source)
    check("second unit holds only its entry", "first" not in units[1].source)
    check("marker lines dropped",
          all("ASCEND_KERNEL_SECTION" not in unit.source for unit in units))

    entry_before_marker = (
        "__global__ __vector__ void zero(__gm__ uint8_t* x, uint32_t n) {}\n"
        f"{KERNEL_SECTION}\n"
        "__global__ __vector__ void one(__gm__ uint8_t* x, uint32_t n) {}\n"
    )
    try:
        split_device_units(entry_before_marker)
    except DeepJitBuildError as exc:
        check("entry before the first marker is refused", "exactly one" in str(exc))
    else:
        check("entry before the first marker is refused", False)

    empty_unit = (
        "#include <kernel_operator.h>\n"
        f"{KERNEL_SECTION}\n"
        "__global__ __vector__ void one(__gm__ uint8_t* x, uint32_t n) {}\n"
        f"{KERNEL_SECTION}\n"
    )
    try:
        split_device_units(empty_unit)
    except DeepJitBuildError as exc:
        check("unit without an entry is refused", "no kernel entry" in str(exc))
    else:
        check("unit without an entry is refused", False)

    duplicate = (
        "#include <kernel_operator.h>\n"
        f"{KERNEL_SECTION}\n"
        "__global__ __vector__ void same(__gm__ uint8_t* x, uint32_t n) {}\n"
        f"{KERNEL_SECTION}\n"
        "__global__ __vector__ void same(__gm__ uint8_t* y, uint32_t m) {}\n"
    )
    try:
        split_device_units(duplicate)
    except DeepJitBuildError as exc:
        check("duplicate kernel names refused", "distinct kernel names" in str(exc))
    else:
        check("duplicate kernel names refused", False)


def test_launcher_text() -> None:
    """The launcher prepares every kernel and launches from cached handles."""
    source = (
        "#include <kernel_operator.h>\n"
        f"{KERNEL_SECTION}\n"
        "__global__ __vector__ void partial_sum(__gm__ uint8_t* x, "
        "__gm__ uint8_t* p, uint32_t n) {}\n"
        f"{KERNEL_SECTION}\n"
        "__global__ __vector__ void finalize_sum(__gm__ uint8_t* p, "
        "__gm__ uint8_t* o, uint32_t c) {}\n"
    )
    units = split_device_units(source)
    launcher = _launcher_source(
        units, [Path("kernel_0.hpp"), Path("kernel_1.hpp")]
    )
    check("one prepare helper per kernel",
          launcher.count("int prepare_kernel") == 2)
    check("one launch stub per entry",
          launcher.count("_launch(uint32_t numBlocks") == 2)
    check("prepare is a C ABI status entry",
          'extern "C" int akb_deepjit_prepare(char* error_message, int error_size)'
          in launcher)
    check("prepare stops at the first failure",
          "if (status0 != 0)" in launcher and "if (status1 != 0)" in launcher)
    # Two in each prepare helper (exception and catch-all) plus one in each
    # launch guard.
    check("every message names its kernel",
          launcher.count('std::string("kernel ")') == 3 * len(units),
          str(launcher.count('std::string("kernel ")')))
    check("every launch passes the caller stream",
          launcher.count(".stream = static_cast<aclrtStream>(stream)") == 2)
    check("launch reads the prepared handle",
          launcher.count("        kHandle0,") == 1
          and launcher.count("        kHandle1,") == 1)
    check("launch refuses a missing handle",
          launcher.count("was not prepared") == 2)
    check("a launch cannot compile",
          ".compile(" not in launcher.split('extern "C" void')[1])
    check("no compile call in a launch stub",
          ".compile(kKernelName" not in launcher.split("extern \"C\" void")[-1])
    check("no device-wide synchronize", "aclrtSynchronizeDevice" not in launcher)
    check("one compiled unit per kernel file",
          "kernel_0.hpp" in launcher and "kernel_1.hpp" in launcher)

    decls = "\n".join(_decls_source(unit.signature) for unit in units)
    check("declarations cover both kernels",
          "partial_sum_launch" in decls and "finalize_sum_launch" in decls)


def test_build_facts() -> None:
    """Build facts record the compiled kernel names."""
    paths = SimpleNamespace(
        root=Path("/tmp/deepjit"),
        library_root=Path("/tmp/lib"),
        commit="deadbeef",
        cache_dir=Path("/tmp/cache"),
        kernel_source=Path(__file__),
        signature="sig",
    )
    facts = build_facts(paths, ["partial_sum", "finalize_sum"])
    check("facts name the build backend", facts["build_backend"] == "deepjit")
    check("facts list both kernels",
          facts["deepjit_kernels"] == ["partial_sum", "finalize_sum"])
    check("facts carry the commit", facts["deepjit_commit"] == "deadbeef")


def evaluator(sample_dir: Path) -> SampleEvaluator:
    """Return a deepjit evaluator with a stubbed device runtime."""
    request = DeviceEvalRequest.model_validate(
        {
            "task_py": "",
            "sample_dir": str(sample_dir),
            "cmake_arch": "dav-2201",
            "hardware_name": "ascend910b2",
            "device": "npu:0",
            "measure_performance": True,
            "seed": 42,
            "num_correct_trials": 1,
            "num_perf_trials": 1,
            "num_warmup": 0,
            "precision": "fp32",
            "tolerances": {"fp32": {"atol": 1e-4, "rtol": 1e-4}},
            "excessive_speedup": 10.0,
            "build_timeout": 60,
            "backend": "ascendc",
            "build_backend": "deepjit",
            "deepjit": {"root": "/tmp/deepjit"},
            "soc_version": "Ascend910B4",
        }
    )
    instance = SampleEvaluator(request)
    instance.torch = SimpleNamespace(
        npu=SimpleNamespace(set_device=lambda index: None)
    )
    return instance


def test_device_compile_failure_payload() -> None:
    """A failing device compile is not compiled and names the kernel."""
    source = (
        "#include <kernel_operator.h>\n"
        f"{KERNEL_SECTION}\n"
        "__global__ __vector__ void partial_sum(__gm__ uint8_t* x, "
        "__gm__ uint8_t* p, uint32_t n) {}\n"
        f"{KERNEL_SECTION}\n"
        "__global__ __vector__ void finalize_sum(__gm__ uint8_t* p, "
        "__gm__ uint8_t* o, uint32_t c) {}\n"
    )
    original_build = eval_device.build_custom_op
    original_load = eval_device.load_custom_op
    original_prepare = eval_device.prepare_deepjit_module
    try:
        eval_device.build_custom_op = (
            lambda *args, **kwargs: Path("/tmp/libcustom_op.so")
        )
        eval_device.load_custom_op = lambda *args, **kwargs: None
        with tempfile.TemporaryDirectory() as tmp:
            sample = Path(tmp)
            (sample / "custom_op.asc").write_text(source, encoding="utf-8")

            def failing(library: Path) -> None:
                raise DeepJitBuildError(
                    "deepjit device compile failed (status 1): "
                    "kernel finalize_sum: boom"
                )

            eval_device.prepare_deepjit_module = failing
            failed = evaluator(sample)._build_and_load()
            check("failed prepare returns a payload", failed is not None)
            check("failed prepare is not compiled", failed["compiled"] is False)
            metadata = failed["metadata"]
            check("failure stage is device_compile",
                  metadata["failure_stage"] == "device_compile")
            check("error names the failing kernel",
                  "finalize_sum" in str(metadata["compilation_error"]))
            check("build backend recorded",
                  metadata["build_backend"] == "deepjit")

            eval_device.prepare_deepjit_module = lambda library: None
            done = evaluator(sample)._build_and_load()
            check("successful prepare compiles every kernel", done is None)
    finally:
        eval_device.build_custom_op = original_build
        eval_device.load_custom_op = original_load
        eval_device.prepare_deepjit_module = original_prepare


def test_pyasc_combination_refused() -> None:
    """The deepjit path never claims a pyasc sample."""
    with tempfile.TemporaryDirectory() as tmp:
        sample = Path(tmp)
        (sample / "custom_op.asc").write_text(
            "#include <kernel_operator.h>\n", encoding="utf-8"
        )
        instance = evaluator(sample)
        instance.backend = __import__("src.backend", fromlist=["Backend"]).Backend.PYASC
        failed = instance._build_and_load()
    check("pyasc plus deepjit is refused",
          failed is not None and failed["compiled"] is False)
    check("refusal is a static check failure",
          failed["metadata"]["failure_stage"] == "static_check")


def main() -> int:
    """Run every no-device check."""
    print("deepjit bridge checks")
    for test in (
        test_include_conversion,
        test_marker_scan_ignores_code_only,
        test_unsafe_arguments_are_refused,
        test_signature_tracks_checkout_state,
        test_parameter_whitelist,
        test_single_kernel_units,
        test_multi_kernel_units,
        test_launcher_text,
        test_build_facts,
        test_device_compile_failure_payload,
        test_pyasc_combination_refused,
    ):
        test()
    print("\nall deepjit bridge checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
