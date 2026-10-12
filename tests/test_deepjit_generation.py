"""No-device tests for the deepjit generation path.

Runs with plain asserts and no test framework:

    python tests/test_deepjit_generation.py

Covers the pieces between settings and a candidate on disk: the DeepJIT
wording reaching the prompt and nothing else, the example pool per build
path, the response extractor accepting the shipped examples in both supported
layouts, the static checker refusing a response that lost a marker, and the
run record carrying the build backend. Nothing here imports torch or an NPU.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.checker import check_custom_op_asc, check_model_new  # noqa: E402
from src.cli_util import (  # noqa: E402
    generation_run_config,
    resolve_generation_settings,
)
from src.config import (  # noqa: E402
    BuildBackend,
    DeepJitConfig,
    load_eval_config,
    load_hardware_profile,
)
from src.dataset import load_task  # noqa: E402
from src.deepjit_build import DeepJitBuildError, _prepare_layout  # noqa: E402
from src.llm import extract_generation  # noqa: E402
from src.prompt import (  # noqa: E402
    DEEPJIT_EXAMPLE_SUBDIR,
    build_prompt,
    load_examples,
    system_prompt,
)

TASK_ID = "level1/88_MinGPTNewGelu"
DEEPJIT_ONLY_STRINGS = (
    "ASCEND_KERNEL_SECTION",
    "DeepJIT",
    "deepjit",
)
CMAKE_ONLY_STRINGS = ("CMake project", "CMakeLists.txt")


def check(name: str, condition: bool, detail: str = "") -> None:
    """Record one check result."""
    suffix = "" if condition else " :: " + detail
    print(f"  {'ok  ' if condition else 'FAIL'} - {name}{suffix}")
    if not condition:
        raise AssertionError(f"{name}: {detail}")


def prompt_for(
    build_backend: BuildBackend = BuildBackend.DEEPJIT,
    mode: str = "one_shot",
) -> str:
    """Return one prompt for the bundled fixture task."""
    return build_prompt(
        load_task(TASK_ID),
        load_hardware_profile("ascend910b2"),
        mode=mode,
        build_backend=build_backend,
    )


def test_deepjit_prompt_wording() -> None:
    """The deepjit contract states the conventions the bridge enforces."""
    prompt = prompt_for()
    for fragment in (
        "// ==================== ASCEND_HOST_SECTION ====================",
        "// ==================== ASCEND_KERNEL_SECTION ====================",
        "exactly one entry per compiled unit",
        "shared preamble",
        "repeat in the host section every constant it needs",
        "__gm__ uint8_t*",
        "int64_t",
        "foo_launch(numBlocks, (void*)aclStream",
        "intermediate tensor alive until both launches are enqueued",
        "The build, the kernel cache, and the preparation step belong to the",
        "torch.ops.custom_op",
    ):
        check(f"prompt states: {fragment[:48]!r}", fragment in prompt, "missing")
    check(
        "deepjit prompt does not promise a CMake build",
        "fixed CMake project" not in prompt
        and "CMake arch" not in prompt
        and "compiled with its fixed" not in prompt,
    )
    check(
        "deepjit prompt still forbids emitting build files",
        "You must NOT emit CMakeLists.txt" in prompt,
    )
    system = system_prompt(BuildBackend.DEEPJIT)
    check("deepjit system message names DeepJIT", "DeepJIT" in system)
    check("deepjit system message drops CMake", "CMake" not in system)


def test_other_paths_are_unchanged() -> None:
    """The CMake prompt never carries deepjit-only conventions."""
    cmake = prompt_for(build_backend=BuildBackend.CMAKE)
    for fragment in DEEPJIT_ONLY_STRINGS:
        check(f"cmake prompt has no {fragment!r}", fragment not in cmake)
    check("cmake prompt keeps its CMake wording", "CMake" in cmake)
    check(
        "cmake prompt keeps its own example pool",
        "001_elementwise_add" in cmake and "002_two_stage_sum" not in cmake,
    )
    check(
        "cmake system message unchanged",
        "CMake" in system_prompt(BuildBackend.CMAKE),
    )


def test_example_pool_selection() -> None:
    """Each build path reads its own pool and the mode keeps its counts."""
    deepjit_pool = load_examples(BuildBackend.DEEPJIT)
    cmake_pool = load_examples(BuildBackend.CMAKE)
    check(
        "deepjit pool has the add and two-stage examples",
        [item.name for item in deepjit_pool]
        == ["001_elementwise_add", "002_two_stage_sum"],
        str([item.name for item in deepjit_pool]),
    )
    check("cmake pool unchanged", len(cmake_pool) == 2, str(len(cmake_pool)))
    zero = prompt_for(mode="zero_shot")
    one = prompt_for(mode="one_shot")
    few = prompt_for(mode="few_shot")
    check("zero_shot carries no example", "### Example task" not in zero)
    check(
        "one_shot carries exactly one example",
        one.count("### Example task") == 1,
        str(one.count("### Example task")),
    )
    check(
        "one_shot shows the single-kernel example",
        "001_elementwise_add" in one and "002_two_stage_sum" not in one,
    )
    check(
        "few_shot carries both examples",
        few.count("### Example task") == 2,
        str(few.count("### Example task")),
    )
    check("deepjit pool directory name is stable", DEEPJIT_EXAMPLE_SUBDIR == "deepjit")


def test_response_extraction() -> None:
    """Both shipped examples survive the response extractor unchanged."""
    for example in load_examples(BuildBackend.DEEPJIT):
        fenced = (
            f"```custom_op.asc\n{example.custom_op_asc}```\n\n"
            f"```model_new.py\n{example.model_new_py}```\n"
        )
        parsed = extract_generation(fenced)
        check(
            f"{example.name}: fenced blocks round-trip",
            parsed.custom_op_asc == example.custom_op_asc
            and parsed.model_new_py == example.model_new_py,
        )
        structured = json.dumps(
            {
                "custom_op_asc": example.custom_op_asc,
                "model_new_py": example.model_new_py,
            }
        )
        parsed_json = extract_generation(structured)
        check(
            f"{example.name}: JSON fields round-trip",
            parsed_json.custom_op_asc == example.custom_op_asc,
        )
    two_stage = next(
        item
        for item in load_examples(BuildBackend.DEEPJIT)
        if item.name == "002_two_stage_sum"
    )
    marker_lines = [
        line
        for line in two_stage.custom_op_asc.splitlines()
        if line.strip().strip("= ").strip() == "ASCEND_KERNEL_SECTION"
        or line.strip().replace("//", "").strip().strip("=").strip()
        == "ASCEND_KERNEL_SECTION"
    ]
    check(
        "two-stage example has exactly two kernel marker lines",
        len(marker_lines) == 2,
        str(len(marker_lines)),
    )
    check(
        "both examples pass the static checks",
        check_custom_op_asc(two_stage.custom_op_asc)
        + check_model_new(two_stage.model_new_py)
        == [],
    )


def test_marker_loss_is_reported() -> None:
    """A response that dropped the host marker is refused with a message."""
    source = (
        "```custom_op.asc\n"
        '#include "kernel_operator.h"\n'
        "__global__ __vector__ void k(__gm__ uint8_t* x) {}\n"
        "```\n\n```model_new.py\nimport torch\n"
        "class ModelNew(torch.nn.Module):\n"
        "    def forward(self, x):\n"
        "        return torch.ops.custom_op.run(x)\n```\n"
    )
    parsed = extract_generation(source)
    violations = check_custom_op_asc(parsed.custom_op_asc) + check_model_new(
        parsed.model_new_py
    )
    check("marker loss is a static violation", bool(violations), str(violations))
    check(
        "marker loss is refused instead of silently accepted",
        all("missing" in item or "not" in item for item in violations),
        str(violations),
    )
    with tempfile.TemporaryDirectory() as tmp:
        try:
            _prepare_layout(
                Path(tmp), parsed.custom_op_asc, DeepJitConfig(), "Ascend910B4"
            )
        except DeepJitBuildError as exc:
            check(
                "the deepjit build path names the missing marker",
                "host-section marker" in str(exc),
                str(exc),
            )
        else:
            check("the deepjit build path names the missing marker", False, "accepted")


def test_run_record_carries_the_build_path() -> None:
    """Generation settings and the run record carry the deepjit options."""
    config = load_eval_config()
    settings = resolve_generation_settings(
        config,
        build_backend=BuildBackend.DEEPJIT.value,
        deepjit_root="/tmp/deepjit",
    )
    check(
        "settings keep the build backend",
        settings.build_backend is BuildBackend.DEEPJIT,
    )
    check(
        "settings keep the deepjit root",
        settings.deepjit.get("root") == "/tmp/deepjit",
        str(settings.deepjit),
    )
    record = generation_run_config(
        settings, hardware_name="ascend910b2", task_ids=[TASK_ID]
    )
    check(
        "run record has build_backend",
        record["build_backend"] == BuildBackend.DEEPJIT.value,
    )
    check(
        "run record has the deepjit block",
        isinstance(record.get("deepjit"), dict)
        and record["deepjit"]["root"] == "/tmp/deepjit",
        str(record.get("deepjit")),
    )
    default_settings = resolve_generation_settings(config)
    check(
        "default settings stay cmake",
        default_settings.build_backend is BuildBackend.CMAKE,
    )
    check("default settings carry no deepjit block", default_settings.deepjit == {})


def test_prompt_builds_without_npu() -> None:
    """Prompt building and the run record need no device and no torch."""
    prompt = prompt_for(mode="few_shot")
    check("few_shot prompt is non-empty", len(prompt) > 2000, str(len(prompt)))
    check(
        "prompt mentions the reference model",
        "MinGPTNewGelu" in prompt or "gelu" in prompt.lower(),
    )


def test_deepjit_config_defaults() -> None:
    """The deepjit settings model keeps empty machine paths by default."""
    settings = DeepJitConfig()
    check("no root by default", settings.root == "")
    check("no cache dir by default", settings.cache_dir == "")
    check(
        "the deepjit block carries no timeout of its own",
        "build_timeout" not in DeepJitConfig().model_dump(),
    )
    check(
        "the top-level build timeout survives",
        load_eval_config().build_timeout > 0,
    )


def main() -> int:
    """Run every no-device generation check."""
    print("deepjit generation checks")
    for test in (
        test_deepjit_prompt_wording,
        test_other_paths_are_unchanged,
        test_example_pool_selection,
        test_response_extraction,
        test_marker_loss_is_reported,
        test_run_record_carries_the_build_path,
        test_prompt_builds_without_npu,
        test_deepjit_config_defaults,
    ):
        test()
    print("\nall deepjit generation checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
