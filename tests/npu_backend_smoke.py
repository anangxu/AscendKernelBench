"""Run the shared evaluation protocol on all supported backend paths."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backend import Backend
from src.config import (
    BuildBackend,
    DeepJitConfig,
    load_eval_config,
    load_hardware_profile,
)
from src.dataset import Task
from src.eval import eval_sample
from src.io_util import write_json_atomic, write_text_atomic

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "src/prompts/examples"
CASES = {
    "cmake_add": ("001_elementwise_add", Backend.ASCENDC, BuildBackend.CMAKE),
    "cmake_leaky_relu": ("002_leaky_relu", Backend.ASCENDC, BuildBackend.CMAKE),
    "cmake_two_stage_sum": (
        "deepjit/002_two_stage_sum",
        Backend.ASCENDC,
        BuildBackend.CMAKE,
    ),
    "deepjit_add": (
        "deepjit/001_elementwise_add",
        Backend.ASCENDC,
        BuildBackend.DEEPJIT,
    ),
    "deepjit_two_stage_sum": (
        "deepjit/002_two_stage_sum",
        Backend.ASCENDC,
        BuildBackend.DEEPJIT,
    ),
    "pyasc_add": ("pyasc/001_elementwise_add", Backend.PYASC, BuildBackend.CMAKE),
    "pyasc_rowsum": ("pyasc/003_rowsum", Backend.PYASC, BuildBackend.CMAKE),
}


def main() -> int:
    """Evaluate selected examples and save complete results outside the corpus."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", choices=list(CASES))
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--hardware", default="ascend910b2")
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    config = load_eval_config()
    hardware = load_hardware_profile(args.hardware)
    output = args.output.resolve()
    records = []
    for name in args.case or CASES:
        relative, backend, build_backend = CASES[name]
        example = EXAMPLES / relative
        source = (example / "task.py").read_text()
        task = Task(f"level1/{name}", 1, name, example / "task.py", source)
        sample = output / name
        for filename in (
            "model_new.py",
            "kernel.py" if backend is Backend.PYASC else "custom_op.asc",
        ):
            content = (example / filename).read_text()
            target = sample / filename
            if not target.exists() or target.read_text() != content:
                write_text_atomic(target, content)
        for repeat in range(args.repeat):
            print(f"Evaluating {name}, repetition {repeat + 1}", flush=True)
            started = time.monotonic()
            result = eval_sample(
                task,
                sample,
                hardware=hardware,
                config=config,
                backend=backend,
                build_backend=build_backend,
                deepjit=DeepJitConfig(),
            )
            metadata = result.get("metadata") or {}
            passed = bool(
                result.get("compiled")
                and result.get("correctness")
                and result.get("runtime")
                and result.get("ref_runtime")
                and metadata.get("timing_valid", True)
                and metadata.get("reference") == "npu"
            )
            records.append(
                {
                    "case": name,
                    "repetition": repeat + 1,
                    "passed": passed,
                    "wall_seconds": round(time.monotonic() - started, 3),
                    "result": result,
                }
            )
            write_json_atomic(output / "smoke_results.json", records)
            print(
                f"{'PASS' if passed else 'FAIL'} {name}: "
                f"compiled={result.get('compiled')} correct={result.get('correctness')} "
                f"runtime_ms={result.get('runtime')}",
                flush=True,
            )
            if not passed:
                print(metadata, flush=True)
    return int(not all(record["passed"] for record in records))


if __name__ == "__main__":
    raise SystemExit(main())
