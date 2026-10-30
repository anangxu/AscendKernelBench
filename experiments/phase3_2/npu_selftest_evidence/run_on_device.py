"""Build the fixture, run it on the NPU, compare, and time it."""
import json, sys, time
from pathlib import Path
ROOT = Path("/cache/akb/src"); sys.path.insert(0, str(ROOT))
import torch, torch_npu  # noqa: F401
from src.build import build_custom_op

fx = ROOT / "experiments" / "framework_selftest"
out = {"compile": None, "run": None, "correct": None, "seconds": None, "error": ""}
work = ROOT / "runs" / "s33_selftest" / "cann" / "nn" / "swiglu" / "sample_0"
work.mkdir(parents=True, exist_ok=True)
(work / "custom_op.asc").write_text((fx / "custom_op.asc").read_text(), encoding="utf-8")
(work / "model_new.py").write_text((fx / "model_new.py").read_text(), encoding="utf-8")
try:
    import torch_npu  # noqa: F401
    from src.config import load_hardware_profile
    hw = load_hardware_profile("ascend910b4")
    res = build_custom_op(
        (work / "custom_op.asc").read_text(), work,
        cmake_arch=hw.cmake_arch, build_backend="cmake",
        soc_version=hw.soc_version,
    )
    out["compile"] = "PASS"
    out["build_log"] = str(res)
except Exception as exc:
    out["error"] = f"build: {type(exc).__name__}: {exc}"[:250]
    print(json.dumps(out, indent=1)); raise SystemExit(0)

try:
    sys.path.insert(0, str(work))
    torch.ops.load_library(str(res))
    import importlib.util
    spec = importlib.util.spec_from_file_location("mn", work / "model_new.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    model = mod.ModelNew().npu()
    x = torch.randn(512, 4096, dtype=torch.float32).npu()
    y = model(x)
    half = 2048
    ref = torch.nn.functional.silu(x[:, :half]) * x[:, half:]
    out["run"] = "PASS"
    out["correct"] = bool(torch.allclose(y, ref, atol=1e-3, rtol=1e-3))
    out["max_diff"] = float((y - ref).abs().max().item())
    for _ in range(3):
        model(x)
    torch.npu.synchronize()
    start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(20):
        model(x)
    end.record(); torch.npu.synchronize()
    out["seconds"] = start.elapsed_time(end) / 20 / 1000
    out["timing_valid"] = out["seconds"] > 0
except Exception as exc:
    out["error"] = f"run: {type(exc).__name__}: {exc}"[:250]
print(json.dumps(out, indent=1))
