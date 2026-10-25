# Getting started

AscendKernelBench separates code generation from operator evaluation. You can inspect tasks, prepare prompts, generate candidates through a remote language model, and analyze saved results on macOS. Compiling and evaluating candidates requires a Linux machine with an Ascend NPU and the matching software stack.

## Choose your environment

| Activity | macOS or another machine without an NPU | Linux with Ascend |
| --- | --- | --- |
| Read tasks and build prompts | Yes | Yes |
| Generate source through a remote endpoint | Yes | Yes |
| Run the static source checker | Yes | Yes |
| Analyze an existing `eval_results.json` | Yes | Yes |
| Compile `custom_op.asc` into `libcustom_op.so` | No, requires the Ascend toolchain | Yes, with the required toolchain |
| Check candidate correctness and measure performance | No | Yes |

The evaluator has a CPU **reference** fallback for tasks that the NPU reference cannot execute. The candidate still runs on the NPU; this is not a CPU evaluation mode.

## Install from a checkout

Use Python **3.10 or newer**. Check `python3 --version` first; on macOS, the system-provided interpreter may be older. If necessary, replace `python3` below with the path to your newer interpreter.

```bash
git clone https://github.com/wuzhenqing/AscendKernelBench.git
cd AscendKernelBench
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run the commands in these guides from the repository root, with the environment activated. Scripts add the checkout root to `sys.path` and import `src.*`; there is no editable or wheel install. Tasks, hardware profiles, scripts, and the build template are used directly from the checkout; no dataset submodule initialization is needed.

`requirements.txt` is the only dependency file. On Linux it also installs the pinned PyTorch and `torch_npu` pair. It never installs CANN, the NPU driver, or the native build toolchain.

Optional local checks after you change Python:

```bash
pre-commit install && pre-commit run --all-files
```

### Ascend host experiment environment

On a Linux machine with CANN 9.1.0, use the conda env AscendKernelBench,
which pairs Python 3.12 with PyTorch 2.10.0 and torch-npu 2.10.0.post6:

```bash
conda create -n AscendKernelBench python=3.12 -y
conda activate AscendKernelBench
source /usr/local/Ascend/cann-9.1.0/set_env.sh
python -m pip install -r requirements.txt
```

Confirm the NPU is visible before evaluating:

```bash
python - <<'PY'
import torch
import torch_npu
print(torch.__version__, torch_npu.__version__, torch.npu.is_available())
PY
```

## Explore tasks without an NPU

Task discovery reads Python source and checks its top-level contract using the Python AST. It does not execute the task or import its PyTorch dependencies.

```bash
python - <<'PY'
from collections import Counter
from src.dataset import discover_tasks, load_task

tasks = discover_tasks()
print("Tasks by level:", dict(sorted(Counter(t.level for t in tasks).items())))
task = load_task("level1/19_ReLU")
print("Example:", task.task_id)
print("Source:", task.path)
PY
```

The committed task set contains 270 tasks across four levels. Use a task ID such as `level1/19_ReLU`, without the `KernelBench/` prefix or `.py` extension. See [task authoring](../task_authoring.md) for the task contract.

You can also inspect exactly what will be sent to the model, without making an API request:

```bash
python - <<'PY'
from pathlib import Path
from src.config import load_hardware_profile
from src.dataset import load_task
from src.prompt import build_prompt

prompt = build_prompt(
    load_task("level1/19_ReLU"),
    load_hardware_profile("ascend910b2"),
    mode="one_shot",
)
Path("relu-prompt.txt").write_text(prompt, encoding="utf-8")
print("Wrote relu-prompt.txt")
PY
```

## Generate your first candidate

Configure an endpoint and a model name that your service actually exposes. The model in the default configuration is a project default, not a guarantee that a particular endpoint serves it.

```bash
export OPENAI_BASE_URL='https://your-endpoint.example/v1'
export OPENAI_API_KEY='your-api-key'

python scripts/generate.py \
  --task level1/19_ReLU \
  --hardware ascend910b2 \
  --model your-served-model-name \
  --n-samples 1 \
  --run-name relu-demo
```

Replace the placeholder values before running. Generation sends the task source and prompt examples to the configured service. It writes `custom_op.asc`, `model_new.py`, and prompt/response files under `runs/relu-demo/level1/19_ReLU/sample_0/`. It does not build or evaluate the generated operator.

See [LLM service configuration](../deploy_llm_service.md) for endpoint behavior and [workflows](workflows.md) for moving a run to an Ascend machine.

## Prepare an Ascend evaluation machine

All evaluation paths require an accessible Linux Ascend runtime. The default
Ascend C CMake path expects:

- A Linux host with an accessible Ascend NPU and its driver/runtime.
- A CANN environment providing the `ASC` CMake package/compiler. The template documents CANN 9.1.0 or newer as its requirement.
- A compatible PyTorch and `torch_npu` installation, importable in the active Python environment.
- CMake 3.16 or newer, Python development headers, and a C++17/GCC toolchain with a discoverable `libgcc.a`.

For pyasc, also install the matching `pyasc` package (`pyasc==1.1.1` is the
selected version); the import name is `asc`. For DeepJIT, provide a checkout
through `DEEPJIT_ROOT` and a host compiler with `std::format` support (GCC 13
or newer). These are optional dependencies: generation, analysis, and offline
tests do not import them. See [pyasc](pyasc-backend.md) and
[DeepJIT](deepjit.md) for backend-specific setup.

Use the installation and compatibility instructions for the stack on your machine; this repository does not provide a universal version matrix. Some tasks also import additional packages, particularly at level 4. Inspect the selected task's imports before starting a large run.

Set `CANN_SET_ENV` to your installed environment script if it differs from the project default:

```bash
export CANN_SET_ENV=/path/to/cann/set_env.sh
source "$CANN_SET_ENV"

python - <<'PY'
import torch
import torch_npu

print("PyTorch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("NPU available:", torch.npu.is_available())
PY
```

This checks basic imports and device availability. Validate compilation,
correctness, and timing on the target machine with:

```bash
python tests/run_tests.py
python tests/npu_backend_smoke.py --output /tmp/ascend-kernel-bench-smoke
```

`--output` is a directory containing case subdirectories and the aggregate
`smoke_results.json`. The offline suite needs no NPU. The smoke runner uses
the selected installed backends, with `--case` available to select cases and
`--repeat 2` to exercise warm builds. Passing its cases does not imply support
for all 270 tasks.

Continue with [generation and evaluation workflows](workflows.md), or consult [troubleshooting](troubleshooting.md) if setup fails.
