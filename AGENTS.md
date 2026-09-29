# AGENTS.md

Working agreement for AI coding agents in this repository. It is the fast path
to being productive here without re-deriving the architecture.

**Maintenance rule:** if your change alters a command, directory, protocol
constant, or invariant described here, update this file in the same change.
A stale `AGENTS.md` is a bug.

## What this project is

AscendKernelBench is a *generate, then evaluate* benchmark for LLM-written
kernels on Huawei Ascend NPUs. A model reads a PyTorch reference `Model` from
the vendored KernelBench corpus and must emit exactly two artifacts. The
backend decides their names: Ascend C (the default) uses `custom_op.asc` and
`model_new.py`; pyasc uses `kernel.py` and `model_new.py`. One run is one
backend. The harness builds the sample through that backend's path, loads
the operator inside a worker PyTorch process, checks outputs against the
reference, and times eligible kernels against live `torch_npu` eager
execution. Ascend C compiles `.asc` into a process-local `libcustom_op.so`
loaded with `torch.ops.load_library`; pyasc imports `kernel.py` and JITs on
the first call. Scoring is KernelBench `fast_p` / `pass@k` plus an optional
roofline SOL score.

```text
KernelBench task + hardware profile
  -> prompt.py -> llm.py -> runs/{run}/level{L}/{task}/sample_{i}/
       (prompt.txt, the backend's kernel file, model_new.py, response_raw.txt)
  -> eval.py static checks (checker.py / checks/)
  -> isolated worker (scripts/_eval_worker.py)
  -> Ascend C: build.py + build_template/ -> libcustom_op.so -> load_library
       (marked sources: split kernel/launcher/host units + shared host PCH)
  -> pyasc:    pyasc_runtime.py -> kernel.py imported; the first call JITs
  -> eval_device.py: seeded correctness -> hidden value gate
       (d1..d4) -> NPU-event timing -> fresh-input re-check
  -> per-sample eval_result.json
  -> eval_results.json + pass_at_k_results.json -> score.py / scripts/analyze.py
```

For the Ascend C backend, `custom_op.asc` has two sections separated by one
`// ==================== ASCEND_HOST_SECTION ====================` line:
device kernels above (compiled without torch headers), torch glue below
(compiled as plain C++ against the shared PCH). The host wrapper launches
kernels through generated `<kernel>_launch(numBlocks, stream, args...)`
stubs. Unmarked legacy sources still build as a single ASC unit.

Current scope: generation and evaluation CLIs only. There is **no**
automatic compile-error repair agent and no scheduler / multi-device queue.
The one thing outside that scope is `experiments/pyasc_device_probes/`, which
holds device-only evidence scripts and is imported by nothing.

## Commands

```bash
# Setup (any machine; NPU not required). requirements.txt is the only
# dependency file; it installs the pinned torch pair on Linux.
python -m pip install -r requirements.txt
pre-commit install   # optional

# Optional before you land Python changes
pre-commit run --all-files

# Ascend host experiment environment (CANN 9.1.0 pairing)
conda create -n AscendKernelBench python=3.12 -y && conda activate AscendKernelBench
source /usr/local/Ascend/cann-9.1.0/set_env.sh
python -m pip install -r requirements.txt

# Workflow CLIs (run from the checkout root)
export OPENAI_BASE_URL="https://provider.example/v1" OPENAI_API_KEY="..."
python scripts/generate.py --task level1/19_ReLU --model <model> \
    --hardware ascend910b2 --run-name relu_demo [--backend ascendc|pyasc] \
    [--level N] [--n-samples N] \
    [--prompt-mode zero_shot|one_shot|few_shot] [--temperature T] [--config PATH] \
    [--max-tokens N] [--reasoning-effort low|medium|high]
# Fixed comparison set: 20 level1 tasks, every fifth task of the level.
python scripts/generate.py --tasks-file configs/subsets/level1_20.txt \
    --model <model> --reasoning-effort high --run-name level1_20
# NPU commands need device access: without it aclInit fails with 507899 (see
# Local host notes). Have the user run them, or escalate the sandbox.
ASCEND_RT_VISIBLE_DEVICES=0 python scripts/evaluate.py relu_demo [level]
python scripts/analyze.py relu_demo
python scripts/baseline.py --level 1 --hardware ascend910b2 --device npu:0

# Device probes (NPU only; they re-measure the facts in
# docs/reference/pyasc-device-evidence.md). Standalone: run one, or all.
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd experiments/pyasc_device_probes && python probe_r09_writeback_length.py
for probe in probe_*.py; do python "$probe" || exit 1; done
```

CLI facts worth remembering:

* `scripts/evaluate.py` takes only `run` and an optional positional `level`.
  It has **no** `--config`/`--hardware` override: it always loads
  `configs/eval_default.yaml` and resolves hardware from the run's
  `generation_config.yaml` (falling back to `config.hardware`).
* `scripts/generate.py` and `scripts/baseline.py` accept `--config` /
  `--hardware`. Generation also selects tasks with `--tasks-file` and
  forwards `--max-tokens` / `--reasoning-effort` to the endpoint; both are
  recorded in the run's `generation_config.yaml`.
* Response parsing accepts a JSON object holding `custom_op_asc` and
  `model_new_py`, or two fenced code blocks. Endpoints that reject
  `response_format` (DeepSeek-compatible ones) answer the structured request
  with that JSON, so the JSON path is the one that runs there.
* `scripts/_eval_worker.py` is internal. Never call it by hand; it is spawned
  by `eval.eval_sample`.
* Offline analysis (`analyze.py`, `score.py`) needs neither torch nor an NPU.

## Layout

| Path | What lives there |
| --- | --- |
| `KernelBench/level{1..4}/` | Vendored corpus: 270 tasks (100/100/50/20). Read-only in spirit. |
| `src/` | Engine: dataset, prompt, LLM, checks, eval, build, timing, scoring, reporting. |
| `src/checks/` | Static anti-cheat: `rules.py` (regex catalogs), `python_ast.py` (wrapper AST), `ascend_c.py` (Ascend C), `pyasc.py` (pyasc kernel, launcher, and call graph), `python_source.py`/`text.py` (entry points and masking). |
| `src/backend.py` | Backend registry: names, artifact filenames, and generation field names. |
| `src/pyasc_runtime.py` | Worker-side pyasc runtime: sample-local JIT cache, real-file kernel import, platform and device selection, first-call timing. |
| `src/prompts/examples/` | Few-shot example pairs used by prompt.py. |
| `scripts/` | The four user CLIs plus `_bootstrap.py` / `_eval_worker.py`. |
| `configs/eval_default.yaml` | Default protocol: 5 correctness trials, 10 warmup + 100 perf trials, tolerances, timeouts, generation defaults. |
| `configs/hardware/*.yaml` | Hardware profiles validated by `config.HardwareProfile`. Default `ascend910b2`; `ascend950pr` is reserved and unvalidated. |
| `configs/subsets/*.txt` | Fixed task subsets for comparable runs. `level1_20.txt` is every fifth level1 task, loaded with `--tasks-file`. Keep the lists stable. |
| `requirements.txt` | Third-party dependencies; the only pip install file. |
| `.pre-commit-config.yaml` | Optional Ruff and basic file hooks on `src/` and `scripts/`. |
| `build_template/CMakeLists.txt` | The single build path that produces `libcustom_op.so`. |
| `docs/` | English guides (`guide/`) and references (`reference/`). |
| `docs/reference/pyasc-device-evidence.md` | The measured pyasc facts behind the guide, with versions, commands, inputs, observed outputs, and the boundary of each claim. |
| `docs/reference/pyasc-evaluation-report.md` | The first end-to-end pyasc evaluation: configuration, results, the framework fixes it forced, and what the numbers do not cover. |
| `experiments/pyasc_device_probes/` | Device-only probes that re-measure those facts. Standalone scripts, imported by nothing in `src/`. |
| `tests/` | No-device suites and their fixtures: backend selection, evaluation wiring, and the static checks. Run as `python tests/<file>.py`; they need neither a device nor pyasc. |
| `runs/`, `results/` | Generated artifacts; gitignored. Never commit them. |

Module boundaries are strict; `docs/reference/architecture.md` has the full
API-layer table. In short: host code goes through `eval.evaluate_run`;
`eval_device` and `worker_main` are worker-side only; `score.py` never loads
kernels or tasks; `_paths.py` anchors all data directories to the checkout
root (`src/` sits directly under that root).

## Invariants — do not break these

1. **Process-local operators.** Build in the sample directory and load with
   `torch.ops.load_library`. Never install into site-packages, `$ASCEND_OPP_PATH`,
   or a global vendor directory; never run `cmake --install`; never emit a
   `custom_opp_*.run` package. Parallel jobs must not share mutable install state.
2. **`torch.library` registration only.** `TORCH_LIBRARY` /
   `TORCH_LIBRARY_IMPL` under namespace `custom_op`. pybind11 is banned and the
   checker enforces it.
3. **Compute belongs to the device.** The host wrapper may allocate and launch
   only; no ATen compute, no `aclnn*`/`aclop` vendor ops, no host side effects
   (process, network, dlopen, threads). `model_new.py` may call only
   `torch.ops.custom_op` (plus integer shape arithmetic); `nn` modules may be
   parameter containers but must never be called.
4. **Static checks are advisory; the runtime protocol is the backstop.** The
   fresh-input re-check, the four hidden value transforms (d1..d4), and the
   `excessive_speedup` flag exist because regex/AST checks cannot stop
   determined obfuscation. Do not weaken any of those layers alone. The
   transform factors stay out of the generation prompt.
5. **Scoring semantics are KernelBench-compatible.** `fast_0` is the
   correctness rate over *collected* sample results. Generation failures that
   never wrote a sample are absent from the denominator, not failures.
   Suspicious speedups stay in `fast_0`/`pass@k` but are excluded from positive
   `fast_p` thresholds and the geometric mean. Do not "fix" that silently.
6. **The reference is measured live**, in the same worker, on the same device,
   in the same run. `results/baseline/` exists only as a cross-run archive and
   must never become the evaluation denominator.
7. **Keep the host NPU-free.** NPU imports (`torch`, `torch_npu`) stay inside
   worker and timing functions so prompt building, dataset discovery, and
   offline analysis work on machines without a device.
8. **English everywhere** in code, prompts, checker messages, comments, and docs.
9. **Never reformat or edit vendored `KernelBench/` tasks** to satisfy lint.
10. **pyasc is a second backend, never a mixed one.** `kernel.py` is
    imported from its real file (pyasc reads the source with introspection);
    `PYASC_CACHE_DIR` is exported per sample before asc is imported; only
    `kernel_fn[core_num](...)` compiles and launches, so a successful import
    is never reported as a compiled sample. `first_call_seconds` is a compound
    trace+compile+launch figure and must not be called a compile time.
11. **CPU-reference fallback is not a free pass.** It only changes the
    correctness reference; such samples have no NPU speedup and no SOL score.

## Conventions

* Optional style: `pre-commit run --all-files` runs Ruff on `src/` and
  `scripts/` only.
* `from __future__ import annotations` plus type hints on public signatures;
  Google-style docstrings on public functions (summary, then `Args`/`Returns`/`Raises`
  when they add information). One responsibility per function.
* Frozen Pydantic models for YAML config and worker request payloads
  (`extra="ignore"`); dataclasses for internal value types.
* All file writes go through `io_util` atomic helpers. Result JSON must stay
  KernelBench-compatible and carry the protocol snapshot.
* Logging is loguru (`logger`); Rich is for CLIs and `report.py` only. Fatal CLI
  errors go through `log.die()`.
* Scripts import checkout modules as `src.*`. Engine modules use relative
  imports. `__init__.py` is not a public library surface.
* No GitHub Actions workflow ships with this tree. Kernel build and timing
  stay host-side validation on an Ascend machine.

## Naming and comment style

* Call the project AscendKernelBench in code, prompts, docs, and commit
  messages. Do not abbreviate it as akb; the conda environment carries the
  full name too.
* Comments and docstrings stay short: at most 3 consecutive lines per
  comment, 80 columns maximum, plain prose. No backticks, no markdown
  emphasis, no bullet markers, no decorative quotes. Write identifiers bare,
  as in torch.ops.load_library.
* Public docstrings keep a one-line summary, plus Args/Returns/Raises only
  where a unit, a format, or an exception is not obvious from the signature.
  Module docstrings are at most 3 lines. Private helpers get one line.
* Mark a genuinely important block, such as a protocol step, the build path,
  a scoring formula, or a key config group, with the same banner line above
  and below it: a run of # characters, an uppercase keyword, another run of
  # characters, 80 columns maximum.
* Delete comments that restate the next line or narrate history. A comment
  that has drifted out of date is worse than no comment.

## Common change recipes

| Change | Touch |
| --- | --- |
| New static check | `checks/rules.py` (regex catalog) or `checks/python_ast.py` / `checks/ascend_c.py`. |
| New hardware profile | `configs/hardware/<name>.yaml`; fields must satisfy `config.HardwareProfile`; mirror the comments in `ascend910b2.yaml`. |
| Protocol/timeout/trial change | `configs/eval_default.yaml` + `timing.py`/`eval.py` + `docs/guide/evaluation.md`. |
| Scoring change | `score.py` / `sol.py` + `docs/guide/results.md` + README metrics table. |
| Prompt or output contract | `prompt.py` (`SYSTEM_PROMPT`, `OUTPUT_CONTRACT`) + `prompts/examples/` + docs. |
| New CLI flag | Thin `scripts/*.py` argparse + `cli_util.py` factory + `docs/reference/cli.md`. |
| Build backend | `build.py` (`build_custom_op` / `load_custom_op` are the only surface) + `build_template/CMakeLists.txt`. |
| New task | Add to `KernelBench/levelN/` keeping `Model`, `get_inputs`, `get_init_inputs`; see `docs/task_authoring.md`. Task source is validated statically and never executed at load time. |
| New comparison subset | `configs/subsets/<name>.txt` + `docs/guide/workflows.md`. Fix the sampling rule in the manifest header and keep the list stable. |

## Local host notes (this checkout, not upstream truths)

* Two Python environments exist here: the default base conda env (3.14, has
  `ruff`/`pre-commit`, **no torch**) for optional lint, and the AscendKernelBench
  conda env
  (`/root/miniconda3/envs/AscendKernelBench`, Python 3.12, torch 2.10.0 +
  torch-npu 2.10.0.post6, CANN 9.1.0) for anything NPU-related. A legacy env
  named `akb` holds the same packages and is no longer the documented name.
* **The NPU works here — but only outside the agent's file sandbox.** The host
  is an Ascend 910B2 (`npu-smi` 25.2.1, Health OK, 64 GB HBM) and the
  AscendKernelBench env reaches it: `torch.npu.is_available()` is `True`, `device_count()` is 1,
  and a matmul on `npu:0` succeeds. Under the default `workspace-write`
  sandbox, opening `/dev/davinci*` (mode 0666) fails with `EACCES`, which
  surfaces as `aclInit ... error code is 507899` from torch_npu and
  `dcmi module initialize failed. ret is -8005` from `npu-smi`. That is a
  sandbox artifact, **not** a driver fault — do not "diagnose" it as a broken
  install. Run build/evaluate/baseline with the sandbox escalated
  (`danger-full-access`) or from the user's own terminal, and expect a benign
  "CANN and HDK(driver) versions require processing for 32 padding size"
  warning when the NPU allocator initializes. `/dev/devmm_svm` reports
  `EACCES` even unsandboxed; torch_npu initializes fine regardless.
* The NPU path is smoke-verified on this host: `scripts/baseline.py --task
  level1/19_ReLU --hardware ascend910b2 --device npu:0`, run with the sandbox
  escalated, completed the 10-warmup/100-trial protocol and archived
  `results/baseline/ascend910b2/19_ReLU.json` (mean 10.5 ms). CANN toolchain
  variables (`ASCEND_TOOLKIT_HOME`, `ASCEND_OPP_PATH`, driver `lib64` in
  `LD_LIBRARY_PATH`, `bisheng` on `PATH`) are already present in
  non-interactive shells, so no `.bashrc` sourcing is required.
* `pre-commit run --all-files` needs hook envs cached under `~/.cache/pre-commit`;
  the first run fetches from github.com and may time out.
* Re-evaluation of an unchanged sample reuses the sample-local `build/`
  directory, so builds are incremental. Set
  `ASCEND_KERNEL_BENCH_ENABLE_CCACHE=1` to opt into ccache; set `CANN_SET_ENV`
  to point at a different `set_env.sh`.
* Cold evaluation of a new-style (split-layout) sample costs about 35-45 s:
  ~9 s worker startup, ~15-20 s build (3 s configure + parallel kernel /
  launcher / PCH'd host compiles + link), the rest correctness and timing.
  The host-TU PCH lives in `~/.cache/ascend-kernel-bench/pch/` (override with
  `ASCEND_KERNEL_BENCH_CACHE_DIR`); the first build in an environment emits
  it in ~20 s. Legacy single-TU samples still build in 60-120 s. The
  breakdown and the rejected alternatives are in
  `docs/guide/troubleshooting.md`; the measured numbers are for this host, so
  re-measure before quoting them elsewhere.
* The DeepSeek endpoint used for the reference run is
  `https://api.deepseek.com/v1` with `OPENAI_API_KEY` set from the
  session's `ANTHROPIC_AUTH_TOKEN`; `deepseek-flash` accepts
  `reasoning_effort` and rejects `response_format`, so generation always
  takes the JSON or fenced-block fallback path.

## Definition of done

1. Pre-commit passes when you use it (optional).
2. New behavior validated on an Ascend host when it touches build, eval, or timing.
3. CLI, result-field, or protocol changes update `README.md` and the matching
   `docs/` page — and this file if any of the above changed.
4. No global installs, no reformatted vendored tasks, no committed
   `runs/`, `results/`, `*.so`, or API keys.

## Editing these instructions

`CLAUDE.md` symlinks this file; edit `AGENTS.md` only.
