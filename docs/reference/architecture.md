# Architecture

AscendKernelBench is a Python orchestration layer around a vendored task corpus, an OpenAI-compatible generation endpoint, and a fixed CMake build for Ascend C. Generation and analysis can run independently of the Ascend evaluation machine.

```text
KernelBench task + hardware profile
                  |
             prompt.py
                  |
         llm.py -> LLM endpoint
                  |
     backend kernel artifact + model_new.py
                  |
             runs/{name}/
                  |
          eval.py static checks
                  |
          one worker per sample
                  |
        ACLNN CMake project -> libcustom_op.so (process-local load)
                  |
        correctness -> NPU-event timing
                  |
          per-sample eval_result.json
                  |
       aggregate JSON -> score.py / analyze.py
```

## Components

| Component | Responsibility |
| --- | --- |
| `KernelBench/level*/` | Reference task files in the original `Model`/input-generator format. |
| `dataset.py` | Static contract validation, task loading, and discovery. |
| `config.py` | Evaluation defaults and hardware-profile loading from YAML. |
| `prompt.py` | Task source, hardware information, examples, and output-contract assembly. |
| `llm.py` | Endpoint requests, structured/fenced response handling, and basic deliverable validation. |
| `rundir.py` | Run paths, generated files, per-sample collection, and atomic aggregate writes. |
| `checker.py` | Stable re-export of static checks. Implementation is `checks/` (regex `PatternRule`s, Python AST visitor, Ascend C rules). |
| `eval.py` | Public host API (`evaluate_run`) and isolated worker spawn. |
| `eval_device.py` | Worker-side `SampleEvaluator`: build, seeded correctness, NPU-event timing, and SOL attach. |
| `process.py` | Shared isolated JSON worker (`IsolatedJsonWorker.for_eval` / `for_baseline`). |
| `report.py` | Rich progress bars, sample status, and evaluation tables. |
| `log.py` | loguru setup used by CLIs and workers. |
| `eval_result.py` | KernelBench-compatible result payloads and protocol-snapshot metadata. |
| `build.py` and `build_template/` | CMake build of `libcustom_op.so` and `torch.ops.load_library`. |
| `timing.py` | NPU events, L2 flush sized from the hardware profile, and timing statistics. |
| `compare.py` | Dtype-aware output matching and tensor-byte accounting. |
| `sol.py` | Roofline bound and SOL-ExecBench-style score. |
| `runtime.py` | CANN / PyTorch / torch-npu / device identity recorded in result metadata. |
| `cli_util.py` | Generation/eval setting factories, task selection, and re-exports of `report.py`. |
| `io_util.py` | Atomic JSON/YAML writes, JSON-object reads, and worker cfg loading. |
| `baseline_worker.py` | Isolated eager-reference timing used by `scripts/baseline.py`. |
| `score.py` | Sample speedups, `fast_p`, geometric mean speedup, pass@k, and mean SOL. |
| `scripts/_eval_worker.py` | Internal worker process entry. Bootstraps the checkout root and calls `worker_main`. |

The Python modules are in [`src/`](https://github.com/wuzhenqing/AscendKernelBench/tree/main/src). The scripts under [`scripts/`](https://github.com/wuzhenqing/AscendKernelBench/tree/main/scripts) are the user entry points.

## Evaluation API layers

| Layer | What to call | What not to call |
| --- | --- | --- |
| CLI | `scripts/evaluate.py`, `scripts/analyze.py` | `scripts/_eval_worker.py` |
| Engine host | `evaluate_run`, `eval_sample`, rundir helpers, `score.py` | `eval_sample_on_device`, `worker_main` |
| Isolated worker | `worker_main` → `eval_sample_on_device` → `build_custom_op` / `load_custom_op` | Anything from the host process |

`evaluate.py` is a thin argparse wrapper around `evaluate_run`. The CLI accepts a run name and an optional level.

## Workflow entry points

| Script | Inputs | Main output | Ascend required |
| --- | --- | --- | --- |
| `generate.py` | Tasks, hardware profile, generation settings, endpoint access | Candidate sources in a run directory | No |
| `evaluate.py` | Existing complete samples in a run, optional level | Per-sample results, aggregate JSON, and a printed report | Yes |
| `baseline.py` | Reference tasks, evaluation settings, device | Archived NPU reference latency | Yes |
| `analyze.py` | Existing aggregate evaluation JSON | Terminal score tables | No |

Evaluation is sequential at the batch-script level, with a new worker for each sample. There is no scheduler, multi-device dispatcher, or resumable evaluation queue.

## Repository data

`_paths.py` anchors configurations, tasks, build templates, runs, and baseline results to the checkout root, inferred as the parent of `src/`. Prompt examples live under `src/prompts/examples/`. The vendored corpus and top-level configuration/build directories remain checkout assets.

## Generation boundary

Prompt construction reads task source without executing it. Prompts contain the complete reference file, target hardware information, the selected examples, and a fixed two-file output contract. `zero_shot` omits examples, `one_shot` uses the first example, and `few_shot` uses every available example. The current repository ships two examples (elementwise add, then LeakyReLU). `one_shot` uses the first; `few_shot` uses both.

The LLM client first attempts a structured response with the `custom_op_asc` and `model_new_py` fields. If that request or parsing fails, it attempts a normal completion and extracts two fenced code blocks. The resulting fields pass through the same Pydantic model and marker checks. Generation can retry once after an invalid/failed attempt. Endpoint fallback and retries can therefore issue more than one request for a sample.

These checks establish that response fields resemble the expected files; they do not establish compilability or correctness. Full static checking happens at evaluation. Raw responses, when available, are persisted next to the generated files.

## Evaluation boundary

The host (`evaluate_run` → `eval_sample`) reads sample source and runs static checks before launching `scripts/_eval_worker.py` with the same Python interpreter. That script adds the checkout root to `sys.path` and imports `src.eval`. A temporary config JSON carries the task source, sample path, profile architecture, and resolved evaluation settings into the worker. The worker calls `eval_device.eval_sample_on_device` and writes a JSON result file back. Correctness is the seeded trials plus a four-distribution value gate (original, ×3, ×0.01, ×−1) adapted from KernelBench-Verified; timing still uses the original input draw. The host then stamps `metadata.harness` from `generation_config.yaml`, so a result names the model-harness pair and not only the model.

NPU imports live inside worker/timing functions so source inspection and host-side utilities do not initialize an NPU runtime. The worker loads generated code, compiles native code, and uses the selected NPU. Its process is separate but has the invoking user's privileges; this is not a security sandbox.

Build configuration is controlled by the repository template. The generated `.asc` file supplies kernel logic, host launch wrappers, and a `TORCH_LIBRARY` / `TORCH_LIBRARY_IMPL` binding. The template builds a **process-local** `libcustom_op.so` with RPATH to Torch, `torch_npu`, and CANN libraries. It does not run `cmake --install`, does not produce a `custom_opp_*.run` package, and does not write into site-packages or `$ASCEND_OPP_PATH/vendors`. The worker loads that `.so` with `torch.ops.load_library` so `torch.ops.custom_op` is available. Hardware profile architecture selection is an input to compilation, not runtime hardware verification.

For Ascend C, build and evaluation compile `custom_op.asc` with the CMake template and load `libcustom_op.so`; for pyasc they import `kernel.py` and let pyasc trace and compile on the first call. Sources carrying the `ASCEND_HOST_SECTION` marker build in split mode: `build.py` separates the device and host sections, generates the `extern "C"` launch-stub TU from the kernel signatures, and compiles the host glue as plain C++ with a shared precompiled header cached per environment. Unmarked sources build as a single ASC translation unit. Both modes produce the same process-local library contract; there is no pybind11 path.

The [evaluation guide](../guide/evaluation.md) documents seeded initialization, tolerance rules, CPU-reference fallback, timeout accounting, and event timing. The [task authoring guide](../task_authoring.md) describes both reference and candidate contracts.

## Persistence and scoring boundary

Generation creates the two implementation files and accompanying prompt/response artifacts. Evaluation writes per-sample results; aggregation only includes complete sample directories with stored result JSON. The aggregator adds sample IDs and groups entries by task ID.

`score.py` operates on those dictionaries without loading kernels, task models, or NPU libraries. `analyze.py` reads the existing aggregate rather than reconstructing it, enabling offline analysis after transferring a run from the evaluation machine.

Compiled extensions and build directories are machine-specific. Transfer source artifacts and configuration to the destination machine and let evaluation build there. Preserve separate run copies when comparing machines because result filenames are reused.

The [results guide](../guide/results.md) explains what is persisted, which failures enter the denominator, and how reference mode or flags affect each metric.
