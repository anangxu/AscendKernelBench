# CLI reference

Run these scripts from the checkout root. Each script adds the checkout root to `sys.path` and imports `src.*`. Third-party dependencies still need to be installed; see [getting started](../guide/getting-started.md). Each command supports `-h` / `--help`. The scripts are the only command-line interface.

`scripts/_eval_worker.py` is an internal process entry used by `evaluate_run`. Do not invoke it directly.

## Common conventions

| Concept | Meaning |
| --- | --- |
| Task ID | `level{number}/{file_stem}`, for example `level1/19_ReLU`. Omit `KernelBench/` and `.py`. |
| `--task` and `--level` | In batch generation and baseline measurement, repeat `--task` to select tasks. It overrides `--level`. Omitting both selects all discovered tasks. |
| `--tasks-file` | Generation only. A manifest of task IDs, one per line, with blank lines and `#` comments ignored; see `configs/subsets/`. `--task` overrides it. |
| Run name | A directory name under `<repository>/runs/`. Use a simple, unique name such as `relu-demo`. There is no separate `--run-dir` option. |
| `--hardware` | A profile name such as `ascend910b2`, or the path to an existing YAML file. |
| `--config` | An evaluation YAML file. If omitted, loads `<repository>/configs/eval_default.yaml`. A relative path is resolved from the current working directory. |
| `--device` | Device string used by generation-adjacent tools such as `baseline.py`, default `npu:0`. `evaluate.py` always uses `npu:0`; select a physical card with `ASCEND_RT_VISIBLE_DEVICES`. |

All batch commands process items sequentially. They have no CLI options for parallel workers, resume, or skipping previously completed samples.

## `scripts/generate.py`

Generate source files through the configured LLM endpoint. This command does not require an NPU.

```bash
python scripts/generate.py \
  --task level1/19_ReLU \
  --n-samples 10 \
  --model your-served-model-name \
  --run-name relu-10
```

| Option | Default | Purpose |
| --- | --- | --- |
| `--level INTEGER` | All levels | Discover tasks from one level. |
| `--task TASK_ID` | Unset | Select a task; repeat for multiple tasks. Overrides `--level`. |
| `--tasks-file PATH` | Unset | Select tasks from a manifest file; overrides `--level`. |
| `--n-samples INTEGER` | `generation.num_samples`, otherwise `1` | Number of samples per task. Use a positive integer. |
| `--model NAME` | `generation.model`, otherwise `deepseek-flash` | Endpoint model name. |
| `--hardware NAME_OR_PATH` | `hardware` from configuration | Select the hardware description injected into the prompt. |
| `--prompt-mode MODE` | `generation.prompt_mode`, otherwise `one_shot` | `zero_shot`, `one_shot`, or `few_shot`. |
| `--temperature FLOAT` | `generation.temperature`, otherwise `0.0` | Generation sampling temperature. |
| `--max-tokens INTEGER` | `generation.max_tokens`, otherwise `131072` | Completion token budget, reasoning tokens included. |
| `--reasoning-effort LEVEL` | `generation.reasoning_effort`, otherwise unset | `low`, `medium`, or `high`. The parameter is omitted from the request when unset. |
| `--build-backend NAME` | `build_backend` from configuration, otherwise `cmake` | How `custom_op.asc` is built: `cmake` or `deepjit`. Recorded in the run. |
| `--deepjit-root PATH` | `deepjit.root` | DeepJIT checkout for `--build-backend deepjit`. |
| `--deepjit-commit SHA` | `deepjit.commit` | Pinned DeepJIT commit; a mismatch with the checkout is refused. |
| `--run-name NAME` | `gen_YYYYMMDD_HHMMSS` | Output run name, using local time when generated automatically. |
| `--config PATH` | `configs/eval_default.yaml` | Load generation and hardware settings from an evaluation YAML file. |

Successful samples include `prompt.txt`, `custom_op.asc`, `model_new.py`, and, when nonempty, `response_raw.txt`.

Generation errors are printed and the batch continues. The script exits with status 1 if **all** generation attempts fail; a partially successful batch returns status 0. Check the final saved/expected count before proceeding. Configuration, task loading, and prompt-construction errors can also stop execution.

## `scripts/evaluate.py`

Build, check, and time complete source pairs in an existing run. Requires the Ascend evaluation environment. There are no flags.

```bash
python scripts/evaluate.py relu-10
python scripts/evaluate.py relu-10 1
```

| Argument | Default | Purpose |
| --- | --- | --- |
| `run` | Required | Run name under `runs/`, or an existing directory path. |
| `level` | Every level in the run | Integer level to evaluate, for example `1`. |

This command is a thin wrapper around `evaluate_run`. Embedding code should call that function instead of importing the script. It does not accept a task, sample id, device, hardware, config, or performance-skip flag. Multi-card hosts set `ASCEND_RT_VISIBLE_DEVICES` so the chosen card appears as `npu:0`. Hardware is read from `generation_config.yaml` when present, otherwise from `configs/eval_default.yaml`.

It writes per-sample `eval_result.json`, then the run's `eval_results.json` and `pass_at_k_results.json`, and prints the same report as `analyze.py`. Evaluating one level overwrites that level's per-sample results and rebuilds aggregates from every stored result in the run.

Missing runs or a selection with no samples cause a nonzero exit. Individual compilation and correctness failures are recorded as results; a completed batch does not return a failing exit code merely because candidates failed. Inspect the result files when using this command in automation.

## `scripts/baseline.py`

Measure and archive PyTorch eager reference timings on the NPU, without an LLM call or generated candidate.

```bash
python scripts/baseline.py \
  --task level1/19_ReLU \
  --hardware ascend910b2 \
  --device npu:0
```

| Option | Default | Purpose |
| --- | --- | --- |
| `--level INTEGER` | All levels | Discover tasks from one level. |
| `--task TASK_ID` | Unset | Select a task; repeat to select more. Overrides `--level`. |
| `--hardware NAME_OR_PATH` | `hardware` from configuration | Select the hardware name used for the archive directory. |
| `--device DEVICE` | `npu:0` | Device on which to time the reference. |
| `--config PATH` | `configs/eval_default.yaml` | Precision, seed, warmup, trial count, and timeout settings. |

Results go to `results/baseline/<hardware.name>/<task_file_stem>.json`. The command can record `supported_on_npu: false` for a reference that fails within its measurement phase. Worker startup failures may leave no result file. Per-task exceptions are printed and the batch continues without a failing exit code solely for those exceptions.

The hardware profile labels the archive; it does not change the physical device selected by `--device`. Evaluation independently measures its own live reference and does not consume these archived files.

## `scripts/analyze.py`

Print aggregate and per-problem reports from existing results. Does not need an NPU or endpoint credentials.

```bash
python scripts/analyze.py relu-10
```

| Argument | Default | Purpose |
| --- | --- | --- |
| `run` | Required | Run name under `runs/`, or an existing directory path. |

The script recalculates metrics in memory and prints them, including `fast_p`, geometric-mean speedup, `pass@k`, and mean roofline SOL score when present. It writes no report files, does not collect per-sample results, and does not update `pass_at_k_results.json`. A missing aggregate file causes a nonzero exit.

See [configuration](configuration.md) for defaults and environment variables, and [results](../guide/results.md) for the result schema and scoring rules.
