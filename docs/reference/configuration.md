# Configuration

Evaluation settings live in `configs/eval_default.yaml`. Hardware descriptions live in `configs/hardware/`, and fixed task subsets in `configs/subsets/`. CLI scripts load these files using `src/config.py`.

## Precedence and loading

1. A supplied generation or baseline CLI flag overrides the corresponding configuration setting.
2. `--config PATH` on those scripts loads that YAML file; otherwise, the repository's default file is loaded. Evaluation does not take `--config`.
3. Omitted top-level fields receive `EvalConfig` Pydantic-model defaults. Generation scripts apply their own defaults for missing `generation` keys.

A custom YAML file **replaces** the default file; the loader does not merge files. Nested mappings such as `tolerances` and `generation` are not recursively merged with the default YAML. Unknown top-level keys are silently ignored, so a misspelled key may appear to work while leaving the default in effect. The loader does not provide comprehensive value/range validation.

Evaluation reads `hardware_file` from an existing run's `generation_config.yaml` first, then the `hardware` field, when present. Other evaluation protocol values always come from `configs/eval_default.yaml`. Generation scripts still accept `--config` and `--hardware`.

## Evaluation settings

| Key | Default | Behavior |
| --- | --- | --- |
| `hardware` | `ascend910b2` | Default hardware profile name. Evaluation prefers `hardware_file`, then the name recorded in the run's `generation_config.yaml`. Generation still accepts `--hardware`. |
| `num_correct_trials` | `5` | Number of seeded correctness trials. Every trial must pass. |
| `seed` | `42` | Seed used for model initialization, correctness seed generation, and performance inputs. |
| `precision` | `fp32` | Floating-point precision. Use `fp32`, `fp16`, or `bf16`. Integer and Boolean tensors retain their types. |
| `tolerances` | See below | Absolute and relative tolerances by precision. |
| `num_perf_trials` | `100` | Number of measured trials for each timed implementation. |
| `num_warmup` | `10` | Number of warmup calls before measured trials. |
| `excessive_speedup` | `10.0` | Flag speedup strictly above this ratio for review. Flagged samples remain eligible for correctness metrics but are excluded from speedup metrics. |
| `build_timeout` | `600` | Seconds allowed for CMake configure and for build, separately. |
| `eval_timeout` | `300` | Evaluation component of the host worker time budget; also the standalone baseline worker timeout. |

The host evaluation worker timeout is `eval_timeout + 2 * build_timeout`: **1,500 seconds** with the defaults. This is an outer budget for the entire worker, not a separate 300-second timer around the runtime phase.

Default tolerances are:

```yaml
tolerances:
  fp32: {atol: 1.0e-4, rtol: 1.0e-4}
  fp16: {atol: 1.0e-2, rtol: 1.0e-2}
  bf16: {atol: 1.0e-2, rtol: 1.0e-2}
```

A task's nonempty `TOLERANCE` mapping takes precedence over configured tolerance values. A task can also define `custom_check(ref, out)` to control comparison. See the [task contract](../task_authoring.md) and [evaluation protocol](../guide/evaluation.md).

Use positive trial counts and timeouts. Set `num_warmup` to a nonnegative integer. Configuration changes alter the evaluation protocol; preserve the exact YAML used when comparing results.

## Generation settings

These are nested under `generation`:

| Key | Default | Used by |
| --- | --- | --- |
| `model` | `deepseek-flash` | `generate.py` |
| `temperature` | `0.0` | `generate.py` |
| `max_tokens` | `131072` | `generate.py`, or `--max-tokens` |
| `num_samples` | `1` | `generate.py` |
| `prompt_mode` | `one_shot` | `generate.py` |
| `reasoning_effort` | `high` | `generate.py`, or `--reasoning-effort` |

`num_samples` must be a positive integer; `generate.py` exits with a clear error when the resolved value is zero or negative. `reasoning_effort` is sent as a request parameter only when set; an empty or null YAML value is treated as unset for endpoints that reject it. Reasoning tokens count against `max_tokens`.

The model name is passed to your endpoint. Set it to a model your service supports. `zero_shot` omits examples, `one_shot` uses the first bundled example, and `few_shot` uses all bundled examples. The repository currently ships two examples (elementwise add and LeakyReLU).

For example, save the following as `configs/relu-experiment.yaml`:

```yaml
hardware: ascend910b2
precision: fp32
seed: 42
num_correct_trials: 5
num_warmup: 10
num_perf_trials: 100
excessive_speedup: 10.0
build_timeout: 600
eval_timeout: 300
tolerances:
  fp32: {atol: 1.0e-4, rtol: 1.0e-4}
  fp16: {atol: 1.0e-2, rtol: 1.0e-2}
  bf16: {atol: 1.0e-2, rtol: 1.0e-2}
generation:
  model: your-served-model-name
  temperature: 0.7
  max_tokens: 131072
  num_samples: 10
  prompt_mode: one_shot
  reasoning_effort: high
```

Use a custom YAML for generation. Evaluation loads `configs/eval_default.yaml` and the hardware name recorded in the run:

```bash
python scripts/generate.py \
  --task level1/19_ReLU \
  --config configs/relu-experiment.yaml \
  --run-name relu-experiment

# Run on the Ascend host after transferring the run.
python scripts/evaluate.py relu-experiment
```

## Hardware profiles

The supplied files describe these targets:

| Profile | Stored `soc_version` | Stored `cmake_arch` | Status |
| --- | --- | --- | --- |
| `ascend910b2` | `Ascend910B2` | `dav-2201` | Default profile in this repository. Validate against the machine and installed toolchain. |
| `ascend950pr` | `Ascend950PR` | `dav-3510` | Reserved profile; its file explicitly requires specification validation before use. |

These are configuration values, not automatic hardware detection or a claim that every target has passed validation. Selecting a profile does not change the NPU exposed to the process. Evaluation always uses `npu:0`; set `ASCEND_RT_VISIBLE_DEVICES` to choose a physical card.

| Profile field | Required | Meaning |
| --- | --- | --- |
| `name` | Yes | Result label and baseline archive directory name. |
| `soc_version` | Yes | SoC label included in the prompt. |
| `cmake_arch` | Yes | Passed to `CMAKE_ASC_ARCHITECTURES` during compilation. |
| `ai_core_num` | Yes | Core-count value included in the prompt. |
| `ub_size_kb` | Yes | Per-core UB capacity description included in the prompt. |
| `l2_cache_mb` | No; defaults to `0` | Used to size the timing L2 flush to `max(256 MiB, 2 × L2)`. |
| `hbm_gb` | No; defaults to `0` | HBM capacity description included in the prompt. |
| `memory_bandwidth_gbps` | No; defaults to `0` | Prompt bandwidth and the roofline SOL bound. |
| `cube_core_num` / `vector_core_num` | No; defaults to `0` | Included in the prompt core-count line when set. |
| `peak_tflops` | No; defaults to `{}` | Optional per-precision TFLOPS. Used only when a task also declares a positive top-level `FLOPS` or `NUM_FLOPS` constant; otherwise the SOL bound is memory-only. |
| `supported_dtypes` | No; defaults to `[]` | Data type names included in the prompt; does not itself enforce runtime support. |
| `api_style` | No; defaults to an empty string | Free-form instructions inserted into the hardware prompt block. |

Pass a custom profile with `--hardware configs/hardware/my-device.yaml`, or save it under `configs/hardware/my-device.yaml` and use `--hardware my-device`. An existing file path is tried first, relative to the current working directory when applicable; otherwise, the loader looks for `<repository>/configs/hardware/<value>.yaml`. When generation is given an existing file path, `generation_config.yaml` records its absolute path as `hardware_file`, and evaluation prefers that path before falling back to the profile name.

## Environment variables

| Variable | Purpose |
| --- | --- |
| `OPENAI_BASE_URL` | Endpoint URL used by `LLMClient`. The CLI has no endpoint URL flag. |
| `OPENAI_API_KEY` | Credential passed to the endpoint client. The CLI has no API key flag. |
| `CANN_SET_ENV` | Path to the environment script used by the build helper. Defaults to `/usr/local/Ascend/cann-9.1.0/set_env.sh`. |
| `ASCEND_KERNEL_BENCH_ENABLE_CCACHE` | When set to `1` / `true` / `yes` / `on`, the CMake configure step receives `-DENABLE_CCACHE=ON`. |
| `ASCEND_KERNEL_BENCH_CACHE_DIR` | Override the cache root holding the shared host-TU precompiled header. Defaults to `$XDG_CACHE_HOME/ascend-kernel-bench` or `~/.cache/ascend-kernel-bench`. |
| `ASCEND_KERNEL_BENCH_DISABLE_PCH` | When set to `1` / `true` / `yes` / `on`, the split build compiles the host glue without the precompiled header. |
| `ASCEND_SLOG_PRINT_TO_STDOUT` | The build/evaluation code defaults this to `0` in child environments when it is unset, to reduce runtime log output. |
| `ASCEND_RT_VISIBLE_DEVICES` | Restrict which physical NPUs the process can see. Evaluation always addresses `npu:0` inside that visible set. |

If the selected CANN environment script exists, the build helper sources it through Bash and caches the resulting environment for that process. If it does not exist, the helper uses the current environment. Source the appropriate CANN script in your shell before running the evaluator so Python imports and worker startup can also use it.

The repository root is the parent of `src/`. Configs, tasks, the build template, runs, and results all live under that checkout. See [architecture](architecture.md) for the layout.
