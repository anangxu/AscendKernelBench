# Documentation

```text
/======================================================================\
|                                                                      |
|            ___    _____   ______   ______    _   __    ____          |
|           /   |  / ___/  / ____/  / ____/   / | / /   / __ \         |
|          / /| |  \__ \  / /      / __/     /  |/ /   / / / /         |
|         / ___ | ___/ / / /___   / /___    / /|  /   / /_/ /          |
|        /_/  |_|/____/  \____/   \____/   /_/ |_/   /_____/           |
|            __ __   ______    ____     _   __   ______    __          |
|           / //_/  / ____/   / __ \   / | / /  / ____/   / /          |
|          / ,<    / __/     / /_/ /  /  |/ /  / __/     / /           |
|         / /| |  / /___    / _, _/  / /|  /  / /___    / /___         |
|        /_/ |_|  \____/   /_/ |_|  /_/ |_/   \____/   /_____/         |
|               ____    ______    _   __   ______    __  __            |
|              / __ )  / ____/   / | / /  / ____/   / / / /            |
|             / __  | / __/     /  |/ /  / /       / /_/ /             |
|            / /_/ / / /___    / /|  /  / /___    / __  /              |
|           /_____/  \____/   /_/ |_/   \____/   /_/ /_/               |
|                                                                      |
|                          AscendKernelBench                           |
|                                                                      |
|       [ LLM ]=======>{ Ascend C / pyasc }=======>{ Ascend NPU }       |
|                 fast_p   |   pass@k   |   SOL score                  |
|                                                                      |
\======================================================================/
```

Guides and references for running AscendKernelBench. These pages live in
the repository and are meant to be read on GitHub. There is no published
documentation site.

## Start here

- [Installation and first steps](guide/getting-started.md)
- [Connect an LLM service](deploy_llm_service.md)

## Run the benchmark

- [Generation and evaluation](guide/workflows.md)
- [Evaluation protocol](guide/evaluation.md)
- [pyasc authoring backend](guide/pyasc-backend.md)
- [DeepJIT build backend](guide/deepjit.md)
- [Results and metrics](guide/results.md)
- [Troubleshooting](guide/troubleshooting.md)

## Reference

- [Command line](reference/cli.md)
- [Configuration and hardware](reference/configuration.md)
- [Architecture and Python entry points](reference/architecture.md)
- [Write a benchmark task](task_authoring.md)

## Choose your starting point

| Your environment | What you can do | Start here |
| --- | --- | --- |
| macOS or a machine without an Ascend NPU | Read tasks, construct prompts, generate candidates through a remote service, and analyze existing JSON results | [Installation and first steps](guide/getting-started.md) |
| Linux with a supported Ascend software stack | Compile candidates, check their outputs, collect timings, and record eager baselines | [Generation and evaluation](guide/workflows.md) |
| Adding tasks or integrating an agent | Use the task contract and call individual engine modules | [Task authoring](task_authoring.md) · [Architecture](reference/architecture.md) |

## What the benchmark measures

The input is a PyTorch reference `Model`. A language model produces two files:
`custom_op.asc` for Ascend C or `kernel.py` for pyasc, together with a
compatible `ModelNew` wrapper in `model_new.py`. The evaluator checks the
source, builds or JIT-compiles it inside a worker, compares outputs, and
measures eligible candidates against live `torch_npu` eager execution.
Ascend C uses device-event timing; pyasc measures synchronized end-to-end
invocations, including host dispatch and launch overhead. [Results and metrics](guide/results.md) explains how `fast_p`,
`pass@k`, and the optional roofline SOL score are calculated and which samples
enter each denominator.

The project implements generation, isolated sample evaluation, and analysis.
It does not provide an automatic compile-error repair loop. A saved hardware
profile is a configuration input, not proof that a software or device
combination has passed validation. In particular, `ascend950pr` is reserved
and must be checked before use. See
[configuration and hardware](reference/configuration.md).

> **Working on a machine without an NPU.** Kernel compilation and evaluation
> require the Linux Ascend environment. CPU fallback changes the reference
> only; it does not make candidate evaluation run on a laptop or Mac.
