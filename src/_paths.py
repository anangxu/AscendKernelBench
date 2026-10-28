"""Repository layout constants.

Data directories are inferred from this file: src/ sits under the
checkout root.
"""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
CONFIGS_DIR = REPO_ROOT / "configs"
HARDWARE_DIR = CONFIGS_DIR / "hardware"
EVAL_DEFAULT_CONFIG = CONFIGS_DIR / "eval_default.yaml"
BUILD_TEMPLATE_DIR = REPO_ROOT / "build_template"
KB_ROOT = REPO_ROOT / "KernelBench"
DOMAIN_ROOT = REPO_ROOT / "domain_benchmarks"
RESULTS_DIR = REPO_ROOT / "results"
BASELINE_DIR = RESULTS_DIR / "baseline"
RUNS_DIR = REPO_ROOT / "runs"
PROMPT_EXAMPLES_DIR = PACKAGE_DIR / "prompts" / "examples"
