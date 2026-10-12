"""Hardware profile and evaluation configuration loading.

YAML is parsed with PyYAML and validated by frozen Pydantic models, so
required fields, defaults, and unknown-key handling live in one place.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ._paths import EVAL_DEFAULT_CONFIG, HARDWARE_DIR


class BuildBackend(str, Enum):
    """How custom_op.asc becomes a loadable library.

    cmake stays the default so every existing run keeps its meaning. deepjit
    is an opt-in alternative that compiles the same source through the
    DeepJIT runtime instead of the CMake template.
    """

    CMAKE = "cmake"
    DEEPJIT = "deepjit"


DEFAULT_BUILD_BACKEND = BuildBackend.CMAKE


def parse_build_backend(value: object) -> BuildBackend:
    """Resolve a build-backend value, defaulting when absent.

    Raises:
        ValueError: If the value is not a known build backend.
    """
    if isinstance(value, BuildBackend):
        return value
    if value is None:
        return DEFAULT_BUILD_BACKEND
    text = str(value).strip().lower()
    if not text:
        return DEFAULT_BUILD_BACKEND
    try:
        return BuildBackend(text)
    except ValueError as exc:
        known = ", ".join(item.value for item in BuildBackend)
        raise ValueError(
            f"unknown build backend {value!r}; choose one of: {known}"
        ) from exc


def _default_tolerances() -> dict[str, dict[str, float]]:
    """Return the KernelBench-style per-precision comparison tolerances."""
    return {
        "fp32": {"atol": 1e-4, "rtol": 1e-4},
        "fp16": {"atol": 1e-2, "rtol": 1e-2},
        "bf16": {"atol": 1e-2, "rtol": 1e-2},
    }


class HardwareProfile(BaseModel):
    """Hardware profile: CMake arch for builds, specs for prompts and SOL."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    name: str
    soc_version: str
    cmake_arch: str
    ai_core_num: int
    ub_size_kb: int
    l2_cache_mb: int = 0
    hbm_gb: int = 0
    memory_bandwidth_gbps: float = 0.0
    supported_dtypes: list[str] = Field(default_factory=list)
    api_style: str = ""
    cube_core_num: int = 0
    vector_core_num: int = 0
    peak_tflops: dict[str, float] = Field(default_factory=dict)

    @field_validator("peak_tflops", mode="before")
    @classmethod
    def _coerce_peak_tflops(cls, value: object) -> dict[str, float]:
        """Accept missing, empty, or loosely typed peak-TFLOPS mappings."""
        if not value:
            return {}
        if not isinstance(value, Mapping):
            return {}
        return {str(key): float(item) for key, item in value.items()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> HardwareProfile:
        """Build a profile from a hardware YAML mapping."""
        return cls.model_validate(data)

    def peak_tflops_for(self, precision: str) -> float:
        """Return datasheet peak TFLOPS for precision, or 0.0 when unset.

        Args:
            precision: One of fp32, fp16, bf16.
        """
        return float(self.peak_tflops.get(precision, 0.0) or 0.0)


class DeepJitConfig(BaseModel):
    """Local DeepJIT settings for the optional deepjit build backend.

    Every path is local configuration, never a repository default: the
    checkout location, the isolated host toolchain, and its runtime library
    directories differ per machine. Empty fields fall back to environment
    variables (DEEPJIT_ROOT, DEEPJIT_COMMIT, DEEPJIT_CXX, DEEPJIT_CACHE_DIR)
    so a machine can be configured without editing tracked YAML.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    root: str = ""
    commit: str = ""
    cxx: str = ""
    include_dirs: list[str] = Field(default_factory=list)
    library_dirs: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    extra_cflags: list[str] = Field(default_factory=list)
    extra_ldflags: list[str] = Field(default_factory=list)
    cache_dir: str = ""
    build_timeout: int = 1800


class EvalConfig(BaseModel):
    """Evaluation defaults for one benchmark run."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    hardware: str = "ascend910b2"
    num_correct_trials: int = 5
    seed: int = 42
    tolerances: dict[str, dict[str, float]] = Field(default_factory=_default_tolerances)
    precision: str = "fp32"
    num_perf_trials: int = 100
    num_warmup: int = 10
    excessive_speedup: float = 10.0
    build_timeout: int = 600
    eval_timeout: int = 300
    build_backend: BuildBackend = DEFAULT_BUILD_BACKEND
    deepjit: DeepJitConfig = Field(default_factory=DeepJitConfig)
    generation: dict[str, Any] = Field(default_factory=dict)

    @field_validator("build_backend", mode="before")
    @classmethod
    def _build_backend_value(cls, value: object) -> object:
        """Accept a backend name, a member, or an absent value."""
        return parse_build_backend(value)

    @field_validator("generation", mode="before")
    @classmethod
    def _generation_mapping(cls, value: object) -> dict[str, Any]:
        """Treat a missing or null generation block as an empty mapping."""
        return value if isinstance(value, dict) else {}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EvalConfig:
        """Build an eval config, ignoring unknown top-level keys."""
        return cls.model_validate(data)


def load_hardware_profile(name_or_path: str) -> HardwareProfile:
    """Load a hardware profile by name or filesystem path.

    Args:
        name_or_path: configs/hardware/<name>.yaml stem, or a file path.

    Raises:
        FileNotFoundError: If neither a path nor a named profile exists.
    """
    candidate = Path(name_or_path)
    if not candidate.is_file():
        candidate = HARDWARE_DIR / f"{name_or_path}.yaml"
    if not candidate.is_file():
        available = sorted(path.stem for path in HARDWARE_DIR.glob("*.yaml"))
        raise FileNotFoundError(
            f"Hardware profile not found: {name_or_path}. Available: {available}"
        )
    loaded = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
    return HardwareProfile.from_dict(loaded)


def load_eval_config(path: str | Path | None = None) -> EvalConfig:
    """Load evaluation config; defaults to configs/eval_default.yaml.

    Raises:
        FileNotFoundError: If the resolved path does not exist.
    """
    cfg_path = Path(path) if path else EVAL_DEFAULT_CONFIG
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Eval config not found: {cfg_path}")
    loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    return EvalConfig.from_dict(loaded)
