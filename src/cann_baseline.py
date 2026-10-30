"""AKB native baseline configurations for the CANN Bench tasks.

Each task pins one legal configuration: shapes, dtypes, attributes, and the
value range. The official 1200 cases stay available as a separate generalisation
suite; the main benchmark uses these baselines so that a candidate is measured
on one stable, reproducible workload per operator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .cases import CaseSpec, generate_inputs

BASELINE_PATH = Path("domain_benchmarks/cann/baseline.json")


def load_baseline(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Return the pinned configuration per task id."""
    target = path or BASELINE_PATH
    data = json.loads(target.read_text(encoding="utf-8"))
    return data["tasks"]


def baseline_case(task_id: str, config: dict[str, Any], seed: int = 0) -> CaseSpec:
    """Build the case spec of a task baseline for one seed."""
    return CaseSpec(
        case_id=seed,
        input_shapes=tuple(
            tuple(shape) if shape is not None else None for shape in config["shapes"]
        ),
        dtypes=tuple(config["dtypes"]),
        attrs=dict(config.get("attrs") or {}),
        value_range=tuple(
            tuple(item) if isinstance(item, (list, tuple)) else item
            for item in config["value_range"]
        ),
        seed=seed,
        source_revision=str(config.get("source", "")),
        note="AKB native baseline; not an official case",
    )


def build_inputs(
    task_id: str,
    config: dict[str, Any],
    torch,
    seed: int = 0,
    golden_path: Path | None = None,
) -> list:
    """Generate the baseline inputs for a task and seed.

    When the official golden ships an ``get_input`` fixer, it runs afterwards so
    that cross-input invariants (distinct indices, expert token sums, sequence
    lengths) hold. Math.random is seeded from the case seed, so the same seed
    reproduces the same data.
    """
    torch.manual_seed(seed)
    inputs = generate_inputs(baseline_case(task_id, config, seed), torch)
    if golden_path is None:
        return inputs
    import importlib.util

    spec = importlib.util.spec_from_file_location(f"golden_fix_{seed}", golden_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        return inputs
    fixer = getattr(module, "get_input", None)
    if fixer is None:
        return inputs
    import inspect

    params = [
        p
        for p in inspect.signature(fixer).parameters.values()
        if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    ]
    attrs = dict(config.get("attrs") or {})
    bound, used = {}, 0
    for p in params:
        if p.name in attrs:
            bound[p.name] = attrs[p.name]
        elif used < len(inputs):
            bound[p.name] = inputs[used]
            used += 1
    fixed = fixer(**bound)
    return list(fixed) if isinstance(fixed, (list, tuple)) else inputs
