"""Rich progress bars and evaluation report tables."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rich.console import Console
    from rich.progress import Progress


class SampleStatus(Enum):
    """Terminal style and short label for one evaluation result."""

    OK = ("green", "OK")
    COMPILE_FAIL = ("yellow", "COMPILE-FAIL")
    WRONG = ("red", "WRONG")

    @property
    def style(self) -> str:
        """Rich style name used for this status."""
        return self.value[0]

    @property
    def label(self) -> str:
        """Short status token, for example OK or COMPILE-FAIL."""
        return self.value[1]

    @classmethod
    def from_result(cls, result: dict[str, object]) -> SampleStatus:
        """Classify a KernelBench-compatible sample result."""
        if result.get("correctness"):
            return cls.OK
        if not result.get("compiled"):
            return cls.COMPILE_FAIL
        return cls.WRONG


def sample_status_label(result: dict[str, object]) -> tuple[str, str]:
    """Return the (style, label) pair for one evaluation result."""
    status = SampleStatus.from_result(result)
    return status.style, status.label


def eval_result_lines(result: dict[str, object]) -> tuple[str, list[str]]:
    """Return the Rich style and summary lines for one evaluation result."""
    from .score import sample_speedup

    style, label = sample_status_label(result)
    lines = [
        f"status: {label}",
        f"compiled: {result.get('compiled')}",
        f"correctness: {result.get('correctness')}",
    ]
    runtime = result.get("runtime")
    ref_runtime = result.get("ref_runtime")
    if runtime is not None:
        speedup = sample_speedup(result)
        detail = f"speedup {speedup:.2f}x" if speedup is not None else "no speedup"
        lines.append(
            f"runtime: {runtime:.4f} ms (ref {ref_runtime:.4f} ms, {detail})"
            if isinstance(ref_runtime, int | float)
            else f"runtime: {runtime:.4f} ms ({detail})"
        )
    metadata = result.get("metadata") or {}
    if isinstance(metadata, dict):
        err = metadata.get("compilation_error") or metadata.get("runtime_error")
        if err:
            lines.append(f"error: {str(err)[:800]}")
    return style, lines


def cli_progress(console: Console) -> Progress:
    """Return the shared Rich progress bar used by the batch CLIs."""
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    )


def print_eval_report(
    console: Console, run_name: str, results: dict[str, object]
) -> None:
    """Print the headline and per-problem evaluation tables."""
    from rich.table import Table

    from .score import (
        compute_pass_at_k,
        sample_speedup,
        summarize_eval_results,
    )

    summary = summarize_eval_results(results)
    pass_at_k = compute_pass_at_k(results)

    table = Table(title=f"AscendKernelBench report: {run_name}")
    table.add_column("metric", style="bold")
    table.add_column("value", justify="right")
    table.add_row("problems", str(summary["total_problems"]))
    table.add_row("samples", str(summary["total_samples"]))
    table.add_row(
        "compiled",
        f"{summary['compiled']} "
        f"({summary['compiled'] / max(summary['total_samples'], 1):.1%})",
    )
    table.add_row("correct (fast_0 denominator)", str(summary["correct"]))
    table.add_row("npu-reference samples", str(summary.get("npu_reference", 0)))
    table.add_row("cpu-reference samples", str(summary.get("cpu_reference", 0)))
    table.add_row(
        "flagged excessive speedup",
        str(summary.get("excessive_speedup", 0)),
    )
    # Shown even when zero: a rejected timing keeps its correctness credit but
    # is excluded from every performance number above fast_0.
    table.add_row(
        "invalid timing (excluded from perf)",
        str(summary.get("timing_invalid", 0)),
    )
    table.add_row(
        "hidden-distribution failures",
        str(summary.get("hidden_failed", 0)),
    )
    # Shown even when zero: an unsupported backend is a coverage gap, not a
    # model failure, and must not be readable as a pass.
    table.add_row(
        "unsupported dtype (backend)",
        str(summary.get("unsupported_dtype_backend_wide", 0)),
    )
    table.add_row(
        "unsupported operator (DSL)",
        str(summary.get("unsupported_operator", 0)),
    )
    for key, value in summary["fast_p"].items():
        table.add_row(key, f"{value:.3f}")
    table.add_row(
        "geomean speedup (correct only)",
        f"{summary['geometric_mean_speedup_correct_only']:.3f}",
    )
    sol = summary.get("mean_sol_score")
    table.add_row(
        "mean SOL score (roofline)",
        f"{sol:.3f}" if isinstance(sol, int | float) else "-",
    )
    for key, value in pass_at_k["average"].items():
        table.add_row(key, f"{value:.3f}")
    console.print(table)

    detail = Table(title="per-problem detail")
    detail.add_column("problem", style="bold")
    detail.add_column("samples", justify="right")
    detail.add_column("compiled", justify="right")
    detail.add_column("correct", justify="right")
    detail.add_column("best speedup", justify="right")
    for problem_id, samples in sorted(results.items()):
        if not isinstance(samples, list):
            continue
        compiled = sum(1 for sample in samples if sample.get("compiled"))
        correct = sum(1 for sample in samples if sample.get("correctness"))
        speedups = [s for s in (sample_speedup(x) for x in samples) if s]
        best = f"{max(speedups):.2f}x" if speedups else "-"
        style = "green" if correct else ("yellow" if compiled else "red")
        detail.add_row(
            f"[{style}]{problem_id}[/{style}]",
            str(len(samples)),
            str(compiled),
            str(correct),
            best,
        )
    console.print(detail)
