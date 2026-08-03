"""Reporting — console, markdown, and the regression verdict.

The console output is the product. If a run's result isn't legible in five
seconds on a tired evening, the harness won't get run, and an eval harness
nobody runs is worse than none because it implies coverage that isn't there.
"""
from __future__ import annotations

from rich.console import Console
from rich.table import Table
from rich.text import Text

from .models import SuiteResult
from .store import Comparison

console = Console()


def _score_style(value: float) -> str:
    if value >= 0.9:
        return "green"
    if value >= 0.6:
        return "yellow"
    return "red"


def print_suite_result(result: SuiteResult, comparison: Comparison | None = None) -> None:
    title = f"[bold]{result.suite}[/bold]  ·  {result.target}/{result.scorer}  ·  {result.target_version}"
    if result.dry_run:
        title += "  [yellow](DRY RUN — scores are meaningless)[/yellow]"
    console.print()
    console.print(title)

    # ---- per-category rollup -------------------------------------------------
    table = Table(box=None, pad_edge=False, header_style="dim")
    table.add_column("CATEGORY")
    table.add_column("PASS", justify="right")
    table.add_column("MEAN", justify="right")
    table.add_column("", width=14)

    for category, rows in sorted(result.by_category().items()):
        passed = sum(1 for r in rows if r.passed)
        mean = sum(r.score for r in rows) / len(rows)
        bar = "█" * round(mean * 10) + "·" * (10 - round(mean * 10))
        table.add_row(
            category,
            f"{passed}/{len(rows)}",
            f"{mean:.2f}",
            Text(bar, style=_score_style(mean)),
        )
    console.print(table)

    # ---- headline ------------------------------------------------------------
    passed_n = sum(1 for r in result.scored if r.passed)
    total_n = len(result.scored)
    console.print(
        f"\n  [bold]{passed_n}/{total_n}[/bold] passed"
        f"   mean [bold {_score_style(result.mean_score)}]{result.mean_score:.2f}[/]"
        f"   {result.elapsed_sec}s"
        + (f"   {result.total_tokens:,} tokens" if result.total_tokens else "   $0 (local/subscription)")
    )

    # ---- failures ------------------------------------------------------------
    if result.failures:
        console.print("\n  [red]FAILURES[/red]")
        for r in result.failures:
            console.print(f"    [red]✗[/red] {r.case_id}  [dim]({r.category}, {r.score:.2f})[/dim]")
            console.print(f"      {r.reason}")

    if result.errored:
        console.print("\n  [bold red]ERRORS[/bold red] [dim](target or scorer failed — not scored)[/dim]")
        for r in result.errored:
            console.print(f"    [red]![/red] {r.case_id}: {r.error}")

    skipped = [r for r in result.results if r.skipped]
    if skipped:
        console.print(f"\n  [dim]{len(skipped)} skipped[/dim]")

    # ---- regression verdict --------------------------------------------------
    if comparison is not None:
        console.print()
        if comparison.regressed:
            console.print(f"  [bold red]REGRESSION[/bold red] — {len(comparison.regressed)} case(s) went pass → fail:")
            for cid in comparison.regressed:
                console.print(f"    [red]▼[/red] {cid}")
        if comparison.fixed:
            console.print(f"  [green]FIXED[/green] — {len(comparison.fixed)}: {', '.join(comparison.fixed[:6])}")
        if comparison.new_cases:
            console.print(f"  [dim]new cases: {', '.join(comparison.new_cases[:6])}[/dim]")
        if comparison.removed_cases:
            console.print(f"  [yellow]removed from suite: {', '.join(comparison.removed_cases[:6])}[/yellow]")

        delta = comparison.score_delta
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "=")
        style = "green" if delta > 0 else ("red" if delta < 0 else "dim")
        console.print(f"  vs baseline: [{style}]{arrow} {delta:+.3f} mean score[/{style}]")
        if comparison.is_clean:
            console.print("  [dim]no change vs baseline[/dim]")
    else:
        console.print("\n  [dim]no baseline — run `evalforge baseline <suite>` to set one[/dim]")


def print_summary(rows: list[tuple[str, SuiteResult, bool]]) -> None:
    """One line per suite when several ran."""
    console.print("\n[bold]SUMMARY[/bold]")
    table = Table(box=None, pad_edge=False, header_style="dim")
    table.add_column("SUITE")
    table.add_column("PASS", justify="right")
    table.add_column("MEAN", justify="right")
    table.add_column("VERDICT")
    for name, result, ok in rows:
        passed = sum(1 for r in result.scored if r.passed)
        table.add_row(
            name,
            f"{passed}/{len(result.scored)}",
            f"{result.mean_score:.2f}",
            Text("PASS", style="green") if ok else Text("FAIL", style="bold red"),
        )
    console.print(table)


def to_markdown(result: SuiteResult, comparison: Comparison | None = None) -> str:
    """Markdown report — for a PR body, so review sees the evidence."""
    lines = [
        f"## Eval: {result.suite}",
        "",
        f"- **Target:** `{result.target}` ({result.target_version})",
        f"- **Scorer:** `{result.scorer}`",
        f"- **Run:** {result.started_at} ({result.elapsed_sec}s)",
        "",
        "| Category | Pass | Mean |",
        "|---|---|---|",
    ]
    for category, rows in sorted(result.by_category().items()):
        passed = sum(1 for r in rows if r.passed)
        mean = sum(r.score for r in rows) / len(rows)
        lines.append(f"| {category} | {passed}/{len(rows)} | {mean:.2f} |")

    passed_n = sum(1 for r in result.scored if r.passed)
    lines += ["", f"**{passed_n}/{len(result.scored)} passed · mean {result.mean_score:.2f}**", ""]

    if result.failures:
        lines += ["### Failures", ""]
        lines += [f"- `{r.case_id}` ({r.score:.2f}) — {r.reason}" for r in result.failures]
        lines.append("")
    if result.errored:
        lines += ["### Errors", ""]
        lines += [f"- `{r.case_id}` — {r.error}" for r in result.errored]
        lines.append("")
    if comparison and comparison.regressed:
        lines += ["### ⚠ Regressions vs baseline", ""]
        lines += [f"- `{cid}` pass → fail" for cid in comparison.regressed]
        lines.append("")
    return "\n".join(lines)
