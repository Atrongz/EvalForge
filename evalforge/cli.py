"""EvalForge CLI.

    evalforge list                       # what suites and plugins exist
    evalforge run <suite>                # run one suite (or --all)
    evalforge run <suite> --dry-run      # walk everything, call nothing
    evalforge baseline <suite>           # promote the last run to the baseline
    evalforge gate <env-suite>           # prove tasks are not reward-hackable
    evalforge report <suite> --markdown  # emit a PR-ready report

Exit codes: 0 pass · 1 suite gate failed · 2 regression vs baseline · 3 usage
error. Distinct codes so CI can treat "got worse" differently from "was already
below threshold".
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .envs import build_env, registered_envs
from .hackgate import gate_passed, gate_tasks
from .models import discover_env_suites, discover_suites, load_env_suite, load_suite
from .report import console, print_suite_result, print_summary, to_markdown
from .runner import run_suite, suite_passed
from .scorers import registered_scorers
from .store import ResultStore, compare
from .targets import registered_targets
from .verifiers import registered_verifiers

EXIT_OK, EXIT_FAILED, EXIT_REGRESSION, EXIT_USAGE = 0, 1, 2, 3


def _paths(args: argparse.Namespace) -> tuple[Path, ResultStore]:
    root = Path(args.root).resolve()
    return root / "suites", ResultStore(root / "results")


def _resolve(suites_dir: Path, name: str | None, run_all: bool) -> list[Path]:
    found = discover_suites(suites_dir)
    if run_all:
        return found
    if not name:
        return []
    matches = [p for p in found if p.name == name]
    if not matches:
        available = ", ".join(p.name for p in found) or "(none)"
        console.print(f"[red]no suite named {name!r}[/red]. Available: {available}")
    return matches


def cmd_list(args: argparse.Namespace) -> int:
    suites_dir, _ = _paths(args)
    console.print("\n[bold]SUITES[/bold]")
    found = discover_suites(suites_dir)
    if not found:
        console.print(f"  [dim]none found under {suites_dir}[/dim]")
    for path in found:
        try:
            config, cases = load_suite(path)
            active = sum(1 for c in cases if not c.skip)
            console.print(
                f"  [bold]{path.name}[/bold]  [dim]{active} cases · "
                f"{config.target}/{config.scorer}[/dim]\n    {config.description}"
            )
        except Exception as exc:
            console.print(f"  [red]{path.name} — failed to load: {exc}[/red]")
    console.print("\n[bold]ENVIRONMENTS[/bold]")
    env_suites = discover_env_suites(Path(args.root).resolve() / "envs")
    if not env_suites:
        console.print("  [dim]no environment suites found under envs/[/dim]")
    for path in env_suites:
        try:
            config, tasks = load_env_suite(path)
            active = sum(1 for t in tasks if not t.skip)
            console.print(
                f"  [bold]{path.name}[/bold]  [dim]{active} tasks · {config.env}[/dim]"
                f"\n    {config.description}"
            )
        except Exception as exc:
            console.print(f"  [red]{path.name} — failed to load: {exc}[/red]")

    console.print(f"\n[bold]TARGETS[/bold]\n  {', '.join(registered_targets())}")
    console.print(f"\n[bold]SCORERS[/bold]\n  {', '.join(registered_scorers())}")
    console.print(f"\n[bold]ENVS[/bold]\n  {', '.join(registered_envs())}")
    console.print(f"\n[bold]VERIFIERS[/bold]\n  {', '.join(registered_verifiers())}\n")
    return EXIT_OK


def cmd_gate(args: argparse.Namespace) -> int:
    """Run the reward-hack gate over an environment suite.

    This runs before anything expensive and costs nothing but CPU, so it is the
    cheapest possible place to discover that a task measures nothing.
    """
    root = Path(args.root).resolve() / "envs"
    found = discover_env_suites(root)
    if args.all:
        selected = found
    else:
        selected = [p for p in found if p.name == args.suite]
        if not selected:
            available = ", ".join(p.name for p in found) or "(none)"
            console.print(f"[red]no environment suite named {args.suite!r}[/red]. Available: {available}")
            return EXIT_USAGE

    all_reports = []
    for path in selected:
        config, tasks = load_env_suite(path)
        console.print(f"\n[bold]{config.name}[/bold]  [dim]{config.env}[/dim]")
        reports = gate_tasks(
            lambda c=config: build_env(c.env, c.env_config),
            tasks,
            pass_threshold=config.pass_threshold,
        )
        all_reports.extend(reports)
        for report in reports:
            mark = "[green]PASS[/green]" if report.passed else "[red]FAIL[/red]"
            console.print(f"  {mark}  {report.task_id}")
            for probe in report.probes:
                colour = "green" if probe.ok else "red"
                console.print(
                    f"        [{colour}]{probe.kind:<11}[/{colour}] {probe.name:<18} "
                    f"reward={probe.reward:<5} [dim]{probe.verdict}[/dim]"
                )

    hacks = [p for r in all_reports for p in r.hacks]
    if hacks:
        console.print(
            f"\n[bold red]{len(hacks)} reward hack(s)[/bold red] — a shortcut was rewarded. "
            "Fix the verifier, not the task.\n"
        )
    ok = gate_passed(all_reports)
    console.print(
        f"[bold]{'gate passed' if ok else 'gate failed'}[/bold] "
        f"· {sum(1 for r in all_reports if r.passed)}/{len(all_reports)} tasks sound\n"
    )
    return EXIT_OK if ok else EXIT_FAILED


def cmd_run(args: argparse.Namespace) -> int:
    suites_dir, store = _paths(args)
    selected = _resolve(suites_dir, args.suite, args.all)
    if not selected:
        return EXIT_USAGE

    summary: list[tuple[str, object, bool]] = []
    exit_code = EXIT_OK

    for path in selected:
        config, cases = load_suite(path)
        result = run_suite(
            config, cases, dry_run=args.dry_run, only_category=args.category
        )

        comparison = None if args.dry_run else compare(result, store.load_baseline(config.name))
        print_suite_result(result, comparison)

        if not args.dry_run:
            store.save_run(result)

        ok = suite_passed(config, result)
        summary.append((path.name, result, ok))

        if comparison and comparison.has_regression:
            exit_code = max(exit_code, EXIT_REGRESSION)
        elif not ok:
            exit_code = max(exit_code, EXIT_FAILED)

    if len(summary) > 1:
        print_summary(summary)  # type: ignore[arg-type]

    if args.dry_run:
        console.print("\n[yellow]dry run — nothing was invoked and nothing was saved[/yellow]")
        return EXIT_OK
    return exit_code


def cmd_baseline(args: argparse.Namespace) -> int:
    suites_dir, store = _paths(args)
    selected = _resolve(suites_dir, args.suite, args.all)
    if not selected:
        return EXIT_USAGE

    for path in selected:
        config, _ = load_suite(path)
        latest = store.latest_run(config.name)
        if latest is None:
            console.print(f"[yellow]{config.name}: no run to promote — run it first[/yellow]")
            continue
        if latest.dry_run:
            console.print(f"[red]{config.name}: last run was a dry run; refusing to baseline it[/red]")
            continue
        written = store.save_baseline(latest)
        console.print(
            f"[green]baseline set[/green] {config.name} "
            f"→ {written}  [dim](mean {latest.mean_score:.2f}, {latest.started_at})[/dim]"
        )
    return EXIT_OK


def cmd_report(args: argparse.Namespace) -> int:
    suites_dir, store = _paths(args)
    selected = _resolve(suites_dir, args.suite, args.all)
    if not selected:
        return EXIT_USAGE

    for path in selected:
        config, _ = load_suite(path)
        latest = store.latest_run(config.name)
        if latest is None:
            console.print(f"[yellow]{config.name}: no runs yet[/yellow]")
            continue
        comparison = compare(latest, store.load_baseline(config.name))
        if args.markdown:
            print(to_markdown(latest, comparison))
        else:
            print_suite_result(latest, comparison)
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evalforge",
        description="A general evaluation harness for LLM systems.",
    )
    parser.add_argument(
        "--root", default=".", help="project root holding suites/ and results/ (default: .)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list", help="list suites, targets and scorers")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("run", help="run a suite")
    p.add_argument("suite", nargs="?")
    p.add_argument("--all", action="store_true", help="run every suite")
    p.add_argument("--dry-run", action="store_true", help="walk everything, invoke nothing")
    p.add_argument("--category", help="only cases in this category")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("gate", help="run the reward-hack gate over an environment suite")
    p.add_argument("suite", nargs="?")
    p.add_argument("--all", action="store_true", help="gate every environment suite")
    p.set_defaults(func=cmd_gate)

    p = sub.add_parser("baseline", help="promote the latest run to the baseline")
    p.add_argument("suite", nargs="?")
    p.add_argument("--all", action="store_true")
    p.set_defaults(func=cmd_baseline)

    p = sub.add_parser("report", help="reprint the latest run")
    p.add_argument("suite", nargs="?")
    p.add_argument("--all", action="store_true")
    p.add_argument("--markdown", action="store_true", help="emit markdown for a PR body")
    p.set_defaults(func=cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        console.print("\n[yellow]interrupted[/yellow]")
        return EXIT_USAGE
    except Exception as exc:
        console.print(f"[bold red]{type(exc).__name__}:[/bold red] {exc}")
        return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
