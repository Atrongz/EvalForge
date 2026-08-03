"""The runner — execute a suite's cases against a target and score them.

Design rules this obeys:
  * A target crash is an ERROR, never a zero. Zeros mean "the system answered
    and the answer was wrong"; conflating the two hides outages as regressions.
  * Concurrency is bounded and configurable per suite. Some targets are local
    and cheap, some are rate-limited.
  * --dry-run does the full traversal with no external call, so wiring bugs
    surface before anything is spent.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Callable

from .models import Case, CaseResult, SuiteConfig, SuiteResult
from .scorers import get_scorer
from .targets import EchoTarget, Target, build_target

ProgressFn = Callable[[CaseResult], None]


def run_suite(
    config: SuiteConfig,
    cases: list[Case],
    *,
    dry_run: bool = False,
    only_category: str | None = None,
    on_result: ProgressFn | None = None,
) -> SuiteResult:
    """Run every case and return the aggregate. Never raises for a single bad
    case — that becomes a CaseResult with `error` set."""
    started = datetime.now(timezone.utc)
    t0 = time.monotonic()

    if dry_run:
        # Echo returns the input, so scores are meaningless — but every case is
        # loaded, every scorer is constructed, and every code path is walked.
        target: Target = EchoTarget()
    else:
        target = build_target(config.target, config.target_config)

    selected = [
        c for c in cases
        if not c.skip and (only_category is None or c.category == only_category)
    ]
    skipped = [
        c for c in cases
        if c.skip or (only_category is not None and c.category != only_category)
    ]

    def run_one(case: Case) -> CaseResult:
        started_case = time.monotonic()
        try:
            response = target.invoke(case.input)
        except Exception as exc:  # target failure — not a score of zero
            return CaseResult(
                case_id=case.id,
                category=case.category,
                error=f"{type(exc).__name__}: {exc}"[:400],
                elapsed_sec=round(time.monotonic() - started_case, 3),
            )

        scorer_name = case.scorer or config.scorer
        # `_request` carries the original input so scorers that need to judge
        # output *relative to what was asked* (e.g. refusal) can see both sides.
        # Underscore-prefixed so it never collides with a user config key.
        scorer_config = {
            **config.scorer_config,
            **case.scorer_config,
            "_request": case.input,
        }
        try:
            score = get_scorer(scorer_name)(response.output, case.expected, scorer_config)
        except Exception as exc:  # scorer failure is also an error, not a zero
            return CaseResult(
                case_id=case.id,
                category=case.category,
                output=response.output,
                error=f"scorer {scorer_name}: {type(exc).__name__}: {exc}"[:400],
                elapsed_sec=round(time.monotonic() - started_case, 3),
            )

        return CaseResult(
            case_id=case.id,
            category=case.category,
            output=response.output,
            score=round(score.value, 4),
            passed=score.value >= config.pass_threshold,
            reason=score.reason,
            elapsed_sec=round(time.monotonic() - started_case, 3),
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )

    results: list[CaseResult] = []
    workers = 1 if dry_run else max(1, config.max_parallel)
    if workers == 1:
        for case in selected:
            r = run_one(case)
            results.append(r)
            if on_result:
                on_result(r)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for r in pool.map(run_one, selected):
                results.append(r)
                if on_result:
                    on_result(r)

    results.extend(
        CaseResult(case_id=c.id, category=c.category, skipped=True) for c in skipped
    )
    # Stable ordering so two runs diff cleanly regardless of completion order.
    results.sort(key=lambda r: r.case_id)

    return SuiteResult(
        suite=config.name,
        target=config.target,
        scorer=config.scorer,
        target_version="dry-run" if dry_run else target.version(),
        started_at=started.isoformat(timespec="seconds"),
        elapsed_sec=round(time.monotonic() - t0, 2),
        dry_run=dry_run,
        results=results,
    )


def suite_passed(config: SuiteConfig, result: SuiteResult) -> bool:
    """Suite-level gate. Any errored case fails the suite outright — an
    unreachable target must never read as a green run."""
    if result.errored:
        return False
    return result.pass_rate >= config.required_pass_rate
