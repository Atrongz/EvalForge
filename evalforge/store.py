"""Persistence + regression comparison.

Two kinds of stored run:
  * results/runs/<suite>/<timestamp>.json — the history. Gitignored.
  * results/baselines/<suite>.json        — the committed reference a run is
    compared against. Checked in on purpose: the baseline is the claim you are
    making about how the system behaves, and it belongs in review.

Regression detection compares per-case, not just in aggregate. A run whose mean
score is flat but which flipped two cases from pass to fail is a regression, and
the aggregate would hide it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .models import SuiteResult


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in value).strip("-")


class ResultStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.runs_dir = self.root / "runs"
        self.baselines_dir = self.root / "baselines"

    def save_run(self, result: SuiteResult) -> Path:
        d = self.runs_dir / _slug(result.suite)
        d.mkdir(parents=True, exist_ok=True)
        stamp = result.started_at.replace(":", "").replace("-", "")
        path = d / f"{stamp}.json"
        path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        return path

    def baseline_path(self, suite: str) -> Path:
        return self.baselines_dir / f"{_slug(suite)}.json"

    def load_baseline(self, suite: str) -> SuiteResult | None:
        path = self.baseline_path(suite)
        if not path.exists():
            return None
        return SuiteResult(**json.loads(path.read_text(encoding="utf-8")))

    def save_baseline(self, result: SuiteResult) -> Path:
        self.baselines_dir.mkdir(parents=True, exist_ok=True)
        path = self.baseline_path(result.suite)
        path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        return path

    def latest_run(self, suite: str) -> SuiteResult | None:
        d = self.runs_dir / _slug(suite)
        if not d.exists():
            return None
        runs = sorted(d.glob("*.json"))
        if not runs:
            return None
        return SuiteResult(**json.loads(runs[-1].read_text(encoding="utf-8")))


@dataclass
class Comparison:
    """Per-case diff between a run and its baseline."""

    regressed: list[str] = field(default_factory=list)   # passed -> failed
    fixed: list[str] = field(default_factory=list)       # failed -> passed
    new_cases: list[str] = field(default_factory=list)
    removed_cases: list[str] = field(default_factory=list)
    score_delta: float = 0.0
    pass_rate_delta: float = 0.0

    @property
    def has_regression(self) -> bool:
        return bool(self.regressed)

    @property
    def is_clean(self) -> bool:
        return not (self.regressed or self.fixed or self.new_cases or self.removed_cases)


def compare(current: SuiteResult, baseline: SuiteResult | None) -> Comparison | None:
    if baseline is None:
        return None

    cur = {r.case_id: r for r in current.results if not r.skipped}
    base = {r.case_id: r for r in baseline.results if not r.skipped}

    shared = cur.keys() & base.keys()
    return Comparison(
        regressed=sorted(cid for cid in shared if base[cid].passed and not cur[cid].passed),
        fixed=sorted(cid for cid in shared if not base[cid].passed and cur[cid].passed),
        new_cases=sorted(cur.keys() - base.keys()),
        removed_cases=sorted(base.keys() - cur.keys()),
        score_delta=round(current.mean_score - baseline.mean_score, 4),
        pass_rate_delta=round(current.pass_rate - baseline.pass_rate, 4),
    )
