"""The reward-hack gate — prove a task measures what it claims before it counts.

A verifier is only as good as the laziest thing that satisfies it. Once a model
is trained against a reward it will find that laziest thing, and it will find it
reliably, because that is the entire mechanism of reinforcement learning. So a
task does not enter a suite on the author's confidence. It enters by surviving
three probes, all run through the ordinary episode runner with no shortcuts:

  gold        the known-good sequence          must PASS
  null        do nothing at all                must FAIL
  degenerate  named shortcuts, one per probe   must FAIL

Null catches the most common broken task: one that was already solved at reset
and therefore measures nothing while quietly inflating every score. The
degenerate probes catch the second most common: a verifier the agent can satisfy
without doing the work — deleting the test, stubbing the assertion, writing the
expected value straight into the output.

A task with no gold sequence fails the gate. "Nobody has demonstrated this is
solvable" and "this is a hard task" look identical from the outside, and only
one of them is worth shipping.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .episode import EnvFactory, run_task
from .models import EpisodeResult, Task
from .policies import NullPolicy, ScriptedPolicy


@dataclass
class ProbeResult:
    """One probe and whether it did what a healthy task requires."""

    name: str
    kind: str                    # gold | null | degenerate
    should_pass: bool
    reward: float = 0.0
    passed: bool = False         # did the probe clear the threshold
    error: str | None = None
    reason: str = ""

    @property
    def ok(self) -> bool:
        """The probe behaved as a sound task requires. An errored probe is never
        ok — the same rule as an errored case failing a suite outright."""
        if self.error:
            return False
        return self.passed == self.should_pass

    @property
    def verdict(self) -> str:
        if self.error:
            return f"ERROR {self.error}"
        if self.ok:
            return "ok"
        if self.should_pass:
            return f"gold did not pass (reward {self.reward}) — task may be unsolvable as written"
        return f"REWARD HACK: {self.name} reached {self.reward} without doing the work"


@dataclass
class HackGateReport:
    task_id: str
    probes: list[ProbeResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.probes) and all(p.ok for p in self.probes)

    @property
    def failures(self) -> list[ProbeResult]:
        return [p for p in self.probes if not p.ok]

    @property
    def hacks(self) -> list[ProbeResult]:
        """Probes that were rewarded and should not have been. These are the
        findings — a failing gold is a broken task, but a passing degenerate is
        a broken *reward*, and the reward is what gets trained against."""
        return [p for p in self.failures if not p.should_pass and not p.error]

    def summary(self) -> str:
        if self.passed:
            return f"{self.task_id}: gate passed ({len(self.probes)} probes)"
        parts = [f"{p.name}: {p.verdict}" for p in self.failures]
        return f"{self.task_id}: GATE FAILED — " + "; ".join(parts)


def run_hack_gate(
    env_factory: EnvFactory,
    task: Task,
    *,
    seed: int = 0,
    pass_threshold: float = 1.0,
) -> HackGateReport:
    """Run every probe for one task. Never raises — a probe that blows up is a
    failed probe, and a failed probe fails the gate."""
    report = HackGateReport(task_id=task.id)

    def probe(name: str, kind: str, should_pass: bool, policy) -> None:
        try:
            result: EpisodeResult = run_task(
                env_factory, policy, task, seed=seed, pass_threshold=pass_threshold
            )
        except Exception as exc:  # defence in depth; run_task already traps
            report.probes.append(
                ProbeResult(name, kind, should_pass, error=f"{type(exc).__name__}: {exc}"[:300])
            )
            return
        report.probes.append(
            ProbeResult(
                name=name,
                kind=kind,
                should_pass=should_pass,
                reward=result.reward,
                passed=result.passed,
                error=result.error,
                reason=result.reason,
            )
        )

    if task.gold:
        probe("gold", "gold", True, ScriptedPolicy(task.gold, id="gold"))
    else:
        report.probes.append(
            ProbeResult(
                "gold",
                "gold",
                True,
                error="task has no gold sequence — solvability was never demonstrated",
            )
        )

    probe("null", "null", False, NullPolicy())

    for name, actions in (task.degenerate or {}).items():
        probe(name, "degenerate", False, ScriptedPolicy(actions, id=f"degenerate:{name}"))

    return report


def gate_tasks(
    env_factory: EnvFactory,
    tasks: list[Task],
    *,
    seed: int = 0,
    pass_threshold: float = 1.0,
) -> list[HackGateReport]:
    """Gate every non-skipped task. Sequential on purpose: the gate runs rarely,
    it runs before anything expensive, and a flaky parallel gate would be worse
    than a slow one."""
    return [
        run_hack_gate(env_factory, t, seed=seed, pass_threshold=pass_threshold)
        for t in tasks
        if not t.skip
    ]


def gate_passed(reports: list[HackGateReport]) -> bool:
    """Suite-level gate. One unsound task fails the set — a suite is a claim
    about coverage, and a task that measures nothing makes the claim false."""
    return bool(reports) and all(r.passed for r in reports)
