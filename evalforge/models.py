"""Core data model.

A Suite is a directory: suite.yaml (config) + cases/*.yaml (the golden set).
A Case is one input with a known-correct expectation.
A CaseResult is what came back plus how it scored.

Nothing here talks to a model or a filesystem beyond loading — keeping the
model layer inert makes the whole harness testable without network access.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class Case(BaseModel):
    """One golden example: an input, and what a correct system does with it."""

    id: str
    input: Any
    expected: Any = None
    # Free-form grouping — lets you slice results ("2 bypasses, both indirect-injection").
    category: str = "default"
    # Why this case exists / how the ground truth was established. Load-bearing:
    # an unexplained golden value is a golden value nobody will trust in six months.
    notes: str = ""
    # Per-case scorer override. Falls back to the suite's scorer.
    scorer: str | None = None
    scorer_config: dict[str, Any] = Field(default_factory=dict)
    # Cases can be parked without deleting them (keeps the id stable in history).
    skip: bool = False

    @field_validator("id")
    @classmethod
    def _id_is_slug(cls, v: str) -> str:
        if not v or " " in v:
            raise ValueError(f"case id must be a non-empty slug without spaces: {v!r}")
        return v


class SuiteConfig(BaseModel):
    """suite.yaml — what to run, against what, scored how."""

    name: str
    description: str = ""
    # Which registered target adapter to invoke (see targets.py).
    target: str
    target_config: dict[str, Any] = Field(default_factory=dict)
    # Default scorer for cases that don't override it (see scorers.py).
    scorer: str
    scorer_config: dict[str, Any] = Field(default_factory=dict)
    # A case scoring below this is a failure. Scorers all normalize to 0.0-1.0.
    pass_threshold: float = 1.0
    # Suite-level gate: fraction of cases that must pass for the suite to pass.
    required_pass_rate: float = 1.0
    # Concurrency for target invocation. 1 = strictly sequential.
    max_parallel: int = 4


class CaseResult(BaseModel):
    """What happened for one case."""

    case_id: str
    category: str
    skipped: bool = False
    # Raw output from the target, before scoring.
    output: Any = None
    score: float = 0.0
    passed: bool = False
    # Human-readable reason the scorer gave this score. Shown on failures.
    reason: str = ""
    error: str | None = None
    elapsed_sec: float = 0.0
    # Populated when the target reports usage. Drives the cost column.
    input_tokens: int = 0
    output_tokens: int = 0


class SuiteResult(BaseModel):
    """Aggregate for one suite run. This is what gets persisted and diffed."""

    suite: str
    target: str
    scorer: str
    # Identifies what was under test — model id, git sha, whatever the target reports.
    target_version: str = "unknown"
    started_at: str
    elapsed_sec: float = 0.0
    dry_run: bool = False
    results: list[CaseResult] = Field(default_factory=list)

    @property
    def scored(self) -> list[CaseResult]:
        return [r for r in self.results if not r.skipped and r.error is None]

    @property
    def errored(self) -> list[CaseResult]:
        return [r for r in self.results if r.error]

    @property
    def failures(self) -> list[CaseResult]:
        return [r for r in self.scored if not r.passed]

    @property
    def pass_rate(self) -> float:
        if not self.scored:
            return 0.0
        return sum(1 for r in self.scored if r.passed) / len(self.scored)

    @property
    def mean_score(self) -> float:
        if not self.scored:
            return 0.0
        return sum(r.score for r in self.scored) / len(self.scored)

    @property
    def total_tokens(self) -> int:
        return sum(r.input_tokens + r.output_tokens for r in self.results)

    def by_category(self) -> dict[str, list[CaseResult]]:
        out: dict[str, list[CaseResult]] = {}
        for r in self.scored:
            out.setdefault(r.category, []).append(r)
        return out


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_suite(suite_dir: Path) -> tuple[SuiteConfig, list[Case]]:
    """Load suite.yaml + every cases/*.yaml|json under a suite directory.

    Case ids must be unique within a suite — a duplicate id would silently
    overwrite history in the results store, so it's a hard error.
    """
    suite_dir = Path(suite_dir)
    config_path = suite_dir / "suite.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"no suite.yaml in {suite_dir}")

    config = SuiteConfig(**yaml.safe_load(config_path.read_text(encoding="utf-8")))

    cases_dir = suite_dir / "cases"
    if not cases_dir.exists():
        raise FileNotFoundError(f"no cases/ directory in {suite_dir}")

    cases: list[Case] = []
    seen: set[str] = set()
    for path in sorted(cases_dir.iterdir()):
        if path.suffix.lower() not in {".yaml", ".yml", ".json"}:
            continue
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        # A file may hold one case or a list of them.
        items = raw if isinstance(raw, list) else [raw]
        for item in items:
            case = Case(**item)
            if case.id in seen:
                raise ValueError(f"duplicate case id {case.id!r} in {suite_dir.name}")
            seen.add(case.id)
            cases.append(case)

    if not cases:
        raise ValueError(f"suite {config.name!r} has no cases")
    return config, cases


def discover_suites(root: Path) -> list[Path]:
    """Every immediate subdirectory of `root` containing a suite.yaml."""
    root = Path(root)
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir() if (p / "suite.yaml").exists())


# ---------------------------------------------------------------------------
# Environments: a Task is to an Environment what a Case is to a Target
# ---------------------------------------------------------------------------

class Task(BaseModel):
    """One goal, a starting world, and the verifier that decides success.

    A Case carries a prompt and a known-correct answer. A Task carries a goal
    and a known-correct *ending*, which is a different thing: many action
    sequences reach the same valid end state, so the grader looks at the world
    rather than at what the agent said about it.

    `gold` and `degenerate` exist for the reward-hack gate, not for training.
    A task whose gold sequence does not pass, or whose degenerate sequences do,
    is a broken task — see hackgate.py.
    """

    id: str
    goal: str
    # Environment-specific starting fixture. FileTaskEnv reads {files: {...}}.
    setup: dict[str, Any] = Field(default_factory=dict)
    verifier: str | None = None
    verifier_config: dict[str, Any] = Field(default_factory=dict)
    category: str = "default"
    notes: str = ""
    # The known-good action sequence. Required before a task may enter a suite:
    # a task nobody has demonstrated is solvable is a task that measures nothing.
    gold: list[dict[str, Any]] = Field(default_factory=list)
    # Named shortcuts that must NOT be rewarded — deleting the test, stubbing the
    # assertion, writing the expected value straight into the output.
    degenerate: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    # Hard stop. An agent that loops forever must end the episode, not the run.
    max_steps: int = 20
    skip: bool = False

    @field_validator("id")
    @classmethod
    def _id_is_slug(cls, v: str) -> str:
        if not v or " " in v:
            raise ValueError(f"task id must be a non-empty slug without spaces: {v!r}")
        return v


class TrajectoryStep(BaseModel):
    """One (observation, action, reward) triple.

    The trajectory is not debug output — it is the artifact a buyer trains on,
    so it is recorded for every episode whether or not anyone reads it.
    """

    step: int
    observation: str
    action: str
    action_args: dict[str, Any] = Field(default_factory=dict)
    reward: float = 0.0
    done: bool = False


class EpisodeResult(BaseModel):
    """What happened across one full attempt at one task."""

    task_id: str
    category: str
    policy: str
    env: str
    env_version: str = "unknown"
    steps: list[TrajectoryStep] = Field(default_factory=list)
    # Terminal reward from the verifier. Shaped step rewards sum into shaped_reward.
    reward: float = 0.0
    shaped_reward: float = 0.0
    passed: bool = False
    reason: str = ""
    # Set when the environment or verifier broke. An errored episode is never a
    # reward of zero — same rule as CaseResult, for the same reason.
    error: str | None = None
    truncated: bool = False
    elapsed_sec: float = 0.0

    @property
    def step_count(self) -> int:
        return len(self.steps)


class EnvSuiteConfig(BaseModel):
    """env.yaml — which world to build, and the bar a task must clear."""

    name: str
    description: str = ""
    # Which registered environment adapter to build (see envs.py).
    env: str
    env_config: dict[str, Any] = Field(default_factory=dict)
    # A task scoring below this did not succeed. Verifiers normalize to 0.0-1.0.
    pass_threshold: float = 1.0
    max_parallel: int = 4


def load_env_suite(suite_dir: Path) -> tuple[EnvSuiteConfig, list[Task]]:
    """Load env.yaml + every tasks/*.yaml under an environment suite directory."""
    suite_dir = Path(suite_dir)
    config_path = suite_dir / "env.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"no env.yaml in {suite_dir}")
    config = EnvSuiteConfig(**yaml.safe_load(config_path.read_text(encoding="utf-8")))
    return config, load_tasks(suite_dir / "tasks")


def discover_env_suites(root: Path) -> list[Path]:
    """Every immediate subdirectory of `root` containing an env.yaml."""
    root = Path(root)
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir() if (p / "env.yaml").exists())


def load_tasks(tasks_dir: Path) -> list[Task]:
    """Load every tasks/*.yaml|json under a directory.

    Same duplicate-id rule as cases: two tasks sharing an id would silently
    overwrite each other in the results store.
    """
    tasks_dir = Path(tasks_dir)
    if not tasks_dir.exists():
        raise FileNotFoundError(f"no tasks directory at {tasks_dir}")

    tasks: list[Task] = []
    seen: set[str] = set()
    for path in sorted(tasks_dir.iterdir()):
        if path.suffix.lower() not in {".yaml", ".yml", ".json"}:
            continue
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        items = raw if isinstance(raw, list) else [raw]
        for item in items:
            task = Task(**item)
            if task.id in seen:
                raise ValueError(f"duplicate task id {task.id!r} in {tasks_dir}")
            seen.add(task.id)
            tasks.append(task)

    if not tasks:
        raise ValueError(f"no tasks found in {tasks_dir}")
    return tasks
