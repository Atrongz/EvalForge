"""EvalForge — a general evaluation harness for LLM systems.

Targets are pluggable, scorers are pluggable, and the runner knows about
neither. Adding a system to evaluate is one config file; adding a new kind of
system is one adapter and one registry line.
"""
from .models import (
    Case, CaseResult, EpisodeResult, SuiteConfig, SuiteResult, Task,
    discover_env_suites, discover_suites, load_env_suite, load_suite,
)
from .runner import run_suite, suite_passed
from .scorers import Score, get_scorer, registered_scorers
from .store import Comparison, ResultStore, compare
from .targets import TargetResponse, build_target, registered_targets
from .envs import build_env, registered_envs
from .episode import run_episode, run_task, run_tasks
from .hackgate import gate_passed, gate_tasks, run_hack_gate
from .verifiers import Reward, get_verifier, registered_verifiers
from . import repo_env as _repo_env  # noqa: F401  — registers the "repo" adapter

__version__ = "0.2.0"

__all__ = [
    "Case", "CaseResult", "SuiteConfig", "SuiteResult",
    "load_suite", "discover_suites",
    "run_suite", "suite_passed",
    "Score", "get_scorer", "registered_scorers",
    "TargetResponse", "build_target", "registered_targets",
    "ResultStore", "Comparison", "compare",
    "Task", "EpisodeResult", "load_env_suite", "discover_env_suites",
    "build_env", "registered_envs",
    "run_episode", "run_task", "run_tasks",
    "run_hack_gate", "gate_tasks", "gate_passed",
    "Reward", "get_verifier", "registered_verifiers",
    "__version__",
]
