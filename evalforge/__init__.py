"""EvalForge — a general evaluation harness for LLM systems.

Targets are pluggable, scorers are pluggable, and the runner knows about
neither. Adding a system to evaluate is one config file; adding a new kind of
system is one adapter and one registry line.
"""
from .models import Case, CaseResult, SuiteConfig, SuiteResult, discover_suites, load_suite
from .runner import run_suite, suite_passed
from .scorers import Score, get_scorer, registered_scorers
from .store import Comparison, ResultStore, compare
from .targets import TargetResponse, build_target, registered_targets

__version__ = "0.1.0"

__all__ = [
    "Case", "CaseResult", "SuiteConfig", "SuiteResult",
    "load_suite", "discover_suites",
    "run_suite", "suite_passed",
    "Score", "get_scorer", "registered_scorers",
    "TargetResponse", "build_target", "registered_targets",
    "ResultStore", "Comparison", "compare",
    "__version__",
]
