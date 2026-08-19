"""Offline tests for the repo-mining pipeline — no git, no network, no venv.

The end-to-end proof for RepoEnv is `evalforge gate sentinel-rag-chunking`, which
provisions an interpreter and needs the network. These cover the pure logic that
would otherwise only ever be exercised by that slow path.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import mine_tasks
import resolve_deps

from evalforge.envs import registered_envs
from evalforge.verifiers import get_verifier


# --- the miner's file classification ---------------------------------------

def test_test_files_are_recognised_across_layouts():
    for path in ["tests/test_rag.py", "src/foo_test.py", "app/Button.test.tsx", "a/spec/x.spec.js"]:
        assert mine_tasks.is_test(path), path
    for path in ["src/sentinel/rag/ingest.py", "app/Button.tsx"]:
        assert not mine_tasks.is_test(path), path


def test_source_excludes_tests_and_non_code():
    assert mine_tasks.is_source("src/sentinel/rag/ingest.py")
    assert not mine_tasks.is_source("tests/test_rag.py")   # a test is not source
    assert not mine_tasks.is_source("README.md")
    assert not mine_tasks.is_source("requirements.txt")


# --- module name to distribution name ---------------------------------------

def test_distribution_name_mapping():
    assert resolve_deps.distribution_for("pydantic_settings") == "pydantic-settings"
    assert resolve_deps.distribution_for("tiktoken") == "tiktoken"
    # Aliases exist because the mechanical rule is wrong for these.
    assert resolve_deps.distribution_for("yaml") == "pyyaml"
    assert resolve_deps.distribution_for("sklearn") == "scikit-learn"
    # Submodules resolve to their top-level distribution.
    assert resolve_deps.distribution_for("sentinel.rag.ingest") == "sentinel"


def test_missing_module_is_parsed_from_a_real_traceback():
    trace = (
        "src\\sentinel\\config.py:15: in <module>\n"
        "    from pydantic_settings import BaseSettings\n"
        "E   ModuleNotFoundError: No module named 'pydantic_settings'\n"
    )
    found = resolve_deps.MISSING.search(trace)
    assert found and found.group(1) == "pydantic_settings"


# --- verifier wiring --------------------------------------------------------

def test_command_verifier_substitutes_interpreter_and_root(tmp_path):
    v = get_verifier("command_exit_zero")
    state = {"root": str(tmp_path), "python": sys.executable}
    got = v(state, {"cmd": '"{python}" -c "raise SystemExit(0)"'})
    assert got.value == 1.0


def test_command_verifier_applies_declared_environment(tmp_path):
    """An env declared in terminal state must reach the graded subprocess."""
    v = get_verifier("command_exit_zero")
    state = {"root": str(tmp_path), "python": sys.executable, "env": {"EF_PROBE": "present"}}
    cmd = '"{python}" -c "import os,sys; sys.exit(0 if os.environ.get(\'EF_PROBE\')==\'present\' else 1)"'
    assert v(state, {"cmd": cmd}).value == 1.0
    # Without the declaration the same command must fail, or the test proves nothing.
    bare = {"root": str(tmp_path), "python": sys.executable}
    assert v(bare, {"cmd": cmd}).value == 0.0


def test_braces_in_a_command_are_not_a_format_crash(tmp_path):
    v = get_verifier("command_exit_zero")
    state = {"root": str(tmp_path), "python": sys.executable}
    got = v(state, {"cmd": '"{python}" -c "d={1:2}; raise SystemExit(0)"'})
    assert got.value == 1.0


def test_repo_env_is_registered():
    assert "repo" in registered_envs()
