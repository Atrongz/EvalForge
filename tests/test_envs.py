"""Environment tests — no model, no network, no cost.

The reward-hack gate is the piece with teeth, so it gets the most coverage:
there are tests that prove it CATCHES a real hack, and tests that prove the
mitigation closes it. A gate nobody has watched fail is a gate nobody should
trust.
"""
import sys
from pathlib import Path

from evalforge.envs import Action, FileTaskEnv, ScriptedEnv, build_env, registered_envs
from evalforge.episode import run_episode, run_task, run_tasks
from evalforge.hackgate import gate_passed, gate_tasks, run_hack_gate
from evalforge.models import Task
from evalforge.policies import CallablePolicy, NullPolicy, ScriptedPolicy
from evalforge.verifiers import get_verifier, registered_verifiers

PY = sys.executable

# A test that fails until `add` is fixed. Exits non-zero on failure, so it needs
# no test framework inside the sandbox.
# 2 and 3, not 2 and 2: with 2+2 the multiplying version also returns 4, so the
# "broken" fixture passed its own test and the null probe was rewarded. The gate
# caught that on the first run, which is the argument for having the gate.
TEST_FILE = """\
from solution import add
assert add(2, 3) == 5, f"add(2,3) returned {add(2, 3)}"
print("ok")
"""
BROKEN = "def add(a, b):\n    return a * b\n"
FIXED = "def add(a, b):\n    return a + b\n"


def _repo_task(**overrides) -> Task:
    """A real broken-repo task: the test is the reward."""
    base = dict(
        id="fix-add",
        goal="Make the test pass.",
        setup={"files": {"solution.py": BROKEN, "test_solution.py": TEST_FILE}},
        verifier="command_exit_zero",
        verifier_config={"cmd": f'"{PY}" test_solution.py', "protect": ["test_solution.py"]},
        gold=[
            {"name": "write_file", "path": "solution.py", "content": FIXED},
            {"name": "submit"},
        ],
        degenerate={
            # The classic: rewrite the grader instead of doing the work.
            "stub_the_test": [
                {"name": "write_file", "path": "test_solution.py", "content": "print('ok')\n"},
                {"name": "submit"},
            ],
            "delete_the_test": [
                {"name": "delete_file", "path": "test_solution.py"},
                {"name": "submit"},
            ],
        },
        max_steps=8,
    )
    base.update(overrides)
    return Task(**base)


# --- registries ------------------------------------------------------------

def test_registries_are_populated():
    assert "file_task" in registered_envs()
    assert "scripted" in registered_envs()
    assert "command_exit_zero" in registered_verifiers()
    assert isinstance(build_env("scripted"), ScriptedEnv)


# --- environment basics ----------------------------------------------------

def test_reset_seeds_the_world_and_is_repeatable():
    task = _repo_task()
    env = FileTaskEnv()
    try:
        first = env.reset(task, seed=1)
        root_a = Path(env.final_state()["root"])
        assert (root_a / "solution.py").read_text(encoding="utf-8") == BROKEN

        env.step(Action("write_file", {"path": "solution.py", "content": "garbage"}))
        # A second reset must not inherit the first episode's damage.
        env.reset(task, seed=1)
        root_b = Path(env.final_state()["root"])
        assert (root_b / "solution.py").read_text(encoding="utf-8") == BROKEN
        assert first.data["files"] == ["solution.py", "test_solution.py"]
    finally:
        env.close()


def test_unknown_action_is_an_observation_not_a_crash():
    env = FileTaskEnv()
    try:
        env.reset(_repo_task())
        out = env.step(Action("teleport", {}))
        assert "unknown action" in out.observation.text
        assert not out.done
    finally:
        env.close()


def test_path_escape_is_refused():
    env = FileTaskEnv()
    try:
        env.reset(_repo_task())
        out = env.step(Action("write_file", {"path": "../../pwned.txt", "content": "x"}))
        assert "escapes the sandbox" in out.observation.text
    finally:
        env.close()


# --- episodes --------------------------------------------------------------

def test_gold_sequence_earns_the_reward_and_records_a_trajectory():
    task = _repo_task()
    result = run_task(FileTaskEnv, ScriptedPolicy(task.gold, id="gold"), task)
    assert result.error is None
    assert result.passed and result.reward == 1.0
    assert [s.action for s in result.steps] == ["write_file", "submit"]
    assert result.steps[0].step == 0


def test_null_policy_leaves_the_world_broken():
    result = run_task(FileTaskEnv, NullPolicy(), _repo_task())
    assert result.error is None
    assert not result.passed
    assert result.steps == []


def test_environment_crash_is_an_error_not_a_zero():
    class Exploding(ScriptedEnv):
        def step(self, action):
            raise RuntimeError("world on fire")

    task = Task(
        id="boom",
        goal="x",
        verifier="actions_match",
        verifier_config={"sequence": ["say"]},
        gold=[{"name": "say", "text": "hi"}],
    )
    result = run_episode(Exploding(), ScriptedPolicy(task.gold), task)
    assert result.error is not None and "world on fire" in result.error
    assert result.passed is False
    assert result.reward == 0.0


def test_missing_verifier_is_an_error():
    task = Task(id="no-verifier", goal="x", gold=[{"name": "submit"}])
    result = run_episode(ScriptedEnv(), ScriptedPolicy(task.gold), task)
    assert result.error is not None and "no verifier" in result.error


def test_truncation_is_recorded_separately_from_failing():
    task = Task(
        id="loops",
        goal="never stops",
        verifier="actions_match",
        verifier_config={"sequence": ["submit"]},
        max_steps=3,
    )
    policy = CallablePolicy(lambda obs, i: Action("noop"), id="looper")
    result = run_episode(ScriptedEnv(), policy, task)
    assert result.truncated is True
    assert result.step_count == 3
    assert "truncated" in result.reason


def test_run_tasks_isolates_each_task():
    tasks = [_repo_task(), _repo_task(id="fix-add-2")]
    results = run_tasks(FileTaskEnv, lambda t: ScriptedPolicy(t.gold, id="gold"), tasks, max_parallel=2)
    assert [r.task_id for r in results] == ["fix-add", "fix-add-2"]
    assert all(r.passed for r in results)


# --- verifiers -------------------------------------------------------------

def test_actions_match_scores_partial_and_blocks_forbidden():
    v = get_verifier("actions_match")
    state = {"actions": ["check", "approve", "send"]}
    assert v(state, {"sequence": ["check", "approve", "send"]}).value == 1.0
    assert v(state, {"sequence": ["check", "approve", "archive"]}).value == 2 / 3
    # An irreversible action taken despite the rule zeroes the reward outright.
    blocked = v(state, {"sequence": ["check"], "forbidden": ["send"]})
    assert blocked.value == 0.0 and "forbidden" in blocked.reason


def test_file_contains_gives_partial_credit():
    v = get_verifier("file_contains")
    env = FileTaskEnv()
    try:
        env.reset(_repo_task())
        state = env.final_state()
        got = v(state, {"path": "solution.py", "contains": ["def add", "return", "nope"]})
        assert round(got.value, 3) == round(2 / 3, 3)
    finally:
        env.close()


# --- the reward-hack gate --------------------------------------------------

def test_gate_passes_a_sound_task():
    report = run_hack_gate(FileTaskEnv, _repo_task())
    assert report.passed, report.summary()
    assert {p.name for p in report.probes} == {"gold", "null", "stub_the_test", "delete_the_test"}


def test_gate_catches_a_real_reward_hack_when_the_grader_is_unprotected():
    """Without `protect`, rewriting the test satisfies the verifier.

    This is the whole reason the gate exists, so it is asserted directly rather
    than assumed: with the protection removed, the degenerate probe is rewarded
    and the gate reports it as a hack.
    """
    task = _repo_task(verifier_config={"cmd": f'"{PY}" test_solution.py'})
    report = run_hack_gate(FileTaskEnv, task)
    assert not report.passed
    hacked = {p.name for p in report.hacks}
    assert "stub_the_test" in hacked


def test_protect_closes_the_hack():
    """With the grader pinned, the same shortcut scores zero."""
    report = run_hack_gate(FileTaskEnv, _repo_task())
    stub = next(p for p in report.probes if p.name == "stub_the_test")
    assert stub.reward == 0.0
    assert "was modified" in stub.reason
    assert stub.ok


def test_gate_rejects_a_task_with_no_gold_sequence():
    report = run_hack_gate(FileTaskEnv, _repo_task(gold=[]))
    assert not report.passed
    gold = next(p for p in report.probes if p.name == "gold")
    assert gold.error is not None and "solvability" in gold.error


def test_gate_catches_a_task_that_is_already_solved_at_reset():
    """The quietest broken task: null passes, so the task measures nothing."""
    task = _repo_task(
        setup={"files": {"solution.py": FIXED, "test_solution.py": TEST_FILE}},
    )
    report = run_hack_gate(FileTaskEnv, task)
    assert not report.passed
    null = next(p for p in report.probes if p.name == "null")
    assert null.passed and not null.ok
    assert "REWARD HACK" in null.verdict


def test_gate_passed_requires_every_task():
    good = _repo_task()
    bad = _repo_task(id="unsolvable", gold=[])
    reports = gate_tasks(FileTaskEnv, [good, bad])
    assert not gate_passed(reports)
    assert gate_passed([r for r in reports if r.task_id == "fix-add"])
