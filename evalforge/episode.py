"""The episode runner — drive one policy through one environment, then grade it.

Same design rules as runner.py, because they were right the first time:
  * An environment or verifier crash is an ERROR, never a reward of zero.
    Zero means "the agent acted and the world ended up wrong". Conflating the
    two turns an outage into a phantom regression and teaches people to ignore
    the numbers.
  * Hitting the step limit is `truncated`, recorded separately from failing.
    A policy that ran out of room and one that finished and was wrong are
    different findings with different fixes.
  * Every episode records its full trajectory, whether or not anyone reads it.
    The trajectory is the artifact, not the log.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from .envs import Action, Environment
from .models import EpisodeResult, Task, TrajectoryStep
from .policies import Policy
from .verifiers import Reward, get_verifier, protect_hashes

EnvFactory = Callable[[], Environment]
PolicyFactory = Callable[[Task], Policy]


def run_episode(
    env: Environment,
    policy: Policy,
    task: Task,
    *,
    seed: int = 0,
    pass_threshold: float = 1.0,
) -> EpisodeResult:
    """Run one attempt end to end. Never raises for a bad episode — that
    becomes an EpisodeResult with `error` set."""
    t0 = time.monotonic()
    result = EpisodeResult(
        task_id=task.id,
        category=task.category,
        policy=getattr(policy, "id", "unknown"),
        env=getattr(env, "id", "unknown"),
    )

    try:
        result.env_version = env.version()
    except Exception:
        result.env_version = "unknown"

    try:
        observation = env.reset(task, seed=seed)
    except Exception as exc:
        result.error = f"reset: {type(exc).__name__}: {exc}"[:400]
        result.elapsed_sec = round(time.monotonic() - t0, 3)
        return result

    # Snapshot the files the verifier grades against, before the policy can act.
    # Doing this after the episode would only detect tampering; doing it here
    # means a rewritten test is scored as the failure it is.
    verifier_config = dict(task.verifier_config)
    protect = verifier_config.pop("protect", None)
    if protect:
        try:
            root = (env.final_state() or {}).get("root")
            if root:
                verifier_config["_protected_hashes"] = protect_hashes(root, list(protect))
        except Exception as exc:
            result.error = f"protect: {type(exc).__name__}: {exc}"[:400]
            result.elapsed_sec = round(time.monotonic() - t0, 3)
            return result

    shaped = 0.0
    done = False
    try:
        for step_index in range(task.max_steps):
            action = policy.act(observation, step_index)
            if action is None:
                break
            if not isinstance(action, Action):
                action = Action.from_dict(action)  # type: ignore[arg-type]

            outcome = env.step(action)
            shaped += outcome.reward
            result.steps.append(
                TrajectoryStep(
                    step=step_index,
                    observation=str(outcome.observation.text)[:2000],
                    action=action.name,
                    action_args=dict(action.args),
                    reward=round(outcome.reward, 4),
                    done=outcome.done,
                )
            )
            observation = outcome.observation
            if outcome.done:
                done = True
                break
        else:
            # Loop exhausted without a submit or a None — out of room, not wrong.
            result.truncated = True
    except Exception as exc:
        result.error = f"step: {type(exc).__name__}: {exc}"[:400]
        result.elapsed_sec = round(time.monotonic() - t0, 3)
        return result

    result.shaped_reward = round(shaped, 4)

    if not task.verifier:
        result.error = f"task {task.id!r} has no verifier"
        result.elapsed_sec = round(time.monotonic() - t0, 3)
        return result

    try:
        reward: Reward = get_verifier(task.verifier)(env.final_state(), verifier_config).clamped()
    except Exception as exc:
        result.error = f"verifier {task.verifier}: {type(exc).__name__}: {exc}"[:400]
        result.elapsed_sec = round(time.monotonic() - t0, 3)
        return result

    result.reward = round(reward.value, 4)
    result.passed = reward.value >= pass_threshold
    result.reason = reward.reason
    if result.truncated and not done:
        result.reason = f"{reward.reason} (truncated at {task.max_steps} steps)"
    result.elapsed_sec = round(time.monotonic() - t0, 3)
    return result


def run_task(
    env_factory: EnvFactory,
    policy: Policy,
    task: Task,
    *,
    seed: int = 0,
    pass_threshold: float = 1.0,
) -> EpisodeResult:
    """Build a fresh environment, run one episode, always close the world.

    A leaked sandbox is a slow disk leak in CI and a stale-state bug in a
    training loop, so close() runs even when the episode blew up.
    """
    env = env_factory()
    try:
        return run_episode(env, policy, task, seed=seed, pass_threshold=pass_threshold)
    finally:
        try:
            env.close()
        except Exception:
            pass


def run_tasks(
    env_factory: EnvFactory,
    policy_factory: PolicyFactory,
    tasks: list[Task],
    *,
    seed: int = 0,
    pass_threshold: float = 1.0,
    max_parallel: int = 4,
) -> list[EpisodeResult]:
    """Run every non-skipped task. Each gets its own environment instance —
    sharing one would let task N see task N-1's world, which is the quietest
    possible way to make a suite meaningless."""
    selected = [t for t in tasks if not t.skip]

    def one(task: Task) -> EpisodeResult:
        return run_task(
            env_factory,
            policy_factory(task),
            task,
            seed=seed,
            pass_threshold=pass_threshold,
        )

    results: list[EpisodeResult]
    if max_parallel <= 1:
        results = [one(t) for t in selected]
    else:
        with ThreadPoolExecutor(max_workers=max_parallel) as pool:
            results = list(pool.map(one, selected))

    # Stable ordering so two runs diff cleanly regardless of completion order.
    results.sort(key=lambda r: r.task_id)
    return results
