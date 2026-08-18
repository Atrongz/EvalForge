"""Policies — whatever decides the next action.

A policy is the thing being measured. Keeping it behind a one-method Protocol
means the episode runner is identical whether the actor is a model, a scripted
list of moves, or nothing at all — which is what makes the reward-hack gate
possible: the null and degenerate probes are policies, run through exactly the
same code path as the real agent. A gate that used a special-case shortcut
would not be testing the thing it claims to test.
"""
from __future__ import annotations

from typing import Any, Callable, Protocol

from .envs import Action, Observation


class Policy(Protocol):
    id: str

    def act(self, observation: Observation, step_index: int) -> Action | None:
        """Return the next action, or None to stop. Returning None is a
        deliberate submit, not an error — an agent that decides it is finished
        is behaving correctly and gets graded on what it left behind."""
        ...


class ScriptedPolicy:
    """Replays a fixed list of actions, then stops.

    This is the gold policy, and it is also every degenerate probe. A task's
    `gold` is a scripted policy that must reach reward 1.0; each entry in
    `degenerate` is a scripted policy that must not.
    """

    def __init__(self, actions: list[dict[str, Any]] | list[Action], *, id: str = "scripted") -> None:
        self.id = id
        self._actions = [a if isinstance(a, Action) else Action.from_dict(a) for a in actions]

    def act(self, observation: Observation, step_index: int) -> Action | None:
        if step_index >= len(self._actions):
            return None
        return self._actions[step_index]


class NullPolicy:
    """Does nothing at all.

    Every task is run against this. If doing nothing scores above the pass
    threshold, the task was already solved at reset and measures nothing — the
    most common way a task suite silently inflates a model's score.
    """

    id = "null"

    def act(self, observation: Observation, step_index: int) -> Action | None:
        return None


class CallablePolicy:
    """Wraps any function of (observation, step_index) -> Action | None.

    The adapter point for a real agent: a model loop, an SDK, a subprocess.
    Kept dependency-free so the core has no opinion about how you call a model.
    """

    def __init__(self, fn: Callable[[Observation, int], Action | None], *, id: str = "callable") -> None:
        self.id = id
        self._fn = fn

    def act(self, observation: Observation, step_index: int) -> Action | None:
        return self._fn(observation, step_index)
