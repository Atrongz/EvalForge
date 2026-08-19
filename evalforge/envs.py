"""Environments — stateful worlds an agent acts in over multiple turns.

A Target answers one question and forgets it happened. An Environment holds
state: the agent acts, the world changes, and the next observation reflects the
change. That difference is the reason this file exists. A reward that grades a
*string* can be satisfied by a plausible-sounding sentence; a reward that grades
*terminal world state* has to be earned.

The contract deliberately mirrors targets.py — one Protocol, one class per kind,
one line in the registry. The episode runner, the verifiers and the reward-hack
gate never change when an environment is added.

Isolation, stated plainly: FileTaskEnv sandboxes to a temporary directory and
does nothing else. That is sufficient for fixtures you wrote yourself. It is NOT
sufficient for an acquired third-party repository, which can read your
filesystem and reach the network. Container isolation belongs in a separate
adapter; pretending this one provides it would be worse than not having it.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .models import Task


@dataclass
class Observation:
    """What the agent sees after an action. `text` is what a language model is
    shown; `data` carries structure for programmatic policies and verifiers."""

    text: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class Action:
    """One move. `name` must be in the environment's declared action space —
    an unknown name is an observation saying so, never a crash, because a policy
    hallucinating a tool is normal behaviour to be measured, not an outage."""

    name: str
    args: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Action":
        data = dict(raw)
        name = data.pop("name", None)
        if not name:
            raise ValueError(f"action needs a 'name': {raw!r}")
        # Everything else is arguments, so YAML stays flat and readable.
        return cls(name=str(name), args=data.pop("args", None) or data)


@dataclass
class ActionSpec:
    """A declared, legal action. Environments publish these so a policy can be
    told what it may do instead of guessing."""

    name: str
    args: list[str] = field(default_factory=list)
    doc: str = ""


@dataclass
class StepResult:
    """The result of one step.

    `reward` here is the *shaped* per-step signal and is 0.0 for most
    environments. The reward that decides pass or fail is computed once at the
    end by a verifier against terminal state — a per-step reward is the easiest
    thing in the whole design to accidentally make hackable.
    """

    observation: Observation
    reward: float = 0.0
    done: bool = False
    info: dict[str, Any] = field(default_factory=dict)


class Environment(Protocol):
    id: str

    def version(self) -> str:
        """Identifies the world under test — image tag, fixture sha, adapter
        version. Recorded on every episode so a reward change can be attributed."""
        ...

    def reset(self, task: Task, *, seed: int = 0) -> Observation:
        """Restore the world to this task's starting state and return the first
        observation. Two resets with the same task and seed must produce
        identical worlds, or episodes are not comparable and nothing downstream
        means anything."""
        ...

    def actions(self) -> list[ActionSpec]: ...

    def step(self, action: Action) -> StepResult: ...

    def final_state(self) -> Any:
        """Whatever the verifier needs to grade the ending. Deliberately opaque
        to the runner — only the environment and its verifiers agree on shape."""
        ...

    def close(self) -> None: ...


# ---------------------------------------------------------------------------


class ScriptedEnv:
    """A world with no world. Echoes actions back and ends when told.

    The counterpart to EchoTarget: exercises the episode runner, trajectory
    recording and the hack gate with no filesystem and no subprocess. Terminal
    state is the ordered list of action names, which is enough for a verifier
    to assert "did the policy do the right things in the right order".
    """

    id = "scripted"

    def __init__(self, **_: Any) -> None:
        self._log: list[str] = []
        self._task: Task | None = None

    def version(self) -> str:
        return "scripted"

    def reset(self, task: Task, *, seed: int = 0) -> Observation:
        self._log = []
        self._task = task
        return Observation(text=task.goal, data={"seed": seed})

    def actions(self) -> list[ActionSpec]:
        return [
            ActionSpec("noop", [], "Does nothing. Recorded."),
            ActionSpec("say", ["text"], "Records a string."),
            ActionSpec("submit", [], "Ends the episode."),
        ]

    def step(self, action: Action) -> StepResult:
        self._log.append(action.name)
        if action.name == "submit":
            return StepResult(Observation("submitted"), done=True)
        return StepResult(Observation(f"ok: {action.name}"))

    def final_state(self) -> Any:
        return {"actions": list(self._log)}

    def close(self) -> None:
        self._log = []


class FileTaskEnv:
    """A seeded working directory the agent edits, plus a command runner.

    This is the SWE-bench shape reduced to its core: put the world in a known
    broken state, let the agent act, then let the repository's own tests decide
    whether it was fixed. The tests are the reward, and they are the strongest
    available reward precisely because they were written by people who were not
    trying to satisfy a grader.

    `task.setup` is `{files: {relative/path: contents}}`. Anything else is
    ignored, so a richer setup (a git checkout, a restored database) is a new
    environment rather than a new branch in this one.
    """

    id = "file_task"

    def __init__(
        self,
        *,
        allow_commands: bool = True,
        command_timeout: int = 30,
        max_output_chars: int = 4000,
        **_: Any,
    ) -> None:
        self.allow_commands = allow_commands
        self.command_timeout = command_timeout
        self.max_output_chars = max_output_chars
        self._root: Path | None = None

    def version(self) -> str:
        return "file_task/1"

    # -- lifecycle ---------------------------------------------------------

    def reset(self, task: Task, *, seed: int = 0) -> Observation:
        self.close()
        self._root = Path(tempfile.mkdtemp(prefix="evalforge-env-"))
        files: dict[str, str] = (task.setup or {}).get("files", {}) or {}
        for rel, contents in files.items():
            path = self._safe_path(rel)
            path.parent.mkdir(parents=True, exist_ok=True)
            # utf-8 pinned everywhere: on Windows the default is cp1252 and a
            # non-ASCII fixture otherwise fails three frames away from the cause.
            path.write_text(str(contents), encoding="utf-8")
        return Observation(
            text=f"{task.goal}\n\nFiles: {', '.join(sorted(files)) or '(none)'}",
            data={"root": str(self._root), "files": sorted(files), "seed": seed},
        )

    def close(self) -> None:
        if self._root and self._root.exists():
            shutil.rmtree(self._root, ignore_errors=True)
        self._root = None

    # -- action space ------------------------------------------------------

    def actions(self) -> list[ActionSpec]:
        specs = [
            ActionSpec("list_files", [], "List every file in the working directory."),
            ActionSpec("read_file", ["path"], "Return a file's contents."),
            ActionSpec("write_file", ["path", "content"], "Create or overwrite a file."),
            ActionSpec("delete_file", ["path"], "Delete a file."),
            ActionSpec("submit", [], "End the episode and be graded."),
        ]
        if self.allow_commands:
            specs.insert(-1, ActionSpec("run", ["cmd"], "Run a shell command in the working directory."))
        return specs

    def step(self, action: Action) -> StepResult:
        if self._root is None:
            raise RuntimeError("step() before reset()")
        handler = {
            "list_files": self._list_files,
            "read_file": self._read_file,
            "write_file": self._write_file,
            "delete_file": self._delete_file,
            "run": self._run,
            "submit": self._submit,
        }.get(action.name)
        if handler is None:
            # A hallucinated tool is a measurable behaviour, not an error.
            legal = ", ".join(s.name for s in self.actions())
            return StepResult(Observation(f"unknown action {action.name!r}. Legal: {legal}"))
        try:
            return handler(action.args)
        except Exception as exc:
            return StepResult(Observation(f"{type(exc).__name__}: {exc}"))

    def final_state(self) -> Any:
        return {"root": str(self._root) if self._root else None}

    # -- handlers ----------------------------------------------------------

    def _list_files(self, _: dict[str, Any]) -> StepResult:
        assert self._root
        names = sorted(
            str(p.relative_to(self._root)).replace(os.sep, "/")
            for p in self._root.rglob("*")
            if p.is_file()
        )
        return StepResult(Observation("\n".join(names) or "(empty)", {"files": names}))

    def _read_file(self, args: dict[str, Any]) -> StepResult:
        path = self._safe_path(args["path"])
        if not path.is_file():
            return StepResult(Observation(f"no such file: {args['path']}"))
        return StepResult(Observation(path.read_text(encoding="utf-8", errors="replace")))

    def _write_file(self, args: dict[str, Any]) -> StepResult:
        path = self._safe_path(args["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(args.get("content", "")), encoding="utf-8")
        return StepResult(Observation(f"wrote {args['path']}"))

    def _delete_file(self, args: dict[str, Any]) -> StepResult:
        path = self._safe_path(args["path"])
        if path.is_file():
            path.unlink()
            return StepResult(Observation(f"deleted {args['path']}"))
        return StepResult(Observation(f"no such file: {args['path']}"))

    def _run(self, args: dict[str, Any]) -> StepResult:
        if not self.allow_commands:
            return StepResult(Observation("commands are disabled in this environment"))
        assert self._root
        cmd = args["cmd"]
        proc = subprocess.run(
            cmd,
            shell=isinstance(cmd, str),
            cwd=str(self._root),
            capture_output=True,
            timeout=self.command_timeout,
            encoding="utf-8",
            errors="replace",
        )
        body = ((proc.stdout or "") + (proc.stderr or ""))[: self.max_output_chars]
        return StepResult(
            Observation(f"exit={proc.returncode}\n{body}", {"returncode": proc.returncode}),
            info={"returncode": proc.returncode},
        )

    def _submit(self, _: dict[str, Any]) -> StepResult:
        return StepResult(Observation("submitted"), done=True)

    # -- internals ---------------------------------------------------------

    def _safe_path(self, rel: str) -> Path:
        """Resolve inside the sandbox or refuse. `../../etc/passwd` in a task
        fixture is a bug; the same string from a policy is an escape attempt."""
        assert self._root
        candidate = (self._root / str(rel)).resolve()
        root = self._root.resolve()
        if root not in candidate.parents and candidate != root:
            raise ValueError(f"path escapes the sandbox: {rel!r}")
        return candidate


# ---------------------------------------------------------------------------
# Registry — one line per adapter.
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type] = {
    "scripted": ScriptedEnv,
    "file_task": FileTaskEnv,
}


def build_env(kind: str, config: dict[str, Any] | None = None) -> Environment:
    if kind not in _REGISTRY:
        raise ValueError(
            f"unknown environment {kind!r}. Registered: {', '.join(sorted(_REGISTRY))}"
        )
    return _REGISTRY[kind](**(config or {}))  # type: ignore[return-value]


def registered_envs() -> list[str]:
    return sorted(_REGISTRY)
