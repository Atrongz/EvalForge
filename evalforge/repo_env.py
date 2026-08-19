"""RepoEnv — a real repository, checked out at a ref, in a disposable sandbox.

This is the SWE-bench shape applied to an acquired codebase. Reset extracts the
repo at a git ref into a temp directory, puts it into a known-broken state, and
provisions an interpreter with the dependencies the target test actually needs.
The agent then edits files and runs commands; the repository's own tests decide
the reward.

Three decisions worth stating, because each was forced by a real obstacle:

**`git archive`, never a checkout of the working tree.** The source repository is
read-only to this environment. A `git checkout` or a stray `git clean` inside a
sandbox that shares a worktree with someone's actual repo is how an evaluation
harness eats uncommitted work.

**Install the dependency closure of the target test, not `requirements.txt`.**
sentinel-rag's full requirements include torch and chromadb; the chunking test
needs `tiktoken` and `pytest`. Installing the whole file would turn a two-second
task into a multi-gigabyte one, and most acquired repos have a requirements file
that no longer resolves anyway.

**The virtualenv is cached and holds no repo code.** The sandbox holds the
source; the venv holds only third-party packages; `PYTHONPATH` joins them. That
keeps the venv reusable across episodes — a per-episode `pip install` makes
rollouts too slow to train against — while the code under test is still restored
fresh every reset.

Isolation is process-level: a temp directory, a separate interpreter, and no
network policy. That is honest for a repo you wrote. For third-party code, the
container backend is the one to use, and it is deliberately not faked here.
"""
from __future__ import annotations

import hashlib
import io
import os
import subprocess
import sys
import tarfile
import tempfile
import venv
from pathlib import Path
from typing import Any

from .envs import FileTaskEnv, Observation, _REGISTRY
from .models import Task


class RepoEnv(FileTaskEnv):
    """A git repository at a ref, made broken on purpose, with its tests intact."""

    id = "repo"

    def __init__(
        self,
        *,
        repo: str | None = None,
        ref: str = "HEAD",
        pip: list[str] | None = None,
        python_path: list[str] | None = None,
        venv_cache: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.repo = repo
        self.ref = ref
        self.pip = list(pip or [])
        self.python_path = list(python_path or [])
        self.venv_cache = Path(venv_cache or Path(tempfile.gettempdir()) / "evalforge-venvs")
        self._python: Path | None = None

    def version(self) -> str:
        sha = "unknown"
        if self.repo:
            try:
                sha = subprocess.run(
                    ["git", "-C", str(self.repo), "rev-parse", "--short", self.ref],
                    capture_output=True, text=True, timeout=15,
                ).stdout.strip() or "unknown"
            except Exception:
                pass
        return f"repo/{Path(self.repo).name if self.repo else '?'}@{sha}"

    # -- lifecycle ---------------------------------------------------------

    def reset(self, task: Task, *, seed: int = 0) -> Observation:
        setup = task.setup or {}
        repo = setup.get("repo") or self.repo
        ref = setup.get("ref") or self.ref
        if not repo:
            raise ValueError("RepoEnv needs a 'repo' path in env_config or task.setup")

        self.close()
        self._root = Path(tempfile.mkdtemp(prefix="evalforge-repo-"))
        self._extract(repo, ref, self._root)

        # Put the world into its broken state. When history is minable this is
        # the parent commit instead, and no overwrite is needed.
        for rel, contents in (setup.get("bug_files") or {}).items():
            path = self._safe_path(rel)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(contents), encoding="utf-8")

        pip = list(setup.get("pip") or self.pip)
        self._python = self._ensure_venv(pip) if pip else Path(sys.executable)

        files = sorted(setup.get("bug_files") or {})
        return Observation(
            text=(
                f"{task.goal}\n\n"
                f"Repository {Path(repo).name} at {ref} in a temporary working directory.\n"
                f"Interpreter: {self._python}\n"
                f"Modified from HEAD: {', '.join(files) or '(none)'}"
            ),
            data={"root": str(self._root), "python": str(self._python), "seed": seed},
        )

    def final_state(self) -> Any:
        return {
            "root": str(self._root) if self._root else None,
            "python": str(self._python) if self._python else None,
            # Declared so the verifier runs in the same interpreter and import
            # path the agent had. A verifier grading a repo it cannot import
            # reports a failure that belongs to the harness, not the agent.
            "env": self._env_overrides(),
        }

    # -- wiring ------------------------------------------------------------

    def _env_overrides(self) -> dict[str, str]:
        env: dict[str, str] = {"PYTHONDONTWRITEBYTECODE": "1"}
        if self._root and self.python_path:
            env["PYTHONPATH"] = os.pathsep.join(str(self._root / p) for p in self.python_path)
        if self._python:
            env["PATH"] = str(self._python.parent) + os.pathsep + os.environ.get("PATH", "")
        # Cache tiktoken's BPE download outside the sandbox. Without this every
        # reset re-downloads it, which makes episodes both slow and network-dependent.
        env["TIKTOKEN_CACHE_DIR"] = str(self.venv_cache / "_tiktoken")
        return env

    def subprocess_env(self) -> dict[str, str] | None:
        """Join the sandbox's source tree to the cached interpreter."""
        return {**os.environ, **self._env_overrides()}

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _extract(repo: str, ref: str, dest: Path) -> None:
        """Materialize the repo at `ref` without touching the source worktree."""
        proc = subprocess.run(
            ["git", "-C", str(repo), "archive", "--format=tar", ref],
            capture_output=True, timeout=120,
        )
        if proc.returncode != 0:
            err = proc.stderr.decode("utf-8", "replace").strip()
            raise RuntimeError(f"git archive {ref} failed: {err[:200]}")
        with tarfile.open(fileobj=io.BytesIO(proc.stdout)) as tar:
            tar.extractall(dest)

    def _ensure_venv(self, pip: list[str]) -> Path:
        """One virtualenv per dependency set, reused across episodes.

        Keyed by the dependency list, so changing dependencies makes a new
        environment rather than mutating one other episodes are using.
        """
        key = hashlib.sha256("\n".join(sorted(pip)).encode()).hexdigest()[:12]
        target = self.venv_cache / key
        python = target / ("Scripts" if os.name == "nt" else "bin") / (
            "python.exe" if os.name == "nt" else "python"
        )
        if python.exists():
            return python

        target.parent.mkdir(parents=True, exist_ok=True)
        venv.EnvBuilder(with_pip=True, clear=True).create(target)
        proc = subprocess.run(
            [str(python), "-m", "pip", "install", "--quiet", *pip],
            capture_output=True, text=True, timeout=900,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"pip install {pip} failed: {(proc.stderr or '')[-300:]}")
        return python


_REGISTRY["repo"] = RepoEnv
