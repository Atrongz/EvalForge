"""Verifiers — turn terminal world state into a reward.

A scorer grades a string. A verifier grades what the world looks like after the
agent stopped acting. The distinction matters because the two fail differently:
a scorer is fooled by fluent text, and a verifier is fooled by a shortcut. Every
verifier here is therefore paired with the question "what is the laziest thing
that satisfies this", and the answer belongs in the hack gate as a degenerate
probe rather than in a comment.

Rules carried over from scorers.py, for the same reasons:
  * Normalize to 0.0-1.0 so one runner and one gate serve every environment.
  * The reason string is required. An unexplained reward is a reward nobody
    can debug when a training run starts optimizing against it.

One rule that is specific to rewards:
  * Prefer a verifier the agent cannot author. A test the agent can edit is not
    a verifier, it is a suggestion — which is why command_exit_zero can pin the
    files it grades against tampering.
"""
from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class Reward:
    value: float          # 0.0 – 1.0
    reason: str
    # Named parts of a shaped reward. Kept separate from `value` so partial
    # credit is inspectable rather than an unexplained number in the middle.
    components: dict[str, float] = field(default_factory=dict)

    def clamped(self) -> "Reward":
        return Reward(max(0.0, min(1.0, self.value)), self.reason, dict(self.components))


Verifier = Callable[[Any, dict[str, Any]], Reward]


# ---------------------------------------------------------------------------
# Deterministic verifiers — no model, no network, no cost.
# ---------------------------------------------------------------------------

def command_exit_zero(state: Any, config: dict[str, Any]) -> Reward:
    """Run a command in the final world; exit 0 is success.

    `protect` lists files whose contents are hashed at reset and re-checked
    here. Deleting or rewriting the test that grades you is the single most
    common reward hack against this verifier, and it is cheap to make impossible
    rather than merely detectable after the fact.
    """
    root = (state or {}).get("root")
    if not root:
        return Reward(0.0, "no working directory in final state")

    protected: dict[str, str] = config.get("_protected_hashes") or {}
    for rel, expected_hash in protected.items():
        path = Path(root) / rel
        actual = _hash_file(path)
        if actual != expected_hash:
            verb = "deleted" if actual is None else "modified"
            return Reward(0.0, f"protected file {rel} was {verb} — graded as failure")

    cmd = config.get("cmd")
    if not cmd:
        return Reward(0.0, "verifier command_exit_zero needs a 'cmd'")

    try:
        proc = subprocess.run(
            cmd,
            shell=isinstance(cmd, str),
            cwd=str(root),
            capture_output=True,
            timeout=int(config.get("timeout", 60)),
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        return Reward(0.0, f"verifier command timed out after {config.get('timeout', 60)}s")

    if proc.returncode == 0:
        return Reward(1.0, f"`{cmd}` exited 0")
    tail = ((proc.stdout or "") + (proc.stderr or "")).strip().splitlines()[-3:]
    return Reward(0.0, f"`{cmd}` exited {proc.returncode}: {' / '.join(tail)[:200]}")


def file_contains(state: Any, config: dict[str, Any]) -> Reward:
    """Every required substring must appear in a file. Partial credit by fraction."""
    root = (state or {}).get("root")
    if not root:
        return Reward(0.0, "no working directory in final state")
    path = Path(root) / config["path"]
    if not path.is_file():
        return Reward(0.0, f"{config['path']} does not exist")

    text = path.read_text(encoding="utf-8", errors="replace")
    if config.get("ignore_case", True):
        text = text.lower()
    needles = config.get("contains") or []
    needles = [str(n).lower() if config.get("ignore_case", True) else str(n) for n in needles]
    if not needles:
        return Reward(0.0, "verifier file_contains needs a 'contains' list")

    missing = [n for n in needles if n not in text]
    value = (len(needles) - len(missing)) / len(needles)
    reason = "all present" if not missing else f"missing: {', '.join(missing[:5])}"
    return Reward(value, reason, {"found": len(needles) - len(missing), "total": len(needles)})


def actions_match(state: Any, config: dict[str, Any]) -> Reward:
    """Terminal state's action log must contain an expected subsequence.

    For ScriptedEnv and for environments whose value is the *procedure* rather
    than the artifact — an approval that must precede a send, a check that must
    precede a write.
    """
    log = list((state or {}).get("actions") or [])
    expected = list(config.get("sequence") or [])
    if not expected:
        return Reward(0.0, "verifier actions_match needs a 'sequence'")

    forbidden = set(config.get("forbidden") or [])
    hit = forbidden.intersection(log)
    if hit:
        return Reward(0.0, f"forbidden action taken: {', '.join(sorted(hit))}")

    i = 0
    for name in log:
        if i < len(expected) and name == expected[i]:
            i += 1
    value = i / len(expected)
    reason = "sequence satisfied" if i == len(expected) else f"reached {i}/{len(expected)}: missing {expected[i]!r}"
    return Reward(value, reason, {"matched": float(i), "required": float(len(expected))})


def never(state: Any, config: dict[str, Any]) -> Reward:
    """Always zero. Used to test the gate itself — a task that passes under
    `never` proves the gate is not actually running the verifier."""
    return Reward(0.0, "verifier 'never' always returns 0.0")


# ---------------------------------------------------------------------------


def _hash_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protect_hashes(root: str | Path, relative_paths: list[str]) -> dict[str, str]:
    """Snapshot hashes of the files a verifier grades against.

    Called at reset, before the agent can touch anything. The result goes into
    verifier_config as `_protected_hashes` — underscore-prefixed so it never
    collides with a user key, matching the `_request` convention in runner.py.
    """
    out: dict[str, str] = {}
    for rel in relative_paths or []:
        digest = _hash_file(Path(root) / rel)
        if digest is not None:
            out[rel] = digest
    return out


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, Verifier] = {
    "command_exit_zero": command_exit_zero,
    "file_contains": file_contains,
    "actions_match": actions_match,
    "never": never,
}


def get_verifier(name: str) -> Verifier:
    if name not in _REGISTRY:
        raise ValueError(
            f"unknown verifier {name!r}. Registered: {', '.join(sorted(_REGISTRY))}"
        )
    return _REGISTRY[name]


def registered_verifiers() -> list[str]:
    return sorted(_REGISTRY)
