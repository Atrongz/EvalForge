"""Mine a repository's history for candidate task instances.

    python scripts/mine_tasks.py <repo> [--limit 500] [--json]

A commit is a candidate when it changes source and tests together. The premise
of SWE-bench-style extraction is that such a commit hands you a whole task for
free: the parent is the starting state, the test is the verifier, and the diff
is the gold patch. Nobody has to invent a bug or guess at correctness.

The reason this script exists as its own stage is that it answers a triage
question before any engineering is spent: **can this asset yield tasks at all?**
A repository that was squash-imported, vendored, or committed as one scaffold
has no minable history, and no amount of containerization will change that.
Discovering it here costs seconds. Discovering it after building an environment
costs a day.

Output is deliberately blunt about zero. "No candidates" is a finding, not a
failure, and it should be reported upward rather than worked around quietly.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

SRC_SUFFIXES = {".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rb", ".java", ".rs"}
TEST_MARKERS = ("test_", "_test.", "/tests/", "\\tests\\", ".test.", ".spec.")


@dataclass
class Candidate:
    sha: str
    subject: str
    src_files: list[str]
    test_files: list[str]


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {proc.stderr.strip()[:200]}")
    return proc.stdout


def is_test(path: str) -> bool:
    lowered = path.lower()
    return any(m in lowered for m in TEST_MARKERS)


def is_source(path: str) -> bool:
    return Path(path).suffix in SRC_SUFFIXES and not is_test(path)


def mine(repo: Path, limit: int) -> tuple[list[Candidate], int, int]:
    log = _git(repo, "log", f"-{limit}", "--format=%H\x1f%s")
    commits = [line.split("\x1f", 1) for line in log.splitlines() if "\x1f" in line]

    found: list[Candidate] = []
    skipped_root = 0
    for sha, subject in commits:
        # A root commit has no parent, so there is no starting state to put the
        # agent in. Scaffold imports are almost always root commits and would
        # otherwise dominate the candidate list on a young repository.
        parents = _git(repo, "rev-list", "--parents", "-n", "1", sha).split()
        if len(parents) < 2:
            skipped_root += 1
            continue

        files = _git(repo, "show", "--name-only", "--format=", sha).split()
        src = [f for f in files if is_source(f)]
        tests = [f for f in files if is_test(f) and Path(f).suffix in SRC_SUFFIXES]
        if src and tests:
            found.append(Candidate(sha[:10], subject, src, tests))
    return found, len(commits), skipped_root


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("repo")
    ap.add_argument("--limit", type=int, default=500, help="commits to scan (default 500)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv[1:])

    repo = Path(args.repo).resolve()
    if not (repo / ".git").exists():
        print(f"not a git repository: {repo}")
        return 2

    candidates, scanned, skipped_root = mine(repo, args.limit)

    if args.json:
        print(json.dumps([asdict(c) for c in candidates], indent=2))
        return 0 if candidates else 1

    root_note = f", {skipped_root} root commit(s) skipped" if skipped_root else ""
    print(f"{repo.name}: scanned {scanned} commits{root_note}")
    if not candidates:
        print(
            "  0 candidates — no commit changes source and tests together.\n"
            "  This repository's history cannot yield task instances. Usually squash-imported\n"
            "  or committed as a single scaffold. Either mine a different asset, or synthesize\n"
            "  the broken state from HEAD and use the existing tests as the verifier."
        )
        return 1

    print(f"  {len(candidates)} candidate commit(s):\n")
    for c in candidates:
        print(f"  {c.sha}  {c.subject[:64]}")
        print(f"      source: {len(c.src_files)} file(s)   tests: {', '.join(c.test_files[:3])}")
    print(
        "\n  A candidate is not yet a task. Each still has to be validated: the test must fail\n"
        "  at the parent and pass at the commit, or it does not discriminate."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
