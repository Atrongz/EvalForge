"""Resolve the dependency closure a test actually needs, by running it.

    python scripts/resolve_deps.py <env-suite> [--task <id>] [--max-rounds 12]

`requirements.txt` is the wrong input for this. It is the closure of the whole
project, it is usually stale, and on an acquired codebase it frequently no longer
resolves at all. What a task needs is the closure of one test's *import graph*,
which is both far smaller and knowable only by running it.

So this runs the verifier, reads the `ModuleNotFoundError` off the failure,
installs that one package, and repeats until the test either passes or fails for
a reason that is not a missing module. Each round is one new fact. The output is
the `pip:` list to paste into the task.

Iterating is the point. Static import analysis cannot see conditional imports,
re-exports, or namespace packages, and every real repository has some. Running it
is the only method that is not a guess.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evalforge.envs import build_env  # noqa: E402
from evalforge.models import load_env_suite  # noqa: E402

MISSING = re.compile(r"ModuleNotFoundError: No module named ['\"]([A-Za-z0-9_.]+)['\"]")

# Distribution names that are not just the module with underscores swapped.
ALIASES = {
    "yaml": "pyyaml",
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
    "PIL": "pillow",
    "bs4": "beautifulsoup4",
    "dotenv": "python-dotenv",
    "attr": "attrs",
    "dateutil": "python-dateutil",
}


def distribution_for(module: str) -> str:
    top = module.split(".")[0]
    return ALIASES.get(top, top.replace("_", "-"))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("suite")
    ap.add_argument("--task")
    ap.add_argument("--max-rounds", type=int, default=12)
    args = ap.parse_args(argv[1:])

    config, tasks = load_env_suite(Path(args.suite))
    task = next((t for t in tasks if t.id == args.task), tasks[0]) if args.task else tasks[0]

    pip = list((task.setup or {}).get("pip") or [])
    cmd_template = (task.verifier_config or {}).get("cmd")
    if not cmd_template:
        print("task has no verifier cmd to run")
        return 2

    print(f"{task.id}: starting closure {pip or '(empty)'}\n")

    for round_no in range(1, args.max_rounds + 1):
        task.setup["pip"] = pip
        env = build_env(config.env, config.env_config)
        try:
            env.reset(task)
            state = env.final_state()
            cmd = cmd_template.replace("{python}", str(state["python"]))
            proc = subprocess.run(
                cmd, shell=True, cwd=state["root"], capture_output=True,
                encoding="utf-8", errors="replace", timeout=600,
                env={**__import__("os").environ, **state.get("env", {})},
            )
            output = (proc.stdout or "") + (proc.stderr or "")
        finally:
            env.close()

        if proc.returncode == 0:
            print(f"round {round_no}: test passes. closure is complete.\n")
            print("pip:")
            for p in pip:
                print(f"  - {p}")
            return 0

        found = MISSING.search(output)
        if not found:
            # Exit 1 with no missing module is the healthy end state for a task
            # whose starting point is supposed to be broken: the closure is
            # complete and the test is now failing on the bug.
            tail = "\n".join(output.strip().splitlines()[-4:])
            print(f"round {round_no}: no missing module — closure complete.")
            print(f"  test still fails, which is expected for a broken starting state:\n  {tail[:300]}\n")
            print("pip:")
            for p in pip:
                print(f"  - {p}")
            return 0

        dist = distribution_for(found.group(1))
        if dist in pip:
            print(f"round {round_no}: {dist} already installed but {found.group(1)} still missing.")
            print("  The distribution name is probably wrong — add it to ALIASES.")
            return 1
        print(f"round {round_no}: missing {found.group(1)} -> installing {dist}")
        pip.append(dist)

    print(f"gave up after {args.max_rounds} rounds. closure so far: {pip}")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
