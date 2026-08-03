---
name: evalforge
description: >-
  Evaluate any LLM system against a golden set and catch regressions. Use when the user says
  "/evalforge", "run the evals", "did that change make it better or worse", "build an eval for
  X", "red-team this agent", "is my prompt change an improvement", "test my agent for prompt
  injection", or needs to prove a system's quality with numbers rather than impressions. ALWAYS
  use this instead of eyeballing sample outputs — the whole point is that "it looks fine" is not
  a measurement.
---

# EvalForge

A general evaluation harness. Targets and scorers are pluggable; the runner knows about neither.

**Location:** `C:\Users\atron\EvalForge`
**Run from that directory** (or pass `--root`).

## Commands

```bash
evalforge list                        # suites, targets, scorers
evalforge run <suite>                 # run one
evalforge run --all                   # run everything
evalforge run <suite> --dry-run       # walk every path, invoke nothing
evalforge run <suite> --category X    # slice
evalforge baseline <suite>            # promote last run to the reference
evalforge report <suite> --markdown   # PR-ready report
```

Exit codes: `0` pass · `1` gate failed · `2` regression vs baseline · `3` usage error.

## Building a new suite

1. `suites/<name>/suite.yaml` — pick a target and a scorer (`evalforge list` shows both).
2. `suites/<name>/cases/*.yaml` — the golden set. **Every case needs a `notes` field** explaining
   how the ground truth was established; a test enforces this.
3. `evalforge run <name> --dry-run` — confirm everything loads before spending anything.
4. `evalforge run <name>` — real run.
5. `evalforge baseline <name>` — lock in the reference once you believe the numbers.

## Rules that are load-bearing

- **Prefer a deterministic scorer.** `no_leak`, `numeric`, `citation_grounding`, `json_fields` and
  `regex` cost nothing and never flap. Reach for `llm_judge` only when the question genuinely has
  no crisp answer. A judge is itself a system that can be wrong, and a flapping eval manufactures
  phantom regressions.
- **Errors are not zeros.** If a run shows ERRORS, the target or scorer broke — fix that before
  reading any score. Errored cases fail the suite outright by design.
- **Include negative controls.** A safety suite needs benign cases, or it will reward an agent
  that refuses everything. Any suite measuring "does it block X" needs cases where blocking is wrong.
- **Never baseline a dry run.** The CLI refuses; don't work around it.
- **Investigate before re-baselining.** A regression means either the system got worse or the
  golden set was wrong. Decide which. Re-baselining to make red go green destroys the point.
- **$0 by default.** `claude_cli` runs on the subscription. Only use `anthropic_api` when metered
  token accounting is actually needed.

## Interpreting a result

| Symptom | Meaning |
|---|---|
| ERRORS present | Target/scorer broken. Nothing else in the run is trustworthy. |
| Attack categories 100%, benign low | Over-refusal. The agent is safe and unusable. |
| Same suite, different results run to run | Judge instability. Move that assertion to a deterministic scorer or raise `judge_votes`. |
| Mean flat, cases flipped | A real regression the aggregate hid. Read the per-case diff. |

## Shipped suites

- **`redteam`** — 24 cases, 7 categories: direct/indirect prompt injection, role-play jailbreak,
  encoding bypass, tool misuse, data exfiltration, plus benign controls. Point
  `target_config.system` at a real agent's system prompt to test that agent.

## Extending

- **New target kind** — implement `invoke()` and `version()` in `evalforge/targets.py`, add one
  line to `_REGISTRY`. Nothing else changes.
- **New scorer** — a function `(output, expected, config) -> Score` in `evalforge/scorers.py`,
  add one line to `_REGISTRY`. Always return a useful `reason`; a failing case with no explanation
  gets re-litigated from scratch later.
