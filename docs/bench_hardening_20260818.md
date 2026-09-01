# Bench hardening + eval layer — 2026-08-18

Pre-scaling audit of the benchmark platform (two independent review passes:
bench machinery, harness adapters) plus the new evaluation layer. Context:
the `matrix-v2-20260810` shakeout ran 10 cells; before the paid campaign we
verified the machinery and added code-quality / cost evals.

## Fixed in this change

**Scoring integrity (`bench/contract.py`)**
- NaN predictions at finite ground-truth points scored as PERFECT
  (`nan_to_num` ran on the diff): an all-NaN psi array got rel_l2 = 0.0.
  Now a NaN prediction is a full-magnitude miss, and `frozen_score` reports
  `pred_nan_at_finite`. Existing matrix-v2 numbers shift by <=1.4e-4 (worst:
  L3-dspy, 1.4% NaN contamination) — old and new scores are comparable in
  practice, but re-derive headline numbers with the new code.
- Sample-count mismatches now raise instead of `zip`-truncating silently.
- `predict.py` scoring subprocess: own process group (timeout kills solver
  children too), provider keys scrubbed from its env.
- L3 import audit also catches `importlib.import_module("autotokamak")` /
  `__import__("autotokamak")`.

**Scoring availability (`bench/__main__.py`, `tools/matrix_report.py`)**
- Frozen scoring used to require the FULL contract to pass; a missing README
  key silently cost a paid cell its head-to-head number. Now any cell whose
  three predict gates pass gets scored. Backfilled `L3-claude_sdk`
  (matrix-v2): rel_l2 mean **0.1204 — the best cell in the matrix**,
  previously invisible.
- New `python -m autotokamak.bench score --run-dir <run>` backfills scoring
  for existing runs without re-running the agent.
- Same-second parallel launches no longer share/overwrite one run dir.

**Prompt comparability (`benchmarks/tasks/L{2,3}_mini_v3.yaml`)**
- v2 mini prompts stated "up to 3 rounds of exactly 100 solves" AND
  "each round's 500 evaluations" AND a "10-round cap" (leftovers from the
  full-size tasks) — a ~17x solver-budget ambiguity, identical across
  harnesses but pure interpretation noise. v3 fixes the numbers; use v3 for
  all new mini campaigns. (v2 files untouched per the versioning policy.)

**Harness parity (`harnesses/*`)**
- The cwd-jail RUNTIME NOTE was claude_sdk-only private help; it is now the
  shared `Harness.workspace_note()` appended by claude_sdk, pi, and cursor.
- pi/cursor timeouts no longer discard the partial event stream.
- cursor status now prefers the stream's final `result` verdict over the
  process exit code (parity with claude_sdk's `subtype` logic).
- ursa/dspy: missing-dependency / bad-model failures now produce a proper
  errored `result.json` instead of crashing the bench CLI.
- ursa engine: bounded retry (2, backoff) on connection-class errors only —
  parity with the CLI substrates' internal retries. (A single transient
  `APIConnectionError` killed L2-ursa at 1953s in matrix-v2.)
- dspy path jail no longer blocks the task-provided OpenFUSIONToolkit
  symlink (resolve() escaped the workspace by construction).
- Cost/usage capture everywhere it exists: claude_sdk (`usage` +
  `total_cost_usd`), pi (per-message `usage.cost` — exact dollars), cursor
  (token counts; dollars need a pinned model + price table), dspy (litellm
  cost/usage from `lm.history`), ursa (OpenAI callback). Lands in
  `RunResult.cost_usd` / `extra.usage`.

## New evaluation layer (`tools/`)

- `eval_code_metrics.py` — static, deterministic, zero-LLM per-workspace
  metrics (SLOC, structure, complexity proxy, imports incl. LLM-in-loop
  detection and the L3 no-autotokamak cross-check, seeds, ruff counts).
- `judge_code.py` — blind + outcome-blind LLM-as-judge, 7-dimension rubric
  (correctness, methodology, structure, robustness, reproducibility,
  efficiency, documentation; 1–5 with anchors, evidence required), plus
  decision-style / adaptive-sampling classification and red-flag extraction
  (fabrication, leakage). `score` per run, `compare` for the cross-cell
  synthesis. Use `--samples 3` and an Opus-class judge for publishable
  numbers; the judge's own cost is recorded.
- `cost_report.py` — measured/derived/proxy cost per run; harvests old runs'
  raw event streams retroactively (matrix-v2: pi cells got exact dollars
  after the fact).

## Known-open (accepted for now, revisit before/with the paper)

1. **No timeout enforcement inside ursa/dspy adapters** — enforcing it
   properly means running the engine in a killable child process. Turn/wall
   budgets differ per substrate anyway (claude_sdk MAX_TURNS=300, dspy 40
   iters/step, cursor/pi/ursa uncapped) — report compute budgets per cell.
2. **L3 read-isolation is prompt-deep only**: workspaces live inside the
   repo, every substrate can `cat ../../..` into `src/autotokamak` or read
   `.env` (all provider keys), and `predict.py` could in principle read the
   frozen `test_set.h5` during scoring. Real fix: workspaces outside the
   repo, scrubbed env, ground truth unreadable during predict. None of the
   10 matrix-v2 workspaces show any sign of this (verified honest).
3. **Empty-workspace cwd-escape guard is claude_sdk-only**; hoisting it to
   `cmd_run` (or a git-status snapshot before/after) would cover all
   substrates and catch partial escapes.
4. Gate denominators vary by failure mode (0/3 vs 6/7 vs 9/9) — fine
   mechanically since `passed` is the AND, but display "gates x/y" with
   care.
5. `bench compare`/`matrix_report` do not segregate `prompt_version` — do
   not mix versions under one tag.
