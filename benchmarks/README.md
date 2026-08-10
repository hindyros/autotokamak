# Benchmarks — the agent-capability experiment matrix

Can a coding agent execute the 4-step surrogate pipeline this repo implements
— (1) generate Grad-Shafranov data, (2) feature-engineer, (3) train
surrogates, (4) decide what to do next — and how much does pre-written,
proven library code help? Every experiment in this repo is one cell of a
2-axis matrix, named `<level>-<harness>`:

## Axis B — access level

| Level | Meaning | Where it runs |
|---|---|---|
| **L0** scripted | Pre-written pipeline, seeded heuristic decisions, zero LLM. The reproducible golden baseline. | `python -m autotokamak.pipelines <phase> --level L0` |
| **L1** structured | Pre-written pipeline; an LLM makes only typed decisions (search rounds, meta actions) via the DSPy pickers. | `python -m autotokamak.pipelines <phase> --level L1` |
| **L2** library-assisted | The agent writes the glue code and MAY import `autotokamak`. | `python -m autotokamak.bench run --task benchmarks/tasks/L2_library.yaml --harness <name>` |
| **L3** from-scratch | The agent writes everything; importing `autotokamak` is forbidden and AUDITED (hard contract gate). | `python -m autotokamak.bench run --task benchmarks/tasks/L3_from_scratch.yaml --harness <name>` |

## Axis A — harness (agent substrate)

`ursa` (URSA PlanningAgent+ExecutionAgent) · `dspy` (DSPy plan→ReAct→review) ·
`claude_sdk` (Claude Agent SDK) · `pi` (Pi Code CLI) · `cursor` (Cursor CLI) ·
`echo` (no-LLM mock for CI). L0 has no agent (`L0-none`); L1 is DSPy-typed by
construction (`L1-dspy`).

## The shared contract

Every condition emits the same machine-checkable deliverables, so all cells
are directly comparable (`src/autotokamak/bench/contract.py`):

- `report.json` with `n_solves_*` and `metrics.{test,baseline}_rel_l2.{mean,median,p90}`
- `predict.py --input params.json --output pred.npz` with `psi (N, 96, 64)`
  on the frozen grid (`assets/eval_grid.json`: R 64 pts in [0.15, 0.80] m,
  Z 96 pts in [−0.40, 0.40] m; physical psi in Wb, NaN outside the plasma)
- head-to-head scoring against the frozen test set: `assets/test_params.json`
  (60 params, seed 20260809, committed) solved once by
  `python -m autotokamak.bench freeze-testset` into `assets/test_set.h5`
  (gitignored, reproducible)

## Running a condition

```bash
# 1. cheap smoke first — auth, jailing, trace capture, exit handling:
python -m autotokamak.bench run --task benchmarks/tasks/smoke.yaml --harness claude_sdk
# 2. see exactly what would run without spending anything:
python -m autotokamak.bench run --task benchmarks/tasks/L3_from_scratch.yaml --harness cursor --dry-run
# 3. the real campaign:
python -m autotokamak.bench run --task benchmarks/tasks/L3_from_scratch.yaml --harness ursa --tag aug09
# 4. compare every run under a tag:
python -m autotokamak.bench compare --tag aug09
```

Each run writes `experiments/<tag>/<condition>/<run_id>/{workspace/,
trace.json, result.json}`; `result.json` bundles the harness outcome, the
contract gates, and (when the frozen test set exists) the head-to-head score.

## reference_runs/

Archived agent-generated workspaces from the pre-refactor capability tests
(`L3-ursa/`, `L3-dspy/` — formerly top-level `just_ursa/` and `just_dspy/`).
The READMEs are tracked documentation of those experiments; the workspaces
are agent output, kept on disk, gitignored, and NEVER edited.

## Prompt versioning

Task problem texts are frozen comparability assets: **runs are only
comparable within one prompt version.** Any change to a problem text goes
into a new `_v2`/`_v3` file with a bumped `prompt_version:` field and a
header explaining what changed and why — never an in-place edit. Each run's
`result.json` records `task.prompt_version` and the trace records the YAML's
sha256, so every result is attributable to its exact prompt. Version changes
must be process-level (engineering-discipline gates, identical for every
harness) — never physics/ML hints, and never per-harness. The version ladder
is itself data: what each added gate does to where agents fail is part of
the experiment.

Current versions: v1 = original capability-test text; v2 (2026-08-10) adds
the STORAGE VALIDATION GATE and DELIVERABLE SELF-TEST after the URSA agent
stored all-NaN datasets as successes and shipped an untested predict.py.

## Adding a harness

One adapter module in `src/autotokamak/harnesses/` implementing
`Harness.run(task, workspace, *, run_dir, model, timeout_seconds) → RunResult`
plus a registry entry in `harnesses/registry.py`. Smoke it with
`smoke.yaml` before any paid run.
