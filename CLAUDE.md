# CLAUDE.md — Repo guide for `autotokamak`

## What this repo is

The **`autotokamak`** package — a platform for one research question: **can a
coding agent execute the 4-step surrogate-ML pipeline for the Grad-Shafranov
equation, and how much does pre-written, proven library code help?** The four
steps: (1) generate data via TokaMaker sweeps, (2) feature-engineer (PCA,
folded inside training), (3) train surrogate models (model zoo + Optuna),
(4) decide what to do next (active sampling, random regen, extend search,
terminate).

Every experiment is one cell of a 2-axis matrix, named `<level>-<harness>`:

- **Axis A — harness (agent substrate)**: none / `ursa` / `dspy` /
  `claude_sdk` / `pi` / `cursor` (+ `echo`, a no-LLM mock for CI).
- **Axis B — access level**:
  - **L0 scripted** — pre-written pipeline, seeded heuristic decisions, zero
    LLM. The reproducible golden baseline (`policies/scripted.py`).
  - **L1 structured** — pre-written pipeline; an LLM makes only typed
    decisions via the DSPy pickers (`policies/llm.py`).
  - **L2 library-assisted** — agent writes the glue code, MAY import
    `autotokamak` (`benchmarks/tasks/L2_library.yaml`).
  - **L3 from-scratch** — agent writes everything; `autotokamak` import is
    forbidden and AUDITED by a hard contract gate
    (`benchmarks/tasks/L3_from_scratch.yaml`).

All conditions emit the same contract (manifest/trace, `report.json`,
`predict.py` on the frozen 64×96 R/Z grid, relative-L2 vs a frozen test set)
so every cell is directly comparable. See `benchmarks/README.md`.

It builds on **OpenFUSIONToolkit (OFT) / TokaMaker** — the ground-truth
Grad-Shafranov solver (`pip install OpenFUSIONToolkit>=26.6`) — and, for the
`ursa` harness, **[URSA](https://github.com/lanl/ursa)** (LangChain/LangGraph
`PlanningAgent` + `ExecutionAgent`).

## Two entry points

### 1. Pre-written pipeline (levels L0/L1)

```bash
python -m autotokamak.pipelines phase1 [--n-samples 500]          # data gen (no decisions)
python -m autotokamak.pipelines phase2 --level L0|L1 [opts]       # surrogate AutoML
python -m autotokamak.pipelines meta   --level L0|L1 [opts]       # autonomous outer loop
```

Outputs land in `examples/<workspace>/<level>/` with a `manifest.json`
(keys: `pipeline`, `level`, `condition`, `run_id`, metrics). L0 is seeded and
LLM-free — same seed ⇒ same decisions.

### 2. Agent benchmark (levels L2/L3, any harness)

```bash
python -m autotokamak.bench run  --task benchmarks/tasks/L3_from_scratch.yaml \
                                 --harness claude_sdk [--model ...] [--tag aug09] [--dry-run]
python -m autotokamak.bench validate --workspace <ws> --task <task.yaml>
python -m autotokamak.bench compare  --tag aug09
python -m autotokamak.bench freeze-testset     # ground truth for head-to-head scoring
```

Each run writes `experiments/<tag>/<condition>/<run_id>/{workspace/,
trace.json, result.json}` — the result bundles harness outcome, contract
gates, and the frozen-test-set score. Always smoke a harness with
`benchmarks/tasks/smoke.yaml` before a paid campaign.

## Top-level layout

```
autotokamak/                        # repo root
├── pyproject.toml                  # deps; extras: [ml] [dev] [harnesses]
├── src/autotokamak/
│   ├── core/                       # geometry, OFT solver wrapper, io, schema
│   ├── data/                       # sweeps (run_sweep), HDF5 io, active-learning acquire, envelope
│   ├── surrogate/                  # dataset/PCA/metrics, model zoo, optuna_search, automl_loop
│   ├── policies/                   # L0 scripted + L1 LLM decision providers (search + meta pickers)
│   ├── pipelines/                  # phase1/phase2/meta CLI (--level L0|L1) + discover.py
│   ├── bench/                      # TaskSpec, RunTrace, scoring, deliverable contract, compare CLI
│   ├── harnesses/                  # one adapter per agent substrate + registry + echo mock
│   └── agent/                      # URSA engine (runners/), DSPy pickers + GEPA (dspy/),
│                                   #   meta-loop actions/schema (orchestrator/), prompts/
├── benchmarks/                     # task YAMLs, frozen eval assets, matrix README, reference_runs/
├── examples/                       # runnable demos + L0/L1 pipeline workspaces
│   ├── fixed_boundary/             # legacy analytic + EQDSK demo (hardcoded physics)
│   ├── config_driven_equilibrium/  # reference template: YAML-driven runner + sweep + ψ inverter
│   ├── dataset_generation/         # phase1 workspace (L0/; dataset_config.yaml is canonical)
│   └── surrogate_meta/             # meta workspace (L1/ + archives)
├── notebooks/                      # manual surrogate + scaling-law notebooks (run dirs gitignored)
├── tests/                          # pytest (offline mocks + slow OFT smoke behind -m slow)
├── tools/                          # post-run analysis scripts (see tools/README.md)
├── experiments/                    # gitignored: per-run artifacts, incl. all bench runs
└── docs/                           # design notes; paper drafts under docs/paper/
```

## Where each pipeline step lives

| Step | Library | Decision point (L0/L1) |
|---|---|---|
| 1 data gen | `data.sweep.run_sweep` (config or explicit (N,5) matrix) | — |
| 2 features | `surrogate.reduce.fit_pca` (inside phase-2, refit per fold/growth) | PCA components picked per round |
| 3 training | `surrogate.zoo` + `optuna_search` + `automl_loop(decision_fn=...)` | `RoundDecision` per round |
| 4 meta | `agent.orchestrator.actions` (regen_dataset / enrich_active / extend_search / terminate) + `data.acquire` (residual-UCB, PCA-GP) | `ActionDecision` per iteration via `meta_loop.run(pick_action=...)` |

The decision providers come from `autotokamak.policies.get_search_policy` /
`get_meta_policy` (`"scripted"` = L0, `"llm"` = L1). Tests inject scripted
callables the same way.

## Harness adapters (`src/autotokamak/harnesses/`)

One module per substrate, all implementing
`Harness.run(task, workspace, *, run_dir, model, timeout_seconds) → RunResult`:

| Adapter | Invocation | Env |
|---|---|---|
| `ursa.py` | `agent.runners.plan_execute_feedback.run_feedback_loop` | `OPENAI_API_KEY` |
| `dspy_harness.py` | DSPy plan → per-step ReAct → review → fix rounds | `OPENAI_API_KEY` |
| `claude_sdk.py` | `claude_agent_sdk.query()`; `setting_sources=[]` is **essential** (keeps this repo's `.claude/` out of the benchmarked agent) | `ANTHROPIC_API_KEY` / CLI login |
| `pi.py` | `pi --mode json -p --no-session` (subprocess `cwd=workspace` is load-bearing) | provider key |
| `cursor.py` | `cursor-agent -p --output-format stream-json --trust --force` + hard timeout | `CURSOR_API_KEY` / login |
| `echo.py` | no LLM; CI path for the whole bench machinery | — |

Adding a substrate = one adapter module + a `registry.py` entry.

## Setup

```bash
python3.11 -m venv venv && source venv/bin/activate
pip install -e ".[ml,dev,harnesses]"     # OFT comes in via dependencies

# Optional side-clones for reference docs (read-only, gitignored):
git clone https://github.com/OpenFUSIONToolkit/OpenFUSIONToolkit.git
git clone https://github.com/lanl/ursa.git

cp .env.example .env    # fill in the keys for the substrates you run
```

Python **must be 3.11 or 3.12** (`ursa-ai==0.15.1` deps don't support 3.13+).
`pi` and `cursor-agent` are npm-installed CLIs (see `pyproject.toml` comment);
`claude_sdk` also needs the Claude Code CLI on PATH.

## Physics one-liner

TokaMaker solves the **Grad–Shafranov equation** `Δ*ψ = −μ₀R²p'(ψ) − F(ψ)F'(ψ)`
on a 2D triangular mesh of a D-shaped plasma cross-section. Inputs: LCFS shape
(r0, a, κ, δ), profiles, plasma current Ip. Output: flux ψ(R,Z) — physical Wb,
NaN outside the boundary — plus derived quantities.

## Things to keep in mind when editing

- **Use `autotokamak.core`** for geometry/solver/IO, `autotokamak.data` for
  sweeps/acquisition, `autotokamak.surrogate` for training. Don't duplicate —
  extend the library. `examples/config_driven_equilibrium/run_equilibrium_from_config.py`
  is the reference template for new sweep-style code.
- **Never edit agent-generated output** (benchmark workspaces,
  `benchmarks/reference_runs/`, `experiments/`). Fix the platform — prompts,
  tasks, contract, policies — and re-run.
- **OFT singleton**: only one `OpenFUSIONToolkit.OFT_env` per Python process.
  `core.solver.make_solver` accepts `env=` for reuse; harness/agent code runs
  solves in child processes.
- **Benchmark comparability is sacred**: don't change the frozen grid
  (`benchmarks/assets/eval_grid.json`), the test params
  (`benchmarks/assets/test_params.json`), or the deliverable contract
  (`bench/contract.py`) without versioning the change — old runs become
  incomparable silently.
- Task YAMLs in `benchmarks/tasks/` and prompts in `agent/prompts/` contain
  hard `CONSTRAINTS:` blocks (no `git`, no `pip install`, no `input()`).
  Preserve them. The L2/L3 problem texts must stay identical except the
  access-level paragraphs.
- Run artifacts belong under `experiments/<tag>/`; never create new repo-root
  dirs for generated runs. `outputs/`, `logs/`, example run dirs
  (`examples/*/L0|L1|L2-*`), heavy artifacts (`*.h5`, `*.pkl`) are gitignored.
- DSPy serves two distinct roles — L1 typed pickers + GEPA prompt optimization
  (`agent/dspy/optimize_meta.py`), and the `dspy` harness substrate
  (`harnesses/dspy_harness.py`). Don't conflate them.
- Run `pytest tests/ -v` after structural changes; `-m slow` adds the full OFT
  solve smoke test. Echo-harness bench runs are free — use them to test bench
  machinery changes.
