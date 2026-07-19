# CLAUDE.md — Repo guide for `autotokamak`

## What this repo is

The **`autotokamak`** package — an evolving platform for two complementary research threads:

1. **ML surrogate models** for the Grad–Shafranov equation (fast approximations of TokaMaker's FEM solver).
2. **LLM-driven agentic workflows** (via [URSA](https://github.com/lanl/ursa)) that plan and run equilibrium computations end-to-end.

It builds on:

- **URSA** — LangChain/LangGraph-based `PlanningAgent` + `ExecutionAgent` pair.
- **OpenFUSIONToolkit (OFT) / TokaMaker** — the ground-truth Grad–Shafranov solver. Installed via `pip install OpenFUSIONToolkit>=26.6`.

## Top-level layout

```
autotokamak/                        # repo root
├── pyproject.toml                  # package metadata, deps, optional [ml] [dev]
├── src/autotokamak/                # the importable package
│   ├── core/                       # shared utilities (geometry, solver, io, diagnostics, logging, schema)
│   ├── pipelines/                  # PRIMARY entry point: phase1/phase2/meta CLI (--mode fast|ursa)
│   ├── agent/                      # URSA runners + prompts + DSPy + orchestrator
│   │   ├── runners/                # plan_execute, plan_execute_feedback, meta_loop, scoring, trace
│   │   ├── prompts/                # YAML prompts the agent consumes
│   │   ├── dspy/                   # DSPy signatures/modules/metrics for the meta + search pickers
│   │   └── orchestrator/           # meta-loop actions + schema (MetaConfig, EnvelopeConfig)
│   ├── data/                       # sweep generation, HDF5 io, active-learning acquisition + envelope
│   ├── surrogate/                  # structured AutoML: automl_loop, automl, zoo, schema (Optuna model zoo)
│   ├── models/                     # trained-model loaders (still mostly a stub)
│   └── eval/                       # metrics, diagnostics, reduce (PCA), eval data loaders
├── examples/                       # runnable demos + generated agent workspaces
│   ├── fixed_boundary/             # analytic + EQDSK demo (hardcoded physics)
│   ├── config_driven_equilibrium/  # YAML-driven runner + sweep + ψ inverter
│   ├── dataset_generation/         # Phase-1 workspace (fast/ + ursa/ outputs)
│   └── surrogate_meta/             # meta-loop workspace (fast/ + ursa/ outputs)
├── tests/                          # pytest suite (smoke + schema + geometry + e2e mocks)
├── data/                           # gitignored: raw/, processed/ for training datasets
├── models/                         # gitignored: checkpoints/
├── experiments/                    # gitignored: per-experiment configs and logs
├── docs/                           # architecture diagrams and design notes
└── outputs/                        # gitignored: per-run artifacts from example scripts
```

The **primary entry point** is the unified pipelines CLI:

```bash
python -m autotokamak.pipelines <phase1|phase2|meta> --mode <fast|ursa> [opts]
```

`fast` runs in-process library code; `ursa` has a URSA agent write and run the code.
Each run writes `examples/<workspace>/<mode>/manifest.json`. The `agent/runners/*`
modules below are the lower-level building blocks the CLI dispatches to.

## Two layers of code

### Layer 1 — Agent drivers (`src/autotokamak/agent/`)
These are the **agentic runners** that read a YAML prompt and let URSA do the work.

| File | What it does |
|---|---|
| `runners/plan_execute.py` | Plain plan → execute. PlanningAgent emits steps; ExecutionAgent runs each in turn, threading "previous-step summary" through the prompts. |
| `runners/plan_execute_feedback.py` | Same, plus a **feedback loop**: after execution, re-invoke the planner with the execution history so it can patch failures. Configurable via `feedback_rounds`, `validate_after`. |
| `runners/meta_loop.py` | The autonomous **meta-loop**: drives Phase-1 → Phase-2 and, each round, has the LLM pick `regen_dataset` / `extend_search` / `enrich_active` / `terminate`. Backs `pipelines meta`. |
| `runners/scoring.py` | Shared scorer used to rank rounds / decide early-stop gates. |
| `runners/trace.py` | Structured trace emission consumed by `report.py` and the DSPy trace loader. |
| `runners/config.py` | Shared YAML loader and workspace-path resolver. |

Two sibling packages support the agent layer:
- `agent/dspy/` — DSPy signatures + modules + metrics for the meta-action picker and the Phase-2 search-round picker (`signatures.py`, `module.py`, `metric_meta.py`, …).
- `agent/orchestrator/` — meta-loop `actions.py` and `schema.py` (`MetaConfig`, `EnvelopeConfig`).

Both runners:
- Load `.env` for `OPENAI_API_KEY`.
- `init_chat_model(model=...)` for both planner and executor (default `openai:o4-mini`).
- Create a workspace dir; symlink in `./ursa` and `./OpenFUSIONToolkit` so the agent can read them.

### Layer 2 — Generated example workspaces
These are **artifacts produced by Layer 1 agents** — concrete, hand-runnable TokaMaker examples. They have been committed to the repo so you can run them directly without invoking the agent.

| Dir | What it is | Key entry point |
|---|---|---|
| `examples/fixed_boundary/` | First demo: a fixed-boundary GS equilibrium with two cases (`analytic` vs `eqdsk`). Hardcoded physics; useful as a smoke test. | `python run_fixed_boundary_equilibrium.py --case analytic` |
| `examples/config_driven_equilibrium/` | More sophisticated: **fully config-driven** (no hardcoded `mesh_dx`, order, targets). Adds a discretization sweep runner and a ψ-inverter that tunes parameters to match a target flux map. | `python run_equilibrium_from_config.py discretization_config.yaml` |
| `examples/dataset_generation/` | Phase-1 workspace. `fast/` holds library-mode sweep output; `ursa/` holds the agent-generated `run_dataset_sweep.py` and its output. | `python -m autotokamak.pipelines phase1 --mode fast` |
| `examples/surrogate_meta/` | Meta-loop workspace with `fast/` and `ursa/` subdirs, each carrying a `manifest.json`. | `python -m autotokamak.pipelines meta --mode fast` |

## Prompts dir

`agent/prompts/*.yaml` — these are the inputs to Layer 1. Each contains:
- `problem:` — the natural-language task description given to the planner.
- `workspace:` — where the agent's outputs go (matches the Layer 2 dir names).
- `model:` — LLM string for `init_chat_model`.
- `symlinks:` — what to link into the workspace (always `./ursa` and `./OpenFUSIONToolkit`).
- Sometimes a `discretization_config_schema:` block that documents the expected YAML the agent should produce.

| Prompt file | Produced workspace | Purpose |
|---|---|---|
| `oft_example_generation.yaml` | `examples/fixed_boundary/` | "Build a fixed-boundary equilibrium example by reading the OFT notebook." |
| `oft_discretization_example.yaml` | `examples/config_driven_equilibrium/` | "Build a config-driven equilibrium example with a specified API surface." |
| `dataset_generation.yaml` | `examples/dataset_generation/ursa/` | "Build a fixed-boundary GS parameter sweep that writes a surrogate-training dataset." (Phase-1, ursa mode) |
| `surrogate_automl.yaml` | `examples/surrogate_automl/ursa/` | "Run a surrogate AutoML search over a generated dataset." (Phase-2, ursa mode) |
| `surrogate_meta.yaml` | `examples/surrogate_meta/ursa/` | "Drive the self-improving meta-loop over Phase-1 + Phase-2." (meta, ursa mode) |
| `representation_search.yaml` | (agent workspace) | "Search input/output representations for the surrogate." |

## How a typical run flows

```
src/autotokamak/agent/prompts/oft_discretization_example.yaml
        │
        ▼
python -m autotokamak.agent.runners.plan_execute --config src/autotokamak/agent/prompts/oft_discretization_example.yaml
        │
        ▼
PlanningAgent (LLM) reads problem → emits N steps
        │
        ▼
For each step:
   ExecutionAgent (LLM + tools) writes code, runs it, inspects output,
   passes summary to next step
        │
        ▼
Workspace (e.g. examples/config_driven_equilibrium/) populated with:
   - YAML config the agent generated
   - Python runner script
   - outputs/ dir with NPZ, JSON, PNG plots
   - README.md
```

After the agent finishes, the workspace is self-contained: you can re-run the example without involving any LLM at all.

## Setup

```bash
python3.11 -m venv venv && source venv/bin/activate

# Editable install of autotokamak with all dev tools
pip install -e ".[ml,dev]"

# OpenFUSIONToolkit binary + Python bindings (PyPI as of v26.6 — no /Applications install needed)
# Already included as a dependency in pyproject.toml; pip install above pulls it in.

# Optional: side-clones of OFT and URSA source for reference (not needed at runtime)
git clone https://github.com/OpenFUSIONToolkit/OpenFUSIONToolkit.git
git clone https://github.com/lanl/ursa.git

# Agent runners need OpenAI access:
echo 'OPENAI_API_KEY=sk-...' > .env
```

Python **must be 3.11 or 3.12** (some `ursa-ai==0.15.1` deps don't support 3.13+).

## What's gitignored

- `venv/`, `.env`
- `OpenFUSIONToolkit/`, `ursa/` (you side-clone these; they're not part of this repo)
- All build/cache dirs

## Physics one-liner

TokaMaker solves the **Grad–Shafranov equation**

```
Δ*ψ = −μ₀ R² p'(ψ) − F(ψ) F'(ψ)
```

on a 2D triangular mesh of a D-shaped plasma cross-section. Inputs: LCFS shape (R0, a, κ, δ), pressure/current profiles, total plasma current Ip. Output: flux function ψ(R,Z) plus derived quantities (q-profile, magnetic axis, etc.).

## Things to keep in mind when editing

- **Use `autotokamak.core`** for any geometry / solver / IO / logging logic. Don't duplicate it — extend the library.
- `examples/config_driven_equilibrium/run_equilibrium_from_config.py` is the **reference template** — config-driven, uses `core/`, extensible. Build new sweeps on this pattern.
- `examples/fixed_boundary/run_fixed_boundary_equilibrium.py` is a legacy first-pass demo. It still works but does not yet route through `core/`; treat it as a reference for the EQDSK-loading workflow.
- **OFT singleton**: only one `OpenFUSIONToolkit.OFT_env` can ever be created per Python kernel. `core.solver.make_solver` accepts an optional `env=` to reuse the existing one — required for any retry path or for batched solves in one process.
- Never write into side-cloned `./ursa/` or `./OpenFUSIONToolkit/` if you have them locally — they're read-only and gitignored.
- Agent prompts in `src/autotokamak/agent/prompts/*.yaml` contain hard `CONSTRAINTS:` blocks (no `git`, no `pip install`, no `input()`). Preserve these when editing.
- `outputs/`, `data/raw/`, `data/processed/`, `models/checkpoints/`, `experiments/` are all gitignored. Don't commit generated artifacts.
- Run `pytest tests/ -v` after structural changes; `pytest tests/ -v -m slow` to include the full OFT solve smoke test.
