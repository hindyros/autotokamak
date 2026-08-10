# Agent Workflows

This document covers the `agent/` subtree only.

## Directory Map

- `agent/prompts/`: YAML prompts with `problem`, `workspace`, `model`, and `symlinks`.
- `agent/runners/config.py`: shared config loading and workspace path resolution.
- `agent/runners/plan_execute.py`: one-shot plan then execute.
- `agent/runners/plan_execute_feedback.py`: iterative re-plan and execute loop. Exposes `run_feedback_loop(...)` (the importable engine) plus `main()` (the YAML CLI).
- `agent/runners/meta_loop.py`: the autonomous **meta-loop** — drives Phase-1 → Phase-2 and, each round, picks `regen_dataset` / `extend_search` / `enrich_active` / `terminate`. This is what `python -m autotokamak.pipelines meta` runs; `run()` accepts a `phase2_decision_fn=` so the L0 scripted policy can drive it with no LLM. Phase-2 always runs in `structured` (library AutoML) mode under the pipelines CLI; the URSA-codegen hybrid is reachable only via the meta YAML config.
- `bench/scoring.py`, `bench/trace.py` (in `src/autotokamak/bench/`): shared run scorer and structured trace emission — moved here from `agent/runners/`.
- `agent/dspy/`: DSPy signatures + modules + metrics for the meta-action picker and the Phase-2 search-round picker (the L1 decision substrate; L0 scripted counterparts live in `src/autotokamak/policies/scripted.py`).
- `agent/orchestrator/`: meta-loop `actions.py` and `schema.py` (`MetaConfig`, `EnvelopeConfig`).

> Most users should drive the agent layer through the pipelines CLI
> (`python -m autotokamak.pipelines <phase1|phase2|meta> [--level L0|L1]`) rather than
> invoking these runners directly. The direct invocations below are the lower-level path.

## Typical Run

From repository root:

```bash
python -m autotokamak.agent.runners.plan_execute --config src/autotokamak/agent/prompts/oft_example_generation.yaml
```

Feedback variant:

```bash
python -m autotokamak.agent.runners.plan_execute_feedback --config src/autotokamak/agent/prompts/oft_discretization_example.yaml
```

## Benchmark CLI — the harness × access-level matrix

Agent-*written* pipeline code is not a pipelines-CLI mode; it is a benchmark
condition. Every experiment is one cell of a 2-axis matrix, named
`<level>-<harness>` (see `benchmarks/README.md` for the full framing):

- **Access level (Axis B):** L0 scripted / L1 DSPy-typed decisions run via
  `python -m autotokamak.pipelines <phase> --level <L0|L1>`. L2 (agent writes
  glue code, may import `autotokamak`) and L3 (from scratch, importing
  `autotokamak` is forbidden and audited) run via the bench CLI.
- **Harness (Axis A):** `ursa` · `dspy` · `claude_sdk` · `pi` · `cursor` ·
  `echo` (no-LLM mock for CI), implemented in `src/autotokamak/harnesses/`.

```bash
python -m autotokamak.bench run --task benchmarks/tasks/<smoke|L2_library|L3_from_scratch>.yaml \
    --harness <echo|ursa|dspy|claude_sdk|pi|cursor> [--model ...] [--tag ...] [--dry-run]
python -m autotokamak.bench compare --tag <tag>       # comparison table across a tag's runs
```

Each run writes `experiments/<tag>/<condition>/<run_id>/{workspace/,
trace.json, result.json}`, with the shared deliverable contract checked by
`src/autotokamak/bench/contract.py`.

## Prompt Conventions

- `workspace` should point to a path under `examples/` for generated simulation workspaces.
- `symlinks` usually includes read-only links to `./ursa` and `./OpenFUSIONToolkit`.
- Keep `CONSTRAINTS` blocks explicit and unchanged unless intentional.
