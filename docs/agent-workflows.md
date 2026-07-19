# Agent Workflows

This document covers the `agent/` subtree only.

## Directory Map

- `agent/prompts/`: YAML prompts with `problem`, `workspace`, `model`, and `symlinks`.
- `agent/runners/config.py`: shared config loading and workspace path resolution.
- `agent/runners/plan_execute.py`: one-shot plan then execute.
- `agent/runners/plan_execute_feedback.py`: iterative re-plan and execute loop.
- `agent/runners/meta_loop.py`: the autonomous **meta-loop** — drives Phase-1 → Phase-2 and, each round, has the LLM choose `regen_dataset` / `extend_search` / `enrich_active` / `terminate`. This is what `python -m autotokamak.pipelines meta` runs.
- `agent/runners/scoring.py`, `agent/runners/trace.py`: shared round scorer and structured trace emission.
- `agent/dspy/`: DSPy signatures + modules + metrics for the meta-action picker and the Phase-2 search-round picker.
- `agent/orchestrator/`: meta-loop `actions.py` and `schema.py` (`MetaConfig`, `EnvelopeConfig`).

> Most users should drive the agent layer through the pipelines CLI
> (`python -m autotokamak.pipelines <phase1|phase2|meta> --mode <fast|ursa>`) rather than
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

## Prompt Conventions

- `workspace` should point to a path under `examples/` for generated simulation workspaces.
- `symlinks` usually includes read-only links to `./ursa` and `./OpenFUSIONToolkit`.
- Keep `CONSTRAINTS` blocks explicit and unchanged unless intentional.
