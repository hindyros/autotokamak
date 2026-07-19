# Development Notes

## Current Layout Convention

All code lives under the `src/autotokamak/` package namespace:

- `pipelines/` — the primary `phase1 | phase2 | meta` CLI (`--mode fast|ursa`).
- `core/` — shared geometry/solver/io/diagnostics/schema library.
- `data/` — sweep generation, HDF5 io, active-learning acquisition + envelope.
- `surrogate/` — structured Phase-2 AutoML (Optuna model zoo + search loop).
- `agent/` — URSA runners, the meta-loop, DSPy pickers (`agent/dspy/`), and the orchestrator (`agent/orchestrator/`).
- `eval/` — metrics, diagnostics, PCA reduction.
- `examples/` — hand-runnable simulation workspaces and generated agent workspaces.

## Migration Notes

Repository paths were refactored in two stages. First, the flat agent layout:
- `oft_generation_example/` -> `examples/fixed_boundary/`
- `oft_discretization_example/` -> `examples/config_driven_equilibrium/`
- `agent/plan_execute.py` -> `agent/runners/plan_execute.py`
- `agent/plan_execute_feedback.py` -> `agent/runners/plan_execute_feedback.py`
- `agent/config.py` -> `agent/runners/config.py`

Then everything moved under the `autotokamak` package namespace, so all modules are now
imported as `autotokamak.agent.runners.plan_execute` (etc.), **not** the bare `agent.…` path
that older docs and commands used. The unified `pipelines/` CLI was added on top as the
front door that dispatches to these runners.

## Keep In Mind

- Do not modify side clones `OpenFUSIONToolkit/` and `ursa/`.
- Keep prompt `workspace:` values aligned with paths under `examples/`.
- The `pipelines/` layer deliberately unifies the agent and simulation phases behind one CLI; keep phase-specific logic in `data/`, `surrogate/`, and `agent/runners/` rather than in the thin CLI wrappers.
