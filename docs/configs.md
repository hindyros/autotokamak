# Configuration Types

This repo uses a few distinct YAML configuration categories. The physics/sweep/surrogate
ones are backed by Pydantic v2 schemas with a `from_yaml` classmethod — validate against the
schema rather than hand-writing YAML.

## 1) Agent Task Configs (`agent/prompts/*.yaml`)

Used by URSA runners to define **what the agent should do**.

Common fields:
- `problem`
- `workspace`
- `model`
- `symlinks`

Example:
- `agent/prompts/oft_discretization_example.yaml`

Used with:

```bash
python -m autotokamak.agent.runners.plan_execute --config src/autotokamak/agent/prompts/oft_discretization_example.yaml
```

## 2) Simulation Physics/Discretization Configs — `EquilibriumConfig`

Schema: `autotokamak.core.schema.EquilibriumConfig` (`.from_yaml`). Defines a single
equilibrium solve. Top-level blocks:

- `boundary` (`BoundaryConfig`) — LCFS shape: `R0`, `a`, `kappa`, `delta`, `npts`.
- `mesh` (`MeshConfig`) — `regions` with per-region `dx`.
- `solver` (`SolverConfig`) — FE `order`, tolerances.
- `targets` (`TargetsConfig`) — e.g. total plasma current `Ip`, profiles.
- `init_psi` (`InitPsiConfig`) — initial-ψ `method` (`isoflux` | `tokamaker_default`).
- `outputs` (`OutputsConfig`) — output grid + artifact toggles.

Example: `examples/config_driven_equilibrium/discretization_config.yaml`, run with
`python examples/config_driven_equilibrium/run_equilibrium_from_config.py <config>.yaml`.

## 3) Phase-1 Dataset Sweep Configs — `SweepConfig`

Schema: `autotokamak.data.schema.SweepConfig` (`.from_yaml`). This is the
`dataset_config.yaml` the Phase-1 agent emits and that the meta-loop's `regen_dataset`
action builds programmatically. Blocks:

- `sampling` (`SamplingConfig`) — `method` ∈ `{lhs, uniform, sobol}` (default `lhs`), `n_samples` (1–10000), `seed`.
- `parameters` (`Dict[str, ParamBounds]`) — `low`/`high` for **exactly** the keys `r0, a, kappa, delta, Ip`.
- `fixed` (`FixedKnobs`) — non-per-sample knobs: `z0`, `F0` (default 0.10752), `npts` (default 80), `mesh_dx` (default 0.015), `solver_order` (1–2), `Ip_ratio`, `init_psi_method` (`isoflux` | `tokamaker_default`).
- `output_grid` (`OutputGrid`) — `R` and `Z` axes, each `{min, max, n}`.
- `output_path` — default `dataset.h5`.

Run with `python -m autotokamak.pipelines phase1 --mode fast --config <config>.yaml`.

## 4) Phase-2 Surrogate Configs — `SurrogateConfig`

Schema: `autotokamak.surrogate.schema.SurrogateConfig` (`.from_yaml`). Run-wide settings the
agent does **not** iterate on per round (per-round search decisions live in
`search_spec.json`). Fields:

- `dataset_h5` — relative path to the Phase-1 `dataset.h5`.
- `time_budget_seconds` — Optuna budget (10–3600, default 300).
- `n_pca_components_default` — default PCA rank (1–64, default 12).
- `seed`, `k_folds` (2–8, default 4), `test_frac` (default 2/16), `output_dir` (default `outputs`).

Run with `python -m autotokamak.pipelines phase2 --mode fast --dataset <dataset>.h5`.

## 5) Meta-loop Configs — `MetaConfig` / `EnvelopeConfig`

Schema: `autotokamak.agent.orchestrator.schema` — `MetaConfig` wraps the Phase-1/Phase-2
configs plus meta-loop budgets and an optional `eval_envelope` (`EnvelopeConfig`: `h5`,
`n_eval` (default 256), `parameters`, `seed`, `n_bins` (default 2)) used for the
worst-cell accuracy gate. Driven by `python -m autotokamak.pipelines meta`.

## Rule of Thumb

- Starts with `problem:` and mentions tasks/constraints → **agent task** config.
- `boundary`/`mesh`/`solver`/`targets` → **`EquilibriumConfig`** (single solve).
- `sampling`/`parameters`/`output_grid` → **`SweepConfig`** (Phase-1 dataset).
- `dataset_h5`/`time_budget_seconds` → **`SurrogateConfig`** (Phase-2 AutoML).
