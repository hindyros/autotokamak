# Repository Architecture

This repository is the **`autotokamak`** package — a platform for ML surrogate
models of the Grad-Shafranov equation plus agentic LLM workflows that drive
TokaMaker simulations.

It is organized into these layers:

1. **`autotokamak.core`** — shared library of geometry, solver, I/O, diagnostics, schema, and logging utilities used by every higher layer.
2. **`autotokamak.pipelines`** — the unified `phase1 | phase2 | meta` CLI (`--mode fast|ursa`); the primary entry point that orchestrates everything below.
3. **Data layer** (`autotokamak.data`) — parameter-sweep dataset generation, HDF5 I/O, and active-learning acquisition + envelope evaluation.
4. **Surrogate layer** (`autotokamak.surrogate`) — structured Phase-2 AutoML: an Optuna model zoo plus the outer search loop.
5. **Agent orchestration layer** (`src/autotokamak/agent/`) — URSA runners, the meta-loop, DSPy pickers, and the orchestrator.
6. **Runnable simulation examples layer** (`examples/`) — hand-runnable OFT workflows and generated agent workspaces.
7. **Evaluation layer** (`autotokamak.eval`) — metrics, diagnostics, and PCA reduction.

`models/` is reserved for trained-model loaders and is still largely a stub; the active
surrogate model zoo lives in `autotokamak.surrogate.zoo`.

## Layer Boundaries

- `pipelines/` is the front door: each phase runs in `fast` mode (in-process library code) or `ursa` mode (a URSA agent writes and runs the code), and writes `examples/<workspace>/<mode>/manifest.json`.
- `agent/prompts/` contains task YAML prompts for URSA-driven runs.
- `agent/runners/` contains Python entrypoints: the URSA `plan_execute*` runners plus `meta_loop.py` (the autonomous Phase-1 → Phase-2 outer loop).
- `agent/dspy/` and `agent/orchestrator/` provide the LLM decision surface (action/search pickers) and the meta-loop action/schema layer.
- `examples/` contains hand-runnable OpenFUSIONToolkit workflows and generated artifacts.

The agent layer can generate or update content in `examples/`, but the examples are runnable without LLM involvement.

## Data Flow

```mermaid
flowchart TD
  cli[autotokamak.pipelines CLI<br/>phase1 pipe phase2 pipe meta, --mode fast pipe ursa]

  subgraph dataLayer [Data layer]
    sweep[data/sweep.py run_sweep]
    acquire[data/acquire.py + data/envelope.py<br/>active learning]
    dataset[dataset.h5]
  end

  subgraph surrogateLayer [Surrogate layer]
    automl[surrogate/automl_loop.py<br/>Optuna over surrogate/zoo.py]
    winner[winning surrogate + metrics]
  end

  subgraph agentLayer [Agent layer]
    meta[agent/runners/meta_loop.py]
    dspy[agent/dspy pickers]
  end

  core[autotokamak.core + OFT TokaMaker]
  examples[examples/&lt;workspace&gt;/&lt;mode&gt;/<br/>manifest.json + artifacts + report]

  cli --> sweep
  cli --> automl
  cli --> meta
  sweep --> core
  sweep --> dataset
  acquire --> dataset
  dataset --> automl
  automl --> winner
  meta --> dspy
  meta --> sweep
  meta --> automl
  winner --> examples
  meta --> examples
```

## Entry Points

- **Primary — pipelines CLI:** `python -m autotokamak.pipelines <phase1|phase2|meta> --mode <fast|ursa>`
  - Phase-1 dataset: `python -m autotokamak.pipelines phase1 --mode fast --n-samples 500`
  - Phase-2 AutoML: `python -m autotokamak.pipelines phase2 --mode fast --time-budget 600`
  - Meta-loop: `python -m autotokamak.pipelines meta --mode fast --target-accuracy-pct 90`
- Lower-level agent run: `python -m autotokamak.agent.runners.plan_execute --config src/autotokamak/agent/prompts/oft_example_generation.yaml`
- Lower-level feedback run: `python -m autotokamak.agent.runners.plan_execute_feedback --config src/autotokamak/agent/prompts/oft_discretization_example.yaml`
- Fixed-boundary example: `python examples/fixed_boundary/run_fixed_boundary_equilibrium.py --case analytic`
- Config-driven example: `python examples/config_driven_equilibrium/run_equilibrium_from_config.py examples/config_driven_equilibrium/discretization_config.yaml`
- Tests: `pytest tests/ -v` (`-m slow` to include the full-solve smoke test)

## `autotokamak.core` API summary

| Module | Public functions | Used by |
|---|---|---|
| `geometry` | `build_lcfs`, `build_mesh`, `build_mesh_from_config` | Both example runners; future data sweeps |
| `solver` | `make_solver`, `solve_equilibrium` (retry-on-isoflux-fail) | `config_driven_equilibrium` runner; future surrogate evaluation |
| `io` | `atomic_write_text`, `atomic_savez`, `unified_output_dir`, `utc_run_id` | All runners that write artifacts |
| `diagnostics` | `extract_scalars` | Summary generation, eval metrics |
| `logging` | `log`, `section`, `kv` (libc-flushed) | All runners; keeps OFT compiled output in order |
| `schema` | `EquilibriumConfig`, `SweepConfig`, `InvertConfig` (Pydantic v2, `from_yaml`) | Config validation; sweep + pipeline tooling |

## OFT singleton constraint

OpenFUSIONToolkit enforces **one `OFT_env` per Python kernel** — calling
`oft.OFT_env(...)` a second time raises. `core.solver.make_solver` accepts an
optional `env=` parameter for this reason: the retry path in
`solve_equilibrium` reuses the existing env rather than trying to create a
fresh one. Any batched solver (e.g. a training-data sweep that wants many
solves in one process) must follow the same pattern, OR run each solve in a
subprocess (which is what `examples/config_driven_equilibrium/forward_once.py`
does — recommended for sweeps).
