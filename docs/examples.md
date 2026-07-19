# Examples Guide

This document covers runnable simulation content under `examples/`.

## `examples/fixed_boundary`

Purpose:
- Demonstrates a fixed-boundary Grad-Shafranov solve with OpenFUSIONToolkit/TokaMaker.
- Includes analytic and EQDSK boundary modes.

Main script:
- `examples/fixed_boundary/run_fixed_boundary_equilibrium.py`

Quick run:

```bash
python examples/fixed_boundary/run_fixed_boundary_equilibrium.py --case analytic
```

## `examples/config_driven_equilibrium`

Purpose:
- Config-driven equilibrium workflow with discretization controls.
- Includes sweep and inversion helpers.

Main script:
- `examples/config_driven_equilibrium/run_equilibrium_from_config.py`

Quick run:

```bash
python examples/config_driven_equilibrium/run_equilibrium_from_config.py examples/config_driven_equilibrium/discretization_config.yaml
```

## `examples/dataset_generation`

Purpose:
- Phase-1 workspace: a fixed-boundary GS parameter sweep that writes a surrogate-training `dataset.h5`.

Layout:
- `fast/` — output of `python -m autotokamak.pipelines phase1 --mode fast` (in-process `run_sweep`).
- `ursa/` — the agent-generated `run_dataset_sweep.py` and its output (`--mode ursa`).

Quick run:

```bash
python -m autotokamak.pipelines phase1 --mode fast --n-samples 500
```

## `examples/surrogate_automl`

Purpose:
- Standalone Phase-2 workspace (created on first `pipelines phase2` run; not
  committed). Distinct from `surrogate_meta` below: this is one AutoML search
  over an existing dataset, not the self-improving loop.

Quick run:

```bash
python -m autotokamak.pipelines phase2 --mode fast --time-budget 600
```

## `examples/surrogate_meta`

Purpose:
- Meta-loop workspace: the self-improving Phase-1 → Phase-2 outer loop.

Layout:
- `fast/` and `ursa/`, each with a `manifest.json` (run_id, key paths, score).

Quick run:

```bash
python -m autotokamak.pipelines meta --mode fast --target-accuracy-pct 90 --max-iterations 5
```

## Outputs

- The two hand-authored examples (`fixed_boundary`, `config_driven_equilibrium`) write
  timestamped or hashed run artifacts under each example's local `outputs/` directory.
- Pipeline runs (`phase1`/`phase2`/`meta`) write under `examples/<workspace>/<mode>/` and
  emit a `manifest.json` plus a self-contained HTML report for the run.
