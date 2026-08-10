# Matching a target poloidal flux ψ with TokaMaker (parameter inversion)

This note describes **`run_invert_psi.py`**, an outer loop around the existing config-driven equilibrium in `run_equilibrium_from_config.py`. You list scalar knobs in the YAML (for example `solver.F0`, `targets.Ip`, or **`boundary.r0` / `boundary.a` / …**), run forward solves, and adjust those knobs so that the solved ψ is close to a **target** ψ (see **`optimize.psi_loss`** below when the mesh changes).

## Prerequisites

- Same as the main discretization example: **OpenFUSIONToolkit** on `PYTHONPATH`, working TokaMaker.
- **SciPy** (a core dependency in `pyproject.toml`, used for `scipy.optimize`).
- Your OFT build must expose **`TokaMaker.get_psi()`** returning a one-dimensional `numpy` array (some builds wrap the array in a length-one tuple; that case is handled).

## What is being minimized? (`optimize.psi_loss`)

Two modes are supported:

### `dof` (default)

Candidate ψ and reference ψ\* must be **the same length** (identical mesh and FE layout). The loss is **mean squared error** dof-by-dof after subtracting the **mean** from each vector (affine gauge in ψ) and **scaling** by the RMS of the reference. Use this when you only change `solver.*`, `targets.*`, etc., and **leave `boundary` and mesh resolution unchanged**.

### `sorted`

When **`boundary.*`** or **`mesh.*`** changes, the mesh and the length of **`get_psi()`** usually change, so `dof` mode cannot run. **`sorted`** compares **sorted samples of ψ** (resampled to `optimize.sorted_bins` points on the cumulative “rank” axis): it measures similarity of the **global distribution** of ψ values, **not** spatial agreement. It is weaker and non-physical as a full-field metric; use it for demos or coarse tuning, or prefer `dof` with a fixed mesh.

Optional: **`optimize.sorted_bins`** (default `512`), must be ≥ 16.

### Integer parameters

Paths ending in **`npts`** or **`order`** are rounded to **integers** when written into the config so `load_config` validation still passes.

## Commands

Run from `examples/config_driven_equilibrium/` (or pass absolute paths in the YAML).

### 1. Export a ψ vector to an NPZ file (optional)

Use this when you want `target.mode: npz` instead of a reference solve inside the invert YAML.

```bash
python run_invert_psi.py export-target discretization_config.yaml -o outputs/psi_target.npz
```

Optional overrides (merged on top of the base config before the solve):

```bash
python run_invert_psi.py export-target discretization_config.yaml \
  --overrides my_overrides.yaml -o outputs/psi_target.npz
```

The file contains at least the array **`psi`**.

### 2. Run inversion (default small example)

```bash
python run_invert_psi.py invert invert_psi_example.yaml
```

`invert_psi_example.yaml` keeps the **same LCFS / mesh** as `discretization_config.yaml`, varies only **`solver.F0`** and **`targets.Ip`**, uses **`psi_loss: dof`**, and adds a small **`regularization_lambda`** so the objective is well posed and SciPy usually converges cleanly.

## Inversion YAML schema

Top-level keys:

| Key | Meaning |
|-----|--------|
| `base_config` | Path to the usual equilibrium YAML (validated by `load_config` in `run_equilibrium_from_config.py`). Relative paths are resolved relative to the inversion YAML file. |
| `target` | How ψ\* is defined (see below). |
| `optimize` | Optimizer settings and the list of tunable parameters. |
| `outputs` | Optional. `out_dir` (default `outputs/invert_runs`) for artifacts; `plot_psi_stages` (default `true`) enables **ψ** comparison PNGs after the run. |

### `target`

- **`mode: reference`** — Merge `target.reference_overrides` onto a copy of `base_config`, run one forward solve, and treat its `get_psi()` as ψ\*.
- **`mode: npz`** — Set `target.npz_path` to an NPZ produced by `export-target`. The array **`psi`** must have the same length as `get_psi()` for runs that use the **same** `base_config` mesh and solver order.

### `optimize`

- **`method`**: `L-BFGS-B` (default, respects bounds), `Powell`, or `Nelder-Mead`. For the last two, trial vectors are **clipped** to `[min, max]` before each forward solve.
- **`psi_loss`**: `dof` (default) or `sorted` — see [What is being minimized?](#what-is-being-minimized-optimizepsi_loss).
- **`sorted_bins`**: Used only when `psi_loss: sorted`.
- **`maxiter`**, **`ftol`**: Passed through to SciPy where applicable.
- **`fail_penalty`**: Objective value if a forward solve raises (helps the optimizer skip bad regions).
- **`regularization_lambda`** (optional, default `0`): Adds a scaled quadratic penalty toward **`reference_parameters`** (difference from each reference value, divided by that parameter’s box width). Multiplied by `lambda` and added to the ψ loss. Stabilizes underdetermined ψ-only fits; set to `0` to disable.
- **`reference_parameters`**: Mapping of dotted path → float; required (non-empty) when `regularization_lambda > 0`.
- **`parameters`**: List of entries, each with:
  - **`path`**: Dotted path into the merged config, e.g. `solver.F0`, `targets.Ip`.
  - **`initial`**, **`min`**, **`max`**: Box constraints and starting point.

## Outputs (under `outputs.out_dir`)

| File | Content |
|------|--------|
| `invert_summary.json` | Final loss, optimizer message, best parameter values, tail of iteration log, and **`psi_plots`** paths when plotting succeeds. |
| `config_best.yaml` | Full equilibrium YAML with `base_config` plus best parameters applied. |
| `invert_result.npz` | `psi_best`, `psi_ref`, `x_best` for quick comparison in Python. |
| `psi_reference.png` | (If `target.mode: reference`) **TokaMaker.plot_psi** for the target equilibrium (`reference_overrides` on the base config). |
| `psi_initial.png` | ψ plot for the **starting** optimized parameters (same as the first objective evaluation after clipping to bounds). |
| `psi_best.png` | ψ plot after **inversion** (`config_best.yaml`). |

Each PNG is produced by a separate **`forward_plot_psi.py`** subprocess (full solve + plot), so inversion ends with a few extra solves when `outputs.plot_psi_stages` is true (default). Set `plot_psi_stages: false` to skip. For `target.mode: npz`, only **`psi_initial.png`** and **`psi_best.png`** are written (no reference YAML to plot).

## Multiple solves in one process (OFT quirk)

If you see errors like **`Path length exceeds OFT library allowable lenght of 0`** on the **second** forward solve while the first succeeds, that usually means TokaMaker / OFT does not tolerate a **second `OFT_env` in the same Python process**.

By default, **`run_invert_psi.py` runs each forward solve in a subprocess** via `forward_once.py`, so every evaluation starts a clean interpreter. To force the old single-process behavior (for debugging only), set:

```bash
export OFT_INVERT_SUBPROCESS=0
```

## Limitations (by design)

- This is **black-box** optimization: each iteration is a full equilibrium solve. Many parameters or tight tolerances can be expensive.
- **Uniqueness** is not guaranteed: different profiles can yield similar ψ; use a small parameter set and engineering judgment.
- If **`get_psi()`** is missing in your OFT build, export/invert will fail with a clear error; use a build that exposes the ψ vector.

## Related files

- `run_equilibrium_from_config.py` — forward solve and mesh build reused here.
- `invert_psi_example.yaml` — **default demo**: fixed boundary, **`psi_loss: dof`**, two parameters (`F0`, `Ip`) plus **`regularization_lambda`**. For harder cases (e.g. varying `boundary.*` so mesh dof counts differ), set **`psi_loss: sorted`** and expect a weaker, less physical metric.
