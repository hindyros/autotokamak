# Surrogate models

autotokamak's Phase-2 surrogate goal: predict `ψ(R, Z)` given the five physics inputs `(r0, a, κ, δ, Ip)` faster than TokaMaker.

## The pipeline

```
inputs (N, 5)                        outputs (N, 96, 64)
      │                                     │
      │           dataset.h5                │
      └──────────────┬──────────────────────┘
                     │
              autotokamak.surrogate.dataset.load_dataset
                     │
                     ├─→ k-fold splits (surrogate.dataset.kfold)
                     │
                     ├─→ input featurization (currently just raw 5-vector)
                     │
                     ├─→ output reduction: PCA (surrogate.reduce.fit_pca)
                     │
                     ├─→ model zoo:
                     │       GP  |  kernel_ridge  |  poly_ridge  |  small MLP
                     │
                     ├─→ Optuna search per model (surrogate.automl.run_study)
                     │
                     └─→ winner.pkl, report.json
```

## Model zoo

Four classical sklearn models, defined in `autotokamak.surrogate.zoo`:

| Model | Strengths | Search space keys |
|---|---|---|
| `gp` (RBF + WhiteKernel) | Uncertainty estimates; small-N friendly | `length_scale`, `noise_level`, `alpha` |
| `kernel_ridge` | Cheaper than GP; no uncertainty | `alpha`, `gamma`, `kernel ∈ {rbf, laplacian}` |
| `poly_ridge` | Interpretable; fast; low-frequency bias | `alpha`, `degree ∈ {1, 2, 3}` |
| `mlp` (small MLPRegressor) | Universal approximator | `n_layers ≤ 2`, `layer_width ≤ 256`, `alpha`, `learning_rate_init` |

Deep-learning families (PINN, DeepONet, FNO) are **out of PoC scope** — the advisor's constraint per `docs/project_agenda.md`.

## Dataset shape

Written by `run_dataset_sweep.py`. Keys inside the HDF5:

- `inputs` — `(N, 5)`, columns `[r0, a, kappa, delta, Ip]`.
- `psi` — `(N, n_Z, n_R)`. Cells outside the LCFS are `NaN`. Shipped shape `(N, 96, 64)`.
- `R`, `Z` — 1D grid axes; shape `(n_R,)`, `(n_Z,)`.
- `success` — `(N,)` bool array, True where isoflux succeeded.
- `params_bounds`, `fixed_knobs`, `config_hash`, `oft_version` — provenance.

## Splits

`autotokamak.surrogate.dataset.kfold(bundle, k=4, test_frac=..., seed=0)` — random split, no stratification. At small N (~16), split luck dominates variance — LOO CV is sometimes better; see `docs/search_space.md` §E.

## Reduction

`autotokamak.surrogate.reduce.fit_pca(psi_train, n_components)` fits a PCA on the flattened, NaN-imputed `psi` array. Currently the only reduction supported. `n_components` is one of the AutoML search dimensions.

## Metrics

- **Inner objective:** `autotokamak.surrogate.metrics.psi_rmse(true, pred)` — full-grid RMSE in ψ units, averaged over folds.
- **Baseline:** `baseline_mean_predictor_rmse(train_psi, val_psi)` — mean-of-training-ψ predictor. Any winner must beat this.

## Predict-with-winner

```python
from autotokamak.surrogate.automl import predict_with_winner
pred = predict_with_winner(payload, X)   # X shape (M, 5); returns (M, n_Z, n_R)
```

`payload` is the dict pickled to `winner.pkl` — includes the pca handle, the trained regressor, and the winning hyperparams.

## Search space (full)

`docs/search_space.md` in the repo is the canonical reference — 400+ lines mapping every knob that can affect surrogate quality across ten layers (physics scope → geometry sampling → discretization → featurization → reduction → splits → model → training → search strategy → objective).

Key insight: **data quality dominates every downstream knob.** If `used_fallback=True` on every Phase-1 sample, no hyperparameter search recovers. If `n_samples=16`, no reduction choice helps. Fix Phase-1 first, then optimize Phase-2.

## What the agent controls today

`plan_execute_feedback` on `surrogate_automl.yaml`:

- Which subset of `{gp, kernel_ridge, poly_ridge, mlp}` to try each round.
- Per-model search space (widen, tighten, add).
- `n_trials` per model.
- `n_pca_components`.
- When to stop / widen / add / terminate.

Everything upstream (dataset generation, splits, featurization) is fixed per run.

## What `meta_loop` adds

The Phase-3 outer loop lets the agent break out of Phase 2's fixed dataset. Each iteration it picks:

- `regen_dataset(overrides)` — reshape or resample the dataset.
- `extend_search(focus)` — another Phase-2 with directives (e.g. "focus on MLP").
- `terminate(reason)` — stop.

Diagnosis is deterministic (`autotokamak.surrogate.diagnostics.run_all`); the action pick is LLM (DSPy-optimizable).
