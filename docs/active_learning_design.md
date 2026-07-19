# Active-learning design — residual-driven acquisition over a target envelope

Status: implemented 2026-07-17. This doc records the design decisions behind the
phase-3 ("smart active sampling") subsystem and the judgment calls made.

## Goal

1. Generate seed data over a fixed geometry box (Phase-1, unchanged).
2. Train a good surrogate with data engineering + HPO (Phase-2, unchanged).
3. **Based on observed model performance**, actively sample new tokamak
   geometries where the model is weak, and grow the training pool (Phase-3).

## Decisions (with rationale)

### D1 — Acquisition signal: out-of-fold residuals of the CURRENT winner

The original `data.acquire` scored candidates by the predictive variance of an
*independent* PCA-GP emulator — model-agnostic uncertainty, not "observed model
performance". The residual-driven path replaces the middle of that pipeline:

- Compute **out-of-fold (OOF) residuals** of the current winner on the train
  pool: refit the winner's architecture + hyperparameters per CV fold, record
  each sample's mean-|ψ error| when it was in the val fold. OOF (not
  train-set) residuals are required because GP/KRR interpolate their training
  points to ~0 error.
- Fit an **error model** `g : geometry(5) → log |residual|` (ARD-RBF GP) on
  those residuals.
- Acquisition score (log space): `μ_g(x) + β·σ_g(x) + log P(feasible)` — a
  **UCB**: exploit measured weakness, explore where the error model is
  uncertain (which automatically covers envelope regions with no data yet).
- Greedy batch with exact posterior-variance conditioning (kriging believer:
  the GP *mean* is unchanged by a hallucinated observation at its own mean, so
  only σ shrinks near picked points → batch diversity).

**Why not score residuals on the eval set?** Steering training with the frozen
eval set's residuals would make the headline metric adaptive (mild leakage).
OOF residuals on the train pool are leakage-free, and UCB exploration handles
the not-yet-sampled envelope.

Fallback ladder (each recorded in `AcquisitionResult.notes`):
`residual_ucb` → `gp_variance` (proxy, no winner needed) → `maximin_fallback`.

### D2 — Evaluation: one frozen, full-envelope, per-region eval set

Two nested boxes:

- **Seed box** — the Phase-1 sampling bounds (shipped defaults:
  R₀∈[0.35,0.55], a∈[0.10,0.20], κ∈[1.0,1.6], δ∈[0.0,0.4], Iₚ∈[80,200] kA).
- **Target envelope** — the full geometry space the surrogate must cover.
  Recommended: seed box widened ~25% per side, clipped to physically sane
  values (κ ≥ 1.0, δ ≥ 0.0, a < 0.5·R₀ low-end; expect the feasibility model
  to absorb the extra fallback rate near the envelope edges — run
  `probe_feasible` before committing to an envelope).

The eval set is drawn **once by LHS over the envelope**, solved, frozen, and
never grown (`meta_loop` uses it in place of the legacy seed-region
`test_shard.h5`; the whole initial dataset then becomes train pool). Configured
via `MetaConfig.eval_envelope` — either a pre-solved `h5:` path or a
generation spec (`n_eval`, `parameters`, `seed`). Default `n_eval=256`.

Scoring is stratified: each eval sample is binned into a joint geometry cell
(2 bins/dim → 32 cells) plus per-parameter tercile marginals
(`surrogate.metrics.per_cell_errors`). Headline numbers: `worst_cell_rmse`,
`mean_cell_rmse`, `cells_covered` — these show *which* geometries improved,
which is the whole point of active sampling.

When `eval_envelope` is not configured, the legacy behavior (split a shard
from the seed dataset) is preserved — every existing config keeps working.

### D3 — Acquisition knobs exposed to the meta-agent

`EnrichActivePayload` gains `beta` (UCB exploration weight, clamped [0, 3],
default 1.0) and `n_new` (clamped [1, 2000]). β=0 → pure exploitation (may
cluster); β large → approaches pure exploration.

Two more fields are also LLM-exposed (coercion repairs junk):
- `strategy` ∈ {`auto`, `residual_ucb`, `uncertainty`, `space_filling`} (unknown/omitted → `auto`, letting the system pick the acquisition path). Note these agent-facing names map onto the fallback-ladder `AcquisitionResult.method` values `residual_ucb → gp_variance → maximin_fallback`: `uncertainty` ≙ `gp_variance`, `space_filling` ≙ `maximin_fallback`.
- `feasibility_weighting` (bool, default `True`).

Everything else (candidate pool size, GP kernels, k-folds) stays deterministic
library code — reproducible science, not an LLM surface.

Acquisition samples over the **envelope** bounds when an envelope is
configured, else over the base sweep bounds (legacy).

### D4 — Division of labor: DSPy decides, URSA authors

| Job | Owner |
|---|---|
| Meta action choice (regen/enrich/extend/terminate) | DSPy `MetaActionPicker` (GEPA-optimizable) |
| Nested Phase-2 round choice | DSPy `SearchRoundPicker` |
| Acquisition knobs (`n_new`, `beta`) | DSPy (via `MetaActionPicker` payload) |
| Acquisition core (residual GP, UCB, batch) | Deterministic library (`data.acquire`) |
| **Representation / data-engineering search** | **URSA codegen** (`prompts/representation_search.yaml`) |

URSA's genuine open-ended role: the data-engineering layer (currently frozen
at physical-ψ → PCA) becomes URSA's search surface — it reads the model
diagnostics and authors/evaluates alternative representations (input
featurization, output transforms, PCA alternatives) under a fixed eval
protocol. This satisfies the funder constraint with real work, not ceremony.

### D5 — Control arm

`regen_dataset` (blind LHS append) is retained unchanged as the A/B control.
The headline experiment: same seed dataset, same budgets, `enrich_active` vs
`regen_dataset`, both scored on the same frozen envelope eval set —
active sampling must win on `worst_cell_rmse` per OFT solve spent.

## Data flow (enrich_active, with winner + envelope)

```
train_pool.h5 ──► OOF residuals of winner (k-fold refit)
                        │
                        ▼
             GP  g: x → log|resid|      feasibility GPC (successes+failures)
                        │                     │
                        └────► UCB × P(feas) over Sobol pool in ENVELOPE box
                                      │  greedy diverse batch (n_new)
                                      ▼
                      run_sweep(X) ──► merge ──► refit winner ──► shard credit
```
