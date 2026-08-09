# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Library smoke tests for the Phase-2 surrogate stack.

These exercise eval/ + surrogate/ end-to-end with a tiny synthetic dataset
(no h5 file, no LLM). They catch wiring bugs before any expensive agent run.
Runs in seconds on a laptop.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from autotokamak.surrogate.dataset import DatasetBundle, kfold
from autotokamak.surrogate.metrics import (
    baseline_mean_predictor_rmse,
    pearson_r,
    pixelwise_max_err,
    psi_mae,
    psi_rmse,
    r2_score,
    relative_l2,
    summarize_psi_errors,
    within_rel_tolerance,
)
from autotokamak.surrogate.reduce import fit_pca, inverse_transform, transform


# ---------------- fixtures ----------------

def _synthetic_bundle(n: int = 16, nz: int = 8, nr: int = 6, seed: int = 0) -> DatasetBundle:
    """Build a low-rank synthetic dataset with NaN border (mimics outside-LCFS).

    psi[i] = a0[i] * basis0 + a1[i] * basis1 + small noise, then the outer
    ring is set to NaN. That gives PCA an easy target and lets us verify
    the NaN handling path.
    """
    rng = np.random.default_rng(seed)
    R = np.linspace(0.2, 0.7, nr)
    Z = np.linspace(-0.3, 0.3, nz)
    RR, ZZ = np.meshgrid(R, Z, indexing="xy")

    basis0 = np.exp(-((RR - 0.4) ** 2 + ZZ**2) / 0.05)
    basis1 = np.exp(-((RR - 0.55) ** 2 + ZZ**2) / 0.03)

    inputs = rng.uniform(size=(n, 5))
    a0 = inputs[:, 0]
    a1 = inputs[:, 1]
    psi = a0[:, None, None] * basis0[None, :, :] + a1[:, None, None] * basis1[None, :, :]
    psi += rng.normal(scale=1e-3, size=psi.shape)
    # Outside-LCFS ring -> NaN
    psi[:, 0, :] = np.nan
    psi[:, -1, :] = np.nan
    psi[:, :, 0] = np.nan
    psi[:, :, -1] = np.nan

    return DatasetBundle(inputs=inputs, psi=psi, R=R, Z=Z, source_path="<synthetic>")


# ---------------- tests ----------------

def test_kfold_partitions_all_non_test_samples():
    bundle = _synthetic_bundle()
    splits = kfold(bundle, k=4, test_frac=2 / 16, seed=0)
    assert splits.test_idx.size == 2
    # No overlap between test and any fold val/train.
    for _, tr, va in splits.iter_folds():
        assert np.intersect1d(tr, splits.test_idx).size == 0
        assert np.intersect1d(va, splits.test_idx).size == 0
        # Train + val cover the non-test set with no overlap.
        assert np.intersect1d(tr, va).size == 0


def test_kfold_val_union_equals_non_test():
    bundle = _synthetic_bundle()
    splits = kfold(bundle, k=4, test_frac=2 / 16, seed=0)
    all_val = np.concatenate([va for _, va in splits.folds])
    non_test = np.setdiff1d(np.arange(bundle.n_samples), splits.test_idx)
    assert np.array_equal(np.sort(all_val), non_test)


def test_pca_recovers_low_rank_signal_through_nan():
    bundle = _synthetic_bundle()
    splits = kfold(bundle, k=4, test_frac=2 / 16, seed=0)
    tr, va = splits.folds[0]

    model = fit_pca(bundle.psi[tr], n_components=4)
    coeffs = transform(model, bundle.psi[va])
    recon = inverse_transform(model, coeffs)

    # The synthetic data is rank-2 + noise; 4 components should capture > 99%.
    assert model.total_explained_variance > 0.99
    # NaN cells should NOT propagate into the dense reconstruction.
    assert np.all(np.isfinite(recon))


def test_metrics_handle_nan_alignment():
    bundle = _synthetic_bundle()
    truth = bundle.psi
    # "Prediction" identical to truth -> RMSE 0, relative_l2 0, max 0.
    assert psi_rmse(truth, truth) == pytest.approx(0.0)
    assert psi_mae(truth, truth) == pytest.approx(0.0)
    assert relative_l2(truth, truth) == pytest.approx(0.0)
    assert pixelwise_max_err(truth, truth) == pytest.approx(0.0)
    assert r2_score(truth, truth) == pytest.approx(1.0)
    assert pearson_r(truth, truth) == pytest.approx(1.0)
    assert within_rel_tolerance(truth, truth, rel_tol=0.05) == pytest.approx(1.0)
    summary = summarize_psi_errors(truth, truth, baseline_rmse=1.0)
    assert summary["pct_within_5pct"] == pytest.approx(100.0)
    assert summary["r2"] == pytest.approx(1.0)
    assert summary["rmse_vs_baseline"] == pytest.approx(0.0)


def test_baseline_beats_random_prediction():
    bundle = _synthetic_bundle()
    splits = kfold(bundle, k=4, test_frac=2 / 16, seed=0)
    tr, va = splits.folds[0]
    rng = np.random.default_rng(1)
    bad = rng.normal(scale=10.0, size=bundle.psi[va].shape)
    assert baseline_mean_predictor_rmse(bundle.psi[tr], bundle.psi[va]) < psi_rmse(
        bundle.psi[va], bad
    )


def test_run_study_minimal_two_models(tmp_path: Path):
    """End-to-end Optuna study + refit on synthetic data; tests every
    code path in automl.run_study + refit_winner + predict_with_winner.
    """
    from autotokamak.surrogate import optuna_search as automl
    from autotokamak.surrogate.schema import SearchSpec
    from autotokamak.surrogate.zoo import DEFAULT_SEARCH_SPACES

    bundle = _synthetic_bundle(n=16)
    splits = kfold(bundle, k=4, test_frac=2 / 16, seed=0)
    spec = SearchSpec.model_validate(
        {
            "round": 1,
            "n_pca_components": 4,
            "val_metric": "psi_rmse",
            "action": "initial",
            "rationale": "smoke",
            "models": [
                {
                    "name": "poly_ridge",
                    "n_trials": 3,
                    "search_space": DEFAULT_SEARCH_SPACES["poly_ridge"],
                },
                {
                    "name": "kernel_ridge",
                    "n_trials": 3,
                    "search_space": DEFAULT_SEARCH_SPACES["kernel_ridge"],
                },
            ],
        }
    )

    result = automl.run_study(spec, bundle=bundle, splits=splits, workdir=tmp_path)
    assert Path(result.storage_path).is_file()
    assert len(result.per_model) == 2
    for m in result.per_model:
        assert np.isfinite(m.best_value)
        assert m.n_trials >= 1

    summary = automl.summarize_study(result)
    assert summary["overall_best"]["model"] in {"poly_ridge", "kernel_ridge"}

    winner_path = tmp_path / "winner.pkl"
    automl.refit_winner(result, bundle=bundle, splits=splits, save_to=winner_path)
    assert winner_path.is_file()

    import joblib

    payload = joblib.load(winner_path)
    pred = automl.predict_with_winner(payload, bundle.inputs[splits.test_idx])
    assert pred.shape == (len(splits.test_idx), bundle.psi.shape[1], bundle.psi.shape[2])


def test_mlp_factory_rejects_oversized_layers():
    from autotokamak.surrogate.zoo import make_mlp

    with pytest.raises(ValueError):
        make_mlp(n_layers=3, layer_width=64)
    with pytest.raises(ValueError):
        make_mlp(n_layers=1, layer_width=512)
