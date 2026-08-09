# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Library smoke test for the five bottleneck diagnostics.

Uses the synthetic low-rank fixture from ``test_surrogate_smoke.py`` so this
file has no dependency on a physical TokaMaker run. ~5s.
"""

from __future__ import annotations

import numpy as np
import pytest

from autotokamak.surrogate.dataset import DatasetBundle


def _synthetic_bundle(n: int = 16, nz: int = 8, nr: int = 6, seed: int = 0) -> DatasetBundle:
    """Same fixture used in test_surrogate_smoke; low-rank + NaN border."""
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
    psi[:, 0, :] = np.nan
    psi[:, -1, :] = np.nan
    psi[:, :, 0] = np.nan
    psi[:, :, -1] = np.nan
    return DatasetBundle(inputs=inputs, psi=psi, R=R, Z=Z, source_path="<synth>")


def _factory():
    from autotokamak.surrogate.zoo import make_poly_ridge

    return make_poly_ridge(alpha=0.1, degree=2)


def test_learning_curve_returns_curve_and_slope():
    from autotokamak.surrogate.diagnostics import learning_curve

    bundle = _synthetic_bundle()
    out = learning_curve(bundle, model_factory=_factory, sub_sizes=(8, 12, 16))
    assert isinstance(out["curve"], dict)
    assert len(out["curve"]) >= 2
    # Slope should be a finite number; interpretation should be one of two strings.
    assert out["slope_log_log"] is None or np.isfinite(out["slope_log_log"])


def test_cross_seed_variance_reports_cv():
    from autotokamak.surrogate.diagnostics import cross_seed_variance

    bundle = _synthetic_bundle()
    out = cross_seed_variance(bundle, model_factory=_factory, seeds=(0, 1, 2))
    assert "cv" in out
    assert "interpretation" in out


def test_pca_spectrum_explains_low_rank_data():
    from autotokamak.surrogate.diagnostics import pca_spectrum

    bundle = _synthetic_bundle()
    out = pca_spectrum(bundle, max_components=8)
    # Synthetic data is rank-2 + noise; first 2 components should explain >0.95.
    cum = out["cumulative"]
    assert cum[1] > 0.95


def test_residual_structure_after_real_fit(tmp_path):
    """End-to-end: train a real winner via automl, then probe residuals."""
    from autotokamak.surrogate.dataset import kfold
    from autotokamak.surrogate.diagnostics import residual_structure
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
            "rationale": "diag smoke",
            "models": [
                {
                    "name": "poly_ridge",
                    "n_trials": 2,
                    "search_space": DEFAULT_SEARCH_SPACES["poly_ridge"],
                }
            ],
        }
    )
    result = automl.run_study(spec, bundle=bundle, splits=splits, workdir=tmp_path)
    info = automl.refit_winner(
        result, bundle=bundle, splits=splits, save_to=tmp_path / "winner.pkl"
    )
    import joblib

    payload = joblib.load(tmp_path / "winner.pkl")
    out = residual_structure(payload, bundle, splits)
    assert "input_correlations" in out
    assert "spatial_pattern_variance" in out


def test_edge_hit_summary_collates_per_model():
    from autotokamak.surrogate.diagnostics import edge_hit_summary
    from autotokamak.surrogate.schema import (
        ModelStudyResult,
        SearchSpec,
        StudyResult,
        TrialRecord,
    )
    from autotokamak.surrogate.zoo import DEFAULT_SEARCH_SPACES

    spec = SearchSpec.model_validate(
        {
            "round": 1,
            "n_pca_components": 4,
            "val_metric": "psi_rmse",
            "action": "initial",
            "rationale": "synthetic",
            "models": [
                {
                    "name": "poly_ridge",
                    "n_trials": 1,
                    "search_space": DEFAULT_SEARCH_SPACES["poly_ridge"],
                }
            ],
        }
    )
    result = StudyResult(
        spec=spec,
        per_model=[
            ModelStudyResult(
                model_name="poly_ridge",
                n_trials=1,
                best_value=0.5,
                best_params={"alpha": 0.1, "degree": 2},
                edge_hit={"alpha": True, "degree": False},
                trials=[TrialRecord(number=0, value=0.5, params={"alpha": 0.1})],
            )
        ],
        storage_path="/tmp/synthetic",
    )
    out = edge_hit_summary(result)
    assert "poly_ridge" in out["per_model"]
    assert out["per_model"]["poly_ridge"]["edge_hit_params"] == ["alpha"]
