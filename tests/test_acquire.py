# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Tests for the active-learning acquisition module (``data.acquire``).

All synthetic, no OFT: inputs live in [0,1]^5, psi fields are smooth
low-rank functions of the inputs (same construction as
``conftest.make_synthetic_h5``). Assertions on WHERE points land are kept
statistical-but-seeded so they are deterministic yet not brittle.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import smooth_psi as _smooth_psi

from autotokamak.data.acquire import (
    MIN_SUCCESS_FOR_GP,
    select_acquisition_points,
    select_from_dataset,
)

BOUNDS_LOW = np.zeros(5)
BOUNDS_HIGH = np.ones(5)


def _clustered_train(n: int = 20, hi: float = 0.35, seed: int = 3):
    """Successful solves confined to the low-corner sub-box [0, hi]^5."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, hi, size=(n, 5))
    psi = _smooth_psi(X)
    success = np.ones(n, dtype=bool)
    return X, success, psi


def test_gp_path_shape_bounds_determinism():
    X, success, psi = _clustered_train()
    kwargs = dict(
        attempted_inputs=X,
        success=success,
        psi=psi,
        bounds_low=BOUNDS_LOW,
        bounds_high=BOUNDS_HIGH,
        n_points=8,
        n_candidates=256,
        seed=42,
    )
    res1 = select_acquisition_points(**kwargs)
    res2 = select_acquisition_points(**kwargs)

    assert res1.method == "gp_variance"
    assert res1.X_new.shape == (8, 5)
    assert np.all(res1.X_new >= BOUNDS_LOW) and np.all(res1.X_new <= BOUNDS_HIGH)
    assert res1.scores.shape == (8,)
    assert res1.feasibility.shape == (8,)
    # No failures in the data -> feasibility weighting inactive.
    assert np.allclose(res1.feasibility, 1.0)
    # Seeded end to end: identical call -> identical batch.
    np.testing.assert_array_equal(res1.X_new, res2.X_new)
    # summary_dict must round-trip through JSON (goes into result.json).
    json.dumps(res1.summary_dict())


def test_gp_variance_targets_unexplored_region():
    """Train data lives in [0, 0.35]^5; acquisition must leave the cluster."""
    X, success, psi = _clustered_train()
    res = select_acquisition_points(
        attempted_inputs=X,
        success=success,
        psi=psi,
        bounds_low=BOUNDS_LOW,
        bounds_high=BOUNDS_HIGH,
        n_points=8,
        n_candidates=256,
        seed=0,
    )
    assert res.method == "gp_variance"
    # Predictive variance is minimal inside the sampled corner, so nearly
    # every selected point should escape it.
    outside = (res.X_new > 0.5).any(axis=1)
    assert outside.sum() >= 6, f"only {outside.sum()}/8 escaped the sampled corner"


def test_greedy_batch_is_diverse():
    """Exact variance updates must spread the batch, not pile on one peak."""
    X, success, psi = _clustered_train()
    res = select_acquisition_points(
        attempted_inputs=X,
        success=success,
        psi=psi,
        bounds_low=BOUNDS_LOW,
        bounds_high=BOUNDS_HIGH,
        n_points=8,
        n_candidates=256,
        seed=1,
    )
    d = np.sqrt(((res.X_new[:, None, :] - res.X_new[None, :, :]) ** 2).sum(axis=2))
    min_pairwise = d[np.triu_indices(8, k=1)].min()
    assert min_pairwise > 0.05, f"batch collapsed: min pairwise distance {min_pairwise}"
    # Scores are recorded at selection time; consuming points shrinks
    # variance, so the greedy sequence should be (weakly) decreasing.
    assert res.scores[0] >= res.scores[-1]


def test_feasibility_weighting_avoids_failure_region():
    """Failures concentrated at x0 > 0.7 must repel the batch from there."""
    rng = np.random.default_rng(7)
    # Successes: spread over x0 < 0.6.
    X_ok = rng.uniform(0.0, 0.6, size=(16, 5))
    # Failures: dense in the x0 > 0.7 slab.
    X_bad = rng.uniform(0.0, 1.0, size=(20, 5))
    X_bad[:, 0] = rng.uniform(0.72, 1.0, size=20)
    X = np.vstack([X_ok, X_bad])
    success = np.array([True] * 16 + [False] * 20)
    psi = np.full((36, 8, 6), np.nan)
    psi[:16] = _smooth_psi(X_ok)

    res = select_acquisition_points(
        attempted_inputs=X,
        success=success,
        psi=psi,
        bounds_low=BOUNDS_LOW,
        bounds_high=BOUNDS_HIGH,
        n_points=10,
        n_candidates=512,
        seed=5,
    )
    in_bad_slab = (res.X_new[:, 0] > 0.72).sum()
    assert in_bad_slab <= 3, f"{in_bad_slab}/10 points landed in the failure slab"
    assert (res.feasibility < 1.0).any(), "feasibility weighting never engaged"


def test_maximin_fallback_when_too_few_successes():
    rng = np.random.default_rng(0)
    n = MIN_SUCCESS_FOR_GP - 2
    X = rng.uniform(size=(n, 5))
    res = select_acquisition_points(
        attempted_inputs=X,
        success=np.ones(n, dtype=bool),
        psi=_smooth_psi(X),
        bounds_low=BOUNDS_LOW,
        bounds_high=BOUNDS_HIGH,
        n_points=5,
        n_candidates=256,
        seed=0,
    )
    assert res.method == "maximin_fallback"
    assert res.X_new.shape == (5, 5)
    assert np.all(res.X_new >= BOUNDS_LOW) and np.all(res.X_new <= BOUNDS_HIGH)
    assert "successful solves" in res.notes


def test_select_from_dataset_roundtrip(tmp_path: Path):
    """Wrapper reads the canonical h5 layout, failures included."""
    from tests.conftest import make_synthetic_h5

    h5 = make_synthetic_h5(tmp_path / "pool.h5", n=20, seed=0, n_failures=4)
    res = select_from_dataset(
        h5,
        bounds_low=BOUNDS_LOW,
        bounds_high=BOUNDS_HIGH,
        n_points=6,
        n_candidates=256,
        seed=0,
    )
    assert res.X_new.shape == (6, 5)
    assert res.method == "gp_variance"  # 16 successes >= MIN_SUCCESS_FOR_GP
    json.dumps(res.summary_dict())


def test_input_validation():
    with pytest.raises(ValueError, match="attempted_inputs"):
        select_acquisition_points(
            attempted_inputs=np.zeros((4, 3)),
            success=np.ones(4, dtype=bool),
            psi=np.zeros((4, 8, 6)),
            bounds_low=BOUNDS_LOW,
            bounds_high=BOUNDS_HIGH,
            n_points=2,
        )
    with pytest.raises(ValueError, match="n_points"):
        select_acquisition_points(
            attempted_inputs=np.zeros((4, 5)),
            success=np.ones(4, dtype=bool),
            psi=np.zeros((4, 8, 6)),
            bounds_low=BOUNDS_LOW,
            bounds_high=BOUNDS_HIGH,
            n_points=0,
        )
