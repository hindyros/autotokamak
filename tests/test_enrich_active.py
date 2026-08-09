# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Tests for the ``enrich_active`` meta-action (dispatcher + DSPy coercion).

The OFT solver never runs: ``actions.run_sweep`` is monkeypatched with a
fake that writes a canonical-layout HDF5 for exactly the acquisition-
selected points, so the test exercises the real path
acquire → sweep(X) → merge → refit without any physics.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import (
    fake_run_sweep_factory as _fake_run_sweep_factory,
    make_sweep_config as _make_sweep_config,
)


def test_enrich_active_dispatch(tmp_path: Path, monkeypatch):
    from tests.conftest import make_synthetic_h5

    from autotokamak.agent.orchestrator import actions
    from autotokamak.agent.orchestrator.actions import MetaState, dispatch
    from autotokamak.agent.orchestrator.schema import ActionDecision

    pool = make_synthetic_h5(tmp_path / "pool.h5", n=16, seed=0)
    calls: dict = {}
    monkeypatch.setattr(actions, "run_sweep", _fake_run_sweep_factory(calls))

    state = MetaState(
        workspace=tmp_path / "ws",
        current_dataset_h5=pool,
        base_sweep_config=_make_sweep_config(),
    )
    decision = ActionDecision.model_validate(
        {
            "action": "enrich_active",
            "enrich": {"n_new": 5, "rationale": "high variance off-corner"},
            "diagnosis": "sample-bottlenecked; uncertainty concentrated off-corner",
        }
    )
    result = dispatch(decision, state)

    assert result["kind"] == "enrich_active"
    assert calls["X"].shape == (5, 5)
    assert np.all(calls["X"] >= 0.0) and np.all(calls["X"] <= 1.0)
    assert result["n_new_requested"] == 5
    assert result["n_total"] == 16 + 5
    assert result["acquisition"]["method"] in {"gp_variance", "maximin_fallback"}
    assert result["acquisition"]["n_selected"] == 5

    # State advanced to the merged dataset; both artifacts on disk.
    merged = Path(result["dataset_path"])
    assert state.current_dataset_h5 == merged and merged.is_file()
    acq_json = tmp_path / "ws" / "datasets" / "iter0_acquisition.json"
    assert acq_json.is_file()
    assert json.loads(acq_json.read_text())["n_selected"] == 5

    # Merged pool loads cleanly and has grown.
    from autotokamak.surrogate.dataset import load_dataset

    assert load_dataset(merged).n_samples == 16 + 5


def test_enrich_active_requires_base_sweep_config(tmp_path: Path):
    from tests.conftest import make_synthetic_h5

    from autotokamak.agent.orchestrator.actions import MetaState, enrich_active
    from autotokamak.agent.orchestrator.schema import EnrichActivePayload

    state = MetaState(
        workspace=tmp_path / "ws",
        current_dataset_h5=make_synthetic_h5(tmp_path / "pool.h5", n=16),
    )
    with pytest.raises(RuntimeError, match="base_sweep_config"):
        enrich_active(EnrichActivePayload(n_new=3), state)


def test_enrich_active_refits_winner(tmp_path: Path, monkeypatch):
    """With a prior winner, enrichment must trigger the immediate-credit refit."""
    from tests.conftest import make_synthetic_h5, train_winner

    from autotokamak.agent.orchestrator import actions
    from autotokamak.agent.orchestrator.actions import MetaState, enrich_active
    from autotokamak.agent.orchestrator.schema import EnrichActivePayload
    from autotokamak.surrogate.dataset import load_dataset

    pool = make_synthetic_h5(tmp_path / "pool.h5", n=16, seed=0)
    shard = make_synthetic_h5(tmp_path / "shard.h5", n=4, seed=9)
    monkeypatch.setattr(actions, "run_sweep", _fake_run_sweep_factory({}))

    bundle = load_dataset(pool)
    state = MetaState(
        workspace=tmp_path / "ws",
        current_dataset_h5=pool,
        base_sweep_config=_make_sweep_config(),
        test_shard_h5=shard,
        best_winner_payload=train_winner(bundle.inputs, bundle.psi),
        best_winner_path=tmp_path / "winner.pkl",
        best_rmse=1e9,
    )
    result = enrich_active(EnrichActivePayload(n_new=4), state)
    assert "refit_shard_rmse" in result and result["refit_shard_rmse"] is not None
    assert result["refit_n_samples"] == 16 + 4


def test_coerce_enrich_active_payload():
    """DSPy coercion: enrich_active payload clamps n_new, tolerates junk."""
    dspy = pytest.importorskip("dspy")

    from autotokamak.agent.dspy.module import _coerce_to_action_decision

    pred = dspy.Prediction(
        action="enrich_active",
        diagnosis="uncertainty is concentrated near high kappa",
        rationale="targeted samples reduce variance fastest",
        payload_json=json.dumps({"n_new": 250}),
    )
    dec = _coerce_to_action_decision(pred)
    assert dec.action == "enrich_active"
    assert dec.enrich is not None and dec.enrich.n_new == 250
    assert dec.selected_payload() is dec.enrich

    # Mangled payload -> schema default, not a crash.
    pred_bad = dspy.Prediction(
        action="enrich_active",
        diagnosis="d",
        rationale="r",
        payload_json='{"n_new": "many"}',
    )
    dec_bad = _coerce_to_action_decision(pred_bad)
    assert dec_bad.action == "enrich_active"
    assert dec_bad.enrich.n_new == 100

    # Out-of-range clamps.
    pred_big = dspy.Prediction(
        action="enrich_active", diagnosis="d", rationale="r",
        payload_json='{"n_new": 99999}',
    )
    assert _coerce_to_action_decision(pred_big).enrich.n_new == 2000
