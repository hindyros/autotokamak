"""End-to-end tests for the structured AutoML loop with a scripted decision_fn.

No LLM: decisions are supplied by plain functions, mirroring how
``test_meta_loop_e2e_mock`` scripts the meta action picker.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from autotokamak.surrogate.automl_loop import run_automl_loop
from autotokamak.surrogate.schema import RoundDecision, SurrogateReport
from autotokamak.surrogate.zoo import DEFAULT_SEARCH_SPACES

from tests.conftest import make_synthetic_h5


def _round_models(n_trials: int = 3) -> list[dict]:
    return [
        {"name": name, "n_trials": n_trials, "search_space": DEFAULT_SEARCH_SPACES[name]}
        for name in ("poly_ridge", "kernel_ridge")
    ]


def _scripted(decisions: list[RoundDecision]):
    it = iter(decisions)

    def decision_fn(ctx: dict) -> RoundDecision:
        assert "round" in ctx and "default_search_spaces" in ctx and "budget" in ctx
        return next(it)

    return decision_fn


def test_automl_loop_run_then_terminate(tmp_path: Path):
    ds = make_synthetic_h5(tmp_path / "ds.h5", n=16)
    shard = make_synthetic_h5(tmp_path / "shard.h5", n=4, seed=9)
    workdir = tmp_path / "ws"

    decision_fn = _scripted(
        [
            RoundDecision.model_validate(
                {
                    "action": "run_round",
                    "models": _round_models(),
                    "n_pca_components": 4,
                    "rationale": "initial round",
                }
            ),
            RoundDecision(action="terminate", rationale="good enough"),
        ]
    )

    out = run_automl_loop(
        dataset_h5=ds,
        workdir=workdir,
        decision_fn=decision_fn,
        max_rounds=4,
        time_budget_seconds=300,
        k_folds=4,
        seed=0,
        test_shard_h5=shard,
    )

    outputs = workdir / "outputs"
    for name in ("winner.pkl", "report.json", "study.db", "search_history.jsonl"):
        assert (outputs / name).is_file(), f"missing {name}"
    assert (workdir / "surrogate_config.yaml").is_file()

    report = SurrogateReport.model_validate_json((outputs / "report.json").read_text())
    assert report.terminated_by == "agent"
    assert report.n_outer_rounds == 1
    assert set(report.models_tried) == {"poly_ridge", "kernel_ridge"}
    assert out["terminated_by"] == "agent"
    assert out["n_rounds"] == 1

    # test RMSE was measured on the frozen shard (4 samples) and is finite.
    assert np.isfinite(out["test_psi_rmse"])
    assert report.model_extra["test_shard_path"] == str(shard)

    # Winner predicts shard-shaped output.
    import joblib

    from autotokamak.surrogate.optuna_search import predict_with_winner
    from autotokamak.surrogate.dataset import load_dataset

    payload = joblib.load(outputs / "winner.pkl")
    shard_bundle = load_dataset(shard)
    pred = predict_with_winner(payload, shard_bundle.inputs)
    assert pred.shape == shard_bundle.psi.shape

    # With an external shard, the refit used EVERY train-pool sample.
    assert len(payload["fit_indices"]) == 16
    assert payload["test_indices"] == []


def test_automl_loop_rounds_cap(tmp_path: Path):
    ds = make_synthetic_h5(tmp_path / "ds.h5", n=16)

    def never_terminate(ctx: dict) -> RoundDecision:
        return RoundDecision.model_validate(
            {"action": "run_round", "models": _round_models(2), "rationale": "again"}
        )

    out = run_automl_loop(
        dataset_h5=ds,
        workdir=tmp_path / "ws",
        decision_fn=never_terminate,
        max_rounds=2,
        time_budget_seconds=300,
        seed=0,
    )
    assert out["terminated_by"] == "rounds_cap"
    assert out["n_rounds"] == 2
    report = json.loads((tmp_path / "ws" / "outputs" / "report.json").read_text())
    assert report["terminated_by"] == "rounds_cap"
    # Two rounds of history were appended.
    lines = (tmp_path / "ws" / "outputs" / "search_history.jsonl").read_text().splitlines()
    assert len(lines) == 2


def test_automl_loop_no_winner_when_immediate_terminate(tmp_path: Path):
    ds = make_synthetic_h5(tmp_path / "ds.h5", n=16)
    out = run_automl_loop(
        dataset_h5=ds,
        workdir=tmp_path / "ws",
        decision_fn=_scripted([RoundDecision(action="terminate", rationale="nothing to do")]),
        max_rounds=2,
        seed=0,
    )
    assert out["winner"] is None
    assert out["n_rounds"] == 0
    assert not (tmp_path / "ws" / "outputs" / "winner.pkl").exists()


def test_automl_loop_grid_mismatch_raises(tmp_path: Path):
    ds = make_synthetic_h5(tmp_path / "ds.h5", n=16)
    bad_shard = make_synthetic_h5(tmp_path / "shard.h5", n=4, nz=10, nr=5)
    with pytest.raises(ValueError, match="grid"):
        run_automl_loop(
            dataset_h5=ds,
            workdir=tmp_path / "ws",
            decision_fn=_scripted([RoundDecision(action="terminate")]),
            test_shard_h5=bad_shard,
        )


def test_structured_scorer_on_loop_output(tmp_path: Path):
    ds = make_synthetic_h5(tmp_path / "ds.h5", n=16)
    shard = make_synthetic_h5(tmp_path / "shard.h5", n=4, seed=9)
    workdir = tmp_path / "ws"

    run_automl_loop(
        dataset_h5=ds,
        workdir=workdir,
        decision_fn=_scripted(
            [
                RoundDecision.model_validate(
                    {"action": "run_round", "models": _round_models(), "n_pca_components": 4}
                ),
                RoundDecision(action="terminate", rationale="done"),
            ]
        ),
        max_rounds=3,
        seed=0,
        test_shard_h5=shard,
    )

    from autotokamak.agent.dspy.metric_surrogate import score_surrogate_run

    rep = score_surrogate_run(workdir, mode="structured")
    assert rep.all_gates_pass, rep.summary()
    assert rep.total > 0.0
    assert rep.quality["runner_cleanliness"] == pytest.approx(1.0)
    assert "frozen shard" in rep.details.get("test_set", "")
