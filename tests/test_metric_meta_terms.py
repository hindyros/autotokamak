# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Unit tests for the corrected ``metric_meta`` quality terms."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autotokamak.agent.dspy.metric_meta import WEIGHTS, score_meta_run


def _write_meta_workspace(
    ws: Path,
    *,
    baseline_rmse: float,
    final_rmse: float | None,
    rmse_afters: list[float | None],
    terminated_by: str = "agent",
) -> Path:
    """Minimal meta workspace that passes all hard gates."""
    ws.mkdir(parents=True, exist_ok=True)

    report = {
        "n_iterations": len(rmse_afters),
        "terminated_by": terminated_by,
        "final_rmse": final_rmse,
        "baseline_rmse": baseline_rmse,
        "winner_model_name": "poly_ridge",
        "winner_hyperparams": {"alpha": 0.1},
        "rmse_history": [v for v in rmse_afters if v is not None],
        "actions_taken": ["extend_search"] * len(rmse_afters),
    }
    (ws / "report.json").write_text(json.dumps(report))

    iterations = [
        {
            "iteration": i,
            "started_utc": "2026-01-01T00:00:00Z",
            "decision": {"action": "extend_search", "diagnosis": "edge hits on alpha range"},
            "rmse_after": v,
        }
        for i, v in enumerate(rmse_afters)
    ]
    (ws / "meta_trace.json").write_text(json.dumps({"iterations": iterations}))

    # winner.pkl only needs the {estimator, pca} keys to pass winner_predicts.
    import joblib

    joblib.dump({"estimator": object(), "pca": object()}, ws / "winner.pkl")
    return ws


def test_weights_sum_to_one_and_no_diagnosis_term():
    assert "diagnosis_consistency" not in WEIGHTS
    assert sum(WEIGHTS.values()) == pytest.approx(1.0)


def test_budget_efficiency_early_improvement_scores_high(tmp_path: Path):
    # 4 iterations; all improvement (1.0 -> 0.5) lands in the first half.
    ws = _write_meta_workspace(
        tmp_path / "ws",
        baseline_rmse=1.0,
        final_rmse=0.5,
        rmse_afters=[0.5, 0.5, 0.5, 0.5],
    )
    rep = score_meta_run(ws)
    assert rep.all_gates_pass
    assert rep.quality["budget_efficiency"] == pytest.approx(1.0)


def test_budget_efficiency_late_only_improvement_scores_zero(tmp_path: Path):
    # No improvement over baseline in the first half; all of it in the last iteration.
    ws = _write_meta_workspace(
        tmp_path / "ws",
        baseline_rmse=1.0,
        final_rmse=0.5,
        rmse_afters=[1.2, 1.1, 1.0, 0.5],
    )
    rep = score_meta_run(ws)
    assert rep.quality["budget_efficiency"] == pytest.approx(0.0)


def test_budget_efficiency_worsening_scores_zero(tmp_path: Path):
    # RMSE never beats baseline -> no efficiency credit (the OLD formula gave
    # this case a perfect 1.0 because final/best_early > 1 clipped to 1).
    ws = _write_meta_workspace(
        tmp_path / "ws",
        baseline_rmse=1.0,
        final_rmse=1.5,
        rmse_afters=[1.2, 1.5],
    )
    rep = score_meta_run(ws)
    assert rep.quality["budget_efficiency"] == pytest.approx(0.0)


def test_budget_efficiency_single_improving_iteration(tmp_path: Path):
    ws = _write_meta_workspace(
        tmp_path / "ws",
        baseline_rmse=1.0,
        final_rmse=0.7,
        rmse_afters=[0.7],
    )
    rep = score_meta_run(ws)
    assert rep.quality["budget_efficiency"] == pytest.approx(1.0)


def test_final_rmse_none_parses_and_scores_zero_quality(tmp_path: Path):
    ws = _write_meta_workspace(
        tmp_path / "ws",
        baseline_rmse=1.0,
        final_rmse=None,
        rmse_afters=[None],
    )
    rep = score_meta_run(ws)
    # Gates pass (report parses with final_rmse null); the headline quality
    # term is zero and noted in details.
    assert rep.hard_gates["report_parseable"]
    assert rep.quality["final_rmse_vs_baseline"] == 0.0
    assert rep.details.get("final_rmse") == "no winner produced"


def test_no_waste_flat_run_scores_zero(tmp_path: Path):
    # A winner exists after iteration 1; iterations 2-3 don't move the RMSE.
    ws = _write_meta_workspace(
        tmp_path / "ws",
        baseline_rmse=1.0,
        final_rmse=0.5,
        rmse_afters=[0.5, 0.5, 0.5],
    )
    rep = score_meta_run(ws)
    assert rep.quality["no_waste"] == pytest.approx(0.0)
    assert rep.details["wasted_iterations"] == 2


def test_no_waste_improving_run_scores_one(tmp_path: Path):
    ws = _write_meta_workspace(
        tmp_path / "ws",
        baseline_rmse=1.0,
        final_rmse=0.4,
        rmse_afters=[0.8, 0.6, 0.4],  # each step >1% better than best-so-far
    )
    rep = score_meta_run(ws)
    assert rep.quality["no_waste"] == pytest.approx(1.0)


def test_no_waste_single_measured_iteration_is_not_waste(tmp_path: Path):
    ws = _write_meta_workspace(
        tmp_path / "ws",
        baseline_rmse=1.0,
        final_rmse=0.5,
        rmse_afters=[0.5],
    )
    rep = score_meta_run(ws)
    assert rep.quality["no_waste"] == pytest.approx(1.0)


def test_no_waste_mixed_run(tmp_path: Path):
    # iteration 2 improves, iteration 3 is flat -> 1 of 2 productive.
    ws = _write_meta_workspace(
        tmp_path / "ws",
        baseline_rmse=1.0,
        final_rmse=0.4,
        rmse_afters=[0.8, 0.4, 0.4],
    )
    rep = score_meta_run(ws)
    assert rep.quality["no_waste"] == pytest.approx(0.5)
    assert rep.details["wasted_iterations"] == 1


def test_target_reached_earns_decisiveness_credit(tmp_path: Path):
    ws = _write_meta_workspace(
        tmp_path / "ws",
        baseline_rmse=1.0,
        final_rmse=0.3,
        rmse_afters=[0.3],
        terminated_by="target_reached",
    )
    rep = score_meta_run(ws)
    assert rep.hard_gates["report_parseable"]
    assert rep.quality["terminated_by_agent"] == pytest.approx(1.0)


def test_iterations_cap_earns_no_decisiveness_credit(tmp_path: Path):
    ws = _write_meta_workspace(
        tmp_path / "ws",
        baseline_rmse=1.0,
        final_rmse=0.3,
        rmse_afters=[0.3],
        terminated_by="iterations_cap",
    )
    rep = score_meta_run(ws)
    assert rep.quality["terminated_by_agent"] == pytest.approx(0.0)


def test_diagnosis_consistency_is_advisory_only(tmp_path: Path):
    ws = _write_meta_workspace(
        tmp_path / "ws",
        baseline_rmse=1.0,
        final_rmse=0.5,
        rmse_afters=[0.5],
    )
    rep = score_meta_run(ws)
    assert "diagnosis_consistency" not in rep.quality
    assert "diagnosis_consistency_advisory" in rep.details
    # The fixture's diagnosis mentions "edge" + "range" -> matches extend_search.
    assert rep.details["diagnosis_consistency_advisory"] == pytest.approx(1.0)
