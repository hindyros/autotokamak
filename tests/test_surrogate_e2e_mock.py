# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""End-to-end mock of the Phase-2 pipeline against the real Phase-1 dataset.

Simulates what the agent's runner would produce, without invoking the LLM:
- writes ``surrogate_config.yaml`` + a minimal ``run_surrogate_automl.py`` +
  ``README.md`` into a tmp workspace
- symlinks the real ``dataset.h5`` in
- runs one ``SearchSpec`` round via ``automl.run_study``
- saves ``winner.pkl`` + ``report.json``
- scores the workspace via ``metric_surrogate.score_surrogate_run`` (called
  directly here AND through the runner's ``try_score`` dispatch)

Catches scorer-dispatch wiring, HDF5 schema drift, PCA NaN-handling on the
real outside-LCFS regions, and Optuna SQLite write path issues.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_DATASET = REPO_ROOT / "examples" / "dataset_generation" / "outputs" / "dataset.h5"


@pytest.mark.skipif(
    not REAL_DATASET.is_file(),
    reason=f"Phase-1 dataset not present at {REAL_DATASET}",
)
def test_phase2_pipeline_against_real_dataset(tmp_path: Path):
    from autotokamak.surrogate.dataset import kfold, load_dataset
    from autotokamak.surrogate import optuna_search as automl
    from autotokamak.surrogate.schema import SearchSpec, SurrogateReport
    from autotokamak.surrogate.zoo import DEFAULT_SEARCH_SPACES

    # 1. Build a mock workspace.
    ws = tmp_path / "surrogate_automl"
    (ws / "outputs").mkdir(parents=True)

    # Symlink the real dataset.h5 so the scorer can find it (matches the
    # `symlinks:` entry the prompt will declare).
    os.symlink(REAL_DATASET, ws / "dataset.h5")

    # The surrogate_config the agent would write.
    (ws / "surrogate_config.yaml").write_text(
        "dataset_h5: dataset.h5\n"
        "time_budget_seconds: 60\n"
        "n_pca_components_default: 8\n"
        "seed: 0\n"
        "k_folds: 4\n"
        "test_frac: 0.125\n"
        "output_dir: outputs\n"
    )
    # A runner stub that the scorer's runner_cleanliness term will grade.
    (ws / "run_surrogate_automl.py").write_text(
        "from autotokamak.surrogate.dataset import load_dataset, kfold\n"
        "from autotokamak.surrogate import optuna_search as automl\n"
        "# (this is a test stub; real runner is agent-authored)\n"
    )
    (ws / "README.md").write_text("# surrogate_automl test workspace\n")

    # 2. Run one round of the search.
    bundle = load_dataset(REAL_DATASET)
    splits = kfold(bundle, k=4, test_frac=2 / bundle.n_samples, seed=0)
    spec = SearchSpec.model_validate(
        {
            "round": 1,
            "n_pca_components": 8,
            "val_metric": "psi_rmse",
            "action": "initial",
            "rationale": "e2e mock",
            "models": [
                {
                    "name": "poly_ridge",
                    "n_trials": 4,
                    "search_space": DEFAULT_SEARCH_SPACES["poly_ridge"],
                },
                {
                    "name": "kernel_ridge",
                    "n_trials": 4,
                    "search_space": DEFAULT_SEARCH_SPACES["kernel_ridge"],
                },
            ],
        }
    )
    result = automl.run_study(spec, bundle=bundle, splits=splits, workdir=ws / "outputs")
    refit_info = automl.refit_winner(
        result, bundle=bundle, splits=splits, save_to=ws / "outputs" / "winner.pkl"
    )

    # 3. Write the report the agent would write.
    from autotokamak.surrogate.metrics import psi_rmse

    import joblib

    payload = joblib.load(ws / "outputs" / "winner.pkl")
    test_pred = automl.predict_with_winner(payload, bundle.inputs[splits.test_idx])
    test_rmse = psi_rmse(bundle.psi[splits.test_idx], test_pred)
    # Use the best CV value as a proxy for val rmse.
    val_rmse = result.best_overall.best_value

    report = SurrogateReport(
        winner_model_name=refit_info["winner_model_name"],
        winner_hyperparams=refit_info["winner_hyperparams"],
        val_psi_rmse=float(val_rmse),
        test_psi_rmse=float(test_rmse),
        baseline_mean_psi_rmse=0.0,  # scorer will recompute
        pca_n_components=int(refit_info["pca_n_components"]),
        pca_explained_var=float(refit_info["pca_explained_var"]),
        n_total_trials=sum(m.n_trials for m in result.per_model),
        n_outer_rounds=1,
        terminated_by="agent",
        models_tried=[m.model_name for m in result.per_model],
    )
    (ws / "outputs" / "report.json").write_text(report.model_dump_json(indent=2))

    # 4. Score directly.
    from autotokamak.agent.dspy.metric_surrogate import score_surrogate_run

    report_obj = score_surrogate_run(ws)
    assert all(report_obj.hard_gates.values()), (
        f"hard gates failed: {report_obj.hard_gates}; details={report_obj.details}"
    )
    # Quality terms should all be in [0, 1].
    for name, val in report_obj.quality.items():
        assert 0.0 <= val <= 1.0, f"quality term {name}={val} out of [0,1]"
    assert report_obj.total > 0.0

    # 5. Same score via the runner's try_score dispatch.
    import sys

    sys.path.insert(0, str(REPO_ROOT / "src" / "autotokamak"))
    try:
        from autotokamak.bench.scoring import try_score
    finally:
        sys.path.pop(0)

    dispatched = try_score(
        ws,
        "autotokamak.agent.dspy.metric_surrogate:score_surrogate_run",
    )
    assert dispatched is not None
    assert dispatched.total == pytest.approx(report_obj.total, abs=1e-6)


def test_try_score_handles_missing_scorer_gracefully(tmp_path: Path):
    """Sanity: the dispatcher swallows import/attribute errors."""
    import sys

    sys.path.insert(0, str(REPO_ROOT / "src" / "autotokamak"))
    try:
        from autotokamak.bench.scoring import try_score
    finally:
        sys.path.pop(0)

    assert try_score(tmp_path, "no.such.module:fn") is None
    assert try_score(tmp_path, "autotokamak.agent.dspy.metric:NOT_A_FN") is None
