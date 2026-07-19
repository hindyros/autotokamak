"""Deterministic Phase-2 AutoML outer loop with typed per-round decisions.

This is the structured replacement for the codegen path (an URSA agent
writing a runner script from scratch each run). The loop itself is plain
Python; the ONLY model-driven part is ``decision_fn`` — one call per round
that receives a JSON-serializable round context and returns a validated
``RoundDecision``. The meta-loop's ``extend_search`` action dispatches here
in ``phase2_mode="structured"``; tests inject a scripted ``decision_fn``.

Output layout matches the codegen contract exactly, so scorers and report
tooling keep working:

    <workdir>/surrogate_config.yaml
    <workdir>/outputs/{winner.pkl, report.json, study.db, search_history.jsonl}
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

from autotokamak.surrogate.schema import RoundDecision, SearchSpec, StudyResult

DecisionFn = Callable[[dict], RoundDecision]


def build_round_context(
    *,
    round_index: int,
    max_rounds: int,
    time_budget_seconds: int,
    elapsed_seconds: float,
    trials_used: int,
    focus: Optional[dict],
    bundle,
    dataset_h5: Path,
    n_pca_components_default: int,
    history: list[dict],
) -> dict:
    """The JSON-serializable contract between the loop and ``decision_fn``.

    ``history`` carries one entry per completed round:
    ``{"round", "spec", "summary"}`` where ``summary`` is the
    ``summarize_study`` output (per-model best_value / best_params /
    edge_hit / best_value_at_25pct_trials).
    """
    from autotokamak.surrogate.zoo import DEFAULT_SEARCH_SPACES

    return {
        "round": round_index,
        "max_rounds": max_rounds,
        "rounds_remaining": max_rounds - round_index + 1,
        "budget": {
            "time_budget_seconds": int(time_budget_seconds),
            "elapsed_seconds": float(elapsed_seconds),
            "seconds_remaining": max(0.0, float(time_budget_seconds) - float(elapsed_seconds)),
            "trials_used": int(trials_used),
        },
        "focus": focus,
        "dataset": {
            "n_samples": bundle.n_samples,
            "grid_shape": list(bundle.grid_shape),
            "path": str(dataset_h5),
        },
        "n_pca_components_default": int(n_pca_components_default),
        "default_search_spaces": DEFAULT_SEARCH_SPACES,
        "history": history,
    }


def run_automl_loop(
    *,
    dataset_h5: str | Path,
    workdir: str | Path,
    decision_fn: DecisionFn,
    max_rounds: int = 4,
    time_budget_seconds: int = 300,
    n_pca_components_default: int = 8,
    k_folds: int = 4,
    seed: int = 0,
    focus: Optional[dict] = None,
    test_shard_h5: str | Path | None = None,
) -> Dict[str, Any]:
    """Run the structured AutoML search; return a summary dict.

    When ``test_shard_h5`` is given, ``test_psi_rmse`` is measured on that
    frozen shard and the internal split holds out no test samples
    (``test_frac=0.0``) so every train-pool sample feeds the CV folds and
    the final refit. Without a shard, falls back to the legacy internal
    2-sample holdout.

    Returns ``{"winner": None, ...}`` when ``decision_fn`` terminates before
    any round ran — the caller decides what a winnerless search means.
    """
    from autotokamak.core.io import atomic_write_text
    from autotokamak.surrogate.dataset import kfold, load_dataset
    from autotokamak.surrogate.metrics import baseline_mean_predictor_rmse, psi_rmse
    from autotokamak.surrogate.optuna_search import (
        predict_with_winner,
        refit_winner,
        run_study,
        summarize_study,
    )

    dataset_h5 = Path(dataset_h5)
    workdir = Path(workdir)
    outputs = workdir / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)

    bundle = load_dataset(dataset_h5)
    shard = None
    if test_shard_h5 is not None:
        shard = load_dataset(test_shard_h5)
        if not (np.array_equal(shard.R, bundle.R) and np.array_equal(shard.Z, bundle.Z)):
            raise ValueError(
                f"Frozen test shard grid does not match dataset grid "
                f"(shard {test_shard_h5} vs dataset {dataset_h5})"
            )
    test_frac = 0.0 if shard is not None else 2 / bundle.n_samples
    splits = kfold(bundle, k=k_folds, test_frac=test_frac, seed=seed)

    # Run-wide config so metric_surrogate's dataset resolution works on
    # structured workspaces the same way it does on codegen ones.
    atomic_write_text(
        workdir / "surrogate_config.yaml",
        json.dumps(  # JSON is valid YAML; avoids a yaml dependency here
            {
                "dataset_h5": str(dataset_h5),
                "time_budget_seconds": int(time_budget_seconds),
                "n_pca_components_default": int(n_pca_components_default),
                "seed": int(seed),
                "k_folds": int(k_folds),
                "test_frac": test_frac,
                "output_dir": "outputs",
                "mode": "structured",
            },
            indent=2,
        )
        + "\n",
    )

    history: list[dict] = []
    # model_name -> (ModelStudyResult, SearchSpec of the round it last ran in)
    latest_by_model: dict[str, tuple[Any, SearchSpec]] = {}
    trials_used = 0
    started = time.time()
    terminated_by = "rounds_cap"
    stop_reason: Optional[str] = None
    rounds_completed = 0

    for r in range(1, max_rounds + 1):
        elapsed = time.time() - started
        if elapsed > time_budget_seconds:
            stop_reason = "time_budget"
            break

        ctx = build_round_context(
            round_index=r,
            max_rounds=max_rounds,
            time_budget_seconds=time_budget_seconds,
            elapsed_seconds=elapsed,
            trials_used=trials_used,
            focus=focus,
            bundle=bundle,
            dataset_h5=dataset_h5,
            n_pca_components_default=n_pca_components_default,
            history=history,
        )
        decision = decision_fn(ctx)
        if not isinstance(decision, RoundDecision):
            decision = RoundDecision.model_validate(decision)

        if decision.action == "terminate":
            terminated_by = "agent"
            stop_reason = decision.rationale or "picker terminated"
            break

        spec = SearchSpec(
            round=r,
            models=decision.models,
            n_pca_components=decision.n_pca_components or n_pca_components_default,
            val_metric="psi_rmse",
            action="initial" if r == 1 else "continue",
            rationale=decision.rationale,
        )
        result = run_study(spec, bundle=bundle, splits=splits, workdir=outputs)
        summary = summarize_study(result)
        trials_used += sum(m.n_trials for m in spec.models)
        rounds_completed = r

        with open(outputs / "search_history.jsonl", "a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "spec": spec.model_dump(mode="json"),
                        "summary": summary,
                        "elapsed_seconds": time.time() - started,
                    },
                    default=str,
                )
                + "\n"
            )
        history.append(
            {"round": r, "spec": spec.model_dump(mode="json"), "summary": summary}
        )
        for m in result.per_model:
            # The latest result for a model contains ALL its trials so far
            # (Optuna resumes the study via load_if_exists).
            latest_by_model[m.model_name] = (m, spec)

    if not latest_by_model:
        return {
            "winner": None,
            "terminated_by": terminated_by,
            "stop_reason": stop_reason,
            "n_rounds": 0,
            "val_psi_rmse": None,
            "test_psi_rmse": None,
            "report_path": None,
            "winner_path": None,
            "search_history_path": None,
        }

    best_entry, best_spec = min(latest_by_model.values(), key=lambda t: t[0].best_value)
    final_study = StudyResult(
        spec=best_spec, per_model=[best_entry], storage_path=str(outputs / "study.db")
    )
    winner_path = outputs / "winner.pkl"
    refit_info = refit_winner(final_study, bundle=bundle, splits=splits, save_to=winner_path)

    import joblib

    payload = joblib.load(winner_path)
    if shard is not None:
        test_psi_rmse = float(psi_rmse(shard.psi, predict_with_winner(payload, shard.inputs)))
    elif splits.test_idx.size:
        test_psi_rmse = float(
            psi_rmse(
                bundle.psi[splits.test_idx],
                predict_with_winner(payload, bundle.inputs[splits.test_idx]),
            )
        )
    else:
        test_psi_rmse = float("nan")

    baseline = float(
        sum(
            baseline_mean_predictor_rmse(bundle.psi[tr], bundle.psi[va])
            for _, tr, va in splits.iter_folds()
        )
        / len(splits.folds)
    )

    from autotokamak.surrogate.schema import SurrogateReport

    report = SurrogateReport.model_validate(
        {
            "winner_model_name": refit_info["winner_model_name"],
            "winner_hyperparams": refit_info["winner_hyperparams"],
            "val_psi_rmse": float(best_entry.best_value),
            "test_psi_rmse": test_psi_rmse,
            "baseline_mean_psi_rmse": baseline,
            "pca_n_components": int(refit_info["pca_n_components"]),
            "pca_explained_var": float(refit_info["pca_explained_var"]),
            "n_total_trials": int(trials_used),
            "n_outer_rounds": int(rounds_completed),
            "terminated_by": terminated_by,
            "models_tried": sorted(latest_by_model),
            # provenance extras (SurrogateReport allows extra fields)
            "mode": "structured",
            "seed": int(seed),
            "dataset_h5": str(dataset_h5),
            "test_shard_path": str(test_shard_h5) if test_shard_h5 else None,
        }
    )
    report_path = outputs / "report.json"
    atomic_write_text(report_path, report.model_dump_json(indent=2) + "\n")

    return {
        "winner": refit_info,
        "terminated_by": terminated_by,
        "stop_reason": stop_reason,
        "n_rounds": rounds_completed,
        "val_psi_rmse": float(best_entry.best_value),
        "test_psi_rmse": test_psi_rmse,
        "baseline_mean_psi_rmse": baseline,
        "report_path": str(report_path),
        "winner_path": str(winner_path),
        "search_history_path": str(outputs / "search_history.jsonl"),
    }


__all__ = ["DecisionFn", "build_round_context", "run_automl_loop"]
