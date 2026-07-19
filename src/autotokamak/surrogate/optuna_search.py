"""Optuna study harness — the inner loop of Architecture C.

The OUTER loop is the LLM agent: it emits a ``SearchSpec`` per round
(``schema.SearchSpec``) describing which models to try, with what search
spaces, for how many trials. ``run_study`` then executes that spec
synchronously: one Optuna study per model, all written to one SQLite DB.

The agent inspects ``summarize_study`` between rounds and decides whether to
``widen_range``, ``add_model``, ``tighten_around_best``, or ``terminate``.
When it picks a winner, ``refit_winner`` retrains the best estimator on the
full non-test data and saves it to ``outputs/winner.pkl`` for the scorer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from autotokamak.surrogate.dataset import DatasetBundle, Splits
from autotokamak.surrogate.metrics import psi_rmse
from autotokamak.surrogate.reduce import PCAModel, fit_pca, inverse_transform, transform
from autotokamak.surrogate.schema import (
    ModelSpec,
    ModelStudyResult,
    SearchSpec,
    StudyResult,
    TrialRecord,
)
from autotokamak.surrogate.zoo import make_model


def _suggest(trial, name: str, spec: dict[str, Any]) -> Any:
    """Translate a ``ParamRange`` dict into an Optuna ``suggest_*`` call."""
    ptype = spec["type"]
    if ptype == "float":
        return trial.suggest_float(name, float(spec["low"]), float(spec["high"]))
    if ptype == "loguniform":
        return trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=True)
    if ptype == "int":
        return trial.suggest_int(name, int(spec["low"]), int(spec["high"]))
    if ptype == "categorical":
        return trial.suggest_categorical(name, list(spec["choices"]))
    raise ValueError(f"Unsupported ParamRange type {ptype!r} for hp {name!r}")


def _cv_objective(
    model_name: str,
    n_pca_components: int,
    bundle: DatasetBundle,
    splits: Splits,
):
    """Return an Optuna ``objective(trial)`` closure for one model.

    The closure samples hyperparameters from the trial, builds the estimator
    via ``zoo.make_model``, fits on each CV fold's train, predicts val
    coefficients, inverse-transforms to full grid, and returns the
    fold-averaged ``psi_rmse`` in original ψ units.

    A trial that crashes (e.g. MLP cap violated, GP optimizer blows up) is
    pruned with a very large objective value — Optuna keeps searching.
    """

    def objective(trial) -> float:
        from autotokamak.surrogate.zoo import DEFAULT_SEARCH_SPACES

        hp = {}
        # The trial's "search space" lives in the per-model spec the agent
        # passed; we attach it to the trial via user attrs so this closure
        # can look it up without threading another arg through Optuna.
        search_space = trial.study.user_attrs["search_space"]
        for name, range_spec in search_space.items():
            hp[name] = _suggest(trial, name, range_spec)

        try:
            fold_scores: list[float] = []
            for _, tr_idx, va_idx in splits.iter_folds():
                psi_tr = bundle.psi[tr_idx]
                psi_va = bundle.psi[va_idx]
                X_tr = bundle.inputs[tr_idx]
                X_va = bundle.inputs[va_idx]

                pca = fit_pca(psi_tr, n_components=n_pca_components)
                Y_tr = transform(pca, psi_tr)

                estimator = make_model(model_name, **hp)
                estimator.fit(X_tr, Y_tr)
                Y_pred_va = estimator.predict(X_va)
                psi_pred_va = inverse_transform(pca, Y_pred_va)

                fold_scores.append(psi_rmse(psi_va, psi_pred_va))
            score = float(np.mean(fold_scores))
            if not np.isfinite(score):
                return 1e10
            return score
        except Exception as exc:  # noqa: BLE001
            # Pruned trials in Optuna: record the failure and return a large
            # objective so the search keeps going.
            trial.set_user_attr("exception", f"{type(exc).__name__}: {exc}")
            return 1e10

    return objective


def _detect_edge_hit(model_spec: ModelSpec, best_params: dict[str, Any]) -> dict[str, bool]:
    """Flag hyperparameters whose best value sits at the edge of the searched range.

    The agent uses these flags to decide ``widen_range`` vs ``tighten_around_best``.
    For categorical params we never flag edge-hit (no meaningful "edge").
    """
    flags: dict[str, bool] = {}
    for name, range_spec in model_spec.search_space.items():
        if range_spec.type == "categorical":
            flags[name] = False
            continue
        if name not in best_params:
            flags[name] = False
            continue
        v = float(best_params[name])
        lo = float(range_spec.low) if range_spec.low is not None else float("-inf")
        hi = float(range_spec.high) if range_spec.high is not None else float("inf")
        # 5% of the (log-)range counts as "on the edge".
        span = max(abs(hi - lo), 1e-12)
        flags[name] = (v - lo) / span < 0.05 or (hi - v) / span < 0.05
    return flags


def run_study(
    spec: SearchSpec,
    *,
    bundle: DatasetBundle,
    splits: Splits,
    workdir: Path,
) -> StudyResult:
    """Run one Optuna study per model in ``spec``. Returns aggregated results.

    SQLite storage: one DB at ``workdir/study.db``, one study per model name.
    ``load_if_exists=True`` so subsequent rounds (after the agent widens a
    range or adds a model) resume the same study rather than starting fresh.
    """
    import optuna

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{workdir/'study.db'}"

    per_model: list[ModelStudyResult] = []
    for model_spec in spec.models:
        study = optuna.create_study(
            study_name=model_spec.name,
            storage=storage,
            direction="minimize",
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=splits.seed),
        )
        # Stash the search space on the study so the objective closure can read it.
        study.set_user_attr(
            "search_space",
            {k: v.model_dump() for k, v in model_spec.search_space.items()},
        )
        objective = _cv_objective(
            model_spec.name,
            spec.n_pca_components,
            bundle,
            splits,
        )
        study.optimize(objective, n_trials=model_spec.n_trials, show_progress_bar=False)

        trials = [
            TrialRecord(
                number=int(t.number),
                value=float(t.value if t.value is not None else 1e10),
                params=dict(t.params),
            )
            for t in study.trials
        ]
        best = study.best_trial
        per_model.append(
            ModelStudyResult(
                model_name=model_spec.name,
                n_trials=len(trials),
                best_value=float(best.value),
                best_params=dict(best.params),
                edge_hit=_detect_edge_hit(model_spec, dict(best.params)),
                trials=trials,
            )
        )

    return StudyResult(spec=spec, per_model=per_model, storage_path=str(workdir / "study.db"))


def summarize_study(study_result: StudyResult) -> dict[str, Any]:
    """Per-model summary the agent reads between outer-loop rounds.

    Returns a dict (rather than a model object) so the LLM can ingest it as
    JSON. Contains best score, best params, edge-hit flags, and the
    25th-percentile-trial best (for ``search_efficiency``).
    """
    out: dict[str, Any] = {"overall_best": {}, "per_model": {}}
    best = study_result.best_overall
    out["overall_best"] = {
        "model": best.model_name,
        "value": best.best_value,
        "params": best.best_params,
    }
    for m in study_result.per_model:
        vals = sorted(t.value for t in m.trials if np.isfinite(t.value))
        early = vals[: max(1, len(vals) // 4)] if vals else []
        out["per_model"][m.model_name] = {
            "best_value": m.best_value,
            "best_params": m.best_params,
            "edge_hit": m.edge_hit,
            "n_trials": m.n_trials,
            "best_value_at_25pct_trials": min(early) if early else None,
        }
    return out


def refit_winner(
    study_result: StudyResult,
    *,
    bundle: DatasetBundle,
    splits: Splits,
    save_to: Path,
) -> dict[str, Any]:
    """Retrain the overall best model on all non-test samples; save to disk.

    The saved bundle is the joblib-pickled estimator AND the PCA model AND
    metadata (winner name, params, n_components). The scorer's
    ``winner_loads`` hard gate uses this exact shape.
    """
    import joblib

    save_to = Path(save_to)
    save_to.parent.mkdir(parents=True, exist_ok=True)

    best = study_result.best_overall

    # Train on EVERY non-test sample, not on a single fold's train set, so
    # the saved winner uses all available data.
    test_idx = splits.test_idx
    keep = np.setdiff1d(np.arange(bundle.n_samples), test_idx, assume_unique=False)
    X_full = bundle.inputs[keep]
    psi_full = bundle.psi[keep]

    pca = fit_pca(psi_full, n_components=study_result.spec.n_pca_components)
    Y_full = transform(pca, psi_full)

    estimator = make_model(best.model_name, **best.best_params)
    estimator.fit(X_full, Y_full)

    payload = {
        "estimator": estimator,
        "pca": pca,
        "model_name": best.model_name,
        "hyperparams": best.best_params,
        "n_pca_components": study_result.spec.n_pca_components,
        "fit_indices": keep.tolist(),
        "test_indices": test_idx.tolist(),
    }
    joblib.dump(payload, save_to)

    return {
        "winner_model_name": best.model_name,
        "winner_hyperparams": best.best_params,
        "saved_to": str(save_to),
        "pca_n_components": pca.n_components,
        "pca_explained_var": pca.total_explained_variance,
    }


def predict_with_winner(winner_payload: dict[str, Any], X: np.ndarray) -> np.ndarray:
    """Run a saved winner on raw inputs. Returns full-grid ψ of shape (N, nz, nr).

    Used by the scorer's ``winner_predicts`` hard gate so the same code path
    that scored the model is also what the user would call to use it.
    """
    estimator = winner_payload["estimator"]
    pca: PCAModel = winner_payload["pca"]
    Y_pred = estimator.predict(np.asarray(X, dtype=np.float64))
    if Y_pred.ndim == 1:
        Y_pred = Y_pred.reshape(1, -1)
    return inverse_transform(pca, Y_pred)


__all__ = [
    "predict_with_winner",
    "refit_winner",
    "run_study",
    "summarize_study",
]
