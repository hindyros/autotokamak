# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Bottleneck diagnostics for the meta-agent.

Each function returns a JSON-serializable dict. The meta-agent reads these
between iterations to decide whether the surrogate is sample-bottlenecked
(need more / better data), capacity-bottlenecked (need more flexible model),
search-bottlenecked (need wider ranges), or reduction-bottlenecked (PCA
throwing away signal).

The agent does not have to consume every primitive every iteration; the
meta-loop runner calls ``run_all`` and the prompt teaches the agent how to
interpret the resulting bundle.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from autotokamak.surrogate.dataset import DatasetBundle, kfold
from autotokamak.surrogate.metrics import baseline_mean_predictor_rmse, psi_rmse
from autotokamak.surrogate.reduce import fit_pca, inverse_transform, transform


def learning_curve(
    bundle: DatasetBundle,
    *,
    model_factory: Callable[[], Any],
    sub_sizes: tuple[int, ...] = (4, 8, 12, 16),
    n_pca_components: int = 8,
    k_folds: int = 4,
    seed: int = 0,
) -> dict[str, Any]:
    """RMSE vs N. Slope of log(RMSE) vs log(N) signals sample-bottlenecking.

    For each sub_size <= bundle.n_samples we draw the FIRST sub_size rows
    (deterministic), run k-fold CV using the supplied model_factory, and
    record the fold-averaged RMSE. Returns the curve, the log-log slope of
    the last two points, and a ``plateau_detected`` flag (slope flatter
    than -0.05 → roughly no gain from more data at this regime).
    """
    sub_sizes = tuple(s for s in sub_sizes if s <= bundle.n_samples and s >= max(k_folds + 1, 4))
    if not sub_sizes:
        return {"curve": {}, "slope_log_log": None, "plateau_detected": None, "note": "no usable sub_sizes for bundle.n_samples=%d" % bundle.n_samples}

    curve: dict[int, float] = {}
    for n in sub_sizes:
        sub = DatasetBundle(
            inputs=bundle.inputs[:n],
            psi=bundle.psi[:n],
            R=bundle.R,
            Z=bundle.Z,
            source_path=bundle.source_path,
        )
        try:
            splits = kfold(sub, k=k_folds, test_frac=max(2 / n, 1 / n), seed=seed)
        except ValueError:
            continue
        fold_rmses: list[float] = []
        for _, tr_idx, va_idx in splits.iter_folds():
            psi_tr = sub.psi[tr_idx]
            psi_va = sub.psi[va_idx]
            try:
                pca = fit_pca(psi_tr, n_components=min(n_pca_components, len(tr_idx) - 1))
                Y_tr = transform(pca, psi_tr)
                est = model_factory()
                est.fit(sub.inputs[tr_idx], Y_tr)
                Y_pred = est.predict(sub.inputs[va_idx])
                psi_pred = inverse_transform(pca, Y_pred)
                fold_rmses.append(psi_rmse(psi_va, psi_pred))
            except Exception:  # noqa: BLE001
                continue
        if fold_rmses:
            curve[int(n)] = float(np.mean(fold_rmses))

    # Slope between the last two points in log-log space.
    slope = None
    plateau = None
    if len(curve) >= 2:
        ns = sorted(curve)
        n1, n2 = ns[-2], ns[-1]
        v1, v2 = curve[n1], curve[n2]
        if v1 > 0 and v2 > 0 and n1 > 0 and n2 > 0:
            slope = float((np.log(v2) - np.log(v1)) / (np.log(n2) - np.log(n1)))
            plateau = bool(slope > -0.05)

    return {
        "curve": {str(k): v for k, v in curve.items()},
        "slope_log_log": slope,
        "plateau_detected": plateau,
        "interpretation": (
            "plateau → not sample-bottlenecked; consider model/search changes"
            if plateau
            else "RMSE still dropping with N → sample-bottlenecked; regenerate at higher N"
            if slope is not None
            else "insufficient points to assess slope"
        ),
    }


def cross_seed_variance(
    bundle: DatasetBundle,
    *,
    model_factory: Callable[[], Any],
    seeds: tuple[int, ...] = (0, 1, 2),
    n_pca_components: int = 8,
    k_folds: int = 4,
    test_frac: float = 2 / 16,
) -> dict[str, Any]:
    """Re-run the same train/eval with different RNG seeds.

    High coefficient of variation (CV-of-CV-RMSE > ~0.2) means the splits
    matter too much — dataset is too small for the search to be reliable.
    """
    per_seed: dict[int, float] = {}
    for s in seeds:
        try:
            splits = kfold(bundle, k=k_folds, test_frac=test_frac, seed=int(s))
        except ValueError:
            continue
        fold_rmses: list[float] = []
        for _, tr_idx, va_idx in splits.iter_folds():
            try:
                pca = fit_pca(bundle.psi[tr_idx], n_components=n_pca_components)
                Y_tr = transform(pca, bundle.psi[tr_idx])
                est = model_factory()
                est.fit(bundle.inputs[tr_idx], Y_tr)
                Y_pred = est.predict(bundle.inputs[va_idx])
                psi_pred = inverse_transform(pca, Y_pred)
                fold_rmses.append(psi_rmse(bundle.psi[va_idx], psi_pred))
            except Exception:  # noqa: BLE001
                continue
        if fold_rmses:
            per_seed[int(s)] = float(np.mean(fold_rmses))

    if len(per_seed) < 2:
        return {"per_seed": per_seed, "cv": None, "high_variance": None, "note": "<2 seeds completed"}

    vals = np.asarray(list(per_seed.values()))
    mean = float(vals.mean())
    cv = float(vals.std() / mean) if mean > 0 else None
    return {
        "per_seed": {str(k): v for k, v in per_seed.items()},
        "mean_rmse": mean,
        "cv": cv,
        "high_variance": (cv is not None and cv > 0.2),
        "interpretation": (
            "split-luck dominates → dataset too small or fold scheme too aggressive"
            if (cv is not None and cv > 0.2)
            else "splits stable → measurement is meaningful"
        ),
    }


def pca_spectrum(
    bundle: DatasetBundle,
    *,
    max_components: int = 16,
) -> dict[str, Any]:
    """How much variance lives in the tail components.

    If 95% variance is captured with very few components → PCA is efficient,
    the surrogate's regression task is small. If tail energy is significant
    above n_components the surrogate currently uses, the agent should widen
    n_pca_components.
    """
    n = bundle.n_samples
    max_components = int(min(max_components, n - 1))
    if max_components < 2:
        return {"explained": [], "note": "n_samples too small for PCA spectrum"}

    pca = fit_pca(bundle.psi, n_components=max_components)
    evr = pca.explained_variance_ratio
    cum = np.cumsum(evr).tolist()
    n_for_95 = int(np.searchsorted(np.asarray(cum), 0.95) + 1) if cum and cum[-1] >= 0.95 else None
    tail_energy_above_8 = float(np.sum(evr[8:])) if len(evr) > 8 else 0.0

    return {
        "explained_per_component": [float(v) for v in evr],
        "cumulative": cum,
        "n_components_for_95pct": n_for_95,
        "tail_energy_above_8": tail_energy_above_8,
        "interpretation": (
            f"~{n_for_95} components reach 95% variance"
            if n_for_95 is not None
            else "max_components insufficient to reach 95% variance — increase n_pca_components"
        ),
    }


def residual_structure(
    winner_payload: dict[str, Any],
    bundle: DatasetBundle,
    splits,
) -> dict[str, Any]:
    """Where does the current best surrogate err?

    Looks at test-set residuals. Returns mean residual magnitude, the
    correlation between |residual| and each input dimension, and a coarse
    spatial-pattern flag (mean residual variance over the grid). High
    correlation with one input → that part of the input space is
    undertrained. Strong spatial pattern → model capacity bottleneck.
    """
    from autotokamak.surrogate.optuna_search import predict_with_winner

    test_idx = splits.test_idx
    if test_idx.size == 0:
        return {"note": "no test indices"}

    X_test = bundle.inputs[test_idx]
    psi_true = bundle.psi[test_idx]
    psi_pred = predict_with_winner(winner_payload, X_test)

    residual = psi_true - psi_pred
    abs_res_per_sample = np.array(
        [np.nanmean(np.abs(r)) for r in residual]
    )

    # Correlate |residual| with each input column.
    input_corrs: dict[str, float] = {}
    from autotokamak.surrogate.dataset import PARAM_ORDER

    for j, p in enumerate(PARAM_ORDER):
        col = X_test[:, j]
        if col.std() > 1e-12 and abs_res_per_sample.std() > 1e-12:
            input_corrs[p] = float(np.corrcoef(col, abs_res_per_sample)[0, 1])
        else:
            input_corrs[p] = 0.0

    # Spatial-pattern flag: variance over the grid of per-cell mean |residual|.
    cell_mean_abs = np.nanmean(np.abs(residual), axis=0)  # (nz, nr)
    spatial_var = float(np.nanvar(cell_mean_abs))
    cell_mean = float(np.nanmean(cell_mean_abs))

    return {
        "mean_abs_residual_per_sample": [float(v) for v in abs_res_per_sample],
        "input_correlations": input_corrs,
        "spatial_pattern_variance": spatial_var,
        "spatial_pattern_mean": cell_mean,
        "interpretation": (
            "residual correlates strongly with "
            + ",".join(p for p, v in input_corrs.items() if abs(v) > 0.5)
            if any(abs(v) > 0.5 for v in input_corrs.values())
            else "residual is roughly uniform across inputs → likely capacity- or noise-bottlenecked"
        ),
    }


def edge_hit_summary(study_result) -> dict[str, Any]:
    """Per-model count of hyperparameters that landed on the edge of their range.

    Wraps the edge-hit flags already populated by
    ``surrogate.optuna_search._detect_edge_hit``; presents them as a JSON-friendly
    summary the agent can read directly.
    """
    out: dict[str, Any] = {}
    for m in study_result.per_model:
        hits = [k for k, v in m.edge_hit.items() if v]
        out[m.model_name] = {
            "best_value": float(m.best_value),
            "edge_hit_params": hits,
            "n_trials": int(m.n_trials),
        }
    persistent_edges = {
        m.model_name: [k for k, v in m.edge_hit.items() if v]
        for m in study_result.per_model
        if any(m.edge_hit.values())
    }
    return {
        "per_model": out,
        "models_with_edge_hits": persistent_edges,
        "interpretation": (
            "best params hit range edges → widen those ranges in extend_search"
            if persistent_edges
            else "no edge hits → search ranges look adequate"
        ),
    }


def run_all(
    bundle: DatasetBundle,
    *,
    model_factory: Callable[[], Any] | None = None,
    winner_payload: dict[str, Any] | None = None,
    splits=None,
    study_result=None,
) -> dict[str, Any]:
    """Convenience wrapper the meta-loop calls each iteration.

    Each primitive is skipped if its inputs are missing — diagnostics
    accumulate as the meta-agent's state grows. The runner doesn't need to
    know in advance which signals will be available.
    """
    out: dict[str, Any] = {}
    if model_factory is not None:
        out["learning_curve"] = learning_curve(bundle, model_factory=model_factory)
        out["cross_seed_variance"] = cross_seed_variance(bundle, model_factory=model_factory)
    out["pca_spectrum"] = pca_spectrum(bundle)
    if winner_payload is not None and splits is not None:
        out["residual_structure"] = residual_structure(winner_payload, bundle, splits)
    if study_result is not None:
        out["edge_hit_summary"] = edge_hit_summary(study_result)
    return out


__all__ = [
    "cross_seed_variance",
    "edge_hit_summary",
    "learning_curve",
    "pca_spectrum",
    "residual_structure",
    "run_all",
]
