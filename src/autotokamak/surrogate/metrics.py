"""Error metrics for ψ(R,Z) predictions.

All metrics operate on full-grid arrays of shape ``(N, nz, nr)`` after any
PCA inverse-transform. They share a NaN-handling convention: if both true and
pred have NaN at the same cell, that cell is excluded; if only one has NaN,
the cell is excluded and counted in a returned ``n_excluded`` field when the
function returns a diagnostic dict (the scalar variants just exclude).

This module is intentionally tiny — the agent's runner should call these
rather than re-implementing them. The DSPy scorer also imports them so the
score uses the same definitions as the agent's reported numbers.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np


def _valid_mask(true: np.ndarray, pred: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Return a boolean array of cells to include in the metric.

    Excludes any cell where either array is non-finite. If ``mask`` is given,
    its False entries are also excluded (i.e. the metric is computed only
    over cells where the mask is True AND both arrays are finite).
    """
    valid = np.isfinite(true) & np.isfinite(pred)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    return valid


def _valid_values(
    true: np.ndarray, pred: np.ndarray, mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray] | None:
    valid = _valid_mask(true, pred, mask)
    if not valid.any():
        return None
    return (
        np.asarray(true, dtype=np.float64)[valid],
        np.asarray(pred, dtype=np.float64)[valid],
    )


def psi_rmse(true: np.ndarray, pred: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Cell-averaged RMSE of ψ over included cells.

    Computed in physical ψ units (whatever the dataset HDF5 stored). For
    surrogate scoring we usually want this in the ORIGINAL ψ space, not in
    PCA-coefficient space — call ``inverse_transform`` first.
    """
    vals = _valid_values(true, pred, mask)
    if vals is None:
        return float("nan")
    t, p = vals
    err = t - p
    return float(np.sqrt(np.mean(err * err)))


def psi_mae(true: np.ndarray, pred: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Mean absolute error of ψ over included cells (same units as ψ)."""
    vals = _valid_values(true, pred, mask)
    if vals is None:
        return float("nan")
    t, p = vals
    return float(np.mean(np.abs(t - p)))


def relative_l2(true: np.ndarray, pred: np.ndarray, mask: np.ndarray | None = None) -> float:
    """||true - pred||_2 / ||true||_2 over included cells.

    Scale-invariant — useful when comparing across samples with different
    overall ψ magnitudes. Returns NaN if ``true`` is identically zero on
    included cells. 0 = perfect, 0.1 ≈ 10% relative field error.
    """
    vals = _valid_values(true, pred, mask)
    if vals is None:
        return float("nan")
    t, p = vals
    denom = float(np.linalg.norm(t))
    if denom < 1e-30:
        return float("nan")
    return float(np.linalg.norm(t - p) / denom)


def pixelwise_max_err(true: np.ndarray, pred: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Max absolute error over included cells. Useful for sanity-checking outliers."""
    vals = _valid_values(true, pred, mask)
    if vals is None:
        return float("nan")
    t, p = vals
    return float(np.max(np.abs(t - p)))


def r2_score(true: np.ndarray, pred: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Coefficient of determination R² over included cells.

    1.0 = perfect, 0.0 = no better than predicting the mean of ``true``,
    negative = worse than that mean. Familiar ML regression score.
    """
    vals = _valid_values(true, pred, mask)
    if vals is None:
        return float("nan")
    t, p = vals
    ss_res = float(np.sum((t - p) ** 2))
    ss_tot = float(np.sum((t - np.mean(t)) ** 2))
    if ss_tot < 1e-30:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def pearson_r(true: np.ndarray, pred: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Pearson correlation between true and predicted cells. 1.0 = perfect linear match."""
    vals = _valid_values(true, pred, mask)
    if vals is None:
        return float("nan")
    t, p = vals
    if t.size < 2:
        return float("nan")
    t_c = t - t.mean()
    p_c = p - p.mean()
    denom = float(np.linalg.norm(t_c) * np.linalg.norm(p_c))
    if denom < 1e-30:
        return float("nan")
    return float(np.dot(t_c, p_c) / denom)


def within_abs_tolerance(
    true: np.ndarray,
    pred: np.ndarray,
    *,
    abs_tol: float,
    mask: np.ndarray | None = None,
) -> float:
    """Fraction of included cells with |true - pred| ≤ ``abs_tol`` (ψ units).

    Accuracy-style score in [0, 1]. Example: ``abs_tol=1e-3`` → “% of cells
    within 0.001 Wb of truth.”
    """
    vals = _valid_values(true, pred, mask)
    if vals is None:
        return float("nan")
    t, p = vals
    return float(np.mean(np.abs(t - p) <= float(abs_tol)))


def within_rel_tolerance(
    true: np.ndarray,
    pred: np.ndarray,
    *,
    rel_tol: float = 0.05,
    mask: np.ndarray | None = None,
) -> float:
    """Fraction of included cells with |err| ≤ ``rel_tol`` * |true|.

    Accuracy-style score in [0, 1]. Default ``rel_tol=0.05`` → “% of cells
    within 5% of the true value.” Cells with ``|true|`` near zero use a tiny
    floor so they are not all counted wrong.
    """
    vals = _valid_values(true, pred, mask)
    if vals is None:
        return float("nan")
    t, p = vals
    scale = np.maximum(np.abs(t), 1e-12)
    return float(np.mean(np.abs(t - p) <= float(rel_tol) * scale))


def baseline_mean_prediction(psi_train: np.ndarray, n_val: int) -> np.ndarray:
    """The trivial predictor: per-pixel mean of train ψ, tiled over ``n_val``.

    Shared by ``baseline_mean_predictor_rmse`` (aggregate baseline) and the
    meta-loop's per-cell baseline so the two never drift apart.
    """
    if psi_train.ndim != 3:
        raise ValueError("psi_train must be shape (N, nz, nr)")
    # Outside-LCFS pixels are all-NaN columns; silence the resulting warning.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_psi = np.nanmean(psi_train, axis=0)  # (nz, nr)
    return np.broadcast_to(mean_psi[None, :, :], (n_val, *mean_psi.shape))


def baseline_mean_predictor_rmse(psi_train: np.ndarray, psi_val: np.ndarray) -> float:
    """RMSE of the trivial cell-wise mean predictor.

    Predict val ψ as the per-pixel mean of train ψ. This is the "did our
    surrogate beat doing nothing" reference point and is the denominator in
    the scorer's ``val_rmse_vs_baseline`` quality term.
    """
    if psi_val.ndim != 3:
        raise ValueError("psi_train and psi_val must be shape (N, nz, nr)")
    pred = baseline_mean_prediction(psi_train, psi_val.shape[0])
    return psi_rmse(psi_val, pred)


def summarize_psi_errors(
    true: np.ndarray,
    pred: np.ndarray,
    *,
    baseline_rmse: float | None = None,
    abs_tol: float | None = None,
    rel_tol: float = 0.05,
    mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Bundle of easy-to-read metrics for one true/pred ψ batch.

    If ``abs_tol`` is None, defaults to 5% of the RMS of ``true`` on valid
    cells (a scale-aware absolute tolerance).
    """
    vals = _valid_values(true, pred, mask)
    if vals is None:
        return {
            "n_cells": 0,
            "rmse": float("nan"),
            "mae": float("nan"),
            "rel_l2": float("nan"),
            "max_err": float("nan"),
            "r2": float("nan"),
            "corr": float("nan"),
            "pct_within_5pct": float("nan"),
            "pct_within_abs_tol": float("nan"),
            "abs_tol_used": float("nan"),
            "baseline_rmse": baseline_rmse,
            "rmse_vs_baseline": float("nan"),
            "pct_better_than_baseline": float("nan"),
        }

    t, _ = vals
    rms_true = float(np.sqrt(np.mean(t * t)))
    if abs_tol is None:
        abs_tol = 0.05 * rms_true if rms_true > 0 else 1e-6

    rmse = psi_rmse(true, pred, mask)
    out: dict[str, Any] = {
        "n_cells": int(t.size),
        "rmse": rmse,
        "mae": psi_mae(true, pred, mask),
        "rel_l2": relative_l2(true, pred, mask),
        "max_err": pixelwise_max_err(true, pred, mask),
        "r2": r2_score(true, pred, mask),
        "corr": pearson_r(true, pred, mask),
        "pct_within_5pct": 100.0 * within_rel_tolerance(true, pred, rel_tol=rel_tol, mask=mask),
        "pct_within_abs_tol": 100.0
        * within_abs_tolerance(true, pred, abs_tol=abs_tol, mask=mask),
        "abs_tol_used": float(abs_tol),
        "baseline_rmse": baseline_rmse,
        "rmse_vs_baseline": float("nan"),
        "pct_better_than_baseline": float("nan"),
    }
    if baseline_rmse is not None and np.isfinite(baseline_rmse) and baseline_rmse > 0:
        out["rmse_vs_baseline"] = float(rmse / baseline_rmse)
        out["pct_better_than_baseline"] = float(100.0 * (1.0 - rmse / baseline_rmse))
    return out


def per_cell_errors(
    inputs: np.ndarray,
    true: np.ndarray,
    pred: np.ndarray,
    *,
    bounds_low: np.ndarray,
    bounds_high: np.ndarray,
    n_bins: int = 2,
    param_names: tuple[str, ...] = ("r0", "a", "kappa", "delta", "Ip"),
) -> dict[str, Any]:
    """Stratified per-geometry-cell error breakdown for an eval batch.

    Bins each sample into a joint geometry cell (``n_bins`` equal-width bins
    per input dimension over ``[bounds_low, bounds_high]``, samples outside
    the box clipped into the edge bins) and computes ``psi_rmse`` per
    occupied cell, plus per-parameter tercile marginals. This is the metric
    that shows WHERE the surrogate is weak — aggregate RMSE alone cannot
    distinguish "uniformly mediocre" from "great in the seed box, broken at
    high kappa", which is exactly the distinction active sampling must close.

    Returns a JSON-serializable dict:
      n_cells_total / n_cells_occupied  — joint-cell coverage
      cells            — {cell_key: {"n", "rmse"}}; key = "-".join(bin indices)
      worst_cell       — {"key", "rmse", "n"} (max-RMSE occupied cell)
      mean_cell_rmse   — unweighted mean over occupied cells (each geometry
                         region counts equally, however many samples landed in it)
      overall_rmse     — plain aggregate psi_rmse for reference
      marginals        — {param: [{"bin", "low", "high", "n", "rmse"}, ...]}
                         with 3 equal-width bins per parameter
    """
    inputs = np.asarray(inputs, dtype=np.float64)
    lows = np.asarray(bounds_low, dtype=np.float64)
    highs = np.asarray(bounds_high, dtype=np.float64)
    if inputs.ndim != 2 or inputs.shape[1] != lows.size:
        raise ValueError(f"inputs must be (N, {lows.size}); got {inputs.shape}")
    n, d = inputs.shape
    span = np.maximum(highs - lows, 1e-12)

    def _bin(n_b: int) -> np.ndarray:
        u = (inputs - lows[None, :]) / span[None, :]
        return np.clip((u * n_b).astype(int), 0, n_b - 1)  # (N, d)

    joint = _bin(int(n_bins))
    cells: dict[str, dict[str, Any]] = {}
    for key in np.unique(joint, axis=0):
        in_cell = np.all(joint == key[None, :], axis=1)
        cells["-".join(str(int(b)) for b in key)] = {
            "n": int(in_cell.sum()),
            "rmse": psi_rmse(true[in_cell], pred[in_cell]),
        }

    occupied = {k: v for k, v in cells.items() if np.isfinite(v["rmse"])}
    worst = (
        max(occupied.items(), key=lambda kv: kv[1]["rmse"]) if occupied else None
    )

    tercile = _bin(3)
    marginals: dict[str, list[dict[str, Any]]] = {}
    for j, name in enumerate(param_names[:d]):
        rows = []
        for b in range(3):
            in_bin = tercile[:, j] == b
            rows.append(
                {
                    "bin": b,
                    "low": float(lows[j] + b * span[j] / 3),
                    "high": float(lows[j] + (b + 1) * span[j] / 3),
                    "n": int(in_bin.sum()),
                    "rmse": psi_rmse(true[in_bin], pred[in_bin]) if in_bin.any() else float("nan"),
                }
            )
        marginals[name] = rows

    return {
        "n_samples": int(n),
        "n_bins": int(n_bins),
        "n_cells_total": int(n_bins**d),
        "n_cells_occupied": len(occupied),
        "cells": cells,
        "worst_cell": (
            {"key": worst[0], "rmse": worst[1]["rmse"], "n": worst[1]["n"]}
            if worst
            else None
        ),
        "mean_cell_rmse": (
            float(np.mean([v["rmse"] for v in occupied.values()])) if occupied else float("nan")
        ),
        "overall_rmse": psi_rmse(true, pred),
        "marginals": marginals,
    }


def format_metric_report(summary: dict[str, Any], *, title: str = "ψ metrics") -> str:
    """Human-readable multi-line report for notebook / CLI printing."""
    lines = [f"=== {title} ==="]
    lines.append(f"  cells scored          : {summary.get('n_cells', '—')}")
    lines.append(f"  RMSE                  : {summary['rmse']:.6g}   (ψ units; lower better)")
    lines.append(f"  MAE                   : {summary['mae']:.6g}   (average |error|)")
    lines.append(f"  relative L2           : {summary['rel_l2']:.4%}  (0%=perfect)")
    lines.append(f"  R²                    : {summary['r2']:.4f}   (1=perfect, 0=mean predictor)")
    lines.append(f"  correlation           : {summary['corr']:.4f}   (1=perfect)")
    lines.append(f"  within 5% of truth    : {summary['pct_within_5pct']:.1f}%   (“accuracy-like”)")
    lines.append(
        f"  within abs tol        : {summary['pct_within_abs_tol']:.1f}%   "
        f"(tol={summary['abs_tol_used']:.4g})"
    )
    lines.append(f"  max |error|           : {summary['max_err']:.6g}")
    if summary.get("baseline_rmse") is not None and np.isfinite(summary["baseline_rmse"]):
        lines.append(f"  baseline RMSE         : {summary['baseline_rmse']:.6g}   (predict train mean)")
        lines.append(
            f"  RMSE / baseline       : {summary['rmse_vs_baseline']:.3f}   "
            "(<1 beats mean; 0.28 = 72% error reduction)"
        )
        lines.append(
            f"  error reduction       : {summary['pct_better_than_baseline']:.1f}%   "
            "vs mean predictor"
        )
    return "\n".join(lines)


__all__ = [
    "baseline_mean_prediction",
    "baseline_mean_predictor_rmse",
    "format_metric_report",
    "pearson_r",
    "per_cell_errors",
    "pixelwise_max_err",
    "psi_mae",
    "psi_rmse",
    "r2_score",
    "relative_l2",
    "summarize_psi_errors",
    "within_abs_tolerance",
    "within_rel_tolerance",
]
