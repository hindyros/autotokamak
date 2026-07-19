"""Evaluate a trained surrogate and dump diagnostic plots.

Usage:
    python tools/eval_surrogate.py                       # workspace=examples/surrogate_automl
    python tools/eval_surrogate.py --workspace <path>    # custom workspace
    python tools/eval_surrogate.py --out <dir>           # custom output dir

Reads:
    <workspace>/dataset.h5
    <workspace>/outputs/winner.pkl
    <workspace>/outputs/report.json
    <workspace>/outputs/study.db   (optional, for Optuna history plot)

Writes PNGs to <workspace>/outputs/eval_plots/.

The seven figures:
    1. true_pred_residual.png       — 3-panel heatmap per test sample
    2. per_sample_rmse.png          — bar chart, surrogate vs mean-predictor baseline
    3. pred_vs_true_scatter.png     — pixel-wise scatter with 1:1 line
    4. residual_histogram.png       — distribution of per-pixel errors
    5. pca_variance.png             — cumulative explained-variance curve
    6. pca_reconstruction.png       — PCA-only vs full-pipeline error, isolates the bottleneck
    7. optuna_history.png           — per-model best-value convergence (if study.db exists)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from autotokamak.eval.data import kfold, load_dataset
from autotokamak.eval.discover import (
    find_eval_dataset,
    find_report,
    find_winner,
)
from autotokamak.eval.metrics import baseline_mean_predictor_rmse, psi_rmse
from autotokamak.eval.reduce import inverse_transform, transform
from autotokamak.surrogate.automl import predict_with_winner


def _find_study_db(workspace: Path) -> Path | None:
    """Locate an Optuna study.db across direct and meta-loop sub-run layouts."""
    candidates = [workspace / "outputs" / "study.db"]
    sur = workspace / "surrogate_runs"
    if sur.is_dir():
        # latest sub-run's study first
        candidates += sorted(sur.glob("*/outputs/study.db"), reverse=True)
    for p in candidates:
        if p.exists():
            return p
    return None


def _load_run(workspace: Path):
    """Load winner + eval dataset + report, resolving both workspace layouts.

    The eval dataset is the winner's held-out set (the meta-loop's frozen shard
    when present), so true-vs-predicted plots stay honest. See
    ``autotokamak.eval.discover``.
    """
    winner_path = find_winner(workspace)
    if winner_path is None:
        raise FileNotFoundError(f"no winner.pkl found under {workspace}")
    payload = joblib.load(winner_path)

    ds_path = find_eval_dataset(workspace)
    if ds_path is None:
        raise FileNotFoundError(f"no dataset found under {workspace}")
    bundle = load_dataset(ds_path)

    report_path = find_report(workspace)
    report = json.loads(report_path.read_text()) if report_path else {}

    seed = 0
    k = 4
    test_frac = 2 / bundle.n_samples
    splits = kfold(bundle, k=k, test_frac=test_frac, seed=seed)
    return payload, bundle, splits, report


def _symmetric_vlim(*arrays):
    m = 0.0
    for a in arrays:
        finite = a[np.isfinite(a)]
        if finite.size:
            m = max(m, float(np.max(np.abs(finite))))
    return -m, m


def _apply_lcfs_mask(psi_true, psi_pred):
    """Mask predicted values wherever the truth is NaN (outside the LCFS)."""
    mask = ~np.isfinite(psi_true)
    out = psi_pred.copy()
    out[mask] = np.nan
    return out


def plot_true_pred_residual(bundle, payload, splits, out_dir):
    idx = list(splits.test_idx)
    X = bundle.inputs[idx]
    pred = predict_with_winner(payload, X)
    true = bundle.psi[idx]
    pred = _apply_lcfs_mask(true, pred)
    n = len(idx)
    fig, axes = plt.subplots(n, 3, figsize=(11, 3.5 * n), squeeze=False)
    R, Z = bundle.R, bundle.Z
    for row, (i_ds, sample_true, sample_pred) in enumerate(zip(idx, true, pred)):
        residual = sample_pred - sample_true
        vt = _symmetric_vlim(sample_true, sample_pred)
        vr = _symmetric_vlim(residual)
        rmse = float(psi_rmse(sample_true[None], sample_pred[None]))
        rel = float(
            np.sqrt(np.nansum(residual ** 2))
            / max(np.sqrt(np.nansum(sample_true ** 2)), 1e-12)
        )
        for col, (data, title, vlim) in enumerate([
            (sample_true, f"true ψ", vt),
            (sample_pred, f"predicted ψ", vt),
            (residual, f"residual (pred − true)", vr),
        ]):
            im = axes[row, col].pcolormesh(R, Z, data, cmap="RdBu_r", vmin=vlim[0], vmax=vlim[1], shading="auto")
            axes[row, col].set_aspect("equal")
            axes[row, col].set_xlabel("R [m]")
            axes[row, col].set_ylabel("Z [m]")
            axes[row, col].set_title(f"sample {i_ds} — {title}")
            fig.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
        axes[row, 2].text(
            0.02, 0.98,
            f"RMSE = {rmse:.3g}\nrel L2 = {rel:.3g}",
            transform=axes[row, 2].transAxes, va="top", ha="left",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9),
        )
    fig.suptitle(f"True vs. predicted ψ on the {n} test samples", y=1.0)
    fig.tight_layout()
    fig.savefig(out_dir / "true_pred_residual.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_per_sample_rmse(bundle, payload, splits, out_dir):
    idx = list(splits.test_idx)
    X = bundle.inputs[idx]
    pred = _apply_lcfs_mask(bundle.psi[idx], predict_with_winner(payload, X))
    true = bundle.psi[idx]

    # Per-sample RMSE from surrogate
    per_sample = [float(psi_rmse(true[i:i+1], pred[i:i+1])) for i in range(len(idx))]

    # Baseline predictor: mean of the *training* psi on this fold's train split.
    # Compute per-fold baseline then average — matches what the metric does.
    fold_baselines = []
    for _, tr, va in splits.iter_folds():
        fold_baselines.append(baseline_mean_predictor_rmse(bundle.psi[tr], bundle.psi[va]))
    baseline_avg = float(np.mean(fold_baselines))

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(idx))
    ax.bar(x, per_sample, color="#0366d6", label="surrogate test RMSE")
    ax.axhline(baseline_avg, ls="--", color="#a10000",
               label=f"CV mean-predictor baseline ({baseline_avg:.3g})")
    ax.set_xticks(x)
    ax.set_xticklabels([f"sample {i}" for i in idx])
    ax.set_ylabel("ψ RMSE")
    ax.set_title("Per-sample test RMSE vs. mean-predictor baseline")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "per_sample_rmse.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_pred_vs_true_scatter(bundle, payload, splits, out_dir):
    idx = list(splits.test_idx)
    pred = _apply_lcfs_mask(bundle.psi[idx], predict_with_winner(payload, bundle.inputs[idx]))
    true = bundle.psi[idx]
    m = np.isfinite(true) & np.isfinite(pred)
    t = true[m].ravel()
    p = pred[m].ravel()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.hexbin(t, p, gridsize=60, cmap="Blues", mincnt=1)
    lo, hi = float(min(t.min(), p.min())), float(max(t.max(), p.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x (perfect)")
    r = np.corrcoef(t, p)[0, 1]
    ax.set_xlabel("true ψ (pixel value)")
    ax.set_ylabel("predicted ψ (pixel value)")
    ax.set_title(f"Predicted vs. true ψ, per pixel (test split, N={t.size:,})")
    ax.text(0.02, 0.98, f"Pearson r = {r:.4f}", transform=ax.transAxes,
            va="top", ha="left", family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9))
    ax.legend(loc="lower right", fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_dir / "pred_vs_true_scatter.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_residual_histogram(bundle, payload, splits, out_dir):
    idx = list(splits.test_idx)
    pred = _apply_lcfs_mask(bundle.psi[idx], predict_with_winner(payload, bundle.inputs[idx]))
    true = bundle.psi[idx]
    residuals = (pred - true)[np.isfinite(pred) & np.isfinite(true)].ravel()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(residuals, bins=80, color="#0366d6", edgecolor="white", linewidth=0.5)
    mu, sigma = float(np.mean(residuals)), float(np.std(residuals))
    ax.axvline(0.0, color="k", lw=1)
    ax.set_xlabel("residual  (predicted − true)")
    ax.set_ylabel("count of pixels")
    ax.set_title("Distribution of per-pixel residuals on the test split")
    ax.text(0.98, 0.98,
            f"mean = {mu:+.3g}\nstd  = {sigma:.3g}\nN = {residuals.size:,}",
            transform=ax.transAxes, va="top", ha="right", family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9))
    fig.tight_layout()
    fig.savefig(out_dir / "residual_histogram.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_pca_variance(payload, out_dir):
    pca = payload["pca"]
    ratios = np.asarray(pca.explained_variance_ratio, dtype=float)
    cum = np.cumsum(ratios)
    n = pca.n_components

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(1, n + 1), cum, "-o", color="#0366d6")
    ax.axhline(0.95, color="#a10000", ls="--", label="0.95 target")
    ax.set_xlabel("PCA component (rank)")
    ax.set_ylabel("cumulative explained variance")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"PCA reduction: {n} components used  ·  total = {cum[-1]:.3f}")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "pca_variance.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_pca_reconstruction(bundle, payload, splits, out_dir):
    """Compare (a) surrogate prediction to (b) PCA-encode-then-decode of the truth.

    If (b) is close to truth but (a) is not, the regression is the bottleneck.
    If (b) itself has large error, the PCA reduction is the bottleneck.
    """
    pca = payload["pca"]
    idx = list(splits.test_idx)
    true = bundle.psi[idx]
    coeffs = transform(pca, true)
    reconstructed = inverse_transform(pca, coeffs)
    reconstructed = _apply_lcfs_mask(true, reconstructed)

    surrogate = _apply_lcfs_mask(true, predict_with_winner(payload, bundle.inputs[idx]))

    pca_rmse = [float(psi_rmse(true[i:i+1], reconstructed[i:i+1])) for i in range(len(idx))]
    surr_rmse = [float(psi_rmse(true[i:i+1], surrogate[i:i+1])) for i in range(len(idx))]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(idx))
    width = 0.35
    ax.bar(x - width / 2, pca_rmse, width, color="#28a745",
           label="PCA-only reconstruction error (encode → decode the truth)")
    ax.bar(x + width / 2, surr_rmse, width, color="#0366d6",
           label="Full-pipeline error (predict from inputs, then decode)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"sample {i}" for i in idx])
    ax.set_ylabel("ψ RMSE")
    ax.set_title(
        "Where does the error come from? PCA reduction vs. full surrogate.\n"
        "If green ≈ blue, the regressor is fine and PCA is the ceiling. "
        "If green ≪ blue, the regressor needs work."
    )
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "pca_reconstruction.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_optuna_history(study_db, out_dir):
    if study_db is None or not Path(study_db).exists():
        return
    try:
        import optuna
        storage = f"sqlite:///{study_db}"
        summaries = optuna.study.get_all_study_summaries(storage=storage)
    except Exception as exc:
        print(f"  (skipping optuna_history: {type(exc).__name__}: {exc})")
        return
    if not summaries:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    plotted = 0
    for s in summaries:
        try:
            study = optuna.load_study(study_name=s.study_name, storage=storage)
            trials = [t for t in study.trials if t.value is not None]
            if not trials:
                continue
            best = np.minimum.accumulate([t.value for t in trials])
            ax.plot(np.arange(1, len(best) + 1), best, label=s.study_name)
            plotted += 1
        except Exception:
            continue
    if plotted == 0:
        plt.close(fig)
        return
    ax.set_xlabel("trial index")
    ax.set_ylabel("best objective value so far (lower is better)")
    ax.set_title("Optuna convergence — best value per study")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "optuna_history.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def _summary(workspace, payload, bundle, splits, report):
    idx = list(splits.test_idx)
    pred = _apply_lcfs_mask(bundle.psi[idx], predict_with_winner(payload, bundle.inputs[idx]))
    test_rmse = float(psi_rmse(bundle.psi[idx], pred))
    baseline = float(np.mean(
        [baseline_mean_predictor_rmse(bundle.psi[tr], bundle.psi[va])
         for _, tr, va in splits.iter_folds()]
    ))
    beats = "YES" if test_rmse < baseline else "no"
    return (
        f"workspace       : {workspace}\n"
        f"winner          : {report.get('winner_model_name')}  "
        f"({report.get('n_total_trials')} trials, {len(report.get('models_tried') or [])} models)\n"
        f"test_psi_rmse   : {test_rmse:.4f}   (report says {report.get('test_psi_rmse')})\n"
        f"baseline (mean) : {baseline:.4f}\n"
        f"beats baseline? : {beats}\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workspace", default=str(REPO_ROOT / "examples/surrogate_automl"))
    ap.add_argument("--out", default=None, help="Output dir (default: <workspace>/outputs/eval_plots)")
    args = ap.parse_args()

    workspace = Path(args.workspace)
    out_dir = Path(args.out) if args.out else workspace / "outputs/eval_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    payload, bundle, splits, report = _load_run(workspace)

    plots = [
        ("true_pred_residual", lambda: plot_true_pred_residual(bundle, payload, splits, out_dir)),
        ("per_sample_rmse", lambda: plot_per_sample_rmse(bundle, payload, splits, out_dir)),
        ("pred_vs_true_scatter", lambda: plot_pred_vs_true_scatter(bundle, payload, splits, out_dir)),
        ("residual_histogram", lambda: plot_residual_histogram(bundle, payload, splits, out_dir)),
        ("pca_variance", lambda: plot_pca_variance(payload, out_dir)),
        ("pca_reconstruction", lambda: plot_pca_reconstruction(bundle, payload, splits, out_dir)),
        ("optuna_history", lambda: plot_optuna_history(_find_study_db(workspace), out_dir)),
    ]
    for name, fn in plots:
        try:
            fn()
            print(f"  wrote {out_dir / (name + '.png')}")
        except Exception as exc:
            print(f"  SKIP {name}: {type(exc).__name__}: {exc}")

    print()
    print(_summary(workspace, payload, bundle, splits, report))


if __name__ == "__main__":
    main()
