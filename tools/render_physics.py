# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Render tokamak physics visualizations from a dataset.h5.

Usage:
    python tools/render_physics.py --workspace examples/surrogate_meta
    python tools/render_physics.py --dataset examples/dataset_generation/outputs/dataset.h5

Writes two PNGs to <workspace>/outputs/physics_plots/:
    1. psi_samples.png         — 6 sample ψ(R,Z) contour plots with LCFS overlay
    2. param_distributions.png — histograms of r0, a, kappa, delta, Ip

For meta-loop workspaces the dataset is typically symlinked at
surrogate_runs/iter*/dataset.h5. This script probes those locations too.

Reused by tools/trace_to_html.py, which embeds the PNGs into the run's HTML
report page (silent no-op if this hasn't been run yet).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from autotokamak.surrogate.dataset import PARAM_ORDER, load_dataset
from autotokamak.pipelines.discover import find_training_dataset


def _find_dataset(workspace: Path) -> Path | None:
    """Locate the dataset to visualize for a workspace.

    Handles both direct phase workspaces (``dataset.h5`` / ``outputs/dataset.h5``)
    and meta-loop workspaces (``datasets/train_pool.h5`` and the grown
    ``datasets/iterN_dataset.h5`` pools). See ``autotokamak.pipelines.discover``.
    """
    return find_training_dataset(workspace)


def _pick_sample_indices(inputs: np.ndarray, n: int) -> np.ndarray:
    """Pick n samples spread across the kappa (elongation) axis so the reader
    sees a range of plasma shapes, not n near-duplicates."""
    n = min(n, inputs.shape[0])
    if n == 0:
        return np.array([], dtype=int)
    kappa_col = PARAM_ORDER.index("kappa")
    kappa = inputs[:, kappa_col]
    order = np.argsort(kappa)
    # Evenly spaced picks along the sorted-by-kappa axis.
    picks = np.linspace(0, len(order) - 1, num=n, dtype=int)
    return order[picks]


def _plot_psi_grid(bundle, out_path: Path, n: int = 6) -> None:
    idx = _pick_sample_indices(bundle.inputs, n)
    if idx.size == 0:
        return
    ncols = 3
    nrows = (idx.size + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), squeeze=False)
    R, Z = bundle.R, bundle.Z
    for panel_i, sample_i in enumerate(idx):
        ax = axes[panel_i // ncols][panel_i % ncols]
        psi = bundle.psi[sample_i]
        # Contour lines over a filled background.
        finite = np.isfinite(psi)
        if not finite.any():
            ax.set_visible(False)
            continue
        vmin, vmax = np.nanmin(psi), np.nanmax(psi)
        pc = ax.pcolormesh(R, Z, psi, cmap="RdBu_r", vmin=vmin, vmax=vmax, shading="auto")
        # Overlay ψ contours to make the flux surfaces visible.
        try:
            levels = np.linspace(vmin, vmax, 12)
            ax.contour(R, Z, psi, levels=levels, colors="k", linewidths=0.4, alpha=0.5)
        except Exception:  # noqa: BLE001
            pass
        # LCFS approximation from the analytic Miller-style shape parameters.
        r0, a, kappa, delta, Ip = bundle.inputs[sample_i]
        theta = np.linspace(0, 2 * np.pi, 256)
        r_lcfs = r0 + a * np.cos(theta + delta * np.sin(theta))
        z_lcfs = a * kappa * np.sin(theta)
        ax.plot(r_lcfs, z_lcfs, "k-", linewidth=1.5, label="LCFS")
        ax.set_aspect("equal")
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
        ax.set_title(
            f"#{sample_i}: R0={r0:.2f}, a={a:.2f}, κ={kappa:.2f}, δ={delta:.2f}, Ip={Ip / 1e6:.2f}MA",
            fontsize=9,
        )
        fig.colorbar(pc, ax=ax, shrink=0.75, label="ψ [Wb/rad]")
    # Hide unused panels.
    for panel_i in range(idx.size, nrows * ncols):
        axes[panel_i // ncols][panel_i % ncols].set_visible(False)
    fig.suptitle(f"Sample equilibria from {Path(bundle.source_path).name} (N={bundle.n_samples})", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _plot_param_distributions(bundle, out_path: Path) -> None:
    n_params = len(PARAM_ORDER)
    fig, axes = plt.subplots(1, n_params, figsize=(3 * n_params, 3), squeeze=False)
    for i, name in enumerate(PARAM_ORDER):
        ax = axes[0][i]
        values = bundle.inputs[:, i]
        ax.hist(values, bins=min(20, max(5, bundle.n_samples // 4)), color="#0366d6", edgecolor="white")
        ax.set_xlabel(name)
        ax.set_ylabel("count" if i == 0 else "")
        ax.set_title(f"{name}: [{values.min():.3g}, {values.max():.3g}]", fontsize=9)
    fig.suptitle(f"Input parameter distribution (N={bundle.n_samples})", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", default=None, help="Workspace dir; script finds dataset.h5 inside it.")
    p.add_argument("--dataset", default=None, help="Path to dataset.h5 directly.")
    p.add_argument("--out", default=None, help="Output dir (default: <workspace>/outputs/physics_plots).")
    p.add_argument("--n-samples", type=int, default=6, help="Number of ψ panels to render.")
    args = p.parse_args()

    if args.dataset:
        dataset_path = Path(args.dataset).resolve()
        workspace = Path(args.workspace).resolve() if args.workspace else dataset_path.parent
    elif args.workspace:
        workspace = Path(args.workspace).resolve()
        dataset_path = _find_dataset(workspace)
        if dataset_path is None:
            raise SystemExit(f"No dataset.h5 found under {workspace} (or its surrogate_runs/).")
    else:
        raise SystemExit("Pass --workspace or --dataset.")

    out_dir = Path(args.out).resolve() if args.out else workspace / "outputs" / "physics_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_dataset(dataset_path)
    print(f"Loaded {bundle.n_samples} samples from {dataset_path}; grid {bundle.grid_shape}.")

    _plot_psi_grid(bundle, out_dir / "psi_samples.png", n=args.n_samples)
    print(f"Wrote {out_dir / 'psi_samples.png'}")
    _plot_param_distributions(bundle, out_dir / "param_distributions.png")
    print(f"Wrote {out_dir / 'param_distributions.png'}")


if __name__ == "__main__":
    main()
