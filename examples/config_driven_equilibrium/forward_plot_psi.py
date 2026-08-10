#!/usr/bin/env python
# provenance: URSA-generated (URSA plan->execute agent)
"""One isolated TokaMaker solve and ψ plot (subprocess helper for run_invert_psi).

Usage:
  python forward_plot_psi.py CASE.yaml OUT.png [--title \"...\"]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent


def main() -> int:
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from run_equilibrium_from_config import build_mesh_and_boundary, load_config, setup_and_solve

    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path, help="Equilibrium YAML")
    ap.add_argument("output", type=Path, help="Output PNG path")
    ap.add_argument("--title", default="", help="Figure title")
    args = ap.parse_args()

    cfg = load_config(str(args.config.resolve()))
    gs_mesh, lcfs, mesh_pts, mesh_lc, mesh_reg = build_mesh_and_boundary(cfg)
    gs = setup_and_solve(cfg, mesh_pts, mesh_lc, mesh_reg, lcfs)

    out = args.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 5), constrained_layout=True)
    gs.plot_psi(fig, ax)
    ax.plot(lcfs[:, 0], lcfs[:, 1], "r-", lw=1.2, label="LCFS")
    if args.title:
        ax.set_title(args.title, fontsize=11)
    ax.legend(loc="best", fontsize=8)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
