#!/usr/bin/env python3
# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Reference entry point for a new autotokamak example workspace.

Structure follows examples/config_driven_equilibrium/run_equilibrium_from_config.py:
  1. Load YAML → EquilibriumConfig.
  2. Build LCFS + mesh from cfg.
  3. Solve; check get_last_solve_info() for isoflux fallback.
  4. Save summary.json, raw_arrays.npz, mesh.png, psi.png, config_used.yaml.

Never import OFT directly — use autotokamak.core.solver.get_oft_env() for the
singleton, or (preferred) call solve_equilibrium() which manages it internally.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from autotokamak.core.diagnostics import extract_scalars
from autotokamak.core.geometry import build_mesh_from_config
from autotokamak.core.io import atomic_savez, atomic_write_text, unified_output_dir, utc_run_id
from autotokamak.core.logging import kv, log, section
from autotokamak.core.schema import EquilibriumConfig
from autotokamak.core.solver import get_last_solve_info, solve_equilibrium


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="Equilibrium YAML")
    args = ap.parse_args()

    section("Load config")
    cfg_model = EquilibriumConfig.from_yaml(args.config)
    cfg = cfg_model.model_dump()
    kv("case", cfg.get("case_id", "<unnamed>"))

    section("Build LCFS + mesh")
    lcfs, gs_mesh, mesh_pts, mesh_lc, mesh_reg = build_mesh_from_config(cfg)
    kv("n_nodes", int(np.asarray(mesh_pts).shape[0]))
    kv("n_elements", int(np.asarray(mesh_lc).shape[0]))

    section("Solve")
    gs = solve_equilibrium(
        mesh_pts=mesh_pts, mesh_lc=mesh_lc, mesh_reg=mesh_reg,
        lcfs=lcfs, cfg=cfg,
    )
    info = get_last_solve_info()
    kv("isoflux_used", info["isoflux_used"])
    if not info["isoflux_used"]:
        log(f"WARNING: fallback path — {info['fallback_reason']}")

    section("Diagnostics")
    scalars = extract_scalars(gs)
    for k, v in scalars.items():
        if k != "stats":
            kv(k, v)

    section("Save")
    out_root = unified_output_dir(cfg["outputs"]["out_dir"], run_id=utc_run_id())
    atomic_savez(
        out_root / "raw_arrays.npz",
        mesh_pts=np.asarray(mesh_pts),
        mesh_lc=np.asarray(mesh_lc),
        mesh_reg=np.asarray(mesh_reg) if mesh_reg is not None else np.asarray([]),
        lcfs=np.asarray(lcfs),
    )

    summary = {
        "case": cfg.get("case_id"),
        "solve_info": info,
        "diagnostics": {k: v for k, v in scalars.items() if k != "stats"},
    }
    atomic_write_text(out_root / "summary.json", json.dumps(summary, indent=2, default=str))
    atomic_write_text(out_root / "config_used.yaml",
                      Path(args.config).read_text(encoding="utf-8"))

    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    gs_mesh.plot_mesh(fig, ax)
    ax.plot(lcfs[:, 0], lcfs[:, 1], "r-", lw=1.2, label="LCFS")
    ax.legend(loc="best", fontsize=8)
    fig.savefig(out_root / cfg["outputs"]["mesh_png"], dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    gs.plot_psi(fig, ax)
    ax.plot(lcfs[:, 0], lcfs[:, 1], "r-", lw=1.2)
    fig.savefig(out_root / cfg["outputs"]["psi_png"], dpi=200)
    plt.close(fig)

    log(f"Wrote outputs to {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
