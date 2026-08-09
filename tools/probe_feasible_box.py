# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Probe which parameter box gives a clean isoflux success rate.

For each candidate box, draw N LHS samples and run solve_equilibrium.
Reports (isoflux_used == True) fraction per box. The one with highest
fraction is a safe target for scaling n_samples.

Usage:
    python tools/probe_feasible_box.py            # runs the built-in matrix
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
from scipy.stats import qmc

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from autotokamak.core.geometry import build_lcfs, build_mesh
from autotokamak.core.solver import get_last_solve_info, solve_equilibrium


BOXES = {
    "A_original":   dict(r0=(0.35, 0.55), a=(0.10, 0.20), kappa=(1.0, 1.6), delta=(0.0, 0.40)),
    "B_medium":     dict(r0=(0.40, 0.55), a=(0.12, 0.20), kappa=(1.0, 1.4), delta=(0.0, 0.25)),
    "C_tight":      dict(r0=(0.42, 0.52), a=(0.13, 0.19), kappa=(1.0, 1.3), delta=(0.0, 0.20)),
    "D_low_shape":  dict(r0=(0.40, 0.55), a=(0.12, 0.20), kappa=(1.0, 1.3), delta=(0.0, 0.15)),
}


def _solve_one(r0: float, a: float, kappa: float, delta: float, Ip: float) -> dict:
    """Return {'ok': bool, 'isoflux_used': bool, 'error': str|None}."""
    try:
        lcfs = build_lcfs(r0=r0, z0=0.0, a=a, kappa=kappa, delta=delta, npts=80)
        _gs_mesh, mesh_pts, mesh_lc, mesh_reg = build_mesh(
            lcfs, mesh_dx=0.015, region_name="plasma", region_tag="plasma"
        )
    except Exception as exc:
        return {"ok": False, "isoflux_used": False, "error": f"mesh: {type(exc).__name__}: {exc}"}
    cfg = {
        "equation": {"name": "gs"},
        "boundary": {"type": "isoflux", "r0": r0, "z0": 0.0, "a": a,
                     "kappa": kappa, "delta": delta, "npts": 80},
        "mesh": {"method": "gs_domain",
                 "regions": [{"name": "plasma", "type": "plasma", "dx": 0.015}]},
        "solver": {"order": 1, "F0": 0.10752, "free_boundary": False},
        "targets": {"Ip": Ip, "Ip_ratio": 1.0},
        "init_psi": {"method": "isoflux"},
    }
    stderr = io.StringIO()
    stdout = io.StringIO()
    try:
        with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(stdout):
            _ = solve_equilibrium(mesh_pts=mesh_pts, mesh_lc=mesh_lc,
                                  mesh_reg=mesh_reg, lcfs=lcfs, cfg=cfg)
        info = get_last_solve_info()
        return {"ok": True, "isoflux_used": bool(info.get("isoflux_used", False)),
                "error": None}
    except Exception as exc:
        return {"ok": False, "isoflux_used": False,
                "error": f"solve: {type(exc).__name__}: {exc}"}


def probe(box: dict, *, n: int, seed: int = 0) -> dict:
    keys = ["r0", "a", "kappa", "delta"]
    lows = np.array([box[k][0] for k in keys])
    highs = np.array([box[k][1] for k in keys])
    # Ip is scaled to a fixed sensible value; feasibility is dominated by shape.
    sampler = qmc.LatinHypercube(d=4, seed=seed)
    samples = qmc.scale(sampler.random(n), lows, highs)
    results = []
    for i, (r0, a, kappa, delta) in enumerate(samples):
        Ip = 1.2e5
        r = _solve_one(float(r0), float(a), float(kappa), float(delta), Ip)
        results.append(r)
        marker = "iso" if r["isoflux_used"] else ("fb" if r["ok"] else "ERR")
        print(f"  [{i+1:3d}/{n}] {marker:3s}  r0={r0:.3f} a={a:.3f} kappa={kappa:.2f} delta={delta:.2f}"
              + ("" if r["ok"] else f"   {r['error'][:60]}"))
    n_iso = sum(1 for r in results if r["isoflux_used"])
    n_ok = sum(1 for r in results if r["ok"])
    n_err = sum(1 for r in results if not r["ok"])
    return {"n": n, "n_isoflux_used": n_iso, "n_solve_ok": n_ok, "n_error": n_err}


def main() -> None:
    n = 30
    print(f"Probing {len(BOXES)} boxes × {n} samples each. Isoflux-used fraction is the goal.\n")
    results: dict = {}
    for name, box in BOXES.items():
        print(f"\n=== box {name}: {box} ===")
        r = probe(box, n=n)
        results[name] = r
        print(f"  → isoflux_used: {r['n_isoflux_used']}/{n}  "
              f"(solve_ok {r['n_solve_ok']}/{n}, errors {r['n_error']})")
    print("\n=== SUMMARY ===")
    print(f"{'box':16s} {'isoflux/N':>10s} {'solve_ok/N':>12s} {'errors':>8s}")
    for name, r in results.items():
        print(f"{name:16s} {r['n_isoflux_used']:>4}/{r['n']:<4}  {r['n_solve_ok']:>5}/{r['n']:<4}    {r['n_error']:>4}")


if __name__ == "__main__":
    main()
