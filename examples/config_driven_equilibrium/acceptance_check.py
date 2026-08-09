#!/usr/bin/env python3
# provenance: URSA-generated (URSA plan->execute agent)
"""Lightweight acceptance/regression checks for OFT config-driven equilibrium.

Usage:
  python acceptance_check.py outputs/<case_dir>

Checks performed:
- required artifacts exist
- key diagnostic scalars are finite and within broad bounds
- mesh resolution sanity checks

Exit code:
  0 = pass
  2 = fail
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


REQUIRED = [
    "summary.json",
    "raw_arrays.npz",
    "config_used.yaml",
]

# plots are name-encoded; accept either png or pdf for each type
PLOT_TYPES = ["mesh_", "psi_"]


def _isfinite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("case_dir")
    args = ap.parse_args(argv)

    case_dir = Path(args.case_dir)
    if not case_dir.exists():
        print(f"FAIL: case_dir not found: {case_dir}")
        return 2

    missing = [fn for fn in REQUIRED if not (case_dir / fn).exists()]

    # plot existence
    for prefix in PLOT_TYPES:
        if not any(p.name.startswith(prefix) and p.suffix in {".png", ".pdf"} for p in case_dir.iterdir() if p.is_file()):
            missing.append(f"{prefix}*.png|pdf")

    if missing:
        print("FAIL: missing required artifacts:")
        for m in missing:
            print("  -", m)
        return 2

    s = json.loads((case_dir / "summary.json").read_text())

    stats = (s.get("diagnostics") or {}).get("stats") or {}
    mesh = s.get("mesh") or {}

    # broad bounds: catch NaNs/explosions but tolerant to changes
    bounds = {
        "Ip": (1e3, 1e8),
        "R_geo": (0.1, 5.0),
        "a_geo": (0.01, 5.0),
        "vol": (1e-8, 1e3),
        "q_0": (0.01, 20.0),
        "q_95": (0.01, 50.0),
        "beta_pol": (0.0, 1e5),
        "l_i": (0.0, 50.0),
    }

    bad = []
    for k, (lo, hi) in bounds.items():
        v = stats.get(k, None)
        if not _isfinite(v):
            bad.append((k, v, "not finite"))
            continue
        fv = float(v)
        if not (lo <= fv <= hi):
            bad.append((k, fv, f"out of bounds [{lo}, {hi}]"))

    # mesh sanity
    npts = mesh.get("n_points")
    nelem = mesh.get("n_elements")
    if not _isfinite(npts) or int(npts) < 50:
        bad.append(("mesh.n_points", npts, "too small"))
    if not _isfinite(nelem) or int(nelem) < 50:
        bad.append(("mesh.n_elements", nelem, "too small"))

    if bad:
        print("FAIL: acceptance checks failed:")
        for k, v, msg in bad:
            print(f"  - {k}: {v} ({msg})")
        return 2

    print("PASS")
    print(f"  mesh: n_points={int(npts)}, n_elements={int(nelem)}")
    print(f"  Ip={stats.get('Ip')}  q0={stats.get('q_0')}  q95={stats.get('q_95')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
