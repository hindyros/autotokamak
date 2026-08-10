#!/usr/bin/env python
# provenance: URSA-generated (URSA plan->execute agent)
"""One isolated TokaMaker forward solve: write ψ to NPZ. Used by run_invert_psi (subprocess).

Usage:
  python forward_once.py CASE.yaml OUT.npz

Must be run with cwd on ``sys.path`` containing ``run_equilibrium_from_config`` (this directory).
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: forward_once.py CASE.yaml OUT.npz", file=sys.stderr)
        return 1
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

    import numpy as np

    from run_equilibrium_from_config import build_mesh_and_boundary, load_config, setup_and_solve
    from run_invert_psi import extract_psi

    cfg_path = Path(sys.argv[1]).resolve()
    out_path = Path(sys.argv[2]).resolve()
    cfg = load_config(str(cfg_path))
    _mesh, lcfs, mesh_pts, mesh_lc, mesh_reg = build_mesh_and_boundary(cfg)
    gs = setup_and_solve(cfg, mesh_pts, mesh_lc, mesh_reg, lcfs)
    psi = extract_psi(gs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, psi=psi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
