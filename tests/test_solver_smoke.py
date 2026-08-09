# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Full TokaMaker solve smoke test. Marked @pytest.mark.slow so default `pytest` skips it.

Run with: ``pytest tests/ -v -m slow``
"""

import pytest

from autotokamak.core.geometry import build_mesh_from_config
from autotokamak.core.solver import solve_equilibrium


@pytest.mark.slow
def test_smoke_solve_completes():
    cfg = {
        "boundary": {"npts": 80, "r0": 0.42, "z0": 0.0, "a": 0.15, "kappa": 1.4, "delta": 0.0},
        "mesh": {"regions": [{"name": "plasma", "type": "plasma", "dx": 0.015}]},
        "solver": {"order": 2, "F0": 0.10752, "free_boundary": False},
        "targets": {"Ip": 120000.0, "Ip_ratio": 1.0},
        "init_psi": {"method": "tokamaker_default"},
    }
    lcfs, _, mesh_pts, mesh_lc, mesh_reg = build_mesh_from_config(cfg)
    # No maxits cap: this OFT build RAISES ValueError('Exceeded "maxits"')
    # rather than returning early, and the reference shape converges in
    # ~10 iterations (milliseconds) anyway.
    gs = solve_equilibrium(
        mesh_pts=mesh_pts,
        mesh_lc=mesh_lc,
        mesh_reg=mesh_reg,
        lcfs=lcfs,
        cfg=cfg,
    )
    assert gs is not None

    from autotokamak.core.solver import get_last_solve_info

    info = get_last_solve_info()
    # Fixed-boundary: the isoflux constraint must NOT have been attempted —
    # the mesh boundary enforces the shape (see solve_equilibrium docstring).
    assert info["isoflux_used"] is False
    assert info["fallback_reason"] is None
    assert info["boundary_enforced_by"] == "mesh"
