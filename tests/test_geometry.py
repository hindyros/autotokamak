# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Smoke tests for autotokamak.core.geometry — LCFS construction shape and bounds."""

import numpy as np
import pytest

from autotokamak.core.geometry import build_lcfs, build_mesh_from_config


def test_build_lcfs_shape():
    lcfs = build_lcfs(r0=0.42, z0=0.0, a=0.15, kappa=1.4, delta=0.0, npts=80)
    assert lcfs.shape == (80, 2), f"Expected (80, 2), got {lcfs.shape}"
    assert lcfs.dtype == np.float64


def test_build_lcfs_within_minor_radius_bounds():
    """All LCFS points should lie within the (R0 - a, R0 + a) horizontal envelope."""
    r0, a = 0.42, 0.15
    lcfs = build_lcfs(r0=r0, z0=0.0, a=a, kappa=1.4, delta=0.4, npts=80)
    R = lcfs[:, 0]
    assert R.min() >= r0 - a - 1e-9, f"R min {R.min()} below R0 - a = {r0 - a}"
    assert R.max() <= r0 + a + 1e-9, f"R max {R.max()} above R0 + a = {r0 + a}"


def test_build_lcfs_circle_when_kappa_1_delta_0():
    """kappa=1.0, delta=0 should produce a near-circular boundary."""
    a = 0.15
    lcfs = build_lcfs(r0=0.42, z0=0.0, a=a, kappa=1.0, delta=0.0, npts=80)
    # distance from center
    radii = np.sqrt((lcfs[:, 0] - 0.42) ** 2 + lcfs[:, 1] ** 2)
    assert np.allclose(radii, a, atol=0.01), f"radii spread = {radii.std()} > 0.01"


@pytest.mark.slow
def test_build_mesh_from_config_yields_expected_shapes():
    cfg = {
        "boundary": {"npts": 80, "r0": 0.42, "z0": 0.0, "a": 0.15, "kappa": 1.4, "delta": 0.0},
        "mesh": {"regions": [{"name": "plasma", "type": "plasma", "dx": 0.015}]},
    }
    lcfs, gs_mesh, mesh_pts, mesh_lc, mesh_reg = build_mesh_from_config(cfg)
    assert lcfs.shape == (80, 2)
    assert mesh_pts.ndim == 2 and mesh_pts.shape[1] == 2
    assert mesh_lc.ndim == 2 and mesh_lc.shape[1] == 3
    assert mesh_reg.shape[0] == mesh_lc.shape[0]
    # Mesh should be reasonable: at least a few hundred nodes for dx=0.015 on a ~0.3m wide region.
    assert mesh_pts.shape[0] > 100, f"mesh too coarse: only {mesh_pts.shape[0]} nodes"
