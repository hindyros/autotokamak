# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Plasma boundary (LCFS) construction and 2D triangular meshing.

Both example runners pre-refactor reimplemented these calls. After this refactor,
they both import from here. The OFT API used here:

- ``create_isoflux(npts, r0, z0, a, kappa, delta)`` — analytic D-shape boundary
- ``gs_Domain()`` + ``define_region`` + ``add_polygon`` + ``build_mesh`` — meshing
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


def build_lcfs(
    *,
    r0: float,
    z0: float,
    a: float,
    kappa: float,
    delta: float,
    npts: int = 80,
) -> np.ndarray:
    """Build an analytic D-shape Last Closed Flux Surface contour.

    Parameters
    ----------
    r0 : float
        Major radius of the LCFS center (m).
    z0 : float
        Vertical center of the LCFS (m).
    a : float
        Minor radius (m).
    kappa : float
        Elongation. 1.0 = circle; >1 = vertically stretched.
    delta : float
        Triangularity. 0 = symmetric; >0 = inward-pointing D-shape.
    npts : int, default 80
        Number of points along the contour.

    Returns
    -------
    np.ndarray of shape (npts, 2)
        The LCFS as ``(R, Z)`` coordinates.
    """
    from OpenFUSIONToolkit.TokaMaker.util import create_isoflux

    lcfs = create_isoflux(int(npts), float(r0), float(z0), float(a), float(kappa), float(delta))
    return np.asarray(lcfs, dtype=float)


def build_mesh(
    lcfs: np.ndarray,
    *,
    mesh_dx: float,
    region_name: str = "plasma",
    region_tag: str = "plasma",
) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
    """Triangulate the inside of an LCFS polygon at the given target spacing.

    Parameters
    ----------
    lcfs : np.ndarray (N, 2)
        LCFS boundary coordinates.
    mesh_dx : float
        Target triangle edge length (m). Smaller = more accurate, more expensive.
    region_name, region_tag : str
        Labels for the gs_Domain region. Defaults are appropriate for single-region GS.

    Returns
    -------
    (gs_mesh, mesh_pts, mesh_lc, mesh_reg)
        - ``gs_mesh`` — the gs_Domain handle (needed for plotting)
        - ``mesh_pts`` — node coordinates, shape (N_nodes, 2)
        - ``mesh_lc`` — element connectivity, shape (N_tri, 3) integer indices
        - ``mesh_reg`` — per-element region tag, shape (N_tri,)
    """
    from OpenFUSIONToolkit.TokaMaker.meshing import gs_Domain

    gs_mesh = gs_Domain()
    gs_mesh.define_region(region_name, float(mesh_dx), region_tag)
    gs_mesh.add_polygon(np.asarray(lcfs, dtype=float), region_name)
    mesh_pts, mesh_lc, mesh_reg = gs_mesh.build_mesh()
    return gs_mesh, mesh_pts, mesh_lc, mesh_reg


def build_mesh_from_config(cfg: Dict[str, Any]) -> Tuple[np.ndarray, Any, np.ndarray, np.ndarray, np.ndarray]:
    """Convenience wrapper: read a config dict (the schema used by both runners)
    and return ``(lcfs, gs_mesh, mesh_pts, mesh_lc, mesh_reg)``.

    Expects ``cfg['boundary']`` with keys ``{npts, r0, z0, a, kappa, delta}`` and
    ``cfg['mesh']['regions'][0]['dx']``.
    """
    b = cfg["boundary"]
    lcfs = build_lcfs(
        r0=float(b["r0"]),
        z0=float(b["z0"]),
        a=float(b["a"]),
        kappa=float(b["kappa"]),
        delta=float(b["delta"]),
        npts=int(b["npts"]),
    )
    dx = float(cfg["mesh"]["regions"][0]["dx"])
    gs_mesh, mesh_pts, mesh_lc, mesh_reg = build_mesh(lcfs, mesh_dx=dx)
    return lcfs, gs_mesh, mesh_pts, mesh_lc, mesh_reg


__all__ = ["build_lcfs", "build_mesh", "build_mesh_from_config"]
