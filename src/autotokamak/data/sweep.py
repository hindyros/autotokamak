"""Library-level Phase-1 dataset sweep.

Extracts the deterministic loop from the agent-authored runner at
``examples/dataset_generation/run_dataset_sweep.py`` so the meta-agent
(``agent/orchestrator``) can regenerate datasets without spawning a nested
LLM. The agent-authored runner stays as-is for the standalone Phase-1
workflow; this is the same code path for cases where the LLM round-trip
adds no value.

The HDF5 layout is owned by ``autotokamak.data.h5io`` (single source of
truth for grid/inputs/outputs groups); this module delegates the write to
``h5io.write_h5_arrays``, which is also atomic — a sweep killed mid-write
leaves either the previous file or the complete new one, never a truncated
dataset. A dataset produced here is indistinguishable from one produced by
the Phase-1 agent runner to downstream consumers (Phase-2 scorer,
``eval/data.py``, the meta-loop's merge/split helpers).
"""

from __future__ import annotations

import hashlib
import json
import sys
import traceback
from pathlib import Path

import numpy as np

from autotokamak.data.schema import PARAM_ORDER, SweepConfig, SweepResult


def sobol_sample(lows: np.ndarray, highs: np.ndarray, n: int, *, seed: int) -> np.ndarray:
    """Seeded scrambled-Sobol draw of ``n`` points in the box [lows, highs].

    Low-discrepancy: covers the box more evenly than i.i.d. random points.
    Shared by the ``sobol`` sweep method and the acquisition candidate pool
    (``data.acquire``) so the two never drift apart.
    """
    import warnings

    from scipy.stats import qmc

    engine = qmc.Sobol(d=lows.size, scramble=True, seed=seed)
    with warnings.catch_warnings():
        # scipy warns when n is not a power of two; exact balance is not
        # required for a training draw or a scoring pool.
        warnings.simplefilter("ignore", category=UserWarning)
        u = np.asarray(engine.random(n), dtype=np.float64)
    return lows + u * (highs - lows)


def _sample_inputs(cfg: SweepConfig) -> np.ndarray:
    """Return (N, 5) sample matrix in ``PARAM_ORDER``.

    Three seeded methods (two calls with the same ``SweepConfig`` produce the
    same matrix):
      - ``lhs``     : ``scipy.stats.qmc.LatinHypercube`` — stratified, good
                      1-D projections, the default.
      - ``sobol``   : ``scipy.stats.qmc.Sobol`` — low-discrepancy, best
                      uniformity of coverage in higher dimensions.
      - ``uniform`` : ``numpy.random.default_rng`` — plain i.i.d. random.
    """
    from scipy.stats import qmc

    n = cfg.sampling.n_samples
    seed = cfg.sampling.seed
    d = len(PARAM_ORDER)
    lows = np.array([cfg.parameters[p].low for p in PARAM_ORDER], dtype=np.float64)
    highs = np.array([cfg.parameters[p].high for p in PARAM_ORDER], dtype=np.float64)

    if cfg.sampling.method == "sobol":
        return sobol_sample(lows, highs, n, seed=seed)
    if cfg.sampling.method == "lhs":
        engine = qmc.LatinHypercube(d=d, seed=seed)
        u = np.asarray(engine.random(n=n), dtype=np.float64)
    else:
        rng = np.random.default_rng(seed)
        u = rng.random((n, d))
    return lows + u * (highs - lows)


def _interp_psi_to_grid(
    mesh_pts: np.ndarray,
    mesh_lc: np.ndarray,
    psi_nodes: np.ndarray,
    R: np.ndarray,
    Z: np.ndarray,
) -> np.ndarray:
    """Project node-valued psi onto the (R, Z) target grid.

    Uses matplotlib's ``LinearTriInterpolator`` (NaN outside the convex hull),
    matching the agent-authored runner's convention. Falls back to
    scipy.interpolate.griddata if matplotlib's path fails (e.g. degenerate
    triangulation).
    """
    import matplotlib.tri as mtri

    RR, ZZ = np.meshgrid(R, Z, indexing="xy")
    try:
        tri = mtri.Triangulation(mesh_pts[:, 0], mesh_pts[:, 1], triangles=mesh_lc)
        interp = mtri.LinearTriInterpolator(tri, psi_nodes)
        psi_grid = np.asarray(interp(RR, ZZ).filled(np.nan), dtype=np.float64)
        return psi_grid
    except Exception:
        from scipy.interpolate import griddata

        pts = np.column_stack([mesh_pts[:, 0], mesh_pts[:, 1]])
        psi_grid = griddata(pts, psi_nodes, (RR, ZZ), method="linear")
        return np.asarray(psi_grid, dtype=np.float64)


def _config_hash(cfg: SweepConfig) -> str:
    payload = json.dumps(cfg.model_dump(mode="json"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def run_sweep(
    cfg: SweepConfig,
    output_dir: Path | str,
    X: "np.ndarray | None" = None,
) -> SweepResult:
    """Run the GS sweep described by ``cfg``; write HDF5 to ``output_dir``.

    Parameters
    ----------
    cfg : SweepConfig
        Validated sweep configuration. ``cfg.output_path`` is taken
        relative to ``output_dir``.
    output_dir : Path
        Directory the HDF5 lands in. Created if it does not exist.
    X : (N, 5) array, optional
        Explicit sample matrix (columns in ``PARAM_ORDER``). When given,
        ``cfg.sampling`` is ignored and exactly these points are solved —
        the entry point for acquisition-selected batches
        (``data.acquire``). When None (default), samples are drawn from
        ``cfg.sampling`` as before.

    Returns
    -------
    SweepResult
        Counts + path to the written file + config hash (for trace).
    """
    from autotokamak.core.geometry import build_lcfs, build_mesh
    from autotokamak.core.solver import get_last_solve_info, solve_equilibrium
    from autotokamak.data.h5io import DatasetArrays, write_h5_arrays

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / cfg.output_path

    if X is None:
        X = _sample_inputs(cfg)
    else:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != len(PARAM_ORDER):
            raise ValueError(
                f"Explicit X must be (N, {len(PARAM_ORDER)}) in PARAM_ORDER; got {X.shape}"
            )
    N = X.shape[0]

    R = np.linspace(cfg.output_grid.R.min, cfg.output_grid.R.max, cfg.output_grid.R.n)
    Z = np.linspace(cfg.output_grid.Z.min, cfg.output_grid.Z.max, cfg.output_grid.Z.n)
    nr, nz = len(R), len(Z)

    psi_out = np.full((N, nz, nr), np.nan, dtype=np.float64)
    success = np.zeros(N, dtype=bool)
    isoflux_used = np.zeros(N, dtype=bool)

    z0 = cfg.fixed.z0
    F0 = cfg.fixed.F0
    npts = cfg.fixed.npts
    mesh_dx = cfg.fixed.mesh_dx
    solver_order = cfg.fixed.solver_order
    Ip_ratio = cfg.fixed.Ip_ratio
    init_method = cfg.fixed.init_psi_method

    for i in range(N):
        r0, a, kappa, delta, Ip = map(float, X[i])
        try:
            lcfs = build_lcfs(r0=r0, z0=z0, a=a, kappa=kappa, delta=delta, npts=npts)
            _gs_mesh, mesh_pts, mesh_lc, mesh_reg = build_mesh(
                lcfs, mesh_dx=mesh_dx, region_name="plasma", region_tag="plasma"
            )

            solver_cfg = {
                "equation": {"name": "gs"},
                "boundary": {
                    "type": "isoflux",
                    "r0": r0, "z0": z0, "a": a,
                    "kappa": kappa, "delta": delta, "npts": npts,
                },
                "mesh": {
                    "method": "gs_domain",
                    "regions": [{"name": "plasma", "type": "plasma", "dx": mesh_dx}],
                },
                "solver": {"order": solver_order, "F0": F0, "free_boundary": False},
                "targets": {"Ip": Ip, "Ip_ratio": Ip_ratio},
                "init_psi": {"method": init_method},
            }

            gs = solve_equilibrium(
                mesh_pts=mesh_pts, mesh_lc=mesh_lc, mesh_reg=mesh_reg,
                lcfs=lcfs, cfg=solver_cfg,
            )
            info = get_last_solve_info()
            # get_psi(False) = PHYSICAL psi (Webers). The default get_psi()
            # returns per-sample NORMALIZED flux psi_N in [0,1], which erases
            # the Ip dependence entirely — a surrogate trained on psi_N can
            # never learn the current amplitude (diagnosed 2026-07-10).
            psi_nodes = np.asarray(gs.get_psi(False), dtype=float).ravel()
            if len(psi_nodes) != len(mesh_pts):
                raise RuntimeError(
                    f"Expected len(psi_nodes)==len(mesh_pts) for order=1; "
                    f"got {len(psi_nodes)} vs {len(mesh_pts)}"
                )
            psi_grid = _interp_psi_to_grid(mesh_pts, mesh_lc, psi_nodes, R, Z)
            psi_out[i] = psi_grid
            success[i] = True
            isoflux_used[i] = bool(info.get("isoflux_used") or False)
        except Exception as exc:  # noqa: BLE001
            print(f"[run_sweep i={i+1}/{N}] FAILED: {exc}", file=sys.stderr)
            traceback.print_exc()
            success[i] = False
            isoflux_used[i] = False

    write_h5_arrays(
        out_path,
        DatasetArrays(
            R=R,
            Z=Z,
            inputs={p: X[:, j].astype(np.float64) for j, p in enumerate(PARAM_ORDER)},
            psi=psi_out,
            success=success,
            isoflux_used=isoflux_used,
        ),
    )

    return SweepResult(
        dataset_path=str(out_path),
        n_requested=int(N),
        n_succeeded=int(success.sum()),
        n_isoflux_used=int(isoflux_used.sum()),
        config_hash=_config_hash(cfg),
    )


__all__ = ["run_sweep", "sobol_sample"]
