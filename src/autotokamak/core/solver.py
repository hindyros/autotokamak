# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""TokaMaker solver setup + solve, with the retry-on-isoflux-fail fallback.

Both example runners pre-refactor implemented this dance themselves. The logic
is delicate: ``set_isoflux()`` can fail at construction time on some
mesh/shape combinations, so we keep the proven try/retry pattern from
``config_driven_equilibrium/run_equilibrium_from_config.py:setup_and_solve``.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import numpy as np


# OFT only permits one OFT_env per Python kernel. Cache the first one we make
# and reuse it across calls so make_solver / solve_equilibrium can be invoked
# inside a sweep loop without hitting "Only one instance of `OFT_env`...".
_OFT_ENV_CACHE: Any = None

# Status of the most recent solve_equilibrium call. Module-level (not a return
# value) so callers can opt in without forcing every existing caller to unpack
# a tuple. Currently records whether the isoflux constraint was honored or
# whether we fell back to the unconstrained solve.
_LAST_SOLVE_INFO: Dict[str, Any] = {
    "isoflux_used": None,
    "fallback_reason": None,
    "boundary_enforced_by": "none",
}


def get_last_solve_info() -> Dict[str, Any]:
    """Return status of the most recent solve_equilibrium call.

    Keys:
      isoflux_used    : bool | None  -- True if the isoflux-constrained solve
                                        succeeded; False if the constraint was
                                        not applied (fixed-boundary solves, or
                                        a free-boundary fallback); None if no
                                        solve has run yet in this process.
      fallback_reason : str | None   -- the exception message that triggered
                                        a free-boundary fallback, if any. None
                                        for fixed-boundary solves (nothing
                                        failed; the constraint is inapplicable).
      boundary_enforced_by : str     -- "mesh" for fixed-boundary solves (the
                                        LCFS is the mesh boundary and psi=const
                                        there by construction); "isoflux_fit"
                                        when the constrained solve succeeded;
                                        "none" for an unconstrained free-
                                        boundary fallback.
    """
    return dict(_LAST_SOLVE_INFO)


def get_oft_env() -> Any:
    """Return a process-wide OFT_env, creating it on first call."""
    global _OFT_ENV_CACHE
    if _OFT_ENV_CACHE is None:
        import OpenFUSIONToolkit as oft  # imported lazily to avoid hard dep at import time
        _OFT_ENV_CACHE = oft.OFT_env(nthreads=int(os.getenv("OFT_NTHREADS", "2")))
    return _OFT_ENV_CACHE


def _targets_kwargs_from_cfg(cfg: Dict[str, Any]) -> Dict[str, float]:
    """Extract the kwargs to forward to ``TokaMaker.set_targets``."""
    t = cfg.get("targets", {}) or {}
    return {
        key: float(t[key])
        for key in ("Ip", "Ip_ratio", "pax", "estore", "R0", "V0")
        if key in t
    }


def _solver_setup_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    sol = cfg["solver"]
    return {
        "order": int(sol["order"]),
        "F0": float(sol["F0"]),
        "full_domain": bool(sol.get("full_domain", False)),
    }


def _apply_solver_settings(gs: Any, cfg: Dict[str, Any]) -> None:
    sol = cfg["solver"]
    gs.settings.free_boundary = bool(sol.get("free_boundary", False))
    if "maxits" in sol:
        gs.settings.maxits = int(sol["maxits"])


def _seed_psi(gs: Any, cfg: Dict[str, Any]) -> None:
    """Run ``init_psi`` per the config's ``init_psi.method``, with shape-aware fallback.

    For the ``isoflux`` method, OFT's internal isoflux fit during ``init_psi`` can
    fail on extreme shapes -- we catch that specific case and downgrade to the
    uniform-current seed. Any other error (e.g. ``ValueError`` from an unknown
    method) is re-raised so real config bugs surface instead of being silently
    masked by ``init_psi(-1.0)``.
    """
    init = cfg.get("init_psi", {}) or {}
    method = init.get("method", "tokamaker_default")
    if method == "isoflux":
        b = cfg["boundary"]
        try:
            gs.init_psi(
                float(b["r0"]),
                float(b["z0"]),
                float(b["a"]),
                float(b["kappa"]),
                float(b["delta"]),
            )
        except Exception as e:  # noqa: BLE001
            # OFT's isoflux fit inside init_psi can fail for extreme shapes;
            # the uniform-current seed is the documented fallback.
            print(f"WARNING: shape-aware init_psi failed ({e}); using init_psi(-1.0).")
            gs.init_psi(-1.0)
    elif method == "tokamaker_default":
        gs.init_psi()
    else:
        raise ValueError(
            f"init_psi.method must be 'tokamaker_default' or 'isoflux', got {method!r}"
        )


def make_solver(
    *,
    mesh_pts: np.ndarray,
    mesh_lc: np.ndarray,
    mesh_reg: np.ndarray | None,
    cfg: Dict[str, Any],
    env: Any | None = None,
) -> Tuple[Any, Any]:
    """Build a ``TokaMaker`` on an OFT env, load the mesh, apply settings & targets.

    OFT has a hard constraint: only ONE ``OFT_env`` can ever be created per Python
    kernel. So if you need a "fresh" solver (e.g. for a retry), pass the existing
    ``env`` in rather than calling this with ``env=None``.

    Returns
    -------
    (env, gs)
        The OFT environment handle (reuse this for any retries) and the
        configured-but-unsolved TokaMaker instance.
    """
    from OpenFUSIONToolkit.TokaMaker import TokaMaker

    if env is None:
        env = get_oft_env()
    gs = TokaMaker(env)
    gs.setup_mesh(mesh_pts, mesh_lc, reg=mesh_reg)
    _apply_solver_settings(gs, cfg)
    gs.setup(**_solver_setup_kwargs(cfg))
    gs.set_targets(**_targets_kwargs_from_cfg(cfg))
    return env, gs


def solve_equilibrium(
    *,
    mesh_pts: np.ndarray,
    mesh_lc: np.ndarray,
    mesh_reg: np.ndarray | None,
    lcfs: np.ndarray,
    cfg: Dict[str, Any],
) -> Any:
    """End-to-end: create solver → seed psi → (maybe constrain) → solve.

    Fixed-boundary solves (``solver.free_boundary == False``) never apply the
    isoflux constraint: the LCFS *is* the mesh boundary, where psi = const
    holds by construction, so the shape is enforced exactly already. OFT's
    isoflux constraint fitting exists to adjust free-boundary COIL currents;
    on a plasma-only mesh there are no coils, the fitting system is singular
    (LAPACK DGETRF info=1), and ``solve()`` fails unconditionally with
    "Isoflux fitting failed" — for every shape, including reference configs.
    Root-caused 2026-07-10; previously this cost a doomed constrained solve
    plus a full solver rebuild on every dataset sample.

    Free-boundary solves keep the try/fallback: attempt the isoflux-
    constrained solve, and on failure rebuild a fresh ``TokaMaker`` on the
    *same* OFT env (kernel-level singleton) and solve unconstrained, with a
    loud warning.

    Returns the solved TokaMaker instance.
    """
    global _LAST_SOLVE_INFO
    _LAST_SOLVE_INFO = {"isoflux_used": None, "fallback_reason": None,
                        "boundary_enforced_by": "none"}

    free_boundary = bool((cfg.get("solver") or {}).get("free_boundary", False))

    env, gs = make_solver(mesh_pts=mesh_pts, mesh_lc=mesh_lc, mesh_reg=mesh_reg, cfg=cfg)
    _seed_psi(gs, cfg)

    if not free_boundary:
        gs.solve()
        _LAST_SOLVE_INFO = {
            "isoflux_used": False,
            "fallback_reason": None,
            "boundary_enforced_by": "mesh",
        }
        return gs

    try:
        gs.set_isoflux_constraints(np.asarray(lcfs, dtype=float))
        gs.solve()
        _LAST_SOLVE_INFO = {"isoflux_used": True, "fallback_reason": None,
                            "boundary_enforced_by": "isoflux_fit"}
        return gs
    except Exception as e:  # noqa: BLE001
        reason = str(e)
        print(
            f"WARNING: isoflux-constrained solve failed ({reason}). "
            f"Falling back to unconstrained solve."
        )

    # Retry path: reuse the existing OFT env (cannot create a new one in the
    # same kernel) and build a fresh TokaMaker on it without the isoflux constraint.
    _, gs2 = make_solver(mesh_pts=mesh_pts, mesh_lc=mesh_lc, mesh_reg=mesh_reg, cfg=cfg, env=env)
    _seed_psi(gs2, cfg)
    gs2.solve()
    _LAST_SOLVE_INFO = {"isoflux_used": False, "fallback_reason": reason,
                        "boundary_enforced_by": "none"}
    return gs2


__all__ = ["get_oft_env", "get_last_solve_info", "make_solver", "solve_equilibrium"]
