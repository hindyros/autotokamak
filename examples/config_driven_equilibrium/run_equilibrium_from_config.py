#!/usr/bin/env python
# provenance: URSA-generated, then hand/Claude-refactored to route through autotokamak.core
"""Config-driven TokaMaker fixed-boundary Grad–Shafranov equilibrium.

Reads a YAML config specifying:
- Equation (currently GS)
- Analytic LCFS boundary (isoflux)
- Discretization / mesh spacing
- Solver settings and targets

Workflow (required method sequence):
  create_isoflux -> gs_Domain mesh -> set_targets -> init_psi -> solve -> plot_mesh/plot_psi

Notes:
- No custom profiles: we use TokaMaker's default profiles (unless config adds them later).
- Do not write into OFT install; outputs go to config.outputs.out_dir.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml


class ConfigError(ValueError):
    pass


# ------------------------ minimal validation helpers ------------------------

def _m(x: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(x, Mapping):
        raise ConfigError(f"{path} must be a mapping/object")
    return x


def _req(d: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in d:
        raise ConfigError(f"Missing required key: {path}.{key}")
    return d[key]


def _num(x: Any, path: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ConfigError(f"{path} must be a number")
    return float(x)


def _int(x: Any, path: str) -> int:
    if isinstance(x, bool) or not isinstance(x, int):
        raise ConfigError(f"{path} must be an integer")
    return int(x)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg_text = f.read()
    cfg = yaml.safe_load(cfg_text)
    cfg = _m(cfg, "$")
    # store the original text for provenance copying
    cfg["_config_text"] = cfg_text

    eq = _m(_req(cfg, "equation", "$"), "equation")
    if _req(eq, "name", "equation") != "gs":
        raise ConfigError("equation.name must be 'gs'")

    b = _m(_req(cfg, "boundary", "$"), "boundary")
    if _req(b, "type", "boundary") != "isoflux":
        raise ConfigError("boundary.type must be 'isoflux'")
    _int(_req(b, "npts", "boundary"), "boundary.npts")
    for k in ["r0", "z0", "a", "kappa", "delta"]:
        _num(_req(b, k, "boundary"), f"boundary.{k}")

    mesh = _m(_req(cfg, "mesh", "$"), "mesh")
    regs = _req(mesh, "regions", "mesh")
    if not isinstance(regs, list) or len(regs) == 0:
        raise ConfigError("mesh.regions must be a non-empty list")
    reg0 = _m(regs[0], "mesh.regions[0]")
    if _req(reg0, "type", "mesh.regions[0]") != "plasma":
        raise ConfigError("First mesh region must be type: plasma")
    _num(_req(reg0, "dx", "mesh.regions[0]"), "mesh.regions[0].dx")

    sol = _m(_req(cfg, "solver", "$"), "solver")
    _int(_req(sol, "order", "solver"), "solver.order")
    _num(_req(sol, "F0", "solver"), "solver.F0")

    _m(_req(cfg, "targets", "$"), "targets")

    out = _m(_req(cfg, "outputs", "$"), "outputs")
    _req(out, "out_dir", "outputs")
    _req(out, "mesh_png", "outputs")
    _req(out, "psi_png", "outputs")

    return cfg  # validated enough for this example


def build_mesh_and_boundary(cfg: Dict[str, Any]):
    """Build LCFS + 2D mesh from a config. Thin wrapper around ``autotokamak.core.geometry``."""
    from autotokamak.core.geometry import build_mesh_from_config

    lcfs, gs_mesh, mesh_pts, mesh_lc, mesh_reg = build_mesh_from_config(cfg)
    # Preserve the legacy return ordering: (gs_mesh, lcfs, pts, lc, reg).
    return gs_mesh, lcfs, mesh_pts, mesh_lc, mesh_reg


def setup_and_solve(cfg: Dict[str, Any], mesh_pts, mesh_lc, mesh_reg, lcfs):
    """Set up TokaMaker, seed psi, apply isoflux constraint, solve.

    Thin wrapper around ``autotokamak.core.solver.solve_equilibrium`` — preserves the
    retry-on-isoflux-fail fallback behaviour that this runner had pre-refactor.
    """
    from autotokamak.core.solver import solve_equilibrium

    return solve_equilibrium(
        mesh_pts=mesh_pts,
        mesh_lc=mesh_lc,
        mesh_reg=mesh_reg,
        lcfs=lcfs,
        cfg=cfg,
    )


def _stable_case_slug(cfg: Dict[str, Any]) -> str:
    """Derive a deterministic case slug from key discretization fields."""
    import re

    case_id = str(cfg.get("case_id") or "").strip()
    if case_id:
        slug = case_id
    else:
        dx = float(cfg["mesh"]["regions"][0]["dx"])
        order = int(cfg["solver"]["order"])
        npts = int(cfg["boundary"]["npts"])
        slug = f"dx{dx:g}_p{order}_n{npts}"

    slug = slug.replace(" ", "_")
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", slug)
    return slug


def _hash_config(cfg: Dict[str, Any]) -> str:
    import hashlib

    # stable hash: sort keys and avoid python object noise
    payload = yaml.safe_dump(cfg, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _mesh_stats(mesh_pts, mesh_lc, mesh_reg) -> Dict[str, Any]:
    import numpy as np

    pts = np.asarray(mesh_pts)
    lc = np.asarray(mesh_lc)
    reg = np.asarray(mesh_reg) if mesh_reg is not None else None

    out: Dict[str, Any] = {
        "n_points": int(pts.shape[0]),
        "n_elements": int(lc.shape[0]),
        "element_nnodes": int(lc.shape[1]) if lc.ndim == 2 else None,
        "points_bbox": {
            "r_min": float(np.min(pts[:, 0])),
            "r_max": float(np.max(pts[:, 0])),
            "z_min": float(np.min(pts[:, 1])),
            "z_max": float(np.max(pts[:, 1])),
        },
    }
    if reg is not None and reg.size:
        out["region_ids"] = sorted({int(x) for x in np.unique(reg).tolist()})
    return out


def _to_builtin(x: Any):
    """Convert numpy/scalar containers into YAML/JSON-serializable builtins."""
    import numpy as np

    if x is None:
        return None
    if isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, (list, tuple)):
        return [_to_builtin(v) for v in x]
    # Mapping (includes some dict-like objects)
    if hasattr(x, "items"):
        try:
            return {str(k): _to_builtin(v) for k, v in dict(x).items()}
        except Exception:
            pass
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (np.generic,)):
        return x.item()
    # numpy scalar fallback (covers np.float64 etc. in some builds)
    try:
        import numpy as np

        if isinstance(x, np.generic):
            return x.item()
    except Exception:
        pass

    # fallback
    try:
        if hasattr(x, "tolist"):
            return x.tolist()
    except Exception:
        pass
    return str(x)


def _collect_tokamaker_scalars(gs) -> Dict[str, Any]:
    """Best-effort collection of scalar diagnostics from TokaMaker.

    OFT/TokaMaker diagnostics vary across versions; keep robust.
    """

    scalars: Dict[str, Any] = {}

    # Stats often includes iteration counts/residuals.
    try:
        stats = gs.get_stats()
        scalars["stats"] = _to_builtin(stats)
    except Exception:
        pass

    # Try a few common methods if present.
    for name in [
        "get_Ip",
        "get_axis",
        "get_o_point",
        "get_x_points",
        "get_q_profile",
        "get_q95",
        "get_beta",
        "get_li",
    ]:
        try:
            fn = getattr(gs, name, None)
            if callable(fn):
                scalars[name] = _to_builtin(fn())
        except Exception:
            continue

    return scalars


def save_case_outputs(
    cfg: Dict[str, Any],
    gs_mesh,
    gs,
    mesh_pts,
    mesh_lc,
    mesh_reg,
    lcfs,
) -> Path:
    import json
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = cfg["outputs"]
    root = Path(out["out_dir"]).resolve()
    root.mkdir(parents=True, exist_ok=True)

    case_slug = _stable_case_slug(cfg)
    cfg_hash = _hash_config(cfg)
    case_dir = root / f"{case_slug}_{cfg_hash}"
    case_dir.mkdir(parents=True, exist_ok=True)

    # ---- write summary ----
    summary = {
        "case": {
            "case_id": cfg.get("case_id"),
            "case_slug": case_slug,
            "config_hash": cfg_hash,
        },
        "oft": {
            "module": "OpenFUSIONToolkit",
            "version": getattr(__import__("OpenFUSIONToolkit"), "__version__", None),
        },
        "mesh": _mesh_stats(mesh_pts, mesh_lc, mesh_reg),
        "solver": {
            "free_boundary": bool(cfg["solver"].get("free_boundary", False)),
            "order": int(cfg["solver"]["order"]),
            "F0": float(cfg["solver"]["F0"]),
            "maxits": int(cfg["solver"].get("maxits", -1)),
        },
        "targets": dict(cfg.get("targets", {})),
        "diagnostics": _collect_tokamaker_scalars(gs),
    }

    # Ensure YAML/JSON friendliness
    summary = _to_builtin(summary)

    # Some YAML builds struggle with numpy scalar subclasses; use JSON as ground truth
    # and write YAML from the already-converted JSON-compatible object.
    with open(case_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    try:
        with open(case_dir / "summary.yaml", "w") as f:
            yaml.safe_dump(summary, f, sort_keys=False)
    except Exception as e:
        # last resort: YAML pointer
        with open(case_dir / "summary.yaml", "w") as f:
            f.write(
                "# YAML serialization failed for this environment; see summary.json\n"
                f"# error: {e}\n"
            )

    # ---- save raw arrays (reproducible) ----
    np.savez_compressed(
        case_dir / "raw_arrays.npz",
        mesh_pts=np.asarray(mesh_pts),
        mesh_lc=np.asarray(mesh_lc),
        mesh_reg=np.asarray(mesh_reg) if mesh_reg is not None else np.asarray([]),
        lcfs=np.asarray(lcfs),
    )

    # ---- plots with embedded discretization ----
    mesh_png = f"mesh_{case_slug}_p{cfg['solver']['order']}_dx{cfg['mesh']['regions'][0]['dx']:g}.png"
    psi_png = f"psi_{case_slug}_p{cfg['solver']['order']}_dx{cfg['mesh']['regions'][0]['dx']:g}.png"

    fig, ax = plt.subplots(1, 1, figsize=(5, 5), constrained_layout=True)
    gs_mesh.plot_mesh(fig, ax)
    ax.plot(lcfs[:, 0], lcfs[:, 1], "r-", lw=1.2, label="LCFS (analytic)")
    ax.legend(loc="best", fontsize=8)
    fig.savefig(case_dir / mesh_png, dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5), constrained_layout=True)
    gs.plot_psi(fig, ax)
    # overlay LCFS for reference
    ax.plot(lcfs[:, 0], lcfs[:, 1], "r-", lw=1.2)
    fig.savefig(case_dir / psi_png, dpi=200)
    plt.close(fig)

    # ---- config provenance ----
    cfg_text = cfg.get("_config_text")
    if isinstance(cfg_text, str) and cfg_text.strip():
        (case_dir / "config_used.yaml").write_text(cfg_text, encoding="utf-8")

    return case_dir


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="YAML config file")
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.validate_only:
        print("Config validated OK.")
        return 0

    gs_mesh, lcfs, mesh_pts, mesh_lc, mesh_reg = build_mesh_and_boundary(cfg)
    gs = setup_and_solve(cfg, mesh_pts, mesh_lc, mesh_reg, lcfs)

    case_dir = save_case_outputs(cfg, gs_mesh, gs, mesh_pts, mesh_lc, mesh_reg, lcfs)

    # brief stats
    try:
        stats = gs.get_stats()
        print("Solve stats:", stats)
    except Exception:
        pass

    print(f"Wrote case outputs to: {case_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
