#!/usr/bin/env python
# provenance: URSA-generated (URSA plan->execute agent)
"""Outer loop: tune selected YAML parameters so TokaMaker ψ matches a target.

Target modes:
  - reference: run a forward solve with ``target.reference_overrides`` merged onto
    the base config; ψ from that solve is the target.
  - npz: load ``psi`` from a file produced by ``export-target`` (same mesh/discretization).

Requires a TokaMaker build whose ``TokaMaker`` instance exposes ``get_psi()`` returning a
1D array (optionally wrapped in a length-1 tuple/list of arrays).

Use ``optimize.psi_loss: sorted`` when the mesh changes (e.g. varying ``boundary.*``) so ψ
vectors may have different lengths; default ``dof`` mode requires identical mesh layout.

Usage:
  python run_invert_psi.py export-target BASE_CONFIG.yaml -o psi_target.npz
  python run_invert_psi.py invert invert_psi_example.yaml
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml

from run_equilibrium_from_config import build_mesh_and_boundary, load_config, setup_and_solve


# ---------------------------------------------------------------------------
# config helpers
# ---------------------------------------------------------------------------


def _deep_merge(dst: Dict[str, Any], src: Mapping[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, Mapping) and isinstance(dst.get(k), Mapping):
            _deep_merge(dst[k], v)  # type: ignore[index]
        else:
            dst[k] = copy.deepcopy(v)
    return dst


def _set_path(d: Dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur = d
    for p in parts[:-1]:
        nxt = cur.get(p)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[p] = nxt
        cur = nxt
    cur[parts[-1]] = value


def _resolve_path(base_file: Path, p: str) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = (base_file.parent / path).resolve()
    return path


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: root must be a mapping")
    return data


def _validate_invert_cfg(inv: Mapping[str, Any], inv_path: Path) -> None:
    if "base_config" not in inv:
        raise ValueError(f"{inv_path}: missing base_config")
    if "target" not in inv or not isinstance(inv["target"], dict):
        raise ValueError(f"{inv_path}: missing target mapping")
    if "optimize" not in inv or not isinstance(inv["optimize"], dict):
        raise ValueError(f"{inv_path}: missing optimize mapping")
    t = inv["target"]
    mode = t.get("mode")
    if mode not in ("reference", "npz"):
        raise ValueError(f"{inv_path}: target.mode must be 'reference' or 'npz'")
    if mode == "npz":
        if not t.get("npz_path"):
            raise ValueError(f"{inv_path}: target.npz_path required for mode npz")
    params = inv["optimize"].get("parameters")
    if not isinstance(params, list) or not params:
        raise ValueError(f"{inv_path}: optimize.parameters must be a non-empty list")
    for i, p in enumerate(params):
        if not isinstance(p, dict):
            raise ValueError(f"{inv_path}: optimize.parameters[{i}] must be a mapping")
        for k in ("path", "initial", "min", "max"):
            if k not in p:
                raise ValueError(f"{inv_path}: optimize.parameters[{i}] missing {k}")
    lm = str(inv.get("optimize", {}).get("psi_loss", "dof")).strip().lower()
    if lm not in ("dof", "full", "dofs", "sorted", "order", "quantile"):
        raise ValueError(
            f"{inv_path}: optimize.psi_loss must be 'dof' or 'sorted' (got {lm!r})"
        )
    sb = inv.get("optimize", {}).get("sorted_bins", 512)
    if sb is not None and int(sb) < 16:
        raise ValueError(f"{inv_path}: optimize.sorted_bins must be >= 16")
    lam = float(inv.get("optimize", {}).get("regularization_lambda", 0.0) or 0.0)
    refp = inv.get("optimize", {}).get("reference_parameters")
    if lam > 0.0:
        if not isinstance(refp, dict) or not refp:
            raise ValueError(
                f"{inv_path}: optimize.reference_parameters (non-empty mapping) "
                "is required when regularization_lambda > 0"
            )


# ---------------------------------------------------------------------------
# ψ extraction + loss
# ---------------------------------------------------------------------------


def extract_psi(gs) -> np.ndarray:
    """Return ψ as a 1D float array from a solved TokaMaker instance."""
    getter = getattr(gs, "get_psi", None)
    if not callable(getter):
        raise RuntimeError(
            "TokaMaker.get_psi is not available in this OFT build; "
            "update OpenFUSIONToolkit or use a version that exposes get_psi()."
        )
    raw = getter()
    if isinstance(raw, (list, tuple)) and raw:
        if isinstance(raw[0], np.ndarray):
            raw = raw[0]
    arr = np.asarray(raw, dtype=float).ravel()
    if arr.size == 0:
        raise RuntimeError("get_psi() returned an empty array.")
    return arr


def psi_loss_dof(psi: np.ndarray, psi_ref: np.ndarray) -> float:
    """Per-dof MSE after removing mean ψ; same length as reference (same mesh / FE layout)."""
    a = np.asarray(psi, dtype=float).ravel()
    b = np.asarray(psi_ref, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError(
            f"ψ shape mismatch: candidate {a.shape} vs target {b.shape}. "
            "Use optimize.psi_loss: sorted when mesh size changes (e.g. boundary varies), "
            "or keep the same boundary and discretization."
        )
    a = a - np.mean(a)
    b = b - np.mean(b)
    denom = float(np.sqrt(np.mean(b**2)) + 1.0e-20)
    d = (a - b) / denom
    return float(np.mean(d**2))


def psi_loss_sorted(psi: np.ndarray, psi_ref: np.ndarray, bins: int = 512) -> float:
    """Compare ψ **distributions** (sorted resampling), not spatial identity.

    Use when reference and trial meshes differ (e.g. different ``boundary`` so dof counts
    differ). This is a weaker diagnostic than ``dof`` mode; see README_INVERT_PSI.md.
    """
    a = np.asarray(psi, dtype=float).ravel()
    b = np.asarray(psi_ref, dtype=float).ravel()
    if a.size < 16 or b.size < 16:
        return 1.0e10
    k = int(min(int(bins), a.size, b.size))
    k = max(k, 16)
    sa = np.sort(a)
    sb = np.sort(b)
    ua = np.linspace(0.0, 1.0, sa.size, dtype=float)
    ub = np.linspace(0.0, 1.0, sb.size, dtype=float)
    t = np.linspace(0.0, 1.0, k, dtype=float)
    qa = np.interp(t, ua, sa)
    qb = np.interp(t, ub, sb)
    qa = qa - np.mean(qa)
    qb = qb - np.mean(qb)
    denom = float(np.sqrt(np.mean(qb**2)) + 1.0e-20)
    d = (qa - qb) / denom
    return float(np.mean(d**2))


def psi_loss(
    psi: np.ndarray,
    psi_ref: np.ndarray,
    *,
    mode: str = "dof",
    sorted_bins: int = 512,
) -> float:
    m = mode.strip().lower()
    if m in ("dof", "full", "dofs"):
        return psi_loss_dof(psi, psi_ref)
    if m in ("sorted", "order", "quantile"):
        return psi_loss_sorted(psi, psi_ref, bins=int(sorted_bins))
    raise ValueError(f"Unknown psi_loss mode: {mode!r}")


def _coerce_param(path: str, val: float) -> Any:
    """Integer YAML fields (``npts``, ``order``) must stay ints for ``load_config``."""
    key = path.rsplit(".", 1)[-1]
    if key in ("npts", "order"):
        return int(round(float(val)))
    return float(val)


def _parameter_regularization(
    x: np.ndarray,
    paths: Sequence[str],
    bounds: Sequence[Tuple[float, float]],
    ref_map: Mapping[str, float],
    lam: float,
) -> float:
    """Scaled distance to preferred parameters (Tikhonov). Paths missing from ``ref_map`` are skipped."""
    if lam <= 0.0 or not ref_map:
        return 0.0
    acc = 0.0
    for i, pth in enumerate(paths):
        if pth not in ref_map:
            continue
        lo, hi = float(bounds[i][0]), float(bounds[i][1])
        span = hi - lo
        if span <= 0.0:
            span = 1.0
        xr = float(ref_map[pth])
        acc += ((float(x[i]) - xr) / span) ** 2
    return float(lam * acc)


# ---------------------------------------------------------------------------
# forward evaluation
# ---------------------------------------------------------------------------

_EXAMPLE_DIR = Path(__file__).resolve().parent
_FORWARD_ONCE = _EXAMPLE_DIR / "forward_once.py"
_PLOT_PSI = _EXAMPLE_DIR / "forward_plot_psi.py"


def _run_psi_plot_subprocess(cfg: Dict[str, Any], png: Path, title: str = "") -> None:
    """Full TokaMaker solve in a subprocess, then save ``TokaMaker.plot_psi`` to ``png``."""
    cfg_write = copy.deepcopy(cfg)
    cfg_write.pop("_config_text", None)
    png = Path(png).resolve()
    png.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        cfg_path = td_path / "case.yaml"
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg_write, f, sort_keys=False)
        cmd = [sys.executable, str(_PLOT_PSI), str(cfg_path), str(png)]
        if title:
            cmd += ["--title", title]
        proc = subprocess.run(
            cmd,
            cwd=str(_EXAMPLE_DIR),
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            msg = proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"
            raise RuntimeError(msg)


def _forward_psi_subprocess(cfg: Dict[str, Any]) -> np.ndarray:
    """Run one solve in a fresh interpreter so OFT global state is reset.

    TokaMaker / OFT often cannot initialize a second ``OFT_env`` in the same process;
    the next run may fail with a bogus path-length error (e.g. 'allowable lenght of 0').
    """
    cfg_write = copy.deepcopy(cfg)
    cfg_write.pop("_config_text", None)
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        cfg_path = td_path / "case.yaml"
        out_path = td_path / "out.npz"
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg_write, f, sort_keys=False)
        proc = subprocess.run(
            [sys.executable, str(_FORWARD_ONCE), str(cfg_path), str(out_path)],
            cwd=str(_EXAMPLE_DIR),
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            msg = proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"
            raise RuntimeError(msg)
        data = np.load(out_path)
        return np.asarray(data["psi"], dtype=float).ravel()


def _forward_psi_in_process(cfg: Dict[str, Any]) -> np.ndarray:
    gs_mesh, lcfs, mesh_pts, mesh_lc, mesh_reg = build_mesh_and_boundary(cfg)
    gs = setup_and_solve(cfg, mesh_pts, mesh_lc, mesh_reg, lcfs)
    return extract_psi(gs)


def _forward_psi(cfg: Dict[str, Any]) -> np.ndarray:
    v = os.environ.get("OFT_INVERT_SUBPROCESS", "1").strip().lower()
    if v in ("0", "false", "no", "off"):
        return _forward_psi_in_process(cfg)
    return _forward_psi_subprocess(cfg)


def evaluate_cfg(
    cfg: Dict[str, Any],
    psi_ref: np.ndarray,
    fail_penalty: float,
    *,
    loss_mode: str = "dof",
    sorted_bins: int = 512,
) -> Tuple[float, np.ndarray]:
    try:
        psi = _forward_psi(cfg)
        return (
            psi_loss(psi, psi_ref, mode=loss_mode, sorted_bins=sorted_bins),
            psi,
        )
    except Exception as e:
        print(f"[invert] forward solve failed: {e}", file=sys.stderr)
        return fail_penalty, np.array([])


# ---------------------------------------------------------------------------
# export-target
# ---------------------------------------------------------------------------


def cmd_export_target(args: argparse.Namespace) -> int:
    cfg_path = Path(args.config).resolve()
    cfg = load_config(str(cfg_path))
    if args.overrides:
        ov = _load_yaml(Path(args.overrides).resolve())
        _deep_merge(cfg, ov)
    psi = _forward_psi(cfg)
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        psi=psi,
        n_psi=int(psi.size),
        source_config=str(cfg_path),
    )
    print(f"Wrote ψ vector (n={psi.size}) to {out}")
    return 0


# ---------------------------------------------------------------------------
# invert
# ---------------------------------------------------------------------------


def cmd_invert(args: argparse.Namespace) -> int:
    try:
        from scipy.optimize import minimize
    except ImportError as e:
        print("This command requires scipy (pip install scipy).", file=sys.stderr)
        raise SystemExit(1) from e

    inv_path = Path(args.invert_yaml).resolve()
    inv = _load_yaml(inv_path)
    _validate_invert_cfg(inv, inv_path)

    base_path = _resolve_path(inv_path, str(inv["base_config"]))
    if not base_path.is_file():
        raise SystemExit(f"base_config not found: {base_path}")

    cfg0 = load_config(str(base_path))
    target = inv["target"]
    opt = inv["optimize"]
    out_dir = Path(inv.get("outputs", {}).get("out_dir", "outputs/invert_runs")).resolve()
    if not out_dir.is_absolute():
        out_dir = (inv_path.parent / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    fail_penalty = float(opt.get("fail_penalty", 1.0e12))
    method_raw = str(opt.get("method", "L-BFGS-B")).strip()
    method_u = method_raw.upper().replace(" ", "")
    if method_u in ("L-BFGS-B", "LBFGSB", "L_BFGS_B"):
        method = "L-BFGS-B"
    elif method_u in ("NELDER-MEAD", "NELDERMEAD"):
        method = "Nelder-Mead"
    elif method_u == "POWELL":
        method = "Powell"
    else:
        method = method_raw
    maxiter = int(opt.get("maxiter", 40))
    ftol = float(opt.get("ftol", 1.0e-8))

    loss_mode_raw = str(opt.get("psi_loss", "dof")).strip().lower()
    if loss_mode_raw in ("dof", "full", "dofs"):
        loss_mode = "dof"
    elif loss_mode_raw in ("sorted", "order", "quantile"):
        loss_mode = "sorted"
    else:
        raise SystemExit(f"optimize.psi_loss must be 'dof' or 'sorted', got {loss_mode_raw!r}")
    sorted_bins = int(opt.get("sorted_bins", 512))

    # --- target ψ_ref ---
    cfg_ref: Optional[Dict[str, Any]] = None
    if target["mode"] == "reference":
        cfg_ref = copy.deepcopy(cfg0)
        ro = target.get("reference_overrides") or {}
        if isinstance(ro, dict) and ro:
            _deep_merge(cfg_ref, ro)
        print("[invert] building reference ψ (reference_overrides merged onto base)...")
        psi_ref = _forward_psi(cfg_ref)
    else:
        npz_p = _resolve_path(inv_path, str(target["npz_path"]))
        data = np.load(npz_p)
        if "psi" not in data.files:
            raise SystemExit(f"{npz_p} must contain array 'psi'")
        psi_ref = np.asarray(data["psi"], dtype=float).ravel()
        print(f"[invert] loaded target ψ from {npz_p} (n={psi_ref.size})")

    params: List[Dict[str, Any]] = list(opt["parameters"])
    paths = [str(p["path"]) for p in params]
    x0 = np.array([float(p["initial"]) for p in params], dtype=float)
    bounds = [(float(p["min"]), float(p["max"])) for p in params]

    history: List[Dict[str, Any]] = []

    def pack_cfg(x: np.ndarray) -> Dict[str, Any]:
        cfg = copy.deepcopy(cfg0)
        for val, path in zip(x, paths):
            _set_path(cfg, path, _coerce_param(path, float(val)))
        return cfg

    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)

    lam = float(opt.get("regularization_lambda", 0.0) or 0.0)
    ref_map_raw = opt.get("reference_parameters") or {}
    ref_map: Dict[str, float] = {}
    if isinstance(ref_map_raw, dict):
        ref_map = {str(k): float(v) for k, v in ref_map_raw.items()}

    def objective(x: np.ndarray) -> float:
        x = np.clip(np.asarray(x, dtype=float), lo, hi)
        cfg = pack_cfg(x)
        loss_psi, _psi = evaluate_cfg(
            cfg,
            psi_ref,
            fail_penalty,
            loss_mode=loss_mode,
            sorted_bins=sorted_bins,
        )
        reg = _parameter_regularization(x, paths, bounds, ref_map, lam)
        total = float(loss_psi + reg)
        rec: Dict[str, Any] = {
            "loss": total,
            "loss_psi": float(loss_psi),
            "loss_reg": float(reg),
            "x": x.tolist(),
            "paths": paths,
        }
        for path, v in zip(paths, x):
            rec[path] = float(v)
        history.append(rec)
        if lam > 0.0:
            print(
                f"[invert] obj={total:.6g} (psi={loss_psi:.6g} reg={reg:.6g})  "
                + "  ".join(f"{paths[i]}={x[i]:.6g}" for i in range(len(paths)))
            )
        else:
            print(
                f"[invert] loss={total:.6g}  "
                + "  ".join(f"{paths[i]}={x[i]:.6g}" for i in range(len(paths)))
            )
        return total

    print(f"[invert] optimizing {len(paths)} parameters: {paths}")
    print(
        f"[invert] method={method} maxiter={maxiter} target.n={psi_ref.size} "
        f"psi_loss={loss_mode}" + (f" (bins={sorted_bins})" if loss_mode == "sorted" else "")
    )
    if lam > 0.0:
        print(f"[invert] regularization_lambda={lam} (pulls x toward reference_parameters)")

    if method == "L-BFGS-B":
        res = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter, "ftol": ftol},
        )
    elif method == "Nelder-Mead":
        res = minimize(objective, x0, method="Nelder-Mead", options={"maxiter": maxiter, "xatol": ftol})
    elif method == "Powell":
        res = minimize(objective, x0, method="Powell", options={"maxiter": maxiter, "ftol": ftol})
    else:
        raise SystemExit(f"Unsupported optimize.method: {method_raw!r} (try L-BFGS-B, Powell, Nelder-Mead)")

    x_best = np.asarray(res.x, dtype=float)
    cfg_best = pack_cfg(x_best)
    loss_psi_best, psi_best = evaluate_cfg(
        cfg_best,
        psi_ref,
        fail_penalty,
        loss_mode=loss_mode,
        sorted_bins=sorted_bins,
    )
    reg_best = _parameter_regularization(x_best, paths, bounds, ref_map, lam)
    loss_total_best = float(loss_psi_best + reg_best)

    summary = {
        "invert_yaml": str(inv_path),
        "base_config": str(base_path),
        "target": {"mode": target["mode"]},
        "psi_loss": {"mode": loss_mode, "sorted_bins": sorted_bins},
        "regularization": {
            "lambda": lam,
            "reference_parameters": ref_map,
            "final_psi_loss": float(loss_psi_best),
            "final_reg_term": float(reg_best),
            "final_objective": float(loss_total_best),
        },
        "optimizer": {
            "method": method,
            "success": bool(res.success),
            "message": str(res.message),
            "nit": int(getattr(res, "nit", -1)),
            "nfev": int(getattr(res, "nfev", -1)),
            "final_objective": float(loss_total_best),
            "final_psi_loss": float(loss_psi_best),
        },
        "best_parameters": {paths[i]: float(x_best[i]) for i in range(len(paths))},
        "history_tail": history[-min(50, len(history)) :],
    }

    plot_paths: Dict[str, str] = {}
    out_block = inv.get("outputs") or {}
    do_plots = out_block.get("plot_psi_stages", True)
    if isinstance(do_plots, str):
        do_plots = do_plots.strip().lower() not in ("0", "false", "no", "off")
    if do_plots:
        try:
            if cfg_ref is not None:
                pr = out_dir / "psi_reference.png"
                _run_psi_plot_subprocess(cfg_ref, pr, "Reference (target)")
                plot_paths["psi_reference"] = str(pr)
            x0_clip = np.clip(x0, lo, hi)
            pi = out_dir / "psi_initial.png"
            _run_psi_plot_subprocess(pack_cfg(x0_clip), pi, "Initial guess (first evaluation)")
            plot_paths["psi_initial"] = str(pi)
            pb = out_dir / "psi_best.png"
            _run_psi_plot_subprocess(cfg_best, pb, "After inversion (best)")
            plot_paths["psi_best"] = str(pb)
            for _k, pth in plot_paths.items():
                print(f"  wrote {pth}")
        except Exception as e:
            print(f"[invert] WARNING: ψ stage plots failed: {e}", file=sys.stderr)
    summary["psi_plots"] = plot_paths

    with open(out_dir / "invert_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    best_cfg_path = out_dir / "config_best.yaml"
    merged_best = copy.deepcopy(cfg0)
    for i, path in enumerate(paths):
        _set_path(merged_best, path, _coerce_param(path, float(x_best[i])))
    with open(best_cfg_path, "w", encoding="utf-8") as f:
        f.write(yaml.safe_dump(merged_best, sort_keys=False))

    np.savez_compressed(
        out_dir / "invert_result.npz",
        psi_best=psi_best,
        psi_ref=psi_ref,
        x_best=x_best,
    )

    print(
        f"[invert] done. final_obj={loss_total_best:.6g} final_psi_loss={loss_psi_best:.6g} "
        f"success={res.success}"
    )
    print(f"  wrote {out_dir / 'invert_summary.json'}")
    print(f"  wrote {best_cfg_path}")
    print(f"  wrote {out_dir / 'invert_result.npz'}")
    if not np.isfinite(loss_total_best) or loss_total_best >= fail_penalty * 0.5:
        return 1
    return 0 if bool(res.success) else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p_exp = sub.add_parser("export-target", help="Run one forward case and save ψ to NPZ.")
    p_exp.add_argument("config", help="Base equilibrium YAML (same schema as discretization_config.yaml)")
    p_exp.add_argument("-o", "--output", required=True, help="Output .npz path (contains 'psi')")
    p_exp.add_argument(
        "--overrides",
        default=None,
        help="Optional YAML merged onto the base config before the solve.",
    )
    p_exp.set_defaults(_run=cmd_export_target)

    p_inv = sub.add_parser("invert", help="Optimize parameters to match target ψ.")
    p_inv.add_argument("invert_yaml", help="Inversion problem YAML")
    p_inv.set_defaults(_run=cmd_invert)

    args = ap.parse_args(argv)
    fn: Callable[[argparse.Namespace], int] = args._run
    return int(fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
