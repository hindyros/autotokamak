#!/usr/bin/env python3
# provenance: URSA-generated (URSA plan->execute agent)
"""OpenFUSIONToolkit / TokaMaker: fixed-boundary equilibrium example (standalone).

This script mirrors the workflow in:
  OpenFUSIONToolkit/src/examples/TokaMaker/fixed_boundary/fixed_boundary_ex1.ipynb

It builds a fixed-boundary Grad-Shafranov equilibrium using TokaMaker.
Two cases are supported:
  1) "analytic"  : LCFS contour from create_isoflux (no external data)
  2) "eqdsk"     : use the bundled gNT_example EQDSK file via read_eqdsk

Outputs are written into an output directory (default: outputs/fixed_boundary_ex1/<run_id>/)
with NPZ data products and (optionally) PNG plots.

Constraints honored:
- Does not write into ./OpenFusionToolkit or ./ursa (treated as read-only)
- Does not modify Python environment
- Headless plotting supported (matplotlib Agg)

This file is intentionally verbose: it prints progress messages and also writes a
run.log file into the output directory.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a fixed-boundary equilibrium with OpenFUSIONToolkit TokaMaker and save outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--case",
        choices=["analytic", "eqdsk"],
        default="analytic",
        help="Which fixed-boundary setup to run.",
    )
    p.add_argument(
        "--outdir",
        default="outputs/fixed_boundary_ex1",
        help="Base output directory (a run_id subdir will be created).",
    )
    p.add_argument(
        "--run-id",
        default=None,
        help="Run identifier subdirectory. Default: UTC timestamp.",
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable writing PNG plots (still writes NPZ + JSON).",
    )
    p.add_argument(
        "--nthreads",
        type=int,
        default=2,
        help="Number of OpenFUSIONToolkit threads.",
    )
    p.add_argument(
        "--mesh-dx",
        type=float,
        default=None,
        help="Override mesh spacing. If not provided, uses notebook defaults for each case.",
    )
    p.add_argument(
        "--maxits",
        type=int,
        default=None,
        help="Override Newton iterations (settings.maxits).",
    )

    # Execution milestone helpers
    p.add_argument(
        "--imports-only",
        action="store_true",
        help="Exit after importing OpenFUSIONToolkit (no geometry/mesh/solve).",
    )
    p.add_argument(
        "--setup-only",
        action="store_true",
        help="Build LCFS + mesh, then exit (no solve / no outputs other than mesh/lcfs).",
    )
    p.add_argument(
        "--smoke-solve",
        action="store_true",
        help="Run a very small-number-iteration solve for a fast smoke test (overrides maxits).",
    )
    return p.parse_args()


def _setup_headless_matplotlib(no_plots: bool) -> None:
    """Force non-interactive backend before importing pyplot."""
    if no_plots:
        return
    import matplotlib

    matplotlib.use("Agg")


def _timestamp_run_id() -> str:
    return _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# Captured at module import — used to compute elapsed time for human-friendly console logs.
_RUN_START_TIME = _dt.datetime.utcnow()

# OFT's compiled code (Fortran/C) writes to libc stdio, which buffers independently of
# Python's sys.stdout. Without explicit fflush(NULL), OFT's banner/iteration output
# dumps to the terminal AFTER all our Python prints, which makes the log impossible to read.
import ctypes as _ctypes
try:
    _LIBC = _ctypes.CDLL(None)
except Exception:
    _LIBC = None


def _flush_all() -> None:
    """Flush Python's stdout and (if available) libc stdio so OFT's compiled output
    appears at the right place in the terminal stream."""
    sys.stdout.flush()
    if _LIBC is not None:
        try:
            _LIBC.fflush(None)  # NULL = flush every open stream
        except Exception:
            pass


def _elapsed() -> str:
    """MM:SS.xx since module load, for compact console stamps."""
    delta = (_dt.datetime.utcnow() - _RUN_START_TIME).total_seconds()
    mm = int(delta // 60)
    ss = delta - mm * 60
    return f"{mm:02d}:{ss:05.2f}"


def _log(msg: str, log_fp) -> None:
    """Log one step. Console gets a short elapsed-time stamp; file gets the full ISO timestamp."""
    iso_line = f"[{_dt.datetime.utcnow().isoformat()}Z] {msg}"
    _flush_all()
    print(f"  [{_elapsed()}]  {msg}", flush=True)
    if log_fp is not None:
        log_fp.write(iso_line + "\n")
        log_fp.flush()


def _section(title: str, log_fp) -> None:
    """Print a section header (console + log) to group related steps visually."""
    bar = "-" * 70
    _flush_all()
    print(f"\n{bar}\n  {title}\n{bar}", flush=True)
    _flush_all()
    if log_fp is not None:
        log_fp.write(f"\n[{_dt.datetime.utcnow().isoformat()}Z] === {title} ===\n")
        log_fp.flush()


def _kv(label: str, value, log_fp, *, width: int = 14) -> None:
    """Print a 'label: value' line aligned for readability."""
    print(f"      {label:<{width}} {value}", flush=True)
    if log_fp is not None:
        log_fp.write(f"[{_dt.datetime.utcnow().isoformat()}Z]   {label}: {value}\n")
        log_fp.flush()


def _atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Write a text file atomically (best-effort).

    Writes into the same directory and then replaces the destination.
    """
    _mkdir(path.parent)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding=encoding) as tf:
        tf.write(text)
        tf.flush()
        os.fsync(tf.fileno())
        tmp_name = tf.name
    os.replace(tmp_name, path)


def _atomic_savez(path: Path, **arrays) -> None:
    """Write an .npz atomically (best-effort)."""
    _mkdir(path.parent)
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=str(path.parent)) as tf:
        tmp_name = tf.name
    try:
        np.savez(tmp_name, **arrays)
        # numpy appends .npz if not present; handle both possibilities
        tmp_npz = Path(tmp_name)
        if not str(tmp_npz).endswith(".npz") and Path(str(tmp_npz) + ".npz").exists():
            tmp_npz = Path(str(tmp_npz) + ".npz")
        os.replace(tmp_npz, path)
    finally:
        # Clean up if something went wrong
        for cand in [Path(tmp_name), Path(str(tmp_name) + ".npz")]:
            if cand.exists() and cand != path:
                try:
                    cand.unlink()
                except Exception:
                    pass


def _assert_nonempty_file(path: Path, *, min_bytes: int = 16) -> None:
    st = path.stat()
    if st.st_size < min_bytes:
        raise RuntimeError(f"Output file appears empty/corrupt (size={st.st_size} bytes): {path}")


def probe_versions(log_fp) -> dict:
    """Collect lightweight environment/version info for reproducibility."""
    info: dict = {
        "utc_start": _dt.datetime.utcnow().isoformat() + "Z",
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "executable": sys.executable,
        "cwd": str(Path.cwd()),
    }
    try:
        import OpenFUSIONToolkit as oft

        info["OpenFUSIONToolkit.__file__"] = getattr(oft, "__file__", None)
        info["OpenFUSIONToolkit.__version__"] = getattr(oft, "__version__", None)
    except Exception as e:
        info["OpenFUSIONToolkit_probe_error"] = f"{type(e).__name__}: {e}"

    _kv("Python:",   info["python"].split()[0], log_fp)
    _kv("Platform:", info["platform"], log_fp)
    _kv("OFT ver:",  info.get("OpenFUSIONToolkit.__version__", "?"), log_fp)
    _kv("OFT path:", info.get("OpenFUSIONToolkit.__file__", "?"), log_fp)
    if log_fp is not None:
        log_fp.write(f"[{_dt.datetime.utcnow().isoformat()}Z] Full environment probe: {info}\n")
        log_fp.flush()
    return info


def build_case(case: str, mesh_dx_override: float | None, log_fp):
    """Return (LCFS_contour, mesh_dx, EQ_in_or_None, toka_setup_kwargs, targets_kwargs, profiles_kwargs_or_None)."""

    from OpenFUSIONToolkit.TokaMaker.util import create_isoflux, read_eqdsk

    if case == "analytic":
        mesh_dx = 0.015 if mesh_dx_override is None else float(mesh_dx_override)
        _log("Building analytic D-shaped boundary via create_isoflux", log_fp)
        _kv("R0 (major):",  "0.42 m",  log_fp)
        _kv("a  (minor):",  "0.15 m",  log_fp)
        _kv("kappa:",       "1.40",    log_fp)
        _kv("delta:",       "0.40",    log_fp)
        _kv("n_pts:",       "80",      log_fp)
        _kv("mesh_dx:",     f"{mesh_dx} m", log_fp)
        # Matches notebook: create_isoflux(80,0.42,0.0,0.15,1.4,0.4)
        LCFS_contour = create_isoflux(80, 0.42, 0.0, 0.15, 1.4, 0.4)

        EQ_in = None
        toka_setup_kwargs = dict(order=2, F0=0.10752)
        Ip_target = 120e3
        Beta_target = 0.5
        targets_kwargs = dict(Ip=Ip_target, Ip_ratio=(1.0 / Beta_target - 1.0))
        profiles_kwargs = None
        return LCFS_contour, mesh_dx, EQ_in, toka_setup_kwargs, targets_kwargs, profiles_kwargs

    if case == "eqdsk":
        # Resolve EQDSK path relative to the OpenFUSIONToolkit source tree.
        eqdsk_default = (
            Path(__file__).resolve().parent
            / ".."
            / "OpenFusionToolkit"
            / "src"
            / "examples"
            / "TokaMaker"
            / "fixed_boundary"
            / "gNT_example"
        )
        eqdsk_path = eqdsk_default.resolve()
        if not eqdsk_path.exists():
            eqdsk_path = Path("gNT_example").resolve()

        _log(f"Loading EQDSK input via read_eqdsk('{eqdsk_path}')", log_fp)
        EQ_in = read_eqdsk(str(eqdsk_path))
        LCFS_contour = EQ_in["rzout"]

        mesh_dx = 0.15 if mesh_dx_override is None else float(mesh_dx_override)
        _log(f"Using LCFS contour from EQDSK. mesh_dx={mesh_dx}", log_fp)

        toka_setup_kwargs = dict(order=2, F0=EQ_in["rcentr"] * EQ_in["bcentr"])
        targets_kwargs = dict(Ip=EQ_in["ip"], pax=EQ_in["pres"][0])

        # Build linterp profiles from EQDSK samples and normalize them (notebook-like).
        psi_eqdsk = np.linspace(0.0, 1.0, int(EQ_in["nr"]))
        psi_sample = np.linspace(0.025, 1.0, 10)
        psi_prof = psi_sample.copy()
        psi_prof[0] = 0.0

        ffp_y = np.interp(psi_sample, psi_eqdsk, EQ_in["ffprim"]).astype(float)
        pp_y = np.interp(psi_sample, psi_eqdsk, EQ_in["pprime"]).astype(float)

        # Normalize and force edge to 0
        if ffp_y[0] != 0:
            ffp_y /= ffp_y[0]
        ffp_y[-1] = 0.0
        if pp_y[0] != 0:
            pp_y /= pp_y[0]
        pp_y[-1] = 0.0

        profiles_kwargs = dict(
            ffp_prof={"type": "linterp", "x": psi_prof, "y": ffp_y},
            pp_prof={"type": "linterp", "x": psi_prof, "y": pp_y},
            __note__="Notebook-normalized linterp profiles (may be skipped if solve fails)",
        )
        return LCFS_contour, mesh_dx, EQ_in, toka_setup_kwargs, targets_kwargs, profiles_kwargs

    raise ValueError(f"Unknown case: {case}")


def build_mesh_from_lcfs(LCFS_contour: np.ndarray, mesh_dx: float, log_fp):
    from OpenFUSIONToolkit.TokaMaker.meshing import gs_Domain

    _log("Triangulating LCFS polygon via gs_Domain", log_fp)
    gs_mesh = gs_Domain()
    gs_mesh.define_region("plasma", mesh_dx, "plasma")
    gs_mesh.add_polygon(LCFS_contour, "plasma")
    mesh_pts, mesh_lc, mesh_reg = gs_mesh.build_mesh()
    _kv("Nodes:",     f"{mesh_pts.shape[0]:,}",   log_fp)
    _kv("Triangles:", f"{mesh_lc.shape[0]:,}",    log_fp)
    return gs_mesh, mesh_pts, mesh_lc, mesh_reg


def run_solver(
    *,
    nthreads: int,
    mesh_pts: np.ndarray,
    mesh_lc: np.ndarray,
    toka_setup_kwargs: dict,
    targets_kwargs: dict,
    profiles_kwargs: dict | None,
    maxits_override: int | None,
    log_fp,
):
    from OpenFUSIONToolkit import OFT_env
    from OpenFUSIONToolkit.TokaMaker import TokaMaker

    _log(f"Initializing OFT_env (threads={nthreads})", log_fp)
    myOFT = OFT_env(nthreads=int(nthreads))
    mygs = TokaMaker(myOFT)

    _log("Loading mesh into TokaMaker", log_fp)
    mygs.setup_mesh(mesh_pts, mesh_lc)

    _log("Configuring solver", log_fp)
    mygs.settings.free_boundary = False
    _kv("Mode:",     "fixed-boundary", log_fp)
    _kv("FE order:", f"{toka_setup_kwargs.get('order', '?')} (quadratic)" if toka_setup_kwargs.get("order") == 2 else str(toka_setup_kwargs.get("order")), log_fp)
    _kv("F0:",       toka_setup_kwargs.get("F0"), log_fp)

    if maxits_override is not None:
        _kv("maxits:", int(maxits_override), log_fp)
        mygs.settings.maxits = int(maxits_override)

    mygs.setup(**toka_setup_kwargs)

    _log("Applying equilibrium targets", log_fp)
    for k, v in targets_kwargs.items():
        if k == "Ip" and isinstance(v, (int, float)):
            _kv(f"{k}:", f"{v / 1e3:.1f} kA", log_fp)
        else:
            _kv(f"{k}:", v, log_fp)
    mygs.set_targets(**targets_kwargs)

    profiles_applied = False
    if profiles_kwargs is not None:
        clean_profiles = {k: v for k, v in profiles_kwargs.items() if not str(k).startswith("__")}
        _log(f"Setting custom profiles (keys={list(clean_profiles.keys())})", log_fp)
        mygs.set_profiles(**clean_profiles)
        profiles_applied = True

    _log("Seeding initial psi", log_fp)
    err_flag = mygs.init_psi()
    if err_flag is not None:
        _kv("init_psi:", err_flag, log_fp)

    _log("Running nonlinear solve (Grad-Shafranov)", log_fp)
    try:
        mygs.solve()
    except Exception as e:
        msg = str(e)

        # For smoke tests we may deliberately exceed maxits; treat as non-fatal.
        if maxits_override is not None and ("Exceeded \"maxits\"" in msg or "Exceeded 'maxits'" in msg):
            _log(
                f"WARNING: Solver stopped due to maxits={maxits_override} (treating as expected for smoke test).",
                log_fp,
            )
        elif profiles_applied:
            _log(f"Solve failed after applying custom profiles ({type(e).__name__}: {e}).", log_fp)
            _log("Retrying once with default profiles (no set_profiles)...", log_fp)

            mygs.reset()
            mygs.setup_mesh(mesh_pts, mesh_lc)
            mygs.settings.free_boundary = False
            if maxits_override is not None:
                mygs.settings.maxits = int(maxits_override)
            mygs.setup(**toka_setup_kwargs)
            mygs.set_targets(**targets_kwargs)
            mygs.init_psi()
            mygs.solve()
            profiles_applied = False
        else:
            raise

    try:
        mygs._oft_example_meta = {"profiles_applied": profiles_applied}
    except Exception:
        pass

    return mygs


def extract_key_scalars(*, case: str, LCFS_contour: np.ndarray, EQ_in, mygs, log_fp) -> dict:
    """Compute a handful of scalars useful for validating a run.

    This is intentionally defensive: it only uses public methods/arrays and
    catches errors rather than failing the run.
    """
    scalars: dict = {
        "case": case,
        "boundary_npts": int(getattr(LCFS_contour, "shape", [0])[0]),
    }

    if EQ_in is not None:
        for k in ["ip", "bcentr", "rcentr", "nr", "nz"]:
            if k in EQ_in:
                try:
                    scalars[f"EQ_in.{k}"] = float(EQ_in[k])
                except Exception:
                    scalars[f"EQ_in.{k}"] = str(EQ_in[k])

    # Axis location: best-effort (TokaMaker APIs vary by build).
    for cand in ["mag_axis", "axis", "R0Z0", "o_point"]:
        if hasattr(mygs, cand):
            try:
                v = getattr(mygs, cand)
                if callable(v):
                    v = v()
                scalars["axis_candidate"] = cand
                scalars["axis_value"] = np.asarray(v).tolist()
                break
            except Exception:
                pass

    # q-profile summary
    try:
        psi_q, qvals, ravgs, dl, rbounds, zbounds = mygs.get_q(psi_pad=0.005)
        scalars["q.psi_min"] = float(np.nanmin(psi_q))
        scalars["q.psi_max"] = float(np.nanmax(psi_q))
        scalars["q.q0"] = float(qvals[0])
        scalars["q.q95"] = float(np.interp(0.95, psi_q, qvals))
        scalars["q.q_edge"] = float(qvals[-1])
    except Exception as e:
        _log(f"WARNING: failed to compute q summary scalars: {type(e).__name__}: {e}", log_fp)

    # Pressure/beta proxy: max(p)
    try:
        psi, f, fp, p, pp = mygs.get_profiles()
        scalars["p.max"] = float(np.nanmax(p))
        scalars["p.edge"] = float(p[-1])
    except Exception as e:
        _log(f"WARNING: failed to compute pressure scalars: {type(e).__name__}: {e}", log_fp)

    return scalars


def _json_sanitize(obj):
    """Convert numpy/scalar-ish objects to JSON-serializable Python types."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    # last resort
    return str(obj)


def write_outputs(
    *,
    out_run_dir: Path,
    case: str,
    run_id: str,
    args_dict: dict,
    env_info: dict,
    LCFS_contour: np.ndarray,
    mesh_pts: np.ndarray,
    mesh_lc: np.ndarray,
    mesh_reg: np.ndarray,
    EQ_in,
    mygs,
    mesh_dx: float,
    toka_setup_kwargs: dict,
    targets_kwargs: dict,
    profiles_kwargs: dict | None,
    log_fp,
) -> dict:
    _log(f"Writing artifacts to {out_run_dir}", log_fp)

    # LCFS contour and mesh
    _atomic_savez(out_run_dir / "lcfs_contour.npz", LCFS_contour=LCFS_contour)
    _atomic_savez(out_run_dir / "mesh.npz", mesh_pts=mesh_pts, mesh_lc=mesh_lc, mesh_reg=mesh_reg)

    _assert_nonempty_file(out_run_dir / "lcfs_contour.npz")
    _assert_nonempty_file(out_run_dir / "mesh.npz")

    # Profiles + q
    _log("Extracting profiles (psi, F, p, F', p')", log_fp)
    psi, f, fp, p, pp = mygs.get_profiles()
    _atomic_savez(out_run_dir / "profiles.npz", psi=psi, f=f, fp=fp, p=p, pp=pp)
    _assert_nonempty_file(out_run_dir / "profiles.npz")

    _log("Extracting safety factor q and flux-surface geometry", log_fp)
    psi_q, qvals, ravgs, dl, rbounds, zbounds = mygs.get_q(psi_pad=0.005)
    _atomic_savez(
        out_run_dir / "q_and_geometry.npz",
        psi_q=psi_q,
        qvals=qvals,
        ravgs=ravgs,
        dl=dl,
        rbounds=rbounds,
        zbounds=zbounds,
    )
    _assert_nonempty_file(out_run_dir / "q_and_geometry.npz")

    # Machine-readable summary
    summary = {
        "case": case,
        "run_id": run_id,
        "out_run_dir": str(out_run_dir),
        "args": args_dict,
        "environment": env_info,
        "parameters": {
            "mesh_dx": float(mesh_dx),
            "toka_setup_kwargs": toka_setup_kwargs,
            "targets_kwargs": targets_kwargs,
            "profiles_kwargs": profiles_kwargs,
        },
        "artifacts": {
            "lcfs_contour": "lcfs_contour.npz",
            "mesh": "mesh.npz",
            "profiles": "profiles.npz",
            "q_and_geometry": "q_and_geometry.npz",
            "run_log": "run.log",
        },
        "solver_meta": getattr(mygs, "_oft_example_meta", {}),
        "deviations_from_notebook": [],
    }

    if (
        isinstance(summary.get("solver_meta"), dict)
        and summary["solver_meta"].get("profiles_applied") is False
        and case == "eqdsk"
    ):
        summary["deviations_from_notebook"].append(
            "EQDSK case: solve retried without custom profiles due to target-matrix failure."
        )

    summary["scalars"] = extract_key_scalars(case=case, LCFS_contour=LCFS_contour, EQ_in=EQ_in, mygs=mygs, log_fp=log_fp)

    _atomic_write_text(
        out_run_dir / "summary.json",
        json.dumps(_json_sanitize(summary), indent=2) + "\n",
    )
    _assert_nonempty_file(out_run_dir / "summary.json")
    _log("Wrote summary.json", log_fp)

    _log("Saved artifacts:", log_fp)
    _kv("lcfs_contour:",    "lcfs_contour.npz", log_fp)
    _kv("mesh:",            "mesh.npz",         log_fp)
    _kv("profiles:",        "profiles.npz",     log_fp)
    _kv("q + geometry:",    "q_and_geometry.npz", log_fp)
    _kv("summary:",         "summary.json",     log_fp)

    return summary


def make_plots(
    *,
    out_run_dir: Path,
    gs_mesh,
    mygs,
    EQ_in,
    log_fp,
) -> None:
    import matplotlib.pyplot as plt
    from OpenFUSIONToolkit.util import mu0

    plots_dir = out_run_dir / "plots"
    _mkdir(plots_dir)

    _log("Plotting triangular mesh -> plots/mesh.png", log_fp)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), constrained_layout=True)
    gs_mesh.plot_mesh(fig, ax)
    fig.savefig(plots_dir / "mesh.png", dpi=200)
    plt.close(fig)

    _log("Plotting flux surfaces psi(R,Z) -> plots/psi.png", log_fp)
    fig, ax = plt.subplots(1, 1, figsize=(6, 5), constrained_layout=True)
    mygs.plot_psi(fig, ax)
    fig.savefig(plots_dir / "psi.png", dpi=200)
    plt.close(fig)

    if EQ_in is not None:
        _log("Creating EQDSK comparison plots (profiles + q + <Jphi>)", log_fp)
        psi, f, fp, p, pp = mygs.get_profiles()
        psi_q, qvals, ravgs, dl, rbounds, zbounds = mygs.get_q(psi_pad=0.005)

        fig, ax = plt.subplots(4, 1, sharex=True, figsize=(7, 9), constrained_layout=True)
        psi_eqdsk = np.linspace(0.0, 1.0, int(EQ_in["nr"]))

        ax[0].plot(psi, f * fp, label="TokaMaker")
        ax[0].plot(psi_eqdsk, -EQ_in["ffprim"], "--", label="EQDSK")
        ax[0].set_ylim(bottom=-10)
        ax[0].set_ylabel("FF'")
        ax[0].legend()

        ax[1].plot(psi, pp)
        ax[1].plot(psi_eqdsk, -EQ_in["pprime"], "--")
        ax[1].set_ylim(top=2e6)
        ax[1].set_ylabel("P'")

        ax[2].plot(psi_q, qvals)
        ax[2].plot(psi_eqdsk, EQ_in["qpsi"], "--")
        ax[2].set_ylabel("q")

        jphi = np.interp(psi_q, psi, pp) * ravgs[0, :] * mu0 + np.interp(psi_q, psi, f * fp) * ravgs[1, :]
        ax[3].plot(psi_q, jphi)
        ax[3].set_ylabel(r"< $J_{\phi}$ >")
        ax[3].set_xlabel(r"$\hat{\psi}$")

        fig.savefig(plots_dir / "eqdsk_comparison.png", dpi=200)
        plt.close(fig)


def main() -> int:
    args = parse_args()

    run_id = args.run_id or _timestamp_run_id()
    out_base = Path(args.outdir)
    out_run_dir = out_base / run_id
    _mkdir(out_run_dir)

    log_path = out_run_dir / "run.log"
    with log_path.open("w", encoding="utf-8") as log_fp:
        try:
            _section(f"STARTUP  Fixed-boundary equilibrium (case: {args.case})", log_fp)
            _kv("Output dir:", out_run_dir, log_fp)
            _kv("Run id:",     run_id,      log_fp)
            _kv("Threads:",    args.nthreads, log_fp)
            if args.imports_only:  _kv("Milestone:", "--imports-only", log_fp)
            if args.setup_only:    _kv("Milestone:", "--setup-only",   log_fp)
            if args.smoke_solve:   _kv("Milestone:", "--smoke-solve",  log_fp)

            env_info = probe_versions(log_fp)

            # Milestone 1: imports-only
            if args.imports_only:
                _section("DONE  --imports-only: probe completed, exiting", log_fp)
                return 0

            _setup_headless_matplotlib(args.no_plots)

            _section("PHASE 1  Build plasma boundary (LCFS)", log_fp)
            LCFS_contour, mesh_dx, EQ_in, toka_setup_kwargs, targets_kwargs, profiles_kwargs = build_case(
                args.case, args.mesh_dx, log_fp
            )

            _section("PHASE 2  Triangulate the plasma region", log_fp)
            gs_mesh, mesh_pts, mesh_lc, mesh_reg = build_mesh_from_lcfs(LCFS_contour, mesh_dx, log_fp)

            # Milestone 2: setup-only
            if args.setup_only:
                _section("DONE  --setup-only: writing LCFS + mesh and exiting", log_fp)
                _atomic_savez(out_run_dir / "lcfs_contour.npz", LCFS_contour=LCFS_contour)
                _atomic_savez(out_run_dir / "mesh.npz", mesh_pts=mesh_pts, mesh_lc=mesh_lc, mesh_reg=mesh_reg)
                _assert_nonempty_file(out_run_dir / "lcfs_contour.npz")
                _assert_nonempty_file(out_run_dir / "mesh.npz")
                _atomic_write_text(
                    out_run_dir / "summary.json",
                    json.dumps(
                        _json_sanitize(
                            {
                                "case": args.case,
                                "run_id": run_id,
                                "out_run_dir": str(out_run_dir),
                                "args": vars(args),
                                "environment": env_info,
                                "parameters": {
                                    "mesh_dx": float(mesh_dx),
                                    "toka_setup_kwargs": toka_setup_kwargs,
                                    "targets_kwargs": targets_kwargs,
                                    "profiles_kwargs": profiles_kwargs,
                                },
                                "artifacts": {
                                    "lcfs_contour": "lcfs_contour.npz",
                                    "mesh": "mesh.npz",
                                    "run_log": "run.log",
                                },
                                "note": "setup-only run: no solver executed",
                            }
                        ),
                        indent=2,
                    )
                    + "\n",
                )
                return 0

            # Milestone 3: smoke solve
            maxits_override = args.maxits
            if args.smoke_solve:
                maxits_override = 2

            _section("PHASE 3  Configure and solve Grad-Shafranov", log_fp)
            if args.smoke_solve:
                _kv("Smoke mode:", "maxits=2 (will not converge)", log_fp)

            # Run solver
            mygs = run_solver(
                nthreads=args.nthreads,
                mesh_pts=mesh_pts,
                mesh_lc=mesh_lc,
                toka_setup_kwargs=toka_setup_kwargs,
                targets_kwargs=targets_kwargs,
                profiles_kwargs=profiles_kwargs,
                maxits_override=maxits_override,
                log_fp=log_fp,
            )

            # Print some solver info
            if hasattr(mygs, "print_info"):
                _section("PHASE 4  Equilibrium statistics (from TokaMaker)", log_fp)
                mygs.print_info()

            _section("PHASE 5  Save outputs to disk", log_fp)
            # Write outputs
            summary = write_outputs(
                out_run_dir=out_run_dir,
                case=args.case,
                run_id=run_id,
                args_dict=vars(args),
                env_info=env_info,
                LCFS_contour=LCFS_contour,
                mesh_pts=mesh_pts,
                mesh_lc=mesh_lc,
                mesh_reg=mesh_reg,
                EQ_in=EQ_in,
                mygs=mygs,
                mesh_dx=mesh_dx,
                toka_setup_kwargs=toka_setup_kwargs,
                targets_kwargs=targets_kwargs,
                profiles_kwargs=profiles_kwargs,
                log_fp=log_fp,
            )

            # Plots
            if not args.no_plots:
                _section("PHASE 6  Generate diagnostic plots", log_fp)
                make_plots(out_run_dir=out_run_dir, gs_mesh=gs_mesh, mygs=mygs, EQ_in=EQ_in, log_fp=log_fp)
                summary["artifacts"]["plots_dir"] = "plots/"
                _atomic_write_text(out_run_dir / "summary.json", json.dumps(_json_sanitize(summary), indent=2) + "\n")

            _section(f"DONE  Total elapsed: {_elapsed()}", log_fp)
            _kv("Outputs:", out_run_dir, log_fp)
            print()
            return 0

        except Exception as e:
            _section("FAILED  Run aborted due to exception", log_fp)
            _log(str(e), log_fp)
            tb = traceback.format_exc()
            _log(tb, log_fp)
            return 2


if __name__ == "__main__":
    raise SystemExit(main())
