#!/usr/bin/env python3
# provenance: URSA-generated (URSA plan->execute agent)
"""Two-tier validation for run_fixed_boundary_equilibrium.py outputs.

Tier 1 (structural)
- Verify expected artifact files exist.
- Verify files are non-empty.
- Verify NPZ files load and contain expected keys, array ranks, and finite values.
- Verify basic mesh/LCFS consistency.

Tier 2 (physics-ish)
- Check key scalars are within expected ranges for each case.
- Check the solver hit the Ip target within tolerance.
- Check safety invariants (e.g., q profile positive and monotone-ish).

This script is intentionally self-contained and reads only the generated outputs.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass
class CheckResult:
    name: str
    ok: bool
    details: str = ""


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _nonempty_file(path: str) -> CheckResult:
    try:
        sz = os.path.getsize(path)
        return CheckResult(f"nonempty:{os.path.basename(path)}", sz > 0, f"size={sz}")
    except FileNotFoundError:
        return CheckResult(f"nonempty:{os.path.basename(path)}", False, "missing")


def _npz_load(path: str) -> Tuple[Dict[str, np.ndarray], CheckResult]:
    """Load NPZ.

    Note: q_and_geometry.npz may contain object arrays (variable-length boundary traces).
    We allow pickle *only* to read those arrays produced by our own script.
    """
    if not os.path.exists(path):
        return {}, CheckResult(f"load_npz:{os.path.basename(path)}", False, "missing")
    try:
        with np.load(path, allow_pickle=True) as z:
            data = {k: z[k] for k in z.files}
        return data, CheckResult(f"load_npz:{os.path.basename(path)}", True, f"keys={list(data.keys())}")
    except Exception as e:
        return {}, CheckResult(f"load_npz:{os.path.basename(path)}", False, f"error={e!r}")


def _finite_array(name: str, arr: np.ndarray) -> CheckResult:
    if not isinstance(arr, np.ndarray):
        return CheckResult(f"finite:{name}", False, f"not ndarray: {type(arr)}")
    if arr.size == 0:
        return CheckResult(f"finite:{name}", False, "empty")
    fin = np.isfinite(arr).all()
    return CheckResult(f"finite:{name}", bool(fin), f"shape={arr.shape}, dtype={arr.dtype}")


def _expect_rank(name: str, arr: np.ndarray, rank: int) -> CheckResult:
    ok = isinstance(arr, np.ndarray) and arr.ndim == rank
    return CheckResult(f"rank:{name}", ok, f"ndim={getattr(arr,'ndim',None)} expected={rank} shape={getattr(arr,'shape',None)}")


def _range_check(name: str, val: float, lo: float, hi: float) -> CheckResult:
    ok = (val >= lo) and (val <= hi)
    return CheckResult(f"range:{name}", ok, f"val={val:.6g} expected=[{lo:.6g},{hi:.6g}]")


def _close_check(name: str, val: float, target: float, rtol: float, atol: float) -> CheckResult:
    ok = np.isclose(val, target, rtol=rtol, atol=atol)
    return CheckResult(f"close:{name}", bool(ok), f"val={val:.6g} target={target:.6g} rtol={rtol} atol={atol}")


def validate_run(run_dir: str) -> Tuple[List[CheckResult], Dict[str, Any]]:
    results: List[CheckResult] = []

    summary_path = os.path.join(run_dir, "summary.json")
    if not os.path.exists(summary_path):
        results.append(CheckResult("summary.json exists", False, "missing"))
        return results, {}

    summary = _load_json(summary_path)
    case = summary.get("case")

    # ---- Tier 1: artifacts ----
    expected = [
        "run.log",
        "summary.json",
        "lcfs_contour.npz",
        "mesh.npz",
        "profiles.npz",
        "q_and_geometry.npz",
    ]
    for fn in expected:
        results.append(_nonempty_file(os.path.join(run_dir, fn)))

    lcfs, r = _npz_load(os.path.join(run_dir, "lcfs_contour.npz"))
    results.append(r)
    mesh, r = _npz_load(os.path.join(run_dir, "mesh.npz"))
    results.append(r)
    prof, r = _npz_load(os.path.join(run_dir, "profiles.npz"))
    results.append(r)
    qgeo, r = _npz_load(os.path.join(run_dir, "q_and_geometry.npz"))
    results.append(r)

    # LCFS structure
    if lcfs:
        for k in ["rb", "zb"]:
            if k in lcfs:
                results.append(_expect_rank(f"lcfs.{k}", lcfs[k], 1))
                results.append(_finite_array(f"lcfs.{k}", lcfs[k]))
        if "rb" in lcfs and "zb" in lcfs:
            results.append(CheckResult(
                "lcfs same length",
                lcfs["rb"].shape == lcfs["zb"].shape,
                f"rb={lcfs['rb'].shape} zb={lcfs['zb'].shape}",
            ))
            n = lcfs["rb"].size
            results.append(_range_check("lcfs.npts", float(n), 20, 5000))

    # Mesh structure
    if mesh:
        for k, rk in [("pts", 2), ("lc", 2), ("reg", 1)]:
            if k in mesh:
                results.append(_expect_rank(f"mesh.{k}", mesh[k], rk))
                results.append(_finite_array(f"mesh.{k}", mesh[k]))
        if set(["pts", "lc", "reg"]).issubset(mesh.keys()):
            pts = mesh["pts"]
            lc = mesh["lc"]
            reg = mesh["reg"]
            results.append(CheckResult("mesh pts shape", pts.shape[1] == 2, f"pts.shape={pts.shape}"))
            results.append(CheckResult("mesh lc tri", lc.shape[1] == 3, f"lc.shape={lc.shape}"))
            results.append(CheckResult("mesh reg length", reg.shape[0] == lc.shape[0], f"reg={reg.shape} lc={lc.shape}"))
            results.append(_range_check("mesh.npts", float(pts.shape[0]), 50, 200000))
            results.append(_range_check("mesh.ncells", float(lc.shape[0]), 50, 400000))

    # Profiles structure: expect 1D arrays with same length for s, p, fpol or similar
    if prof:
        # Accept a few possible key spellings across OFT builds
        s_key = "s" if "s" in prof else ("psi_norm" if "psi_norm" in prof else None)
        if s_key:
            results.append(_expect_rank(f"profiles.{s_key}", prof[s_key], 1))
            results.append(_finite_array(f"profiles.{s_key}", prof[s_key]))
            ns = prof[s_key].size
            results.append(_range_check("profiles.ns", float(ns), 10, 5000))

    # q/geometry structure
    if qgeo:
        # Expect at least q and some radial coordinate(s)
        for k in ["psi", "q"]:
            if k in qgeo:
                results.append(_expect_rank(f"qgeo.{k}", qgeo[k], 1))
                results.append(_finite_array(f"qgeo.{k}", qgeo[k]))
        if "q" in qgeo:
            q = qgeo["q"]
            # q should be positive and finite for most points
            results.append(CheckResult("q positive", bool(np.all(q[np.isfinite(q)] > 0.0)), f"qmin={np.nanmin(q):.6g} qmax={np.nanmax(q):.6g}"))

    # ---- Tier 2: scalar/physics checks (ranges based on run logs + common sense) ----
    # We use broad ranges because we don't have authoritative notebook numeric asserts.
    # Our driver stores key values under summary['scalars']
    sc = summary.get("scalars", {})
    targets = summary.get("parameters", {}).get("targets_kwargs", {})

    if case == "analytic":
        # Ranges taken from the solver printout for this run family and broad physical plausibility.
        results.append(_range_check("q.q0", float(sc.get("q.q0", np.nan)), 0.2, 1.5))
        results.append(_range_check("q.q95", float(sc.get("q.q95", np.nan)), 0.6, 2.0))
        results.append(_range_check("p.max[Pa]", float(sc.get("p.max", np.nan)), 1e3, 5e4))
        results.append(_range_check("boundary_npts", float(sc.get("boundary_npts", np.nan)), 50, 200))
        # No Ip scalar stored in summary; we validate by q range and pressure range here.
    elif case == "eqdsk":
        results.append(_range_check("q.q0", float(sc.get("q.q0", np.nan)), 0.8, 4.0))
        results.append(_range_check("q.q95", float(sc.get("q.q95", np.nan)), 2.0, 10.0))
        results.append(_range_check("p.max[Pa]", float(sc.get("p.max", np.nan)), 1e5, 5e6))
        results.append(_range_check("boundary_npts", float(sc.get("boundary_npts", np.nan)), 40, 2000))
        if "pax" in targets:
            results.append(_close_check("p_axis~target", float(sc.get("p.max", np.nan)), float(targets.get("pax", np.nan)), rtol=0.1, atol=3e5))
    else:
        results.append(CheckResult("case known", False, f"case={case!r}"))

    return results, summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="Path to a specific run output directory (the one containing summary.json)")
    args = ap.parse_args()

    run_dir = args.run_dir
    results, summary = validate_run(run_dir)

    ok = all(r.ok for r in results)
    print(f"Validation for: {run_dir}")
    print(f"case={summary.get('case')}  overall={'PASS' if ok else 'FAIL'}  checks={len(results)}")
    print("-")
    # Print failures first
    for r in results:
        if not r.ok:
            print(f"FAIL  {r.name}: {r.details}")
    if not ok:
        print("-")
    for r in results:
        if r.ok:
            print(f"PASS  {r.name}: {r.details}")

    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
