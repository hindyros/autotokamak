# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Composite scoring function for agent-driven dataset generation runs.

Pure-Python, no DSPy dependency. Use it three ways:

1.  Retrospectively, against any workspace produced by `dataset_generation.yaml`:

        from autotokamak.agent.dspy import score_run
        report = score_run("examples/dataset_generation/", requested_n_samples=16)
        print(report.summary())

2.  Inside the agent runner after a feedback round, to gate continuation.

3.  Later, as a DSPy optimizer metric:

        optimizer = dspy.BootstrapFewShot(metric=lambda ex, pred, trace: score_run(...).total)

Score shape
-----------
Hard gates (boolean, ALL must pass for a nonzero total):
    deliverables_present : the three expected files exist
    dataset_h5_opens     : outputs/dataset.h5 exists and opens cleanly
    at_least_one_success : at least one /outputs/success entry is True

Quality terms (each in [0, 1]; weighted sum becomes the total):
    success_rate            (weight 0.30)  n_succeeded / n_requested
    inside_lcfs_quality     (weight 0.25)  fraction of in-LCFS pixels with REAL
                                           (non-extrapolated) psi values. This
                                           catches the griddata(nearest) silent-
                                           fill bug we saw in the first run.
    outside_lcfs_honesty    (weight 0.20)  outside-LCFS pixels are NaN or have
                                           real variation (not flat-filled).
    shape_fidelity          (weight 0.20)  correlation between requested r0 and
                                           observed magnetic-axis R.

(The old ``isoflux_constraint_held`` term was removed 2026-07-11 — see the
NOTE above WEIGHTS. Fixed-boundary solves never apply the isoflux constraint,
so the term punished every correct run; the fraction is kept in details as
``isoflux_used_fraction_advisory``.)
    runner_cleanliness      (weight 0.05)  did the runner import from
                                           autotokamak.core (heuristic, parses
                                           run_dataset_sweep.py).

Total = product(hard_gates) * weighted_sum(quality_terms).

The weights are starting values; tune once we have ≥10 traces to compare.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

EXPECTED_DELIVERABLES = ("dataset_config.yaml", "run_dataset_sweep.py", "README.md")
DATASET_RELPATH = "outputs/dataset.h5"

# NOTE (2026-07-11): the old ``isoflux_constraint_held`` term (weight 0.20)
# was removed from the weighted score. Root cause analysis showed OFT's
# isoflux constraint fitting is a FREE-BOUNDARY mechanism (it adjusts coil
# currents); on the fixed-boundary plasma-only meshes this scorer grades,
# isoflux_used=False is the CORRECT outcome — the LCFS is enforced exactly
# by the mesh boundary. Keeping the term would dock every valid run 20%.
# Shape honoring is still scored independently via ``shape_fidelity``; the
# isoflux fraction is retained in details as advisory provenance.
WEIGHTS = {
    "success_rate": 0.30,
    "inside_lcfs_quality": 0.25,
    "outside_lcfs_honesty": 0.20,
    "shape_fidelity": 0.20,
    "runner_cleanliness": 0.05,
}


@dataclass
class ScoreReport:
    workspace: Path
    hard_gates: dict[str, bool] = field(default_factory=dict)
    quality: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def all_gates_pass(self) -> bool:
        return bool(self.hard_gates) and all(self.hard_gates.values())

    @property
    def total(self) -> float:
        """Composite score in [0, 1]. Zero if any hard gate fails."""
        if not self.all_gates_pass:
            return 0.0
        return float(sum(WEIGHTS[k] * self.quality.get(k, 0.0) for k in WEIGHTS))

    def summary(self) -> str:
        lines = [f"ScoreReport[{self.workspace}]"]
        lines.append("  hard gates:")
        for k, v in self.hard_gates.items():
            lines.append(f"    [{'PASS' if v else 'FAIL'}] {k}")
        if self.all_gates_pass:
            lines.append("  quality:")
            for k, w in WEIGHTS.items():
                q = self.quality.get(k, 0.0)
                lines.append(f"    {q:.3f}  (weight {w:.2f})  {k}")
            lines.append(f"  --> total = {self.total:.3f}")
        else:
            lines.append("  --> total = 0.000 (hard gate failed)")
        return "\n".join(lines)


def score_run(workspace: str | Path, *, requested_n_samples: int) -> ScoreReport:
    """Score one agent run by inspecting its workspace + dataset.h5.

    Parameters
    ----------
    workspace : Path-like
        The directory the agent wrote to (e.g. ``examples/dataset_generation/``).
    requested_n_samples : int
        The N the prompt asked for. Used to compute ``success_rate``; do not
        rely on the dataset itself for this because a buggy runner could
        produce fewer rows than requested without flagging.
    """
    ws = Path(workspace)
    report = ScoreReport(workspace=ws)

    # -- Hard gate 1: deliverables present --
    missing = [f for f in EXPECTED_DELIVERABLES if not (ws / f).is_file()]
    report.hard_gates["deliverables_present"] = not missing
    report.details["missing_deliverables"] = missing

    # -- Hard gate 2: dataset opens --
    dataset_path = ws / DATASET_RELPATH
    h5_handle = None
    try:
        import h5py
        if dataset_path.is_file():
            h5_handle = h5py.File(dataset_path, "r")
        report.hard_gates["dataset_h5_opens"] = h5_handle is not None
    except Exception as exc:  # noqa: BLE001
        report.hard_gates["dataset_h5_opens"] = False
        report.details["dataset_open_error"] = repr(exc)

    if h5_handle is None:
        # Hard gate 3 can't be evaluated; leave the report partial.
        report.hard_gates["at_least_one_success"] = False
        return report

    try:
        success = np.asarray(h5_handle["outputs/success"][:], dtype=bool)
        psi = np.asarray(h5_handle["outputs/psi"][:], dtype=np.float64)
        R = np.asarray(h5_handle["grid/R"][:], dtype=np.float64)
        Z = np.asarray(h5_handle["grid/Z"][:], dtype=np.float64)
        inputs = {k: np.asarray(h5_handle[f"inputs/{k}"][:], dtype=np.float64)
                  for k in ("r0", "a", "kappa", "delta", "Ip")
                  if f"inputs/{k}" in h5_handle}

        report.hard_gates["at_least_one_success"] = bool(success.any())
        report.quality["success_rate"] = float(success.sum() / max(requested_n_samples, 1))

        # -- isoflux fraction (ADVISORY ONLY, not weighted) ----------------
        # Historically a 0.20-weight term, removed 2026-07-11: fixed-boundary
        # solves NEVER apply the isoflux constraint (it fits free-boundary
        # coil currents; a plasma-only mesh has none), so isoflux_used=False
        # is the correct outcome and the requested shape is enforced exactly
        # by the mesh boundary. Recorded here as provenance; shape honoring
        # is scored by shape_fidelity below.
        if "outputs/isoflux_used" in h5_handle:
            isoflux_used = np.asarray(h5_handle["outputs/isoflux_used"][:], dtype=bool)
            n_success = int(success.sum())
            report.details["isoflux_used_count"] = int(isoflux_used[success].sum())
            report.details["isoflux_used_fraction_advisory"] = (
                float(isoflux_used[success].sum() / n_success) if n_success else 0.0
            )
        else:
            report.details["isoflux_used_missing"] = True

        # -- inside_lcfs_quality -----------------------------------------
        # For each successful sample, look only inside the requested LCFS
        # bbox. The bug we want to catch: nearest-fill making every outside
        # pixel a constant 1.0 while metadata claims NaN. We measure the
        # FRACTION of inside-bbox pixels whose value is finite AND whose
        # local std is nonzero (i.e. real interpolation, not a flat fill).
        if success.any() and inputs:
            quality_scores: list[float] = []
            for i in np.where(success)[0]:
                r0 = inputs["r0"][i] if "r0" in inputs else None
                a = inputs["a"][i] if "a" in inputs else None
                k = inputs["kappa"][i] if "kappa" in inputs else 1.0
                if r0 is None or a is None:
                    continue
                in_r = (R >= r0 - a) & (R <= r0 + a)        # shape (nr,)
                in_z = (Z >= -a * k) & (Z <= a * k)         # shape (nz,)
                mask = in_z[:, None] & in_r[None, :]        # (nz, nr)
                p = psi[i][mask]
                if p.size < 10:
                    continue
                finite = np.isfinite(p)
                if finite.sum() < 5:
                    quality_scores.append(0.0)
                    continue
                # Variance test: if every inside pixel is constant, the
                # interpolator is fake-filling. Real GS psi has measurable
                # variation across the plasma.
                pf = p[finite]
                varied = float(pf.std() > 1e-6)
                # Combine: fraction-finite weighted by variance gate
                quality_scores.append(float(finite.mean()) * varied)
            report.quality["inside_lcfs_quality"] = (
                float(np.mean(quality_scores)) if quality_scores else 0.0
            )
        else:
            report.quality["inside_lcfs_quality"] = 0.0

        # -- outside_lcfs_honesty ---------------------------------------
        # The documented convention is "NaN outside interpolation domain".
        # The agent's first run silently overwrote outside-LCFS NaNs with
        # the nearest boundary value (~1.0 for normalized psi), producing
        # a constant-std=0 fake fill outside the plasma.
        #
        # Honest behavior: outside pixels are either NaN (preferred) or
        # have real variation (would happen with a wider physics-aware
        # extrapolation). Dishonest: outside is finite with zero variance.
        if success.any() and inputs:
            honesty_scores: list[float] = []
            for i in np.where(success)[0]:
                r0 = inputs["r0"][i] if "r0" in inputs else None
                a = inputs["a"][i] if "a" in inputs else None
                k = inputs["kappa"][i] if "kappa" in inputs else 1.0
                if r0 is None or a is None:
                    continue
                in_r = (R >= r0 - a) & (R <= r0 + a)
                in_z = (Z >= -a * k) & (Z <= a * k)
                inside = in_z[:, None] & in_r[None, :]
                outside = ~inside
                p_out = psi[i][outside]
                if p_out.size < 10:
                    continue
                nan_frac = float(np.isnan(p_out).mean())
                finite = p_out[np.isfinite(p_out)]
                if finite.size > 0:
                    finite_std = float(finite.std())
                else:
                    finite_std = 0.0
                # Honesty score: 1.0 if all outside is NaN; 1.0 if outside
                # is finite but varied (std > 1e-3 of normalized scale); 0
                # if outside is finite-and-constant (the bug case).
                if nan_frac > 0.95:
                    honesty_scores.append(1.0)
                elif finite_std > 1e-3:
                    honesty_scores.append(1.0)
                else:
                    # Smooth penalty between 0 (flat fill) and 1 (variation)
                    honesty_scores.append(min(1.0, finite_std * 1000.0))
            report.quality["outside_lcfs_honesty"] = (
                float(np.mean(honesty_scores)) if honesty_scores else 0.0
            )
        else:
            report.quality["outside_lcfs_honesty"] = 0.0

        # -- shape_fidelity ----------------------------------------------
        # Crude proxy: does the location of psi-minimum (the magnetic axis)
        # correlate with requested r0? Strong correlation -> the agent's
        # solver is solving what we asked.
        if success.sum() >= 4 and "r0" in inputs:
            observed_r_axis = []
            valid_idx = []
            for i in np.where(success)[0]:
                p = psi[i]
                if not np.isfinite(p).any():
                    continue
                # axis = global psi-min (normalized psi: axis is 0, edge is 1)
                idx = np.unravel_index(np.nanargmin(p), p.shape)
                observed_r_axis.append(R[idx[1]])
                valid_idx.append(i)
            if len(valid_idx) >= 4:
                obs = np.asarray(observed_r_axis)
                req = inputs["r0"][valid_idx]
                if obs.std() > 1e-6 and req.std() > 1e-6:
                    corr = float(np.corrcoef(obs, req)[0, 1])
                    # remap [-1, 1] to [0, 1]; only positive corr earns credit
                    report.quality["shape_fidelity"] = max(0.0, corr)
                else:
                    report.quality["shape_fidelity"] = 0.0
            else:
                report.quality["shape_fidelity"] = 0.0
        else:
            report.quality["shape_fidelity"] = 0.0

    finally:
        if h5_handle is not None:
            h5_handle.close()

    # -- runner_cleanliness (parse the runner script) --------------------
    runner_path = ws / "run_dataset_sweep.py"
    if runner_path.is_file():
        src = runner_path.read_text(encoding="utf-8", errors="replace")
        clean_signals = [
            r"from autotokamak\.core\.geometry import",
            r"from autotokamak\.core\.solver import",
        ]
        dirty_signals = [
            r"import OpenFUSIONToolkit\b",                 # bypassing core
            r"OpenFUSIONToolkit\.OFT_env\(",               # bypassing the cache
            r"\bdef _seed_psi\b|\bdef _isoflux_fit\b",     # reimplementing core
        ]
        n_clean = sum(1 for pat in clean_signals if re.search(pat, src))
        n_dirty = sum(1 for pat in dirty_signals if re.search(pat, src))
        # 1.0 if all clean signals present and no dirty ones; degrades from there
        clean_frac = n_clean / max(len(clean_signals), 1)
        dirty_penalty = min(n_dirty / max(len(dirty_signals), 1), 1.0)
        report.quality["runner_cleanliness"] = max(0.0, clean_frac - dirty_penalty)
    else:
        report.quality["runner_cleanliness"] = 0.0

    return report


__all__ = ["ScoreReport", "score_run", "WEIGHTS", "EXPECTED_DELIVERABLES"]
