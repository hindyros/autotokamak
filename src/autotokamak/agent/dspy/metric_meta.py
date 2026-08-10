# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Composite scorer for the meta-agent loop.

Mirrors ``metric_surrogate.score_surrogate_run`` shape (same ``ScoreReport``
dataclass, same hard-gates × quality composite). Reads the meta-workspace's
``report.json`` + ``meta_trace.json`` and scores:

Hard gates (all must pass for nonzero total):
    deliverables_present : winner.pkl, report.json, meta_trace.json
    report_parseable     : report.json validates against MetaReport
    iteration_log        : meta_trace.json has ≥1 iteration record
    winner_predicts      : winner.pkl loads and predicts test split

Quality terms:
    final_rmse_vs_baseline (0.35) : 1 - final_rmse/baseline_rmse, clipped
    improvement_over_iterations (0.20) : first-rmse minus last-rmse, normalized
    budget_efficiency (0.10) : fraction of the total improvement (from the
                               baseline) achieved by the halfway iteration
    no_waste (0.15) : fraction of post-first-winner iterations that improved
                      the best shard RMSE by >= 1% relative. An agent that
                      keeps burning budget on flat iterations instead of
                      terminating scores 0 here — this is what makes a
                      decisive short run score differently from a cap-riding
                      one (the two used to tie, starving GEPA of variance).
    terminated_by_agent (0.15) : 1.0 if agent terminated, 0.0 if cap hit
    runner_cleanliness (0.05) : iteration log uses the expected action types

The old ``diagnosis_consistency`` term (keyword-regex match between diagnosis
text and action) was removed from the weighted score: it was trivially
gameable by keyword-stuffing the diagnosis. It is still computed and stored
in ``details["diagnosis_consistency_advisory"]`` for inspection. ``no_waste``
is not gameable the same way — it is computed from measured shard RMSE, not
from agent-reported text.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from autotokamak.agent.orchestrator.schema import VALID_ACTIONS

EXPECTED_DELIVERABLES = ("winner.pkl", "report.json", "meta_trace.json")
EXPECTED_ACTIONS = set(VALID_ACTIONS)

WEIGHTS = {
    "final_rmse_vs_baseline": 0.35,
    "improvement_over_iterations": 0.20,
    "budget_efficiency": 0.10,
    "no_waste": 0.15,
    "terminated_by_agent": 0.15,
    "runner_cleanliness": 0.05,
}

# Relative shard-RMSE improvement below which an iteration counts as wasted.
NO_WASTE_MIN_RELATIVE_IMPROVEMENT = 0.01


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


def _clip01(x: float) -> float:
    if not np.isfinite(x):
        return 0.0
    return float(max(0.0, min(1.0, x)))


def _load_report(path: Path) -> tuple[Any | None, str | None]:
    try:
        from autotokamak.agent.orchestrator.schema import MetaReport

        raw = json.loads(path.read_text(encoding="utf-8"))
        return MetaReport.model_validate(raw), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def _load_trace(path: Path) -> tuple[dict | None, str | None]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or "iterations" not in raw:
            return None, "meta_trace.json missing 'iterations'"
        return raw, None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def score_meta_run(workspace: str | Path) -> ScoreReport:
    ws = Path(workspace)
    report = ScoreReport(workspace=ws)

    # -- Hard gate 1: deliverables present --
    missing = [f for f in EXPECTED_DELIVERABLES if not (ws / f).is_file()]
    report.hard_gates["deliverables_present"] = not missing
    report.details["missing_deliverables"] = missing

    # -- Hard gate 2: report.json parses --
    parsed_report, report_err = _load_report(ws / "report.json")
    report.hard_gates["report_parseable"] = parsed_report is not None
    if report_err:
        report.details["report_parse_error"] = report_err

    # -- Hard gate 3: meta_trace.json valid --
    trace, trace_err = _load_trace(ws / "meta_trace.json")
    report.hard_gates["iteration_log"] = trace is not None and len(trace.get("iterations", [])) >= 1
    if trace_err:
        report.details["trace_parse_error"] = trace_err

    # -- Hard gate 4: winner_predicts --
    winner_predicts = False
    if (ws / "winner.pkl").is_file():
        try:
            import joblib

            payload = joblib.load(ws / "winner.pkl")
            req = {"estimator", "pca"}
            winner_predicts = req.issubset(set(payload))
        except Exception as exc:  # noqa: BLE001
            report.details["winner_load_error"] = f"{type(exc).__name__}: {exc}"
    report.hard_gates["winner_predicts"] = winner_predicts

    if not report.all_gates_pass:
        return report

    assert parsed_report is not None
    assert trace is not None

    # -- final_rmse_vs_baseline --
    if parsed_report.final_rmse is None:
        # No winner was ever produced (the winner_predicts gate normally
        # zeroes such runs already; this keeps scoring exception-free).
        report.quality["final_rmse_vs_baseline"] = 0.0
        report.details["final_rmse"] = "no winner produced"
    else:
        report.quality["final_rmse_vs_baseline"] = _clip01(
            1.0 - parsed_report.final_rmse / max(parsed_report.baseline_rmse, 1e-12)
        )

    # -- improvement_over_iterations --
    history = list(parsed_report.rmse_history)
    if len(history) >= 2:
        first, last = history[0], history[-1]
        improvement = (first - last) / max(first, 1e-12)
        report.quality["improvement_over_iterations"] = _clip01(improvement)
    else:
        report.quality["improvement_over_iterations"] = 0.0

    # -- diagnosis_consistency (ADVISORY ONLY, not weighted): heuristic
    # keyword match between diagnosis text and action type. Removed from the
    # score because it is trivially gameable by keyword-stuffing.
    matches = 0
    total = 0
    for it in trace["iterations"]:
        diagnosis = (it.get("decision", {}) or {}).get("diagnosis", "") or ""
        action = (it.get("decision", {}) or {}).get("action", "")
        total += 1
        if action == "regen_dataset" and re.search(r"\b(sample|data|N\b|coverage|fidelity|noise|mesh)", diagnosis, re.I):
            matches += 1
        elif action == "enrich_active" and re.search(r"\b(sample|data|coverage|uncertain|variance|active|acqui|inform)", diagnosis, re.I):
            matches += 1
        elif action == "extend_search" and re.search(r"\b(edge|range|model|hyper|widen|search|trial|tune)", diagnosis, re.I):
            matches += 1
        elif action == "terminate" and re.search(r"\b(good|enough|done|stop|terminate|converge|plateau)", diagnosis, re.I):
            matches += 1
    report.details["diagnosis_consistency_advisory"] = matches / total if total else 0.0

    # -- budget_efficiency: fraction of the total improvement (relative to
    # the baseline) already achieved by the halfway iteration. Early
    # improvement → 1.0; late-only improvement → 0.0; never beating the
    # baseline → 0.0 (no efficiency credit for a run with no improvement).
    ordered = list(trace["iterations"])
    rmses = [it.get("rmse_after") for it in ordered]
    known = [v for v in rmses if v is not None]
    if not known:
        report.quality["budget_efficiency"] = 0.0
    else:
        start = float(parsed_report.baseline_rmse)
        best_final = min(known)
        total_impr = start - best_final
        half_n = max(1, len(ordered) // 2)
        early = [v for i, v in enumerate(rmses) if v is not None and i < half_n]
        if total_impr <= 0 or not early:
            report.quality["budget_efficiency"] = 0.0
        else:
            report.quality["budget_efficiency"] = _clip01(
                (start - min(early)) / total_impr
            )

    # -- no_waste: after the first winner exists, every further measured
    # iteration must improve the best shard RMSE by >= 1% relative or it
    # counts as wasted budget. terminate iterations carry no rmse_after and
    # are never counted. No measured iterations beyond the first -> 1.0
    # (nothing was wasted).
    measured = [it.get("rmse_after") for it in trace["iterations"] if it.get("rmse_after") is not None]
    if len(measured) <= 1:
        report.quality["no_waste"] = 1.0
    else:
        best = float(measured[0])
        productive = 0
        for v in measured[1:]:
            if (best - float(v)) / max(best, 1e-12) >= NO_WASTE_MIN_RELATIVE_IMPROVEMENT:
                productive += 1
                best = float(v)
        report.quality["no_waste"] = productive / (len(measured) - 1)
        report.details["wasted_iterations"] = (len(measured) - 1) - productive

    # -- terminated_by_agent: full credit for both decisive outcomes — the
    # agent choosing to stop, or the run hitting its quality bar early
    # (target_reached IS the desired cost-saving behavior).
    report.quality["terminated_by_agent"] = (
        1.0 if parsed_report.terminated_by in ("agent", "target_reached") else 0.0
    )

    # -- runner_cleanliness: only expected action types used --
    action_types = {
        (it.get("decision", {}) or {}).get("action") for it in trace["iterations"]
    }
    invalid = action_types - EXPECTED_ACTIONS
    report.quality["runner_cleanliness"] = 0.0 if invalid else 1.0
    if invalid:
        report.details["invalid_actions"] = sorted(a for a in invalid if a)

    return report


__all__ = [
    "EXPECTED_DELIVERABLES",
    "NO_WASTE_MIN_RELATIVE_IMPROVEMENT",
    "ScoreReport",
    "WEIGHTS",
    "score_meta_run",
]
