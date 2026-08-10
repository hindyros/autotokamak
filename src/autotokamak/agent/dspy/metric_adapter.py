# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Bridge ``score_meta_run`` to a GEPA-compatible feedback metric.

GEPA's ``GEPAFeedbackMetric`` protocol expects:

    (gold: Example, pred: Prediction, trace=None, pred_name=None, pred_trace=None)
        -> float | dspy.Prediction(score=..., feedback=...)

The trainset we feed GEPA carries the WHOLE-RUN score as ``run_score`` on
each example (see ``trace_loader.load_meta_traces``). Optimization happens
in two modes depending on context:

1. **Offline / cached** (the default in ``optimize_meta``): the metric reads
   the precomputed ``run_score`` from ``gold`` and returns it. Per-iteration
   feedback text combines the LM's prediction with the gold decision so GEPA's
   reflective mutation has natural-language signal.

2. **Online / live-eval**: not implemented here. Would require actually
   invoking the full meta-loop on each candidate prompt — $$$ per evaluation.
   Mode 2 lives in a future ``live_metric`` if/when we want it.

Multi-criterion: GEPA supports per-predictor feedback. Today the meta-action
picker has just one predictor (``predict.predict``), so we return a single
scalar + feedback string. When we add more (Phase-2 search picker, etc.)
this adapter grows a per-predictor branch.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import dspy


def _action_consistency_feedback(gold_action: str, pred_action: str) -> str:
    if pred_action == gold_action:
        return f"Action matches the gold trajectory's action ({gold_action})."
    return (
        f"Action diverges from gold trajectory: predicted {pred_action!r}, "
        f"gold was {gold_action!r}. This may indicate the prompt is pulling "
        f"the picker toward a different decision than the high-score run took."
    )


def _diagnosis_feedback(gold_diagnosis: str, pred_diagnosis: str) -> str:
    if not gold_diagnosis:
        return ""
    return (
        f"Gold trajectory's diagnosis: {gold_diagnosis!r}. "
        f"Predicted diagnosis: {pred_diagnosis!r}. The scorer rewards alignment."
    )


def gepa_metric(
    gold: dspy.Example,
    pred: dspy.Prediction,
    trace: Optional[Any] = None,
    pred_name: Optional[str] = None,
    pred_trace: Optional[Any] = None,
) -> dspy.Prediction:
    """Cached/offline GEPA metric.

    The score is just the trajectory's whole-run ``run_score`` — every
    iteration of the same run gets the same score, GEPA's Pareto frontier
    handles the per-iteration credit assignment. The feedback string is
    constructed per-iteration so reflection has local signal.
    """
    gold_action = str(gold.get("gold_action", "") or "")
    gold_diagnosis = str(gold.get("gold_diagnosis", "") or "")
    pred_action = str(getattr(pred, "action", "") or "")
    pred_diagnosis = str(getattr(pred, "diagnosis", "") or "")

    score = float(gold.get("run_score", 0.0) or 0.0)

    feedback_lines: list[str] = []
    feedback_lines.append(f"This trajectory's overall meta-run scored {score:.3f}.")
    feedback_lines.append(_action_consistency_feedback(gold_action, pred_action))
    diag_fb = _diagnosis_feedback(gold_diagnosis, pred_diagnosis)
    if diag_fb:
        feedback_lines.append(diag_fb)
    rmse_after = gold.get("rmse_after")
    if rmse_after is not None:
        feedback_lines.append(
            f"After this action was actually taken in the gold run, RMSE became {rmse_after}."
        )

    return dspy.Prediction(
        score=score,
        feedback="\n".join(feedback_lines),
    )


__all__ = ["gepa_metric"]
