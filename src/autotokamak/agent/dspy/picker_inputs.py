# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Single source of truth for the meta-action-picker's LM inputs.

Train/serve skew fix (2026-07-10): the runtime picker
(``meta_loop.pick_action_via_llm``) and the GEPA trainset builder
(``trace_loader.load_meta_traces``) used to construct ``state_summary``
with DIFFERENT keys — the runtime sent ``iterations_remaining`` /
``current_dataset`` / ``best_rmse_so_far`` while training examples carried
``rmse_after_prior_iter``. The signature's cold-start HARD RULE keys off
``best_rmse_so_far``, which therefore never appeared in training inputs.

Now: this module builds the three input strings, the runtime records them
verbatim into each ``MetaIterationRecord`` (``picker_inputs``), and the
trace loader prefers the recorded strings — training inputs are
byte-identical to what the deployed picker saw.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

PICKER_INPUT_KEYS = ("diagnostics_json", "history_summary", "state_summary")

DIAGNOSTICS_MAX_CHARS = 4000
HISTORY_MAX_CHARS = 2000


def build_picker_inputs(
    *,
    diagnostics: dict,
    prior_decisions: List[Dict[str, Any]],
    rmse_history: List[Optional[float]],
    iteration: int,
    iterations_remaining: int,
    current_dataset: str,
    best_rmse_so_far: Optional[float],
    target_rmse: Optional[float] = None,
) -> Dict[str, str]:
    """Render the three LM input strings from plain data.

    ``target_rmse`` is the resolved absolute early-stop bar (None = no bar).
    The runner stops mechanically when it is met; surfacing it lets the
    picker prefer cheap actions when close to the bar.
    """
    return {
        "diagnostics_json": json.dumps(diagnostics, indent=2, default=str)[
            :DIAGNOSTICS_MAX_CHARS
        ],
        "history_summary": json.dumps(
            {"prior_decisions": prior_decisions, "rmse_history": rmse_history},
            indent=2,
            default=str,
        )[:HISTORY_MAX_CHARS],
        "state_summary": json.dumps(
            {
                "iteration": iteration,
                "iterations_remaining": iterations_remaining,
                "current_dataset": current_dataset,
                "best_rmse_so_far": best_rmse_so_far,
                "target_rmse": target_rmse,
            },
            indent=2,
            default=str,
        ),
    }


def picker_inputs_from_runtime(
    meta_config: Any,
    state: Any,
    diagnostics: dict,
    history: List[Any],
) -> Dict[str, str]:
    """Build picker inputs from live meta-loop objects.

    ``meta_config`` / ``state`` / ``history`` are duck-typed (MetaConfig,
    MetaState, list[MetaIterationRecord]) so this module stays import-cheap.
    Deterministic: calling it twice with the same objects yields identical
    strings — the runner records the result verbatim into the trace.
    """
    prior_decisions = [
        {
            "iteration": r.iteration,
            "action": r.decision.action,
            "diagnosis": r.decision.diagnosis,
            "rmse_after": r.rmse_after,
        }
        for r in history
    ]
    best = getattr(state, "best_rmse", float("inf"))
    return build_picker_inputs(
        diagnostics=diagnostics,
        prior_decisions=prior_decisions,
        rmse_history=list(state.rmse_history),
        iteration=len(history),
        iterations_remaining=meta_config.max_iterations - len(history),
        current_dataset=str(state.current_dataset_h5),
        best_rmse_so_far=None if best == float("inf") else best,
        target_rmse=getattr(state, "target_rmse_abs", None),
    )


__all__ = ["PICKER_INPUT_KEYS", "build_picker_inputs", "picker_inputs_from_runtime"]
