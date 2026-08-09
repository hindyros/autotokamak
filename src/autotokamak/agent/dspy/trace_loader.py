# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Convert recorded meta-agent traces into a DSPy trainset.

Each meta-loop run writes two JSON files:
- ``experiments/<run_id>/trace.json``        (RunTrace skeleton)
- ``<workspace>/meta_trace.json``            (per-iteration log + report)

This module reads both and emits one ``dspy.Example`` per per-iteration
``(diagnostics, decision, result)`` tuple. The example carries the score
the meta-run earned (from `score_meta_run`), which is what GEPA uses as
its training signal.

Why per-iteration rather than per-run: GEPA's reflective mutation works
best when it can see the local decision the LLM made plus the *eventual*
score the trajectory earned. The whole-run score gets attached to every
iteration the agent emitted; GEPA's Pareto frontier then naturally
identifies which decision patterns correlate with high scores.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import dspy

from autotokamak.agent.dspy.picker_inputs import PICKER_INPUT_KEYS


META_PROMPT_PATH_SUFFIX = "surrogate_meta.yaml"


def _safe_load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _iter_meta_runs(
    experiments_dir: Path,
    *,
    prompt_path_filter: str | None = META_PROMPT_PATH_SUFFIX,
) -> Iterator[tuple[dict, dict]]:
    """Yield ``(run_trace, meta_trace)`` pairs for every completed meta-run.

    Skips runs whose workspace has no ``meta_trace.json`` (i.e. not a
    meta-loop invocation, or it never finalized).
    """
    if not experiments_dir.is_dir():
        return
    for run_dir in sorted(experiments_dir.iterdir()):
        trace_path = run_dir / "trace.json"
        if not trace_path.is_file():
            continue
        run_trace = _safe_load_json(trace_path)
        if not run_trace:
            continue
        prompt_path = (run_trace.get("prompt") or {}).get("path", "")
        if prompt_path_filter and not str(prompt_path).endswith(prompt_path_filter):
            continue
        workspace = (run_trace.get("prompt") or {}).get("workspace", "")
        if not workspace:
            continue
        meta_trace = _safe_load_json(Path(workspace) / "meta_trace.json")
        if not meta_trace or not meta_trace.get("iterations"):
            continue
        yield run_trace, meta_trace


def _truncate_diag(diagnostics: dict, max_chars: int = 4000) -> str:
    """Render diagnostics dict as a stable JSON string, truncated for LM prompts.

    GEPA serializes inputs into the LM context; very long dicts blow context.
    We trim the long lists (e.g. learning_curve trial values) by serializing
    only the top-level interpretation + numeric summary keys.
    """
    s = json.dumps(diagnostics, indent=2, default=str)
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + f"\n... [truncated, total {len(s)} chars]"


def load_meta_traces(
    experiments_dir: str | Path,
    *,
    prompt_path_filter: str | None = META_PROMPT_PATH_SUFFIX,
) -> list[dspy.Example]:
    """Return one ``dspy.Example`` per meta-iteration across all completed runs.

    Parameters
    ----------
    experiments_dir : path-like
        Directory containing ``<run_id>/trace.json`` subdirectories.
    prompt_path_filter : str | None
        If given, only include runs whose prompt path ends with this suffix.
        Default keeps just surrogate_meta.yaml runs.

    Each ``Example``:

    Inputs (consumed by ``MetaActionPicker``):
        diagnostics_json : str    -- the diagnostics dict the picker saw
        history_summary  : str    -- prior decisions + scores
        state_summary    : str    -- current dataset, best rmse, iter index

    Labels (used only for scoring + few-shot demos; not LM-consumed):
        gold_action      : str    -- what the agent actually picked
        gold_diagnosis   : str    -- the agent's stated diagnosis
        rmse_after       : float | None
        run_score        : float  -- score_meta_run total for the whole run
        run_id           : str    -- provenance
    """
    experiments_dir = Path(experiments_dir)
    examples: list[dspy.Example] = []
    for run_trace, meta_trace in _iter_meta_runs(
        experiments_dir, prompt_path_filter=prompt_path_filter
    ):
        run_score = float((run_trace.get("score") or {}).get("total", 0.0))
        run_id = run_trace.get("run_id", "")
        iterations = meta_trace["iterations"]
        rmse_history: list[float | None] = []
        prior_decisions: list[dict[str, Any]] = []
        for it in iterations:
            decision = it.get("decision", {}) or {}
            diagnostics = it.get("diagnostics", {}) or {}

            # Preferred path: the runner recorded the EXACT strings the
            # deployed picker saw (picker_inputs, since the 2026-07-10
            # train/serve skew fix) — use them verbatim.
            recorded = it.get("picker_inputs") or {}
            if all(isinstance(recorded.get(k), str) for k in PICKER_INPUT_KEYS):
                diagnostics_json = recorded["diagnostics_json"]
                history_summary = recorded["history_summary"]
                state_summary = recorded["state_summary"]
            else:
                # Legacy traces (pre-fix): best-effort reconstruction. Note
                # these carry a different state_summary shape than runtime;
                # prefer regenerating the trace corpus over training on these.
                diagnostics_json = _truncate_diag(diagnostics)
                history_summary = json.dumps(
                    {
                        "prior_decisions": prior_decisions,
                        "rmse_history": rmse_history,
                    },
                    indent=2,
                    default=str,
                )[:2000]
                state_summary = json.dumps(
                    {
                        "iteration": it.get("iteration"),
                        "rmse_after_prior_iter": rmse_history[-1] if rmse_history else None,
                    },
                    indent=2,
                    default=str,
                )

            ex = dspy.Example(
                diagnostics_json=diagnostics_json,
                history_summary=history_summary,
                state_summary=state_summary,
                gold_action=decision.get("action", ""),
                gold_diagnosis=decision.get("diagnosis", ""),
                rmse_after=it.get("rmse_after"),
                run_score=run_score,
                run_id=run_id,
            ).with_inputs("diagnostics_json", "history_summary", "state_summary")
            examples.append(ex)

            prior_decisions.append(
                {
                    "iteration": it.get("iteration"),
                    "action": decision.get("action"),
                    "diagnosis": decision.get("diagnosis"),
                    "rmse_after": it.get("rmse_after"),
                }
            )
            rmse_history.append(it.get("rmse_after"))
    return examples


def split_trainset(
    examples: list[dspy.Example],
    *,
    val_frac: float = 0.25,
    seed: int = 0,
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    """Deterministic train/val split for GEPA.

    Splits by RUN, not by iteration — keeps all iterations from the same
    meta-run on the same side of the split so the val set measures
    generalization across runs, not across decisions within a run.
    """
    import random

    by_run: dict[str, list[dspy.Example]] = {}
    for ex in examples:
        by_run.setdefault(ex["run_id"], []).append(ex)
    run_ids = sorted(by_run)
    rng = random.Random(int(seed))
    rng.shuffle(run_ids)
    n_val = max(1, int(round(val_frac * len(run_ids)))) if len(run_ids) > 1 else 0
    val_run_ids = set(run_ids[:n_val])
    train: list[dspy.Example] = []
    val: list[dspy.Example] = []
    for rid in run_ids:
        (val if rid in val_run_ids else train).extend(by_run[rid])
    return train, val


__all__ = ["META_PROMPT_PATH_SUFFIX", "load_meta_traces", "split_trainset"]
