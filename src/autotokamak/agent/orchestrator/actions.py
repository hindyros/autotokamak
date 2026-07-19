"""Action dispatchers invoked by the meta-loop runner.

Three actions, three dispatchers. Each takes ``(payload, state)`` and
returns a serializable dict the next iteration's diagnostics can consume.

The ``extend_search`` dispatcher is the only one that triggers a nested LLM
call — it programmatically invokes
``agent.runners.plan_execute_feedback.main`` against an overlay prompt
that adds the meta-agent's focus directive to the existing Phase-2
problem text.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import yaml

from autotokamak.agent.orchestrator.schema import (
    ActionDecision,
    EnrichActivePayload,
    ExtendSearchFocus,
    RegenDatasetOverrides,
    TerminateReason,
)
from autotokamak.data.schema import SweepConfig
from autotokamak.data.sweep import run_sweep


@dataclass
class MetaState:
    """Mutable state threaded across iterations.

    Holds the live dataset path, the best surrogate report so far, the
    sweep config used for any regen action, and a running RMSE history.
    """

    workspace: Path
    current_dataset_h5: Path
    base_sweep_config: Optional[SweepConfig] = None
    best_winner_payload: Optional[dict] = None
    best_winner_path: Optional[Path] = None
    best_surrogate_report: Optional[dict] = None
    # Best frozen-shard RMSE so far. All winner comparisons happen on the
    # frozen test shard (test_shard_h5), never on nested-run val splits.
    best_rmse: float = float("inf")
    rmse_history: list[float] = field(default_factory=list)
    actions_taken: list[str] = field(default_factory=list)
    phase2_prompt: Path = Path(
        "src/autotokamak/agent/prompts/surrogate_automl.yaml"
    )
    seed: int = 0
    # Frozen held-out test shard: either carved from the initial dataset at
    # meta-loop start (legacy) or the full-envelope eval set (design doc
    # §D2). Never merged into, never regenerated.
    test_shard_h5: Optional[Path] = None
    # Target-envelope acquisition bounds (lows, highs) in PARAM_ORDER, set by
    # the meta-loop when MetaConfig.eval_envelope is configured. When None,
    # enrich_active acquires over base_sweep_config's (seed-box) bounds.
    envelope_bounds: Optional[tuple] = None
    # In-memory cache of the loaded frozen eval set. The file never changes
    # during a run, so _frozen_shard_rmse loads it once instead of re-parsing
    # the HDF5 on every winner comparison (up to 3x per iteration).
    shard_bundle_cache: Optional[Any] = None
    # "structured" = deterministic automl_loop with typed per-round LLM
    # decisions; "codegen" = legacy nested plan_execute_feedback agent.
    phase2_mode: str = "structured"
    # LLM string for the structured search picker (falls back to a default
    # inside _extend_search_structured when None).
    phase2_model: Optional[str] = None
    phase2_max_rounds: int = 3
    # Test hook: when set, the structured path uses this instead of the DSPy
    # search picker (no LLM required).
    phase2_decision_fn: Optional[Callable[[dict], Any]] = None
    # Resolved ABSOLUTE shard-RMSE target (meta_loop derives it from
    # MetaConfig.target_rmse / target_rmse_ratio x baseline). None = no bar;
    # surfaced to the picker and checked mechanically after each iteration.
    target_rmse_abs: Optional[float] = None
    # Hard budget the Phase-2 agent MUST write into surrogate_config.yaml.
    # Threaded into every extend_search overlay's FOCUS DIRECTIVE. None → the
    # Phase-2 agent picks the budget from its own prompt defaults.
    phase2_time_budget_seconds: Optional[int] = None
    # When True, the meta-loop's pick_action_via_llm uses the in-code
    # baseline DSPy module instead of loading the optimized JSON. Used for
    # A/B comparison post-GEPA-optimization.
    use_baseline_picker: bool = False

    def relative_dataset(self) -> str:
        return str(self.current_dataset_h5)


# ----------------------------- regen_dataset ----------------------------- #

def _deep_set(d: Dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def _merge_datasets(old_path: Path, new_path: Path, merged_path: Path) -> Dict[str, int]:
    """Concatenate old + new HDF5 datasets into ``merged_path``.

    Delegates to ``autotokamak.data.h5io.merge_h5`` (lazy import keeps the
    orchestrator module cheap to import).
    """
    from autotokamak.data.h5io import merge_h5

    return merge_h5(old_path, new_path, merged_path)


def _refit_winner_on_pool(state: MetaState) -> Optional[Dict[str, Any]]:
    """Refit the current winner's architecture on the (grown) train pool.

    Without this, a ``regen_dataset`` action can NEVER show immediate credit:
    ``rmse_after`` re-measures the OLD winner, which by construction is
    unchanged by new data. Refitting the same model + hyperparams + PCA size
    on the enriched pool turns the regen into a competing candidate on the
    frozen shard — data enrichment gets honest, immediate credit (or none,
    if the extra samples genuinely didn't help). One fit, no search, no LLM.

    Returns a result dict (refit shard RMSE + whether it became the new
    best), or None when no winner exists yet / the refit fails.
    """
    payload = state.best_winner_payload
    if payload is None:
        return None
    try:
        from autotokamak.surrogate.dataset import load_dataset
        from autotokamak.surrogate.reduce import fit_pca, transform
        from autotokamak.surrogate.zoo import make_model

        bundle = load_dataset(state.current_dataset_h5)
        n_comp = int(payload.get("n_pca_components") or payload["pca"].n_components)
        pca = fit_pca(bundle.psi, n_components=n_comp)
        est = make_model(payload["model_name"], **payload["hyperparams"])
        est.fit(bundle.inputs, transform(pca, bundle.psi))

        candidate = {
            "estimator": est,
            "pca": pca,
            "model_name": payload["model_name"],
            "hyperparams": dict(payload["hyperparams"]),
            "n_pca_components": n_comp,
        }
        iter_idx = len(state.actions_taken)
        refit_path = state.workspace / "refits" / f"iter{iter_idx}.pkl"
        refit_path.parent.mkdir(parents=True, exist_ok=True)
        import joblib

        joblib.dump(candidate, refit_path)

        shard_rmse, became_best = _maybe_update_best(
            candidate, refit_path, state.best_surrogate_report, state
        )
        return {
            "refit_shard_rmse": shard_rmse,
            "refit_became_best": became_best,
            "refit_path": str(refit_path),
            "refit_n_samples": bundle.n_samples,
        }
    except Exception as exc:  # noqa: BLE001
        return {"refit_error": f"{type(exc).__name__}: {exc}"}


def regen_dataset(payload: RegenDatasetOverrides, state: MetaState) -> Dict[str, Any]:
    """Apply overrides and run a fresh sweep, then ENRICH the current dataset.

    The new sweep uses a per-iteration seed offset so its samples are not
    duplicates of prior iterations'. The resulting HDF5 is then concatenated
    with ``state.current_dataset_h5`` — the meta-loop's dataset only grows.
    Afterward the current winner (if any) is REFIT on the grown pool so the
    regen's value is measured immediately (see ``_refit_winner_on_pool``).
    No LLM involved.
    """
    if state.base_sweep_config is None:
        raise RuntimeError(
            "regen_dataset requires meta_config.base_sweep_config to be set"
        )
    # Apply overrides one key at a time, validating each against SweepConfig.
    # The LLM sometimes hallucinates knobs that don't exist (observed live:
    # "sampling.strategy", "sampling.focus_vars") — a single bad key must not
    # kill the whole action; drop it and record it in the result instead.
    import copy

    raw = state.base_sweep_config.model_dump(mode="json")
    overrides_applied: Dict[str, Any] = {}
    overrides_dropped: Dict[str, Any] = {}
    for k, v in payload.overrides.items():
        candidate = copy.deepcopy(raw)
        _deep_set(candidate, k, v)
        try:
            SweepConfig.model_validate(candidate)
        except Exception:  # noqa: BLE001
            overrides_dropped[k] = v
            continue
        raw = candidate
        overrides_applied[k] = v
    new_cfg = SweepConfig.model_validate(raw)

    datasets_dir = state.workspace / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    iter_idx = len(state.actions_taken)
    # Distinct seed per iteration so the LHS engine produces genuinely new
    # samples on top of the ones we already have.
    bumped_sampling = new_cfg.sampling.model_copy(
        update={"seed": int(new_cfg.sampling.seed) + iter_idx + 1}
    )
    new_cfg = new_cfg.model_copy(
        update={
            "sampling": bumped_sampling,
            "output_path": f"iter{iter_idx}_new.h5",
        }
    )

    result = run_sweep(new_cfg, datasets_dir)
    cfg_path = datasets_dir / f"iter{iter_idx}_config.yaml"
    cfg_path.write_text(yaml.safe_dump(new_cfg.model_dump(mode="json"), sort_keys=False))

    # Enrich: merge previous current dataset with the freshly-generated one.
    prior_path = Path(state.current_dataset_h5)
    new_path = Path(result.dataset_path)
    merged_path = datasets_dir / f"iter{iter_idx}_dataset.h5"
    merge_counts = _merge_datasets(prior_path, new_path, merged_path)

    state.current_dataset_h5 = merged_path
    state.base_sweep_config = new_cfg

    refit_info = _refit_winner_on_pool(state) or {}

    return {
        "kind": "regen_dataset",
        **refit_info,
        "dataset_path": str(merged_path),
        "prior_dataset_path": str(prior_path),
        "new_shard_path": str(new_path),
        "config_path": str(cfg_path),
        "n_new_requested": result.n_requested,
        "n_new_succeeded": result.n_succeeded,
        "n_new_isoflux_used": result.n_isoflux_used,
        "n_total": merge_counts["n_total"],
        "n_total_succeeded": merge_counts["n_succeeded"],
        "n_total_isoflux_used": merge_counts["n_isoflux_used"],
        "config_hash": result.config_hash,
        "seed_used": int(bumped_sampling.seed),
        "overrides_applied": overrides_applied,
        "overrides_dropped": overrides_dropped,
        "rationale": payload.rationale,
    }


# ----------------------------- enrich_active ----------------------------- #

def enrich_active(payload: EnrichActivePayload, state: MetaState) -> Dict[str, Any]:
    """ACTIVE data enrichment: solve the points where the surrogate is weak.

    Where ``regen_dataset`` appends another blind LHS batch, this action
    asks ``data.acquire`` for a targeted batch. With a trained winner in
    state, acquisition is RESIDUAL-DRIVEN (UCB over an error model fit on
    the winner's out-of-fold residuals — "based on observed model
    performance"); before any winner exists it degrades to the PCA-GP
    variance proxy. Both are feasibility-weighted, and the batch is drawn
    over the TARGET ENVELOPE bounds when the meta-loop configured one
    (``state.envelope_bounds``), else over the seed sweep box. The selected
    points go through the SAME sweep/merge/refit path as a regen, so
    downstream consumers can't tell the difference — except, ideally, in
    the shard RMSE.

    No LLM involved: the acquisition itself is deterministic given the
    current train pool, winner, and seed. The LLM's job is only to have
    CHOSEN this action (and its n_new / beta knobs) over the alternatives.
    """
    if state.base_sweep_config is None:
        raise RuntimeError(
            "enrich_active requires meta_config.base_sweep_config to be set "
            "(it supplies the sampling bounds and solver knobs)"
        )
    import numpy as np

    from autotokamak.data.acquire import select_from_dataset
    from autotokamak.data.schema import PARAM_ORDER

    cfg = state.base_sweep_config
    if state.envelope_bounds is not None:
        lows, highs = (np.asarray(b, dtype=np.float64) for b in state.envelope_bounds)
    else:
        lows = np.array([cfg.parameters[p].low for p in PARAM_ORDER], dtype=np.float64)
        highs = np.array([cfg.parameters[p].high for p in PARAM_ORDER], dtype=np.float64)

    iter_idx = len(state.actions_taken)
    acq = select_from_dataset(
        state.current_dataset_h5,
        bounds_low=lows,
        bounds_high=highs,
        n_points=int(payload.n_new),
        winner_payload=state.best_winner_payload,
        beta=float(payload.beta),
        strategy=payload.strategy,
        seed=int(state.seed) + iter_idx + 1,
        # floor=1.0 makes every feasibility weight exactly 1 — weighting off.
        # When weighting is ON we omit the kwarg so acquire.py owns its default.
        **({} if payload.feasibility_weighting else {"feasibility_floor": 1.0}),
    )

    datasets_dir = state.workspace / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    sweep_cfg = cfg.model_copy(update={"output_path": f"iter{iter_idx}_active.h5"})
    result = run_sweep(sweep_cfg, datasets_dir, X=acq.X_new)
    (datasets_dir / f"iter{iter_idx}_acquisition.json").write_text(
        json.dumps(acq.summary_dict(), indent=2, default=str)
    )

    # Enrich: same merge-and-grow path as regen_dataset.
    prior_path = Path(state.current_dataset_h5)
    new_path = Path(result.dataset_path)
    merged_path = datasets_dir / f"iter{iter_idx}_dataset.h5"
    merge_counts = _merge_datasets(prior_path, new_path, merged_path)

    state.current_dataset_h5 = merged_path

    refit_info = _refit_winner_on_pool(state) or {}

    return {
        "kind": "enrich_active",
        **refit_info,
        "acquisition": acq.summary_dict(),
        "dataset_path": str(merged_path),
        "prior_dataset_path": str(prior_path),
        "new_shard_path": str(new_path),
        "n_new_requested": result.n_requested,
        "n_new_succeeded": result.n_succeeded,
        "n_new_isoflux_used": result.n_isoflux_used,
        "n_total": merge_counts["n_total"],
        "n_total_succeeded": merge_counts["n_succeeded"],
        "n_total_isoflux_used": merge_counts["n_isoflux_used"],
        "rationale": payload.rationale,
    }


# ----------------------------- extend_search ----------------------------- #

def _frozen_shard_rmse(winner_payload: dict, state: MetaState) -> Optional[float]:
    """RMSE of ``winner_payload`` on the frozen test shard; None on any failure.

    This is the ONLY number winners are compared on — same fixed samples for
    every candidate across every iteration, regardless of how the train pool
    has grown.
    """
    if winner_payload is None or state.test_shard_h5 is None:
        return None
    try:
        from autotokamak.surrogate.dataset import load_dataset
        from autotokamak.surrogate.metrics import psi_rmse
        from autotokamak.surrogate.optuna_search import predict_with_winner

        if state.shard_bundle_cache is None:
            state.shard_bundle_cache = load_dataset(state.test_shard_h5)
        shard = state.shard_bundle_cache
        pred = predict_with_winner(winner_payload, shard.inputs)
        return float(psi_rmse(shard.psi, pred))
    except Exception:  # noqa: BLE001
        return None


def _build_overlay_prompt(
    phase2_prompt_path: Path,
    focus: ExtendSearchFocus,
    dataset_path: Path,
    workspace: Path,
    time_budget_seconds: Optional[int] = None,
) -> Path:
    """Write a copy of the Phase-2 prompt with a 'FOCUS' block injected.

    The overlay prompt has the same structure as ``surrogate_automl.yaml``
    plus a short ``FOCUS DIRECTIVE`` section the nested LLM reads. The
    overlay's ``workspace`` is the meta-iteration's sub-workspace.
    """
    raw = yaml.safe_load(phase2_prompt_path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Phase-2 prompt {phase2_prompt_path} must be a YAML mapping")

    focus_lines = ["", "FOCUS DIRECTIVE FROM META-AGENT", ""]
    if time_budget_seconds is not None:
        focus_lines.append(
            f"  HARD BUDGET: write time_budget_seconds = {int(time_budget_seconds)} "
            f"into surrogate_config.yaml. Do NOT exceed."
        )
    if focus.models_to_emphasize:
        focus_lines.append(
            "  Emphasize these models: " + ", ".join(focus.models_to_emphasize)
        )
    if focus.widen_params:
        focus_lines.append(
            "  Widen the search range for: " + ", ".join(focus.widen_params)
        )
    if focus.n_trials_hint is not None:
        focus_lines.append(
            f"  Suggested total trial budget: {focus.n_trials_hint}"
        )
    if focus.rationale:
        focus_lines.append(f"  Reason: {focus.rationale}")

    raw["problem"] = (raw.get("problem", "") or "") + "\n" + "\n".join(focus_lines)
    raw["workspace"] = str(workspace)
    # Override the dataset symlink so the nested run sees the current
    # meta-state dataset.
    symlinks = list(raw.get("symlinks", []) or [])
    symlinks = [
        s for s in symlinks
        if not (isinstance(s, dict) and s.get("dest") == "dataset.h5")
    ]
    symlinks.append({"source": str(dataset_path), "dest": "dataset.h5"})
    raw["symlinks"] = symlinks

    overlay_path = workspace / "overlay_prompt.yaml"
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return overlay_path


def extend_search(payload: ExtendSearchFocus, state: MetaState) -> Dict[str, Any]:
    """Run a Phase-2 surrogate search; update ``state.best_*`` on improvement.

    Dispatches by ``state.phase2_mode``:
      - "structured" (default): deterministic ``automl_loop`` with one typed
        LLM decision per round.
      - "codegen": legacy nested ``plan_execute_feedback`` run on an overlay
        of the Phase-2 prompt.

    Both paths compare candidate winners by their RMSE on the FROZEN test
    shard (``_frozen_shard_rmse``), never by nested-run val RMSE — val
    splits differ across iterations as the train pool grows, so they are
    not comparable.
    """
    if state.phase2_mode == "codegen":
        return _extend_search_codegen(payload, state)
    return _extend_search_structured(payload, state)


def _maybe_update_best(
    winner_payload: Optional[dict],
    winner_path: Path,
    nested_report: Optional[dict],
    state: MetaState,
) -> tuple[Optional[float], bool]:
    """Compare a candidate winner on the frozen shard; update state if better.

    Returns ``(shard_rmse, became_best)`` — the explicit flag exists so
    callers never have to reconstruct "did it win?" via float comparisons.
    """
    if winner_payload is None:
        return None, False
    shard_rmse = _frozen_shard_rmse(winner_payload, state)
    if shard_rmse is not None and shard_rmse < state.best_rmse:
        state.best_rmse = shard_rmse
        state.best_winner_payload = winner_payload
        state.best_winner_path = winner_path
        state.best_surrogate_report = nested_report
        return shard_rmse, True
    return shard_rmse, False


def _extend_search_codegen(payload: ExtendSearchFocus, state: MetaState) -> Dict[str, Any]:
    """Legacy path: ``plan_execute_feedback`` as a sub-LLM run on the Phase-2 prompt."""
    iter_idx = len(state.actions_taken)
    sub_ws = state.workspace / "surrogate_runs" / f"iter{iter_idx}"
    sub_ws.mkdir(parents=True, exist_ok=True)

    overlay_path = _build_overlay_prompt(
        phase2_prompt_path=state.phase2_prompt,
        focus=payload,
        dataset_path=state.current_dataset_h5,
        workspace=sub_ws,
        time_budget_seconds=state.phase2_time_budget_seconds,
    )

    # Programmatic invocation. Imported lazily so the orchestrator module
    # has no hard dependency on langchain/ursa at import time.
    from autotokamak.agent.runners.plan_execute_feedback import main as feedback_main

    started = time.time()
    feedback_main(
        config_path=str(overlay_path),
        cli_model=None,
        workspace_override=str(sub_ws),
        trace_enabled=True,
        experiments_dir=state.workspace / "experiments",
    )
    elapsed = time.time() - started

    # Load nested artifacts; winner comparison happens on the frozen shard.
    winner_path = sub_ws / "outputs" / "winner.pkl"
    report_path = sub_ws / "outputs" / "report.json"
    nested_rmse: Optional[float] = None
    shard_rmse: Optional[float] = None
    if winner_path.is_file() and report_path.is_file():
        import joblib

        nested_winner = joblib.load(winner_path)
        nested_report = json.loads(report_path.read_text())
        nested_rmse = float(nested_report.get("val_psi_rmse", float("inf")))
        shard_rmse, _became_best = _maybe_update_best(nested_winner, winner_path, nested_report, state)

    return {
        "kind": "extend_search",
        "mode": "codegen",
        "overlay_prompt": str(overlay_path),
        "sub_workspace": str(sub_ws),
        "elapsed_seconds": elapsed,
        "nested_val_rmse": nested_rmse,
        "shard_rmse": shard_rmse,
        "winner_path": str(winner_path) if winner_path.is_file() else None,
        "models_emphasized": list(payload.models_to_emphasize),
        "widen_params": list(payload.widen_params),
        "rationale": payload.rationale,
    }


def _extend_search_structured(payload: ExtendSearchFocus, state: MetaState) -> Dict[str, Any]:
    """Structured path: deterministic AutoML loop + typed per-round LLM decisions."""
    iter_idx = len(state.actions_taken)
    sub_ws = state.workspace / "surrogate_runs" / f"iter{iter_idx}"
    sub_ws.mkdir(parents=True, exist_ok=True)

    decision_fn = state.phase2_decision_fn
    if decision_fn is None:
        from autotokamak.agent.dspy.module import make_search_decision_fn

        decision_fn = make_search_decision_fn(state.phase2_model or "openai:gpt-5-mini")

    from autotokamak.surrogate.automl_loop import run_automl_loop

    started = time.time()
    out = run_automl_loop(
        dataset_h5=state.current_dataset_h5,
        workdir=sub_ws,
        decision_fn=decision_fn,
        max_rounds=state.phase2_max_rounds,
        time_budget_seconds=state.phase2_time_budget_seconds or 300,
        seed=state.seed,
        focus=payload.model_dump(),
        test_shard_h5=state.test_shard_h5,
    )
    elapsed = time.time() - started

    winner_path = sub_ws / "outputs" / "winner.pkl"
    shard_rmse: Optional[float] = None
    if out.get("winner") is not None and winner_path.is_file():
        import joblib

        nested_winner = joblib.load(winner_path)
        nested_report = None
        report_path = sub_ws / "outputs" / "report.json"
        if report_path.is_file():
            nested_report = json.loads(report_path.read_text())
        shard_rmse, _became_best = _maybe_update_best(nested_winner, winner_path, nested_report, state)

    return {
        "kind": "extend_search",
        "mode": "structured",
        "sub_workspace": str(sub_ws),
        "elapsed_seconds": elapsed,
        "n_rounds": out.get("n_rounds"),
        "terminated_by": out.get("terminated_by"),
        "nested_val_rmse": out.get("val_psi_rmse"),
        "shard_rmse": shard_rmse,
        "winner_path": str(winner_path) if winner_path.is_file() else None,
        "models_emphasized": list(payload.models_to_emphasize),
        "widen_params": list(payload.widen_params),
        "rationale": payload.rationale,
    }


# ----------------------------- terminate ----------------------------- #

def terminate(payload: TerminateReason, state: MetaState) -> Dict[str, Any]:
    return {
        "kind": "terminate",
        "reason": payload.reason,
        "confidence": payload.confidence,
    }


# ----------------------------- dispatch ----------------------------- #

DISPATCH = {
    "regen_dataset": regen_dataset,
    "enrich_active": enrich_active,
    "extend_search": extend_search,
    "terminate": terminate,
}


def dispatch(decision: ActionDecision, state: MetaState) -> Dict[str, Any]:
    payload = decision.selected_payload()
    if payload is None:
        raise ValueError(
            f"ActionDecision.action={decision.action} but the corresponding payload is None"
        )
    handler = DISPATCH[decision.action]
    return handler(payload, state)


__all__ = [
    "DISPATCH",
    "MetaState",
    "dispatch",
    "enrich_active",
    "extend_search",
    "regen_dataset",
    "terminate",
]
