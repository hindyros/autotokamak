# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Meta-loop dispatcher.

Both levels run ``meta_loop.run()`` with ``phase2_mode="structured"``; only
the decision providers differ:

    L0 — scripted meta-action picker + scripted search policy (no LLM)
    L1 — DSPy MetaActionPicker + DSPy SearchRoundPicker

The legacy hybrid (nested Phase-2 via URSA codegen) is still reachable via
``phase2_mode: codegen`` in the meta YAML config; the CLI no longer exposes
it — agent codegen is benchmarked through ``python -m autotokamak.bench``.
"""
from __future__ import annotations

import time
from typing import Optional

from autotokamak.pipelines._common import (
    REPO_ROOT,
    resolve_output_dir,
    write_manifest,
)


def run_meta(
    *,
    level: str,
    max_iterations: int = 3,
    n_samples: Optional[int] = None,
    enrich_n_new: Optional[int] = None,
    time_budget: int = 600,
    model: Optional[str] = None,
    dataset: Optional[str] = None,
    seed: int = 0,
    target_rmse: Optional[float] = None,
    target_accuracy_pct: Optional[float] = None,
    target_worst_cell_accuracy_pct: Optional[float] = None,
) -> dict:
    """Dispatch the meta-loop at the given access level."""
    from autotokamak.agent.runners.meta_loop import run as meta_run
    from autotokamak.policies import get_meta_policy, get_search_policy

    out_dir = resolve_output_dir("meta", level)
    policy_kind = "scripted" if level == "L0" else "llm"

    pick_action = get_meta_policy(policy_kind, model=model, seed=seed)
    # L0 also replaces the nested extend_search round picker so the whole
    # loop is LLM-free; L1 keeps the DSPy default inside meta_loop.
    phase2_decision_fn = (
        get_search_policy("scripted", seed=seed) if policy_kind == "scripted" else None
    )

    prompt_path = REPO_ROOT / "src/autotokamak/agent/prompts/surrogate_meta.yaml"

    print(f"[meta/{level}] Output: {out_dir}  policy={policy_kind}")

    started = time.time()
    report = meta_run(
        config_path=str(prompt_path),
        pick_action=pick_action,
        phase2_decision_fn=phase2_decision_fn,
        workspace_override=str(out_dir),
        phase2_mode_override="structured",
        max_iterations_override=max_iterations,
        n_samples_override=n_samples,
        enrich_n_new_override=enrich_n_new,
        phase2_time_budget_override=time_budget,
        model_override=model,
        target_rmse_override=target_rmse,
        target_accuracy_pct_override=target_accuracy_pct,
        target_worst_cell_accuracy_pct_override=target_worst_cell_accuracy_pct,
    )

    elapsed = time.time() - started
    manifest_extra = {
        "elapsed_seconds": round(elapsed, 1),
        "policy": policy_kind,
        "n_iterations": getattr(report, "n_iterations", None),
        "terminated_by": getattr(report, "terminated_by", None),
        "final_rmse": getattr(report, "final_rmse", None),
        "baseline_rmse": getattr(report, "baseline_rmse", None),
        "final_accuracy_pct": getattr(report, "final_accuracy_pct", None),
        "final_worst_cell_accuracy_pct": getattr(report, "final_worst_cell_accuracy_pct", None),
        "winner_model_name": getattr(report, "winner_model_name", None),
        "max_iterations": max_iterations,
        "enrich_n_new": enrich_n_new,
        "time_budget_seconds": time_budget,
        "seed": seed,
    }
    p = write_manifest(out_dir, pipeline="meta", level=level, **manifest_extra)
    print(f"[meta/{level}] Done in {elapsed:.0f}s — manifest: {p}")
    return manifest_extra
