# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Phase-2 surrogate AutoML dispatcher.

Both levels run the same deterministic ``automl_loop``; only the per-round
decision provider differs:

    L0 — scripted heuristic policy (no LLM, reproducible)
    L1 — DSPy SearchRoundPicker (LLM-typed decisions)

Agent-written Phase-2 code is an L2/L3 benchmark condition — see
``python -m autotokamak.bench``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from autotokamak.pipelines._common import (
    REPO_ROOT,
    default_dataset_path,
    resolve_output_dir,
    write_manifest,
)


def run_phase2(
    *,
    level: str,
    dataset: Optional[str] = None,
    time_budget: int = 600,
    model: Optional[str] = None,
    max_rounds: int = 4,
    seed: int = 0,
) -> dict:
    """Train a surrogate with the platform library (Optuna + policy decisions).

    Returns a dict suitable for write_manifest().
    """
    from autotokamak.policies import get_search_policy
    from autotokamak.surrogate.automl_loop import run_automl_loop

    out_dir = resolve_output_dir("phase2", level)

    dataset_h5 = Path(dataset) if dataset else default_dataset_path("L0")
    if not dataset_h5.is_file():
        # Fallback: canonical examples location from a direct sweep run
        fallback = REPO_ROOT / "examples" / "dataset_generation" / "outputs" / "dataset.h5"
        if fallback.is_file():
            dataset_h5 = fallback
        else:
            raise FileNotFoundError(
                f"No dataset found at {dataset_h5}. "
                "Run `python -m autotokamak.pipelines phase1 --level L0` first."
            )

    print(f"[phase2/{level}] Dataset: {dataset_h5}")
    print(f"[phase2/{level}] Output:  {out_dir}")

    policy_kind = "scripted" if level == "L0" else "llm"
    decision_fn = get_search_policy(policy_kind, model=model, seed=seed)
    result = run_automl_loop(
        dataset_h5=dataset_h5,
        workdir=out_dir,
        decision_fn=decision_fn,
        max_rounds=max_rounds,
        time_budget_seconds=time_budget,
        seed=seed,
    )

    manifest_extra = {
        "dataset_h5": str(dataset_h5),
        "policy": policy_kind,
        "winner_model": (result.get("winner") or {}).get("winner_model_name"),
        "val_psi_rmse": result.get("val_psi_rmse"),
        "test_psi_rmse": result.get("test_psi_rmse"),
        "baseline_mean_psi_rmse": result.get("baseline_mean_psi_rmse"),
        "terminated_by": result.get("terminated_by"),
        "report_path": result.get("report_path"),
        "winner_path": result.get("winner_path"),
        "time_budget_seconds": time_budget,
        "model": model if policy_kind == "llm" else None,
        "seed": seed,
    }
    p = write_manifest(out_dir, pipeline="phase2", level=level, **manifest_extra)
    print(f"[phase2/{level}] Done — manifest: {p}")
    return manifest_extra
