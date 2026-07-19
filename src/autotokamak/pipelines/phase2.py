"""Phase-2 surrogate AutoML dispatcher.

fast mode: calls autotokamak.surrogate.automl_loop.run_automl_loop directly.
ursa mode: invokes URSA plan_execute_feedback on surrogate_automl.yaml.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

import yaml

from autotokamak.pipelines._common import (
    REPO_ROOT,
    default_dataset_path,
    resolve_output_dir,
    write_manifest,
)


def run_phase2_fast(
    *,
    dataset: Optional[str] = None,
    time_budget: int = 600,
    model: Optional[str] = "openai:gpt-5-mini",
    max_rounds: int = 4,
    seed: int = 0,
) -> dict:
    """Train a surrogate using the platform library (Optuna + DSPy, no agent code-gen).

    Returns a dict suitable for write_manifest().
    """
    from autotokamak.agent.dspy.module import make_search_decision_fn
    from autotokamak.surrogate.automl_loop import run_automl_loop

    out_dir = resolve_output_dir("phase2", "fast")

    dataset_h5 = Path(dataset) if dataset else default_dataset_path("fast")
    if not dataset_h5.is_file():
        # Fallback: try current fast output, then canonical examples location
        fallback = REPO_ROOT / "examples" / "dataset_generation" / "outputs" / "dataset.h5"
        if fallback.is_file():
            dataset_h5 = fallback
        else:
            raise FileNotFoundError(
                f"No dataset found at {dataset_h5}. "
                "Run `python -m autotokamak.pipelines phase1 --mode fast` first."
            )

    print(f"[phase2/fast] Dataset: {dataset_h5}")
    print(f"[phase2/fast] Output:  {out_dir}")

    decision_fn = make_search_decision_fn(model or "openai:gpt-5-mini")
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
        "winner_model": result.get("winner", {}).get("model_name") if result.get("winner") else None,
        "val_psi_rmse": result.get("val_psi_rmse"),
        "test_psi_rmse": result.get("test_psi_rmse"),
        "baseline_mean_psi_rmse": result.get("baseline_mean_psi_rmse"),
        "terminated_by": result.get("terminated_by"),
        "report_path": result.get("report_path"),
        "winner_path": result.get("winner_path"),
        "time_budget_seconds": time_budget,
        "model": model,
    }
    p = write_manifest(out_dir, pipeline="phase2", mode="fast", **manifest_extra)
    print(f"[phase2/fast] Done — manifest: {p}")
    return manifest_extra


def run_phase2_ursa(
    *,
    dataset: Optional[str] = None,
    model: Optional[str] = None,
) -> dict:
    """Train a surrogate via URSA (writes run_surrogate_automl.py from scratch).

    Writes agent-authored code + outputs to examples/surrogate_automl/ursa/.
    """
    out_dir = resolve_output_dir("phase2", "ursa")
    prompt_path = REPO_ROOT / "src/autotokamak/agent/prompts/surrogate_automl.yaml"

    raw = yaml.safe_load(prompt_path.read_text())
    raw["workspace"] = str(out_dir)

    # If a specific dataset was given, override the symlink so the agent sees it.
    if dataset:
        dataset_h5 = Path(dataset).resolve()
        symlinks = list(raw.get("symlinks", []) or [])
        symlinks = [s for s in symlinks if not (isinstance(s, dict) and s.get("dest") == "dataset.h5")]
        symlinks.append({"source": str(dataset_h5), "dest": "dataset.h5"})
        raw["symlinks"] = symlinks

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", dir=out_dir, delete=False, prefix="overlay_"
    ) as f:
        yaml.safe_dump(raw, f, sort_keys=False)
        overlay_path = f.name

    print(f"[phase2/ursa] Invoking URSA on {prompt_path.name} → {out_dir}")

    # Import lazily — URSA/langchain not available at module load time
    from autotokamak.agent.runners.plan_execute_feedback import main as feedback_main

    feedback_main(
        config_path=overlay_path,
        cli_model=model,
        workspace_override=str(out_dir),
    )

    manifest_extra: dict = {"overlay_prompt": overlay_path}
    report_candidate = out_dir / "outputs" / "report.json"
    if report_candidate.is_file():
        import json
        try:
            manifest_extra["report"] = json.loads(report_candidate.read_text())
        except Exception:
            manifest_extra["report_path"] = str(report_candidate)

    p = write_manifest(out_dir, pipeline="phase2", mode="ursa", **manifest_extra)
    print(f"[phase2/ursa] Done — manifest: {p}")
    return manifest_extra
