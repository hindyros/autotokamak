"""Meta-loop dispatcher.

fast mode: meta_loop.run() with phase2_mode="structured" (Optuna library, DSPy decisions).
ursa mode: hybrid — meta-agent decisions via DSPy, nested Phase-2 via URSA codegen.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

from autotokamak.pipelines._common import (
    REPO_ROOT,
    resolve_output_dir,
    write_manifest,
)


def run_meta(
    *,
    mode: str,
    max_iterations: int = 3,
    n_samples: Optional[int] = None,
    time_budget: int = 600,
    model: Optional[str] = None,
    dataset: Optional[str] = None,
    target_rmse: Optional[float] = None,
    target_accuracy_pct: Optional[float] = None,
    target_worst_cell_accuracy_pct: Optional[float] = None,
) -> dict:
    """Dispatch the meta-loop in either fast or ursa mode.

    Both modes use the same meta_loop.run() entry point; the difference is
    phase2_mode_override: "structured" (fast) or "codegen" (ursa hybrid).
    """
    if mode not in ("fast", "ursa"):
        raise ValueError(f"mode must be 'fast' or 'ursa', got {mode!r}")

    out_dir = resolve_output_dir("meta", mode)
    phase2_mode = "structured" if mode == "fast" else "codegen"

    sys.path.insert(0, str(REPO_ROOT / "src" / "autotokamak"))
    from agent.runners.meta_loop import run as meta_run  # type: ignore[import-not-found]

    prompt_path = REPO_ROOT / "src/autotokamak/agent/prompts/surrogate_meta.yaml"

    print(f"[meta/{mode}] Output: {out_dir}  phase2_mode={phase2_mode}")

    started = time.time()
    report = meta_run(
        config_path=str(prompt_path),
        workspace_override=str(out_dir),
        phase2_mode_override=phase2_mode,
        max_iterations_override=max_iterations,
        n_samples_override=n_samples,
        phase2_time_budget_override=time_budget,
        model_override=model,
        target_rmse_override=target_rmse,
        target_accuracy_pct_override=target_accuracy_pct,
        target_worst_cell_accuracy_pct_override=target_worst_cell_accuracy_pct,
    )

    elapsed = time.time() - started
    manifest_extra = {
        "elapsed_seconds": round(elapsed, 1),
        "n_iterations": getattr(report, "n_iterations", None),
        "terminated_by": getattr(report, "terminated_by", None),
        "final_rmse": getattr(report, "final_rmse", None),
        "baseline_rmse": getattr(report, "baseline_rmse", None),
        "final_accuracy_pct": getattr(report, "final_accuracy_pct", None),
        "final_worst_cell_accuracy_pct": getattr(report, "final_worst_cell_accuracy_pct", None),
        "winner_model_name": getattr(report, "winner_model_name", None),
        "phase2_mode": phase2_mode,
        "max_iterations": max_iterations,
        "time_budget_seconds": time_budget,
    }
    p = write_manifest(out_dir, pipeline="meta", mode=mode, **manifest_extra)
    print(f"[meta/{mode}] Done in {elapsed:.0f}s — manifest: {p}")
    return manifest_extra
