# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Phase-1 dataset generation dispatcher.

fast mode: calls autotokamak.data.sweep.run_sweep directly.
ursa mode: invokes URSA plan_execute_feedback on dataset_generation.yaml.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

import yaml

from autotokamak.pipelines._common import (
    REPO_ROOT,
    default_dataset_config,
    resolve_output_dir,
    write_manifest,
)


def run_phase1_fast(
    *,
    config: Optional[str] = None,
    n_samples: Optional[int] = None,
) -> dict:
    """Generate a dataset using the platform library (no LLM).

    Returns a dict suitable for write_manifest().
    """
    from autotokamak.data.schema import SweepConfig
    from autotokamak.data.sweep import run_sweep

    cfg_path = Path(config) if config else default_dataset_config()
    cfg = SweepConfig.from_yaml(cfg_path)

    if n_samples is not None:
        bumped = cfg.sampling.model_copy(update={"n_samples": int(n_samples)})
        cfg = cfg.model_copy(update={"sampling": bumped})

    out_dir = resolve_output_dir("phase1", "fast")
    outputs_dir = out_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # run_sweep writes cfg.output_path relative to output_dir; normalise to just the filename
    filename = Path(cfg.output_path).name
    cfg = cfg.model_copy(update={"output_path": filename})

    print(f"[phase1/fast] Sweeping {cfg.sampling.n_samples} samples → {outputs_dir / filename}")
    result = run_sweep(cfg, outputs_dir)

    manifest_extra = {
        "dataset_path": result.dataset_path,
        "n_requested": result.n_requested,
        "n_succeeded": result.n_succeeded,
        "n_isoflux_used": result.n_isoflux_used,
        "config_hash": result.config_hash,
        "config_used": str(cfg_path),
    }
    p = write_manifest(out_dir, pipeline="phase1", mode="fast", **manifest_extra)
    print(f"[phase1/fast] Done — manifest: {p}")
    return manifest_extra


def run_phase1_ursa(
    *,
    model: Optional[str] = None,
) -> dict:
    """Generate a dataset via URSA (PlanningAgent + ExecutionAgent).

    Writes code + outputs to examples/dataset_generation/ursa/.
    Returns a dict suitable for write_manifest().
    """
    out_dir = resolve_output_dir("phase1", "ursa")
    prompt_path = REPO_ROOT / "src/autotokamak/agent/prompts/dataset_generation.yaml"

    # Build an overlay prompt with workspace overridden to the ursa output dir.
    raw = yaml.safe_load(prompt_path.read_text())
    raw["workspace"] = str(out_dir)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", dir=out_dir, delete=False, prefix="overlay_"
    ) as f:
        yaml.safe_dump(raw, f, sort_keys=False)
        overlay_path = f.name

    print(f"[phase1/ursa] Invoking URSA on {prompt_path.name} → {out_dir}")

    # Import lazily — URSA/langchain not available at module load time
    from autotokamak.agent.runners.plan_execute_feedback import main as feedback_main

    feedback_main(
        config_path=overlay_path,
        cli_model=model,
        workspace_override=str(out_dir),
    )

    manifest_extra: dict = {"overlay_prompt": overlay_path}
    dataset_candidate = out_dir / "outputs" / "dataset.h5"
    if dataset_candidate.is_file():
        manifest_extra["dataset_path"] = str(dataset_candidate)

    p = write_manifest(out_dir, pipeline="phase1", mode="ursa", **manifest_extra)
    print(f"[phase1/ursa] Done — manifest: {p}")
    return manifest_extra
