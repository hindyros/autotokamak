# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Phase-1 dataset generation dispatcher.

Phase-1 has no decision points, so it is level-free: it always runs the
library sweep (``autotokamak.data.sweep.run_sweep``) and writes to the L0
workspace. Agent-written dataset generation is an L2/L3 benchmark condition:
``python -m autotokamak.bench run --task benchmarks/tasks/... --harness ...``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from autotokamak.pipelines._common import (
    default_dataset_config,
    resolve_output_dir,
    write_manifest,
)


def run_phase1(
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

    out_dir = resolve_output_dir("phase1", "L0")
    outputs_dir = out_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # run_sweep writes cfg.output_path relative to output_dir; normalise to just the filename
    filename = Path(cfg.output_path).name
    cfg = cfg.model_copy(update={"output_path": filename})

    print(f"[phase1/L0] Sweeping {cfg.sampling.n_samples} samples → {outputs_dir / filename}")
    result = run_sweep(cfg, outputs_dir)

    manifest_extra = {
        "dataset_path": result.dataset_path,
        "n_requested": result.n_requested,
        "n_succeeded": result.n_succeeded,
        "n_isoflux_used": result.n_isoflux_used,
        "config_hash": result.config_hash,
        "config_used": str(cfg_path),
    }
    p = write_manifest(out_dir, pipeline="phase1", level="L0", **manifest_extra)
    print(f"[phase1/L0] Done — manifest: {p}")
    return manifest_extra
