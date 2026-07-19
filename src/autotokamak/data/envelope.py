"""Target-envelope evaluation set: frozen, full-envelope, per-region.

Active learning breaks the "carve a test shard from the seed dataset" logic:
once ``enrich_active`` samples geometries OUTSIDE the seed box, a seed-region
shard measures accuracy on the wrong distribution. The fix (design doc
``docs/active_learning_design.md`` §D2) is one eval set drawn ONCE by LHS
over the full target envelope, solved with OFT, frozen, and never grown —
every winner across every meta-iteration is scored on the same samples.

Two nested boxes:
  - seed box       — ``SweepConfig.parameters`` bounds (Phase-1 sampling)
  - target envelope — ``EnvelopeConfig.parameters`` (defaults to the seed box
    when not given, preserving legacy single-box behavior)

The envelope bounds are also what ``enrich_active`` acquires over, so the
eval distribution and the acquisition distribution agree.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from autotokamak.data.schema import PARAM_ORDER, ParamBounds, SweepConfig


class EnvelopeConfig(BaseModel):
    """``MetaConfig.eval_envelope`` block: use a pre-solved eval H5 or generate one.

    ``h5`` wins when set (no OFT solves). Otherwise ``generate_envelope_eval``
    LHS-samples ``n_eval`` points over ``parameters`` (falling back to the base
    sweep's bounds per-parameter) and solves them once.
    """

    model_config = ConfigDict(extra="forbid")
    h5: Optional[str] = Field(
        default=None,
        description="Path to an existing envelope eval HDF5 (canonical layout). "
        "When set, no generation happens.",
    )
    n_eval: int = Field(default=256, ge=8, le=10_000)
    parameters: Optional[Dict[str, ParamBounds]] = Field(
        default=None,
        description="Envelope bounds per parameter; missing keys (or None) fall "
        "back to the base sweep config's bounds.",
    )
    seed: int = Field(
        default=1234,
        description="LHS seed for the eval draw. Deliberately distinct from the "
        "sweep seed default so eval and seed samples never coincide.",
    )
    n_bins: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Joint-cell bins per dimension for per_cell_errors scoring.",
    )


def resolve_envelope_bounds(
    env: EnvelopeConfig, base_sweep: SweepConfig
) -> tuple[np.ndarray, np.ndarray]:
    """(lows, highs) in ``PARAM_ORDER``: envelope bounds, seed-box fallback per key."""
    lows, highs = [], []
    overrides = env.parameters or {}
    for p in PARAM_ORDER:
        b = overrides.get(p, base_sweep.parameters[p])
        lows.append(float(b.low))
        highs.append(float(b.high))
    return np.asarray(lows, dtype=np.float64), np.asarray(highs, dtype=np.float64)


def generate_envelope_eval(
    env: EnvelopeConfig,
    base_sweep: SweepConfig,
    output_dir: Path | str,
    *,
    filename: str = "eval_envelope.h5",
) -> Path:
    """Materialize the frozen envelope eval set; idempotent.

    If ``env.h5`` is set, that path is returned as-is (must exist). Otherwise:
    if ``output_dir/filename`` already exists it is REUSED (the eval set is
    frozen by definition — regenerating mid-project would silently change the
    metric); else ``n_eval`` LHS points over the envelope bounds are solved
    via ``run_sweep`` with an explicit ``X`` and the solver/grid knobs of the
    base sweep config.
    """
    if env.h5:
        p = Path(env.h5)
        if not p.is_file():
            raise FileNotFoundError(f"eval_envelope.h5 not found: {p}")
        return p

    output_dir = Path(output_dir)
    out_path = output_dir / filename
    if out_path.is_file():
        return out_path

    from autotokamak.data.schema import SamplingConfig
    from autotokamak.data.sweep import run_sweep

    # Reuse run_sweep's own sampler: swap in the envelope bounds and an LHS
    # draw of n_eval points, keeping the base config's solver/grid knobs.
    lows, highs = resolve_envelope_bounds(env, base_sweep)
    cfg = base_sweep.model_copy(
        update={
            "sampling": SamplingConfig(method="lhs", n_samples=env.n_eval, seed=env.seed),
            "parameters": {
                p: ParamBounds(low=float(lo), high=float(hi))
                for p, lo, hi in zip(PARAM_ORDER, lows, highs)
            },
            "output_path": filename,
        }
    )
    run_sweep(cfg, output_dir)
    return out_path


__all__ = ["EnvelopeConfig", "generate_envelope_eval", "resolve_envelope_bounds"]
