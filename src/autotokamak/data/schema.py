# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Pydantic schemas for the Phase-1 dataset sweep, used as a library.

These mirror the YAML the Phase-1 agent emits in
``dataset_config.yaml`` so the meta-agent can construct sweep configs
programmatically. The schema is intentionally close to what the agent's
runner expects so the two stay in sync.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


PARAM_ORDER = ("r0", "a", "kappa", "delta", "Ip")


class SamplingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    method: Literal["lhs", "uniform", "sobol"] = "lhs"
    n_samples: int = Field(ge=1, le=10_000)
    seed: int = 0


class ParamBounds(BaseModel):
    model_config = ConfigDict(extra="forbid")
    low: float
    high: float

    @model_validator(mode="after")
    def _ordered(self) -> "ParamBounds":
        if not (self.low < self.high):
            raise ValueError(f"low ({self.low}) must be < high ({self.high})")
        return self


class FixedKnobs(BaseModel):
    """The non-per-sample knobs the runner needs."""

    model_config = ConfigDict(extra="forbid")
    z0: float = 0.0
    F0: float = 0.10752
    npts: int = Field(default=80, gt=0, le=1000)
    mesh_dx: float = Field(default=0.015, gt=0.0)
    solver_order: int = Field(default=1, ge=1, le=2)
    Ip_ratio: float = 1.0
    init_psi_method: Literal["isoflux", "tokamaker_default"] = "isoflux"


class GridAxis(BaseModel):
    model_config = ConfigDict(extra="forbid")
    min: float
    max: float
    n: int = Field(ge=2, le=4096)

    @model_validator(mode="after")
    def _ordered(self) -> "GridAxis":
        if not (self.min < self.max):
            raise ValueError(f"grid axis: min ({self.min}) must be < max ({self.max})")
        return self


class OutputGrid(BaseModel):
    model_config = ConfigDict(extra="forbid")
    R: GridAxis
    Z: GridAxis


class SweepConfig(BaseModel):
    """Programmatic equivalent of the Phase-1 agent's ``dataset_config.yaml``.

    The meta-agent's ``regen_dataset`` action constructs one of these from a
    base config + an overrides dict and hands it to ``data.sweep.run_sweep``.
    """

    model_config = ConfigDict(extra="allow")
    sampling: SamplingConfig
    parameters: Dict[str, ParamBounds]
    fixed: FixedKnobs = Field(default_factory=FixedKnobs)
    output_grid: OutputGrid
    output_path: str = "dataset.h5"

    @model_validator(mode="after")
    def _param_keys_match(self) -> "SweepConfig":
        missing = set(PARAM_ORDER) - set(self.parameters)
        extra = set(self.parameters) - set(PARAM_ORDER)
        if missing:
            raise ValueError(f"parameters missing required keys: {sorted(missing)}")
        if extra:
            raise ValueError(f"parameters has unexpected keys: {sorted(extra)}")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SweepConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise ValueError("Top-level YAML must be a mapping/object.")
        return cls.model_validate(raw)


class SweepResult(BaseModel):
    """Return value of ``run_sweep``; recorded into meta-trace."""

    model_config = ConfigDict(extra="forbid")
    dataset_path: str
    n_requested: int
    n_succeeded: int
    n_isoflux_used: int
    config_hash: str


__all__ = [
    "FixedKnobs",
    "GridAxis",
    "OutputGrid",
    "PARAM_ORDER",
    "ParamBounds",
    "SamplingConfig",
    "SweepConfig",
    "SweepResult",
]
