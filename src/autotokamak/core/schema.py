# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Pydantic v2 config models for equilibrium runs.

Replaces the ad-hoc ``_m``, ``_req``, ``_num``, ``_int`` helpers in the
config-driven runner with a typed schema. Benefits:

- IDE autocomplete on config fields
- Field-level error messages instead of generic ``Missing required key``
- ``model_dump()`` round-trips to a clean dict for hashing / provenance

Dataset-sweep configs have their own schema in :mod:`autotokamak.data.schema`
(``SweepConfig``); this module only covers a single equilibrium run.

Usage::

    from autotokamak.core.schema import EquilibriumConfig
    cfg = EquilibriumConfig.from_yaml("discretization_config.yaml")
    print(cfg.boundary.r0)
    raw_dict = cfg.model_dump()  # for legacy runners that expect dicts
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


# ----------------------------- equation ----------------------------- #

class EquationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Literal["gs"] = Field(description="Equation name. Currently only 'gs' is supported.")


# ----------------------------- boundary ----------------------------- #

class BoundaryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["isoflux"] = Field(
        default="isoflux",
        description="Boundary construction method. Only 'isoflux' (analytic D-shape) for now.",
    )
    npts: int = Field(gt=0, le=1000, description="Number of points on the LCFS contour.")
    r0: float = Field(gt=0, description="Major radius of LCFS center (m).")
    z0: float = Field(description="Vertical center of LCFS (m).")
    a: float = Field(gt=0, description="Minor radius (m).")
    kappa: float = Field(ge=0.5, le=3.0, description="Elongation (1.0 = circle).")
    delta: float = Field(ge=-1.0, le=1.0, description="Triangularity.")


# ----------------------------- mesh ----------------------------- #

class MeshRegion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = "plasma"
    type: Literal["plasma"] = "plasma"
    dx: float = Field(gt=0, description="Target triangle edge length (m).")
    tag: Optional[str] = None  # passed through unused; some runners include it


class MeshConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    method: Optional[Literal["gs_domain"]] = "gs_domain"
    regions: List[MeshRegion] = Field(min_length=1)


# ----------------------------- solver ----------------------------- #

class SolverConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order: int = Field(ge=1, le=3, description="FE polynomial order (1=linear, 2=quadratic).")
    F0: float = Field(description="Toroidal field constant F0 (Tm).")
    full_domain: bool = False
    maxits: Optional[int] = Field(default=None, ge=1, description="Nonlinear iteration cap.")
    free_boundary: bool = False


# ----------------------------- targets ----------------------------- #

class TargetsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    Ip: Optional[float] = Field(default=None, description="Total plasma current (A).")
    Ip_ratio: Optional[float] = None
    pax: Optional[float] = None
    estore: Optional[float] = None
    R0: Optional[float] = None
    V0: Optional[float] = None

    @model_validator(mode="after")
    def _at_least_one(self) -> "TargetsConfig":
        if not any(
            v is not None
            for v in (self.Ip, self.Ip_ratio, self.pax, self.estore, self.R0, self.V0)
        ):
            raise ValueError(
                "targets: at least one of Ip, Ip_ratio, pax, estore, R0, V0 must be set"
            )
        return self


# ----------------------------- init psi ----------------------------- #

class InitPsiConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    method: Literal["tokamaker_default", "isoflux"] = "tokamaker_default"


# ----------------------------- outputs ----------------------------- #

class OutputsConfig(BaseModel):
    model_config = ConfigDict(extra="allow")  # legacy YAMLs include misc extras
    out_dir: str = "outputs"
    mesh_png: str = "mesh.png"
    psi_png: str = "psi.png"


# ----------------------------- top-level ----------------------------- #

class EquilibriumConfig(BaseModel):
    """One equilibrium run: boundary + mesh + solver + targets."""

    model_config = ConfigDict(extra="allow")  # let legacy fields (meta, case_id, etc.) pass

    equation: EquationConfig
    boundary: BoundaryConfig
    mesh: MeshConfig
    solver: SolverConfig
    targets: TargetsConfig
    init_psi: InitPsiConfig = Field(default_factory=InitPsiConfig)
    outputs: OutputsConfig = Field(default_factory=OutputsConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EquilibriumConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict):
            raise ValueError("Top-level YAML must be a mapping/object.")
        return cls.model_validate(raw)


__all__ = [
    "BoundaryConfig",
    "EquationConfig",
    "EquilibriumConfig",
    "InitPsiConfig",
    "MeshConfig",
    "MeshRegion",
    "OutputsConfig",
    "SolverConfig",
    "TargetsConfig",
]
