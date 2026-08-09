# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Core utilities shared across examples, surrogate training, and agentic runners.

Submodules:
- :mod:`autotokamak.core.geometry` — LCFS construction and meshing
- :mod:`autotokamak.core.solver`   — TokaMaker setup and solve (with retry-on-isoflux-fail)
- :mod:`autotokamak.core.io`       — atomic NPZ/JSON writers and unified output paths
- :mod:`autotokamak.core.schema`   — Pydantic config models for a single equilibrium run
"""

from autotokamak.core import geometry, io, solver

__all__ = ["geometry", "io", "solver"]
