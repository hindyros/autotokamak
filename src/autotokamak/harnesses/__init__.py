# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Harness adapters — one per agent substrate (Axis A of the benchmark matrix).

Every adapter implements the same interface (``base.Harness``): take a
``TaskSpec`` and a workspace, run the substrate, emit a ``bench.trace``
RunTrace, return a ``RunResult``. The scoring/contract layer is therefore
substrate-agnostic; adding a new agent is one adapter module plus a
registry entry.
"""
from autotokamak.harnesses.base import Harness, RunResult
from autotokamak.harnesses.registry import HARNESS_NAMES, get_harness

__all__ = ["Harness", "RunResult", "HARNESS_NAMES", "get_harness"]
