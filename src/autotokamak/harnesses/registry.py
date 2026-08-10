# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Harness registry. Imports are lazy so one substrate's missing optional
dependency (ursa, dspy, claude-agent-sdk, ...) never blocks the others."""
from __future__ import annotations

from autotokamak.harnesses.base import Harness

# name -> "module:ClassName"
_REGISTRY: dict[str, str] = {
    "echo": "autotokamak.harnesses.echo:EchoHarness",
    "ursa": "autotokamak.harnesses.ursa:UrsaHarness",
    "dspy": "autotokamak.harnesses.dspy_harness:DspyHarness",
    "claude_sdk": "autotokamak.harnesses.claude_sdk:ClaudeSdkHarness",
    "pi": "autotokamak.harnesses.pi:PiHarness",
    "cursor": "autotokamak.harnesses.cursor:CursorHarness",
}

HARNESS_NAMES = tuple(_REGISTRY)


def get_harness(name: str) -> Harness:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown harness {name!r}. Choose from {HARNESS_NAMES}")
    module_path, _, cls_name = _REGISTRY[name].partition(":")
    import importlib

    mod = importlib.import_module(module_path)
    return getattr(mod, cls_name)()
