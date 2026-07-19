"""Every public submodule of autotokamak must import without side effects."""

import importlib


SUBMODULES = [
    "autotokamak",
    "autotokamak.core",
    "autotokamak.core.geometry",
    "autotokamak.core.solver",
    "autotokamak.core.io",
    "autotokamak.core.schema",
    "autotokamak.agent",
    "autotokamak.agent.runners",
    "autotokamak.agent.runners.config",
    "autotokamak.data",
    "autotokamak.pipelines",
    "autotokamak.surrogate",
]


def test_all_submodules_import():
    for name in SUBMODULES:
        mod = importlib.import_module(name)
        assert mod is not None, f"failed to import {name}"


def test_version_attribute():
    import autotokamak

    assert hasattr(autotokamak, "__version__")
    assert autotokamak.__version__ == "0.1.0"
