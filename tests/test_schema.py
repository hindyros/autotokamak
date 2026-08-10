# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Pydantic schema validation: accept good configs, reject bad ones."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from autotokamak.core.schema import EquilibriumConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
BUNDLED_CFG = REPO_ROOT / "examples" / "config_driven_equilibrium" / "discretization_config.yaml"


def test_loads_bundled_config():
    cfg = EquilibriumConfig.from_yaml(BUNDLED_CFG)
    assert cfg.boundary.r0 == 0.42
    assert cfg.boundary.a == 0.15
    assert cfg.boundary.kappa == 1.4
    assert cfg.solver.order == 2
    assert cfg.targets.Ip == 120000.0


def test_rejects_non_numeric_kappa():
    bad = _good_dict()
    bad["boundary"]["kappa"] = "tall"
    with pytest.raises(ValidationError):
        EquilibriumConfig.model_validate(bad)


def test_rejects_negative_minor_radius():
    bad = _good_dict()
    bad["boundary"]["a"] = -0.1
    with pytest.raises(ValidationError):
        EquilibriumConfig.model_validate(bad)


def test_rejects_targets_all_none():
    bad = _good_dict()
    bad["targets"] = {}  # no Ip, no Ip_ratio, no pax, etc.
    with pytest.raises(ValidationError):
        EquilibriumConfig.model_validate(bad)


def test_rejects_unsupported_equation():
    bad = _good_dict()
    bad["equation"]["name"] = "mhd"
    with pytest.raises(ValidationError):
        EquilibriumConfig.model_validate(bad)


def test_extra_top_level_fields_allowed():
    """Legacy YAMLs include extras like `meta:`; those should pass through."""
    good = _good_dict()
    good["meta"] = {"oft_version_min": "1.0.0"}
    cfg = EquilibriumConfig.model_validate(good)
    assert cfg.boundary.r0 == 0.42


def _good_dict():
    return {
        "equation": {"name": "gs"},
        "boundary": {
            "type": "isoflux",
            "npts": 80,
            "r0": 0.42,
            "z0": 0.0,
            "a": 0.15,
            "kappa": 1.4,
            "delta": 0.0,
        },
        "mesh": {
            "regions": [{"name": "plasma", "type": "plasma", "dx": 0.015}],
        },
        "solver": {"order": 2, "F0": 0.10752},
        "targets": {"Ip": 120000.0},
    }
