"""Library-level smoke test for the extracted Phase-1 sweep.

Exercises ``data.schema.SweepConfig`` validation + ``data.sweep.run_sweep`` on
a 2-sample config. No LLM; ~2-5s. Catches schema drift, HDF5 layout drift,
and the isoflux_used recording path.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _tiny_config() -> dict:
    return {
        "sampling": {"method": "lhs", "n_samples": 2, "seed": 0},
        "parameters": {
            "r0":    {"low": 0.40, "high": 0.50},
            "a":     {"low": 0.12, "high": 0.18},
            "kappa": {"low": 1.1,  "high": 1.4},
            "delta": {"low": 0.1,  "high": 0.3},
            "Ip":    {"low": 100000.0, "high": 150000.0},
        },
        "output_grid": {
            "R": {"min": 0.20, "max": 0.70, "n": 16},
            "Z": {"min": -0.30, "max": 0.30, "n": 24},
        },
        "output_path": "dataset.h5",
    }


def test_sweep_config_rejects_missing_param_key():
    from autotokamak.data.schema import SweepConfig

    bad = _tiny_config()
    del bad["parameters"]["Ip"]
    with pytest.raises(Exception):
        SweepConfig.model_validate(bad)


def test_sweep_config_rejects_inverted_bounds():
    from autotokamak.data.schema import SweepConfig

    bad = _tiny_config()
    bad["parameters"]["a"] = {"low": 0.20, "high": 0.10}
    with pytest.raises(Exception):
        SweepConfig.model_validate(bad)


def test_run_sweep_produces_phase1_compatible_h5(tmp_path: Path):
    import h5py

    from autotokamak.data.schema import SweepConfig
    from autotokamak.data.sweep import run_sweep

    cfg = SweepConfig.model_validate(_tiny_config())
    result = run_sweep(cfg, tmp_path)

    assert result.n_requested == 2
    assert result.n_succeeded >= 1
    assert Path(result.dataset_path).is_file()

    # Schema must match the Phase-1 contract the scorer + eval/data expect.
    with h5py.File(result.dataset_path, "r") as f:
        assert f["grid/R"].shape == (16,)
        assert f["grid/Z"].shape == (24,)
        assert f["outputs/psi"].shape == (2, 24, 16)
        assert f["outputs/success"].shape == (2,)
        assert f["outputs/isoflux_used"].shape == (2,)
        for p in ("r0", "a", "kappa", "delta", "Ip"):
            assert f[f"inputs/{p}"].shape == (2,)


def test_run_sweep_outputs_load_through_eval_data(tmp_path: Path):
    from autotokamak.data.schema import SweepConfig
    from autotokamak.data.sweep import run_sweep
    from autotokamak.surrogate.dataset import load_dataset

    cfg = SweepConfig.model_validate(_tiny_config())
    result = run_sweep(cfg, tmp_path)

    bundle = load_dataset(result.dataset_path)
    assert bundle.n_samples == result.n_succeeded
    assert bundle.grid_shape == (24, 16)


def test_run_sweep_output_roundtrips_through_h5io(tmp_path: Path):
    """run_sweep delegates the write to h5io — the layout owner must be able
    to read its own output (this is what the meta-loop's merge/split rely on)."""
    from autotokamak.data.h5io import read_h5_arrays
    from autotokamak.data.schema import SweepConfig
    from autotokamak.data.sweep import run_sweep

    cfg = SweepConfig.model_validate(_tiny_config())
    result = run_sweep(cfg, tmp_path)

    arrays = read_h5_arrays(result.dataset_path)
    assert arrays.n_rows == result.n_requested
    assert int(arrays.success.sum()) == result.n_succeeded
    assert arrays.psi.shape == (result.n_requested, 24, 16)
