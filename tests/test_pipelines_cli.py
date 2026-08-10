# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Smoke tests for the unified pipelines CLI (levels L0/L1).

Tests run in-process with mocked platform calls so they are fast and offline.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# _common
# ---------------------------------------------------------------------------

def test_resolve_output_dir_creates_path():
    from autotokamak.pipelines._common import REPO_ROOT as PR, resolve_output_dir
    d = resolve_output_dir("phase1", "L0")
    assert d == PR / "examples" / "dataset_generation" / "L0"
    assert d.is_dir()


def test_resolve_output_dir_bad_pipeline():
    from autotokamak.pipelines._common import resolve_output_dir
    with pytest.raises(ValueError, match="Unknown pipeline"):
        resolve_output_dir("bogus", "L0")


def test_resolve_output_dir_bad_level():
    from autotokamak.pipelines._common import resolve_output_dir
    with pytest.raises(ValueError, match="Unknown level"):
        resolve_output_dir("phase1", "fast")


def test_write_manifest(tmp_path):
    from autotokamak.pipelines._common import write_manifest
    p = write_manifest(tmp_path, pipeline="phase1", level="L0", n_succeeded=10)
    data = json.loads(p.read_text())
    assert data["pipeline"] == "phase1"
    assert data["level"] == "L0"
    assert data["condition"] == "L0-none"
    assert data["n_succeeded"] == 10
    assert "run_id" in data


# ---------------------------------------------------------------------------
# __main__ argparse
# ---------------------------------------------------------------------------

def test_help_exits_cleanly():
    import subprocess, sys
    r = subprocess.run(
        [sys.executable, "-m", "autotokamak.pipelines", "--help"],
        capture_output=True,
        cwd=str(REPO_ROOT),
    )
    assert r.returncode == 0
    assert b"phase1" in r.stdout
    assert b"phase2" in r.stdout
    assert b"meta" in r.stdout


def test_phase2_help():
    import subprocess, sys
    r = subprocess.run(
        [sys.executable, "-m", "autotokamak.pipelines", "phase2", "--help"],
        capture_output=True,
        cwd=str(REPO_ROOT),
    )
    assert r.returncode == 0
    assert b"--level" in r.stdout


def test_meta_help():
    import subprocess, sys
    r = subprocess.run(
        [sys.executable, "-m", "autotokamak.pipelines", "meta", "--help"],
        capture_output=True,
        cwd=str(REPO_ROOT),
    )
    assert r.returncode == 0
    assert b"--level" in r.stdout


def test_removed_mode_flag_errors_with_pointer():
    import subprocess, sys
    r = subprocess.run(
        [sys.executable, "-m", "autotokamak.pipelines", "phase1", "--mode", "ursa"],
        capture_output=True,
        cwd=str(REPO_ROOT),
    )
    assert r.returncode != 0
    assert b"autotokamak.bench" in r.stderr


# ---------------------------------------------------------------------------
# phase1 — mocked run_sweep
# ---------------------------------------------------------------------------

def test_phase1_calls_run_sweep(tmp_path):
    fake_result = SimpleNamespace(
        dataset_path=str(tmp_path / "dataset.h5"),
        n_requested=20,
        n_succeeded=20,
        n_isoflux_used=0,
        config_hash="abc123",
    )

    with (
        patch("autotokamak.pipelines.phase1.resolve_output_dir", return_value=tmp_path),
        patch("autotokamak.data.sweep.run_sweep", return_value=fake_result) as mock_sweep,
        patch("autotokamak.data.schema.SweepConfig") as mock_cfg_cls,
    ):
        mock_cfg = MagicMock()
        mock_cfg.output_path = "dataset.h5"
        mock_cfg.sampling.n_samples = 20
        mock_cfg.model_copy.return_value = mock_cfg
        mock_cfg_cls.from_yaml.return_value = mock_cfg

        from autotokamak.pipelines.phase1 import run_phase1
        result = run_phase1(n_samples=20)

    mock_sweep.assert_called_once()
    assert result["n_succeeded"] == 20
    assert (tmp_path / "manifest.json").is_file()


# ---------------------------------------------------------------------------
# meta dispatcher — mocked meta_loop.run
# ---------------------------------------------------------------------------

def _fake_report():
    return SimpleNamespace(
        n_iterations=1,
        terminated_by="iterations_cap",
        final_rmse=0.001,
        baseline_rmse=0.003,
        winner_model_name="poly_ridge",
    )


def test_meta_l0_wires_scripted_policies(tmp_path):
    calls = []

    def fake_run(**kwargs):
        calls.append(kwargs)
        return _fake_report()

    with (
        patch("autotokamak.pipelines.meta.resolve_output_dir", return_value=tmp_path),
        patch("autotokamak.agent.runners.meta_loop.run", side_effect=fake_run),
    ):
        from autotokamak.pipelines.meta import run_meta
        run_meta(level="L0", max_iterations=1, time_budget=120)

    assert len(calls) == 1
    assert calls[0]["phase2_mode_override"] == "structured"
    # L0 injects both scripted pickers so no LLM is ever configured.
    assert calls[0]["pick_action"] is not None
    assert calls[0]["phase2_decision_fn"] is not None


def test_meta_l1_uses_llm_pickers(tmp_path):
    calls = []

    def fake_run(**kwargs):
        calls.append(kwargs)
        return _fake_report()

    with (
        patch("autotokamak.pipelines.meta.resolve_output_dir", return_value=tmp_path),
        patch("autotokamak.agent.runners.meta_loop.run", side_effect=fake_run),
    ):
        from autotokamak.pipelines.meta import run_meta
        run_meta(level="L1", max_iterations=1, time_budget=120)

    assert len(calls) == 1
    assert calls[0]["phase2_mode_override"] == "structured"
    # L1 keeps the DSPy default inside meta_loop for the nested rounds.
    assert calls[0]["phase2_decision_fn"] is None


# ---------------------------------------------------------------------------
# meta_loop.run() kwargs
# ---------------------------------------------------------------------------

def test_meta_loop_run_accepts_policy_kwargs():
    import inspect
    from autotokamak.agent.runners.meta_loop import run
    sig = inspect.signature(run)
    assert "phase2_mode_override" in sig.parameters
    assert "phase2_decision_fn" in sig.parameters
    assert "pick_action" in sig.parameters


def test_meta_loop_run_rejects_invalid_mode(tmp_path):
    """phase2_mode_override must be 'structured' or 'codegen'."""
    from autotokamak.agent.runners.meta_loop import run
    with pytest.raises((ValueError, Exception)):
        run(
            config_path="nonexistent.yaml",
            phase2_mode_override="invalid",
        )
