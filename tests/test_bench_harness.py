# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Bench framework: TaskSpec, echo harness end-to-end, contract validation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from autotokamak.bench.taskspec import TaskSpec


def _task_yaml(tmp_path: Path, **overrides) -> Path:
    import yaml

    raw = {
        "task_id": "unit_test_task",
        "access_level": "L3",
        "problem": "Write hello.txt containing OK.",
        "expected_artifacts": ["hello.txt"],
        "harness_notes": {"echo": "ECHO-ONLY NOTE"},
        **overrides,
    }
    p = tmp_path / "task.yaml"
    p.write_text(yaml.safe_dump(raw))
    return p


# ------------------------------- TaskSpec --------------------------------- #

def test_taskspec_from_yaml_and_prompt_rendering(tmp_path):
    spec = TaskSpec.from_yaml(_task_yaml(tmp_path))
    assert spec.task_id == "unit_test_task"
    assert spec.access_level == "L3"
    assert spec.render_prompt("echo").endswith("ECHO-ONLY NOTE\n")
    assert spec.render_prompt("ursa") == spec.problem
    assert spec.source_path is not None and spec.source_path.is_file()


def test_taskspec_rejects_unknown_fields(tmp_path):
    with pytest.raises(Exception):
        TaskSpec.from_yaml(_task_yaml(tmp_path, bogus_field=1))


def test_taskspec_model_for_falls_back_to_default(tmp_path):
    spec = TaskSpec.from_yaml(
        _task_yaml(tmp_path, model={"default": "m-default", "ursa": "m-ursa"})
    )
    assert spec.model_for("ursa") == "m-ursa"
    assert spec.model_for("dspy") == "m-default"


# ------------------------------- registry --------------------------------- #

def test_registry_names_and_unknown():
    from autotokamak.harnesses import HARNESS_NAMES, get_harness

    assert {"echo", "ursa", "dspy", "claude_sdk", "pi", "cursor"} <= set(HARNESS_NAMES)
    with pytest.raises(ValueError, match="Unknown harness"):
        get_harness("skynet")


# ----------------------------- echo end-to-end ---------------------------- #

def test_echo_harness_full_run(tmp_path):
    from autotokamak.harnesses import get_harness

    spec = TaskSpec.from_yaml(_task_yaml(tmp_path))
    harness = get_harness("echo")
    run_dir = tmp_path / "run"
    workspace = run_dir / "workspace"

    result = harness.run(spec, workspace, run_dir=run_dir)

    assert result.status == "completed"
    assert result.condition == "L3-echo"
    assert (workspace / "hello.txt").exists()
    assert (workspace / "prompt.txt").read_text().endswith("ECHO-ONLY NOTE\n")
    trace = json.loads(Path(result.trace_path).read_text())
    assert trace["status"] == "completed"
    assert trace["prompt"]["harness"] == "echo"
    assert trace["schema_version"] == 1


# ------------------------------- contract --------------------------------- #

def _write_valid_workspace(ws: Path) -> None:
    """A minimal workspace satisfying the full deliverable contract."""
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "report.json").write_text(json.dumps({
        "n_solves_attempted": 10,
        "n_solves_succeeded": 9,
        "metrics": {
            "test_rel_l2": {"mean": 0.1, "median": 0.09, "p90": 0.2},
            "baseline_rel_l2": {"mean": 1.0, "median": 1.0, "p90": 1.1},
        },
    }))
    (ws / "README.md").write_text("readme")
    (ws / "predict.py").write_text(
        "import argparse, json\n"
        "import numpy as np\n"
        "p = argparse.ArgumentParser()\n"
        "p.add_argument('--input'); p.add_argument('--output')\n"
        "a = p.parse_args()\n"
        "n = len(json.load(open(a.input)))\n"
        "np.savez(a.output, psi=np.zeros((n, 96, 64)),\n"
        "         R=np.linspace(0.15, 0.80, 64), Z=np.linspace(-0.40, 0.40, 96))\n"
    )


def test_contract_passes_on_valid_workspace(tmp_path):
    from autotokamak.bench.contract import validate_deliverables

    spec = TaskSpec.from_yaml(_task_yaml(
        tmp_path, expected_artifacts=["report.json", "predict.py", "README.md"]
    ))
    ws = tmp_path / "ws"
    _write_valid_workspace(ws)
    report = validate_deliverables(ws, spec)
    assert report.passed, report.to_dict()


def test_contract_flags_missing_artifact_and_bad_shape(tmp_path):
    from autotokamak.bench.contract import validate_deliverables

    spec = TaskSpec.from_yaml(_task_yaml(
        tmp_path, expected_artifacts=["report.json", "predict.py", "README.md"]
    ))
    ws = tmp_path / "ws"
    _write_valid_workspace(ws)
    (ws / "README.md").unlink()
    (ws / "predict.py").write_text(
        "import argparse, json\n"
        "import numpy as np\n"
        "p = argparse.ArgumentParser()\n"
        "p.add_argument('--input'); p.add_argument('--output')\n"
        "a = p.parse_args()\n"
        "n = len(json.load(open(a.input)))\n"
        "np.savez(a.output, psi=np.zeros((n, 64, 96)),\n"  # transposed!
        "         R=np.linspace(0.15, 0.80, 64), Z=np.linspace(-0.40, 0.40, 96))\n"
    )
    report = validate_deliverables(ws, spec)
    assert not report.passed
    assert report.gates["artifact:README.md"] is False
    assert report.gates["predict_shape"] is False


def test_contract_l3_import_audit(tmp_path):
    from autotokamak.bench.contract import validate_deliverables

    spec = TaskSpec.from_yaml(_task_yaml(tmp_path, expected_artifacts=[]))
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "cheater.py").write_text("from autotokamak.core import solver\n")
    report = validate_deliverables(ws, spec, run_predict_check=False)
    assert report.gates["no_autotokamak_import"] is False
    assert "cheater.py" in report.notes["no_autotokamak_import"]


def test_rel_l2_ignores_nan_ground_truth():
    from autotokamak.bench.contract import rel_l2_errors

    true = np.ones((1, 4, 4))
    true[0, 0, 0] = np.nan
    pred = np.ones((1, 4, 4)) * 2.0
    errs = rel_l2_errors(pred, true)
    # 15 finite cells, |pred-true|=1 each → sqrt(15)/sqrt(15) = 1
    assert errs.shape == (1,)
    assert errs[0] == pytest.approx(1.0)


# ------------------------------ bench CLI --------------------------------- #

def test_bench_cli_dry_run(tmp_path):
    import subprocess, sys

    task = _task_yaml(tmp_path)
    r = subprocess.run(
        [sys.executable, "-m", "autotokamak.bench", "run",
         "--task", str(task), "--harness", "echo", "--dry-run"],
        capture_output=True, text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert r.returncode == 0, r.stderr
    info = json.loads(r.stdout)
    assert info["harness"] == "echo"
    assert info["condition"] == "L3-echo"
