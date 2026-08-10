# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Trace-loader unit tests.

Synthesizes a tiny experiments/ tree with one fake run_id directory
containing trace.json + meta_trace.json. Confirms load_meta_traces
returns the expected number of dspy.Example objects with the right
input/label fields. No LLM, no real trace.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

dspy = pytest.importorskip("dspy", reason="dspy-ai not installed")


def _write_fake_run(experiments_dir: Path, workspace: Path, run_id: str, *, score: float):
    """Synthesize one run's trace + meta_trace files."""
    run_dir = experiments_dir / run_id
    run_dir.mkdir(parents=True)
    workspace.mkdir(parents=True)

    trace = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "completed",
        "prompt": {
            "path": "src/autotokamak/agent/prompts/surrogate_meta.yaml",
            "model": "openai:gpt-5-mini",
            "workspace": str(workspace),
        },
        "score": {"total": float(score)},
    }
    (run_dir / "trace.json").write_text(json.dumps(trace))

    meta_trace = {
        "iterations": [
            {
                "iteration": 0,
                "diagnostics": {
                    "learning_curve": {"slope_log_log": -0.25, "plateau_detected": False},
                    "cross_seed_variance": {"cv": 0.07, "high_variance": False},
                },
                "decision": {
                    "action": "regen_dataset",
                    "diagnosis": "sample-bottlenecked",
                    "regen": {"overrides": {"sampling.n_samples": 64}, "rationale": "more data"},
                },
                "rmse_after": 0.18,
            },
            {
                "iteration": 1,
                "diagnostics": {
                    "learning_curve": {"slope_log_log": -0.05, "plateau_detected": True},
                },
                "decision": {
                    "action": "terminate",
                    "diagnosis": "plateaued",
                    "terminate": {"reason": "good enough", "confidence": "high"},
                },
                "rmse_after": 0.18,
            },
        ],
        "report": {"final_rmse": 0.18, "baseline_rmse": 0.25, "terminated_by": "agent"},
    }
    (workspace / "meta_trace.json").write_text(json.dumps(meta_trace))


def test_load_meta_traces_basic(tmp_path: Path):
    from autotokamak.agent.dspy.trace_loader import load_meta_traces

    experiments = tmp_path / "experiments"
    workspace = tmp_path / "ws_a"
    _write_fake_run(experiments, workspace, run_id="20260628T120000Z", score=0.85)

    examples = load_meta_traces(experiments)
    assert len(examples) == 2  # one example per iteration

    # Inputs are present
    ex0 = examples[0]
    assert "diagnostics_json" in ex0
    assert "history_summary" in ex0
    assert "state_summary" in ex0

    # Labels carry gold + provenance
    assert ex0["gold_action"] == "regen_dataset"
    assert ex0["gold_diagnosis"] == "sample-bottlenecked"
    assert ex0["run_score"] == pytest.approx(0.85)
    assert ex0["run_id"] == "20260628T120000Z"

    # The second iteration sees iteration-0's decision in its history
    ex1 = examples[1]
    assert "regen_dataset" in ex1["history_summary"]
    assert ex1["gold_action"] == "terminate"

    # inputs() returns only marked input keys
    in_keys = set(ex0.inputs().keys())
    assert in_keys == {"diagnostics_json", "history_summary", "state_summary"}


def test_load_meta_traces_skips_non_meta_runs(tmp_path: Path):
    """Runs whose prompt isn't surrogate_meta.yaml should be filtered out."""
    from autotokamak.agent.dspy.trace_loader import load_meta_traces

    experiments = tmp_path / "experiments"
    experiments.mkdir()
    # Non-meta run (Phase-1 dataset_generation)
    other = experiments / "20260628T130000Z"
    other.mkdir()
    (other / "trace.json").write_text(
        json.dumps({"run_id": "x", "prompt": {"path": "dataset_generation.yaml", "workspace": "."}})
    )
    # Meta run
    workspace = tmp_path / "ws"
    _write_fake_run(experiments, workspace, run_id="20260628T140000Z", score=0.5)

    examples = load_meta_traces(experiments)
    assert all(ex["run_id"] == "20260628T140000Z" for ex in examples)


def test_split_trainset_splits_by_run_not_iteration(tmp_path: Path):
    """Same-run iterations stay on the same split side."""
    from autotokamak.agent.dspy.trace_loader import load_meta_traces, split_trainset

    experiments = tmp_path / "experiments"
    for i in range(4):
        _write_fake_run(
            experiments,
            tmp_path / f"ws_{i}",
            run_id=f"20260628T0000{i:02d}Z",
            score=0.5 + 0.1 * i,
        )
    examples = load_meta_traces(experiments)
    train, val = split_trainset(examples, val_frac=0.25, seed=0)

    train_runs = {ex["run_id"] for ex in train}
    val_runs = {ex["run_id"] for ex in val}
    assert train_runs.isdisjoint(val_runs), "no run should appear in both splits"
    assert len(train) + len(val) == len(examples)


def test_load_meta_traces_handles_empty_dir(tmp_path: Path):
    from autotokamak.agent.dspy.trace_loader import load_meta_traces

    assert load_meta_traces(tmp_path / "nonexistent") == []
    empty = tmp_path / "empty"
    empty.mkdir()
    assert load_meta_traces(empty) == []


def test_recorded_picker_inputs_used_verbatim(tmp_path: Path):
    """Iterations carrying picker_inputs (post skew-fix) bypass reconstruction."""
    from autotokamak.agent.dspy.trace_loader import load_meta_traces

    experiments = tmp_path / "experiments"
    workspace = tmp_path / "ws"
    run_dir = experiments / "20260710T000000Z"
    run_dir.mkdir(parents=True)
    workspace.mkdir(parents=True)

    (run_dir / "trace.json").write_text(
        json.dumps(
            {
                "run_id": "20260710T000000Z",
                "prompt": {
                    "path": "src/autotokamak/agent/prompts/surrogate_meta.yaml",
                    "workspace": str(workspace),
                },
                "score": {"total": 0.7},
            }
        )
    )
    recorded = {
        "diagnostics_json": '{"SENTINEL_DIAG": 1}',
        "history_summary": '{"SENTINEL_HIST": 2}',
        "state_summary": '{"iteration": 0, "iterations_remaining": 3, '
        '"current_dataset": "train_pool.h5", "best_rmse_so_far": null}',
    }
    (workspace / "meta_trace.json").write_text(
        json.dumps(
            {
                "iterations": [
                    {
                        "iteration": 0,
                        "diagnostics": {"something": "else entirely"},
                        "decision": {"action": "terminate", "diagnosis": "d"},
                        "picker_inputs": recorded,
                        "rmse_after": None,
                    }
                ]
            }
        )
    )

    examples = load_meta_traces(experiments)
    assert len(examples) == 1
    ex = examples[0]
    # Verbatim pass-through — not reconstructed from the diagnostics dict.
    assert ex["diagnostics_json"] == recorded["diagnostics_json"]
    assert ex["history_summary"] == recorded["history_summary"]
    assert ex["state_summary"] == recorded["state_summary"]
    # The runtime state_summary shape (incl. best_rmse_so_far) is what trains.
    assert "best_rmse_so_far" in ex["state_summary"]


def test_runtime_and_recorded_inputs_are_identical():
    """picker_inputs_from_runtime is deterministic: the strings the picker
    consumes and the strings the runner records are the same call."""
    from types import SimpleNamespace

    from autotokamak.agent.dspy.picker_inputs import picker_inputs_from_runtime

    meta_config = SimpleNamespace(max_iterations=3)
    state = SimpleNamespace(
        best_rmse=float("inf"),
        rmse_history=[],
        current_dataset_h5="datasets/train_pool.h5",
    )
    diagnostics = {"learning_curve": {"plateau_detected": False}}

    a = picker_inputs_from_runtime(meta_config, state, diagnostics, [])
    b = picker_inputs_from_runtime(meta_config, state, diagnostics, [])
    assert a == b
    assert '"best_rmse_so_far": null' in a["state_summary"]
    assert '"iterations_remaining": 3' in a["state_summary"]
