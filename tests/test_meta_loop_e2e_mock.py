# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""End-to-end meta-loop test with a scripted action picker.

Bypasses the LLM by supplying a hand-written ``ActionPicker`` to
``meta_loop.run``. Verifies that:
- the meta workspace is created with the expected layout
- each iteration writes diagnostics/action/result files
- the terminate action breaks the loop
- the final report.json + meta_trace.json are valid
- the meta scorer returns a non-zero total

Does NOT exercise the nested Phase-2 LLM call (would require a real
``plan_execute_feedback`` run, ~10 min and ~$1-5). The ``extend_search``
action is exercised in a separate test that monkey-patches the nested call
to inject a pre-computed winner.pkl + report.json.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_DATASET = REPO_ROOT / "examples" / "dataset_generation" / "outputs" / "dataset.h5"
PHASE2_PROMPT = REPO_ROOT / "src" / "autotokamak" / "agent" / "prompts" / "surrogate_automl.yaml"


@pytest.fixture
def meta_config_yaml(tmp_path: Path) -> Path:
    """Write a tiny meta_config.yaml pointing at the real Phase-1 dataset."""
    cfg_path = tmp_path / "meta_config.yaml"
    cfg_path.write_text(
        "max_iterations: 2\n"
        "seed: 0\n"
        f"initial_dataset_h5: {REAL_DATASET}\n"
        f"base_sweep_config: {REPO_ROOT}/examples/dataset_generation/dataset_config.yaml\n"
        f"phase2_prompt: {PHASE2_PROMPT}\n"
        f"workspace: {tmp_path / 'meta_ws'}\n"
        "model: openai:gpt-5.2\n"
    )
    return cfg_path


def _make_picker(decisions: list[dict]):
    """Return an ActionPicker that yields the supplied decisions in order."""
    from autotokamak.agent.orchestrator.schema import ActionDecision

    materialized = [ActionDecision.model_validate(d) for d in decisions]
    counter = {"i": 0}

    def picker(meta_config, state, diagnostics, history):
        d = materialized[counter["i"]]
        counter["i"] += 1
        return d

    return picker


@pytest.mark.skipif(
    not REAL_DATASET.is_file(),
    reason=f"Phase-1 dataset not present at {REAL_DATASET}",
)
def test_meta_loop_terminate_path(meta_config_yaml: Path, tmp_path: Path):
    """Simplest path: agent immediately terminates."""
    # Make src/autotokamak runner imports work without installing the package.
    sys.path.insert(0, str(REPO_ROOT / "src" / "autotokamak"))
    try:
        from autotokamak.agent.runners.meta_loop import run
    finally:
        sys.path.pop(0)

    picker = _make_picker(
        [
            {
                "action": "terminate",
                "terminate": {"reason": "baseline is good enough", "confidence": "high"},
                "diagnosis": "good enough for the smoke test",
            }
        ]
    )

    report = run(
        config_path=str(meta_config_yaml),
        pick_action=picker,
        trace_enabled=False,
    )
    assert report.terminated_by == "agent"
    assert report.n_iterations == 1


@pytest.mark.skipif(
    not REAL_DATASET.is_file(),
    reason=f"Phase-1 dataset not present at {REAL_DATASET}",
)
def test_meta_loop_regen_then_terminate(meta_config_yaml: Path, tmp_path: Path):
    """Exercises regen_dataset action without invoking the Phase-2 sub-LLM."""
    sys.path.insert(0, str(REPO_ROOT / "src" / "autotokamak"))
    try:
        from autotokamak.agent.runners.meta_loop import run
    finally:
        sys.path.pop(0)

    picker = _make_picker(
        [
            {
                "action": "regen_dataset",
                "regen": {
                    # "sampling.strategy" is a hallucinated knob (observed live)
                    # — it must be DROPPED, not kill the action.
                    "overrides": {
                        "sampling.n_samples": 3,
                        "sampling.seed": 7,
                        "sampling.strategy": "stratified",
                    },
                    "rationale": "smoke test of regen path",
                },
                "diagnosis": "sample-bottlenecked, regenerate with smaller N for speed",
            },
            {
                "action": "terminate",
                "terminate": {"reason": "regen exercised, end of test", "confidence": "high"},
                "diagnosis": "test complete",
            },
        ]
    )

    report = run(
        config_path=str(meta_config_yaml),
        pick_action=picker,
        trace_enabled=False,
    )
    assert report.terminated_by == "agent"
    assert report.n_iterations == 2

    # Workspace artifacts should exist.
    ws = Path(json.loads(meta_config_yaml.read_text().split("workspace: ", 1)[1].split("\n")[0]).strip() if False else REPO_ROOT)
    # Easier: derive it from meta_config.
    from autotokamak.agent.orchestrator.schema import MetaConfig

    mc = MetaConfig.from_yaml(meta_config_yaml)
    ws = Path(mc.workspace)
    assert (ws / "meta_trace.json").is_file()
    assert (ws / "report.json").is_file()
    assert (ws / "iterations" / "000" / "diagnostics.json").is_file()
    assert (ws / "iterations" / "000" / "action.json").is_file()
    assert (ws / "iterations" / "001" / "action.json").is_file()

    # The regen action should have produced a new dataset, applying the valid
    # overrides and dropping the hallucinated one.
    datasets_dir = ws / "datasets"
    assert any(datasets_dir.iterdir())
    result = json.loads((ws / "iterations" / "000" / "result.json").read_text())
    assert result["kind"] == "regen_dataset"
    assert result["overrides_applied"] == {"sampling.n_samples": 3, "sampling.seed": 7}
    assert result["overrides_dropped"] == {"sampling.strategy": "stratified"}


def test_meta_loop_scorer_requires_winner(tmp_path: Path):
    """A meta workspace with no winner.pkl fails the winner_predicts hard gate."""
    from autotokamak.agent.dspy.metric_meta import score_meta_run

    ws = tmp_path / "empty_meta"
    ws.mkdir()
    (ws / "report.json").write_text("{}")
    (ws / "meta_trace.json").write_text(json.dumps({"iterations": []}))

    rep = score_meta_run(ws)
    assert rep.total == 0.0
    assert not rep.all_gates_pass


# ---------------- frozen-shard invariants ----------------


def _file_hash(p: Path) -> str:
    import hashlib

    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.mark.skipif(
    not REAL_DATASET.is_file(),
    reason=f"Phase-1 dataset not present at {REAL_DATASET}",
)
def test_meta_loop_creates_frozen_shard_and_honest_report(meta_config_yaml: Path):
    """Terminate-only run: shard files exist, report has provenance fields,
    and final_rmse is None (no winner) instead of a baseline fallback."""
    sys.path.insert(0, str(REPO_ROOT / "src" / "autotokamak"))
    try:
        from autotokamak.agent.runners.meta_loop import run
    finally:
        sys.path.pop(0)

    picker = _make_picker(
        [
            {
                "action": "terminate",
                "terminate": {"reason": "shard smoke test", "confidence": "high"},
                "diagnosis": "test",
            }
        ]
    )
    report = run(config_path=str(meta_config_yaml), pick_action=picker, trace_enabled=False)

    from autotokamak.agent.orchestrator.schema import MetaConfig

    mc = MetaConfig.from_yaml(meta_config_yaml)
    ws = Path(mc.workspace)
    assert (ws / "datasets" / "train_pool.h5").is_file()
    assert (ws / "datasets" / "test_shard.h5").is_file()
    split_info = json.loads((ws / "datasets" / "split_info.json").read_text())

    assert report.final_rmse is None
    assert report.test_shard_path == str(ws / "datasets" / "test_shard.h5")
    assert report.n_test_samples == split_info["n_test"]

    # Every iteration records the exact picker inputs (train/serve skew fix).
    meta_trace = json.loads((ws / "meta_trace.json").read_text())
    for it in meta_trace["iterations"]:
        pi = it["picker_inputs"]
        assert set(pi) == {"diagnostics_json", "history_summary", "state_summary"}
        assert "best_rmse_so_far" in pi["state_summary"]

    # Shard + train pool are disjoint and load cleanly.
    from autotokamak.surrogate.dataset import load_dataset

    shard = load_dataset(ws / "datasets" / "test_shard.h5")
    pool = load_dataset(ws / "datasets" / "train_pool.h5")
    assert shard.n_samples == split_info["n_test"]
    shard_rows = {tuple(r) for r in shard.inputs.round(12)}
    pool_rows = {tuple(r) for r in pool.inputs.round(12)}
    assert shard_rows.isdisjoint(pool_rows)


@pytest.mark.skipif(
    not REAL_DATASET.is_file(),
    reason=f"Phase-1 dataset not present at {REAL_DATASET}",
)
def test_meta_loop_shard_untouched_by_regen(meta_config_yaml: Path):
    """regen_dataset grows the train pool; the frozen shard is byte-identical."""
    sys.path.insert(0, str(REPO_ROOT / "src" / "autotokamak"))
    try:
        from autotokamak.agent.runners.meta_loop import run
    finally:
        sys.path.pop(0)

    picker = _make_picker(
        [
            {
                "action": "regen_dataset",
                "regen": {
                    "overrides": {"sampling.n_samples": 3, "sampling.seed": 11},
                    "rationale": "shard invariance test",
                },
                "diagnosis": "need more data coverage",
            },
            {
                "action": "terminate",
                "terminate": {"reason": "done", "confidence": "high"},
                "diagnosis": "test complete",
            },
        ]
    )

    from autotokamak.agent.orchestrator.schema import MetaConfig

    mc = MetaConfig.from_yaml(meta_config_yaml)
    ws = Path(mc.workspace)

    # Snapshot the shard hash mid-run via a wrapper around the picker.
    hashes: list[str] = []

    def picker_with_snapshot(meta_config, state, diagnostics, history):
        shard_p = ws / "datasets" / "test_shard.h5"
        if shard_p.is_file():
            hashes.append(_file_hash(shard_p))
        return picker(meta_config, state, diagnostics, history)

    run(config_path=str(meta_config_yaml), pick_action=picker_with_snapshot, trace_enabled=False)

    final_hash = _file_hash(ws / "datasets" / "test_shard.h5")
    assert hashes, "picker never saw the shard"
    assert all(h == final_hash for h in hashes), "frozen shard changed during the run"

    # The regen merged into the train pool, not the shard: current dataset grew.
    result = json.loads((ws / "iterations" / "000" / "result.json").read_text())
    assert result["kind"] == "regen_dataset"
    split_info = json.loads((ws / "datasets" / "split_info.json").read_text())
    assert result["n_total"] == split_info["n_train_rows"] + result["n_new_requested"]


def test_refit_winner_on_pool_gives_regen_immediate_credit(tmp_path: Path):
    """After a regen grows the pool, the winner refit must compete on the
    shard — without it, rmse_after can never reflect a regen's value."""
    from tests.conftest import make_synthetic_h5

    from autotokamak.agent.orchestrator.actions import MetaState, _refit_winner_on_pool
    from autotokamak.surrogate.dataset import load_dataset
    from autotokamak.surrogate.metrics import psi_rmse
    from autotokamak.surrogate.reduce import fit_pca, transform
    from autotokamak.surrogate.optuna_search import predict_with_winner
    from autotokamak.surrogate.zoo import make_model

    pool = make_synthetic_h5(tmp_path / "pool.h5", n=16, seed=0)
    shard = make_synthetic_h5(tmp_path / "shard.h5", n=4, seed=9)

    # Winner fit on a SMALL slice of the pool (simulates pre-regen training).
    bundle = load_dataset(pool)
    small = slice(0, 6)
    pca = fit_pca(bundle.psi[small], n_components=2)
    est = make_model("poly_ridge", alpha=0.1, degree=1)
    est.fit(bundle.inputs[small], transform(pca, bundle.psi[small]))
    payload = {
        "estimator": est,
        "pca": pca,
        "model_name": "poly_ridge",
        "hyperparams": {"alpha": 0.1, "degree": 1},
        "n_pca_components": 2,
    }
    shard_bundle = load_dataset(shard)
    prior_rmse = float(
        psi_rmse(shard_bundle.psi, predict_with_winner(payload, shard_bundle.inputs))
    )

    state = MetaState(
        workspace=tmp_path / "ws",
        current_dataset_h5=pool,
        test_shard_h5=shard,
        best_winner_payload=payload,
        best_winner_path=tmp_path / "winner.pkl",
        best_surrogate_report={"winner_model_name": "poly_ridge"},
        best_rmse=prior_rmse,
        actions_taken=["regen_dataset"],
    )

    result = _refit_winner_on_pool(state)
    assert result is not None and "refit_error" not in result, result
    assert result["refit_shard_rmse"] is not None
    assert result["refit_n_samples"] == 16
    assert Path(result["refit_path"]).is_file()
    # Best-so-far can only improve or stay (comparison is on the same shard).
    assert state.best_rmse <= prior_rmse
    if result["refit_became_best"]:
        assert state.best_rmse == pytest.approx(result["refit_shard_rmse"])


def test_refit_winner_on_pool_none_without_winner(tmp_path: Path):
    from tests.conftest import make_synthetic_h5

    from autotokamak.agent.orchestrator.actions import MetaState, _refit_winner_on_pool

    state = MetaState(
        workspace=tmp_path / "ws",
        current_dataset_h5=make_synthetic_h5(tmp_path / "pool.h5", n=16),
    )
    assert _refit_winner_on_pool(state) is None


# ---------------- extend_search dispatch (structured vs codegen) ----------------


def _stub_winner_workspace(sub_ws: Path, shard_h5: Path) -> None:
    """Drop a minimal-but-real winner.pkl + report.json into sub_ws/outputs.

    The estimator is a real poly_ridge fit on the shard itself so
    predict_with_winner works when the meta loop evaluates it.
    """
    from autotokamak.surrogate.dataset import load_dataset
    from autotokamak.surrogate.reduce import fit_pca, transform
    from autotokamak.surrogate.zoo import make_model

    shard = load_dataset(shard_h5)
    pca = fit_pca(shard.psi, n_components=2)
    Y = transform(pca, shard.psi)
    est = make_model("poly_ridge", alpha=0.1, degree=1)
    est.fit(shard.inputs, Y)

    outputs = sub_ws / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(
        {
            "estimator": est,
            "pca": pca,
            "model_name": "poly_ridge",
            "hyperparams": {"alpha": 0.1, "degree": 1},
        },
        outputs / "winner.pkl",
    )
    (outputs / "report.json").write_text(
        json.dumps(
            {
                "winner_model_name": "poly_ridge",
                "winner_hyperparams": {"alpha": 0.1, "degree": 1},
                "val_psi_rmse": 0.5,
                "test_psi_rmse": 0.5,
                "baseline_mean_psi_rmse": 1.0,
                "pca_n_components": 2,
                "pca_explained_var": 0.99,
                "n_total_trials": 6,
                "n_outer_rounds": 1,
                "terminated_by": "agent",
                "models_tried": ["poly_ridge"],
            }
        )
    )


@pytest.mark.skipif(
    not REAL_DATASET.is_file(),
    reason=f"Phase-1 dataset not present at {REAL_DATASET}",
)
def test_extend_search_structured_dispatch(meta_config_yaml: Path, monkeypatch):
    """phase2_mode=structured routes through run_automl_loop with the train
    pool + frozen shard, and state.best_rmse is the SHARD RMSE."""
    sys.path.insert(0, str(REPO_ROOT / "src" / "autotokamak"))
    try:
        from autotokamak.agent.runners.meta_loop import run
    finally:
        sys.path.pop(0)

    from autotokamak.agent.orchestrator import actions
    from autotokamak.agent.orchestrator.schema import MetaConfig

    mc = MetaConfig.from_yaml(meta_config_yaml)
    ws = Path(mc.workspace)

    calls: dict = {}

    def fake_run_automl_loop(**kwargs):
        calls.update(kwargs)
        sub_ws = Path(kwargs["workdir"])
        _stub_winner_workspace(sub_ws, Path(kwargs["test_shard_h5"]))
        return {
            "winner": {"winner_model_name": "poly_ridge"},
            "terminated_by": "agent",
            "n_rounds": 1,
            "val_psi_rmse": 0.5,
        }

    import autotokamak.surrogate.automl_loop as loop_mod

    monkeypatch.setattr(loop_mod, "run_automl_loop", fake_run_automl_loop)

    # Avoid constructing a real dspy.LM: the decision_fn is never invoked
    # because run_automl_loop above is faked.
    import autotokamak.agent.dspy.module as dspy_mod

    monkeypatch.setattr(
        dspy_mod, "make_search_decision_fn", lambda model: (lambda ctx: None)
    )

    picker = _make_picker(
        [
            {
                "action": "extend_search",
                "extend": {"models_to_emphasize": ["poly_ridge"], "rationale": "test"},
                "diagnosis": "edge hits need a wider search",
            },
            {
                "action": "terminate",
                "terminate": {"reason": "done", "confidence": "high"},
                "diagnosis": "test complete",
            },
        ]
    )
    report = run(config_path=str(meta_config_yaml), pick_action=picker, trace_enabled=False)

    assert calls["dataset_h5"] == ws / "datasets" / "train_pool.h5"
    assert calls["test_shard_h5"] == ws / "datasets" / "test_shard.h5"
    assert calls["focus"]["models_to_emphasize"] == ["poly_ridge"]

    result = json.loads((ws / "iterations" / "000" / "result.json").read_text())
    assert result["mode"] == "structured"
    # best_rmse must be the shard RMSE (recomputed), not the nested val 0.5.
    assert result["shard_rmse"] is not None
    assert report.final_rmse == pytest.approx(result["shard_rmse"])


@pytest.mark.skipif(
    not REAL_DATASET.is_file(),
    reason=f"Phase-1 dataset not present at {REAL_DATASET}",
)
def test_target_rmse_stops_loop_early(tmp_path: Path, monkeypatch):
    """With a target set, the loop stops after the iteration that meets it —
    remaining iteration budget is saved, terminated_by='target_reached'."""
    sys.path.insert(0, str(REPO_ROOT / "src" / "autotokamak"))
    try:
        from autotokamak.agent.runners.meta_loop import run
    finally:
        sys.path.pop(0)

    cfg_path = tmp_path / "meta_config.yaml"
    cfg_path.write_text(
        "max_iterations: 3\n"
        "seed: 0\n"
        "target_rmse_ratio: 0.9\n"   # stub winner easily beats 90% of baseline
        f"initial_dataset_h5: {REAL_DATASET}\n"
        f"phase2_prompt: {PHASE2_PROMPT}\n"
        f"workspace: {tmp_path / 'meta_ws'}\n"
        "model: openai:gpt-5.2\n"
    )

    def fake_run_automl_loop(**kwargs):
        sub_ws = Path(kwargs["workdir"])
        _stub_winner_workspace(sub_ws, Path(kwargs["test_shard_h5"]))
        return {"winner": {"winner_model_name": "poly_ridge"}, "terminated_by": "agent",
                "n_rounds": 1, "val_psi_rmse": 0.5}

    import autotokamak.surrogate.automl_loop as loop_mod
    import autotokamak.agent.dspy.module as dspy_mod

    monkeypatch.setattr(loop_mod, "run_automl_loop", fake_run_automl_loop)
    monkeypatch.setattr(dspy_mod, "make_search_decision_fn", lambda model: (lambda ctx: None))

    # Picker would happily run 3 extend_search iterations; the target must
    # cut it short after the first.
    picker = _make_picker(
        [
            {
                "action": "extend_search",
                "extend": {"rationale": f"round {i}"},
                "diagnosis": "keep searching",
            }
            for i in range(3)
        ]
    )
    report = run(config_path=str(cfg_path), pick_action=picker, trace_enabled=False)

    assert report.terminated_by == "target_reached"
    assert report.n_iterations == 1  # two iterations of budget saved
    assert report.target_rmse is not None
    assert report.final_rmse is not None and report.final_rmse <= report.target_rmse

    # The picker saw the target in its recorded inputs.
    meta_trace = json.loads((tmp_path / "meta_ws" / "meta_trace.json").read_text())
    assert '"target_rmse"' in meta_trace["iterations"][0]["picker_inputs"]["state_summary"]


@pytest.mark.skipif(
    not REAL_DATASET.is_file(),
    reason=f"Phase-1 dataset not present at {REAL_DATASET}",
)
def test_target_accuracy_pct_stops_loop_early(tmp_path: Path, monkeypatch):
    """The performance stop (target_accuracy_pct) stops the loop and records
    the achieved accuracy in the report."""
    sys.path.insert(0, str(REPO_ROOT / "src" / "autotokamak"))
    try:
        from autotokamak.agent.runners.meta_loop import run
    finally:
        sys.path.pop(0)

    cfg_path = tmp_path / "meta_config.yaml"
    cfg_path.write_text(
        "max_iterations: 3\n"
        "seed: 0\n"
        "target_accuracy_pct: 10\n"   # stub winner clears 10% error reduction
        f"initial_dataset_h5: {REAL_DATASET}\n"
        f"phase2_prompt: {PHASE2_PROMPT}\n"
        f"workspace: {tmp_path / 'meta_ws'}\n"
        "model: openai:gpt-5.2\n"
    )

    def fake_run_automl_loop(**kwargs):
        sub_ws = Path(kwargs["workdir"])
        _stub_winner_workspace(sub_ws, Path(kwargs["test_shard_h5"]))
        return {"winner": {"winner_model_name": "poly_ridge"}, "terminated_by": "agent",
                "n_rounds": 1, "val_psi_rmse": 0.5}

    import autotokamak.surrogate.automl_loop as loop_mod
    import autotokamak.agent.dspy.module as dspy_mod

    monkeypatch.setattr(loop_mod, "run_automl_loop", fake_run_automl_loop)
    monkeypatch.setattr(dspy_mod, "make_search_decision_fn", lambda model: (lambda ctx: None))

    picker = _make_picker(
        [
            {
                "action": "extend_search",
                "extend": {"rationale": f"round {i}"},
                "diagnosis": "keep searching",
            }
            for i in range(3)
        ]
    )
    report = run(config_path=str(cfg_path), pick_action=picker, trace_enabled=False)

    assert report.terminated_by == "target_reached"
    assert report.n_iterations == 1  # stopped after the first qualifying iteration
    extras = report.model_dump()
    assert extras["target_accuracy_pct"] == 10.0
    # Achieved accuracy is reported and clears the requested bar.
    assert extras["final_accuracy_pct"] is not None
    assert extras["final_accuracy_pct"] >= 10.0


@pytest.mark.skipif(
    not REAL_DATASET.is_file(),
    reason=f"Phase-1 dataset not present at {REAL_DATASET}",
)
def test_extend_search_codegen_dispatch(tmp_path: Path, monkeypatch):
    """phase2_mode=codegen still routes through plan_execute_feedback."""
    sys.path.insert(0, str(REPO_ROOT / "src" / "autotokamak"))
    try:
        from autotokamak.agent.runners.meta_loop import run
    finally:
        sys.path.pop(0)

    cfg_path = tmp_path / "meta_config.yaml"
    cfg_path.write_text(
        "max_iterations: 2\n"
        "seed: 0\n"
        "phase2_mode: codegen\n"
        f"initial_dataset_h5: {REAL_DATASET}\n"
        f"phase2_prompt: {PHASE2_PROMPT}\n"
        f"workspace: {tmp_path / 'meta_ws'}\n"
        "model: openai:gpt-5.2\n"
    )

    invoked: dict = {}

    def fake_feedback_main(config_path, cli_model, workspace_override, **kwargs):
        invoked["config_path"] = config_path
        invoked["workspace"] = workspace_override

    import autotokamak.agent.runners.plan_execute_feedback as pef

    monkeypatch.setattr(pef, "main", fake_feedback_main)

    picker = _make_picker(
        [
            {
                "action": "extend_search",
                "extend": {"rationale": "codegen path test"},
                "diagnosis": "widen the search",
            },
            {
                "action": "terminate",
                "terminate": {"reason": "done", "confidence": "high"},
                "diagnosis": "test complete",
            },
        ]
    )
    run(config_path=str(cfg_path), pick_action=picker, trace_enabled=False)

    assert "overlay_prompt.yaml" in invoked["config_path"]
    result = json.loads(
        (tmp_path / "meta_ws" / "iterations" / "000" / "result.json").read_text()
    )
    assert result["mode"] == "codegen"
