"""Tests for the residual-driven active-learning stack.

Covers the four new pieces (design doc ``docs/active_learning_design.md``):
  - ``surrogate.metrics.per_cell_errors``   — stratified per-geometry-cell RMSE
  - ``data.envelope``                  — frozen target-envelope eval set
  - ``data.acquire`` residual path     — OOF-residual UCB acquisition
  - ``enrich_active`` wiring           — winner + envelope bounds threaded in

All synthetic, no OFT, no LLM.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import (
    fake_run_sweep_factory,
    make_picker,
    make_sweep_config as _make_sweep_config,
    smooth_psi as _smooth_psi,
    train_winner as _train_winner,
)

from autotokamak.data.schema import PARAM_ORDER, ParamBounds

REPO_ROOT = Path(__file__).resolve().parents[1]

BOUNDS_LOW = np.zeros(5)
BOUNDS_HIGH = np.ones(5)


def _import_meta_run():
    """Import meta_loop.run with the src/autotokamak runner path on sys.path."""
    sys.path.insert(0, str(REPO_ROOT / "src" / "autotokamak"))
    try:
        from autotokamak.agent.runners.meta_loop import run
    finally:
        sys.path.pop(0)
    return run


# ------------------------- per_cell_errors -------------------------------


def test_per_cell_errors_finds_worst_cell():
    from autotokamak.surrogate.metrics import per_cell_errors

    rng = np.random.default_rng(0)
    n = 64
    inputs = rng.uniform(size=(n, 5))
    true = _smooth_psi(inputs)
    pred = true.copy()
    # Corrupt predictions ONLY in the high-x0 half: those joint cells (first
    # bin index 1) must dominate the worst-cell ranking.
    bad = inputs[:, 0] > 0.5
    pred[bad] += 1.0

    out = per_cell_errors(
        inputs, true, pred, bounds_low=BOUNDS_LOW, bounds_high=BOUNDS_HIGH, n_bins=2
    )
    assert out["n_cells_total"] == 2**5
    assert 0 < out["n_cells_occupied"] <= 2**5
    assert out["worst_cell"] is not None
    # Worst cell sits in the corrupted x0-half and is worse than the aggregate.
    assert out["worst_cell"]["key"].startswith("1")
    assert out["worst_cell"]["rmse"] > out["overall_rmse"]
    # Marginals: the top x0 tercile must be worse than the bottom one.
    m = out["marginals"]["r0"]
    assert m[2]["rmse"] > m[0]["rmse"]
    # JSON round-trip (goes into report.json).
    json.dumps(out)


def test_per_cell_errors_perfect_prediction():
    from autotokamak.surrogate.metrics import per_cell_errors

    rng = np.random.default_rng(1)
    inputs = rng.uniform(size=(16, 5))
    true = _smooth_psi(inputs)
    out = per_cell_errors(
        inputs, true, true.copy(), bounds_low=BOUNDS_LOW, bounds_high=BOUNDS_HIGH
    )
    assert out["worst_cell"]["rmse"] == pytest.approx(0.0, abs=1e-12)
    assert out["mean_cell_rmse"] == pytest.approx(0.0, abs=1e-12)


# ------------------------- envelope --------------------------------------


def test_resolve_envelope_bounds_fallback_and_override():
    from autotokamak.data.envelope import EnvelopeConfig, resolve_envelope_bounds

    base = _make_sweep_config()
    env = EnvelopeConfig(parameters={"kappa": ParamBounds(low=-0.5, high=2.0)})
    lows, highs = resolve_envelope_bounds(env, base)
    k = PARAM_ORDER.index("kappa")
    assert lows[k] == -0.5 and highs[k] == 2.0
    # Every other param falls back to the seed box.
    for j, p in enumerate(PARAM_ORDER):
        if p != "kappa":
            assert lows[j] == 0.0 and highs[j] == 1.0


def test_generate_envelope_eval_h5_passthrough(tmp_path: Path):
    from tests.conftest import make_synthetic_h5

    from autotokamak.data.envelope import EnvelopeConfig, generate_envelope_eval

    existing = make_synthetic_h5(tmp_path / "env.h5", n=8)
    env = EnvelopeConfig(h5=str(existing))
    assert generate_envelope_eval(env, _make_sweep_config(), tmp_path) == existing

    with pytest.raises(FileNotFoundError):
        generate_envelope_eval(
            EnvelopeConfig(h5=str(tmp_path / "missing.h5")), _make_sweep_config(), tmp_path
        )


def test_generate_envelope_eval_generates_and_reuses(tmp_path: Path, monkeypatch):
    from autotokamak.data import sweep as sweep_mod
    from autotokamak.data.envelope import EnvelopeConfig, generate_envelope_eval

    calls: dict = {}
    monkeypatch.setattr(
        sweep_mod, "run_sweep", fake_run_sweep_factory(calls, expect_X=False)
    )

    env = EnvelopeConfig(
        n_eval=32,
        seed=7,
        parameters={"kappa": ParamBounds(low=1.0, high=2.0)},
    )
    out = generate_envelope_eval(env, _make_sweep_config(), tmp_path)
    assert out == tmp_path / "eval_envelope.h5" and out.is_file()
    # The draw goes through run_sweep's own sampler: the config it received
    # must carry the eval size/seed and the envelope (kappa override) bounds.
    cfg = calls["cfg"]
    assert cfg.sampling.n_samples == 32 and cfg.sampling.seed == 7
    assert cfg.parameters["kappa"].low == 1.0 and cfg.parameters["kappa"].high == 2.0
    assert cfg.parameters["r0"].low == 0.0 and cfg.parameters["r0"].high == 1.0
    assert calls["X"].shape == (32, 5)

    # Idempotent: second call reuses the frozen file, no regeneration.
    calls.clear()
    assert generate_envelope_eval(env, _make_sweep_config(), tmp_path) == out
    assert "X" not in calls


# ------------------------- residual acquisition --------------------------


def test_residual_ucb_shape_bounds_determinism():
    from autotokamak.data.acquire import select_acquisition_points_residual

    rng = np.random.default_rng(2)
    X = rng.uniform(size=(24, 5))
    psi = _smooth_psi(X)
    winner = _train_winner(X, psi)
    kwargs = dict(
        winner_payload=winner,
        attempted_inputs=X,
        success=np.ones(24, dtype=bool),
        psi=psi,
        bounds_low=BOUNDS_LOW,
        bounds_high=BOUNDS_HIGH,
        n_points=6,
        n_candidates=256,
        seed=11,
        beta=1.0,
    )
    res1 = select_acquisition_points_residual(**kwargs)
    res2 = select_acquisition_points_residual(**kwargs)

    assert res1.method == "residual_ucb"
    assert res1.X_new.shape == (6, 5)
    assert np.all(res1.X_new >= BOUNDS_LOW) and np.all(res1.X_new <= BOUNDS_HIGH)
    assert res1.beta == 1.0
    assert res1.residual_summary["n_oof"] == 24
    np.testing.assert_array_equal(res1.X_new, res2.X_new)
    json.dumps(res1.summary_dict())


def test_residual_ucb_targets_model_weakness():
    """A linear winner misfits a kink at x0 > 0.6; acquisition must go there."""
    from autotokamak.data.acquire import select_acquisition_points_residual

    rng = np.random.default_rng(3)
    n = 48
    X = rng.uniform(size=(n, 5))
    psi = _smooth_psi(X, seed=3)
    # Strong nonlinearity only in the high-x0 region — a degree-1 model has
    # near-zero OOF residuals elsewhere and large ones there.
    kink = 25.0 * np.clip(X[:, 0] - 0.6, 0.0, None) ** 2
    psi = psi + kink[:, None, None] * np.ones_like(psi[0])[None, :, :]
    winner = _train_winner(X, psi)

    res = select_acquisition_points_residual(
        winner_payload=winner,
        attempted_inputs=X,
        success=np.ones(n, dtype=bool),
        psi=psi,
        bounds_low=BOUNDS_LOW,
        bounds_high=BOUNDS_HIGH,
        n_points=8,
        n_candidates=512,
        seed=5,
        beta=0.5,  # mostly exploitation
    )
    assert res.method == "residual_ucb"
    # The batch concentrates in the misfit region: mean x0 well above the
    # 0.5 a uniform draw would give.
    assert res.X_new[:, 0].mean() > 0.6


def test_residual_falls_back_without_winner_or_data():
    from autotokamak.data.acquire import select_acquisition_points_residual

    rng = np.random.default_rng(4)
    X = rng.uniform(size=(24, 5))
    psi = _smooth_psi(X)

    # No winner -> proxy path, with the reason recorded.
    res = select_acquisition_points_residual(
        winner_payload=None,
        attempted_inputs=X,
        success=np.ones(24, dtype=bool),
        psi=psi,
        bounds_low=BOUNDS_LOW,
        bounds_high=BOUNDS_HIGH,
        n_points=4,
        n_candidates=128,
        seed=0,
    )
    assert res.method in {"gp_variance", "maximin_fallback"}
    assert "residual_ucb unavailable" in res.notes

    # Too few successes -> same fallback ladder.
    res2 = select_acquisition_points_residual(
        winner_payload=_train_winner(X, psi),
        attempted_inputs=X[:5],
        success=np.ones(5, dtype=bool),
        psi=psi[:5],
        bounds_low=BOUNDS_LOW,
        bounds_high=BOUNDS_HIGH,
        n_points=4,
        n_candidates=128,
        seed=0,
    )
    assert res2.method == "maximin_fallback"
    assert "residual_ucb unavailable" in res2.notes


# ------------------------- enrich_active wiring --------------------------


def test_enrich_active_uses_residual_path_with_winner(tmp_path: Path, monkeypatch):
    from tests.conftest import make_synthetic_h5

    from autotokamak.agent.orchestrator import actions
    from autotokamak.agent.orchestrator.actions import MetaState, enrich_active
    from autotokamak.agent.orchestrator.schema import EnrichActivePayload
    from autotokamak.surrogate.dataset import load_dataset

    pool = make_synthetic_h5(tmp_path / "pool.h5", n=24, seed=0)
    shard = make_synthetic_h5(tmp_path / "shard.h5", n=4, seed=9)
    calls: dict = {}
    monkeypatch.setattr(actions, "run_sweep", fake_run_sweep_factory(calls))

    bundle = load_dataset(pool)
    winner = _train_winner(bundle.inputs, bundle.psi)
    state = MetaState(
        workspace=tmp_path / "ws",
        current_dataset_h5=pool,
        base_sweep_config=_make_sweep_config(),
        test_shard_h5=shard,
        best_winner_payload=winner,
        best_winner_path=tmp_path / "winner.pkl",
        best_rmse=1e9,
    )
    result = enrich_active(EnrichActivePayload(n_new=5, beta=0.7), state)

    assert result["kind"] == "enrich_active"
    assert result["acquisition"]["method"] == "residual_ucb"
    assert result["acquisition"]["beta"] == 0.7
    assert result["acquisition"]["residual_summary"]["n_oof"] == 24
    assert calls["X"].shape == (5, 5)


def test_enrich_active_acquires_over_envelope_bounds(tmp_path: Path, monkeypatch):
    """envelope_bounds on state must widen where the batch can land."""
    from tests.conftest import make_synthetic_h5

    from autotokamak.agent.orchestrator import actions
    from autotokamak.agent.orchestrator.actions import MetaState, enrich_active
    from autotokamak.agent.orchestrator.schema import EnrichActivePayload

    pool = make_synthetic_h5(tmp_path / "pool.h5", n=16, seed=0)
    calls: dict = {}
    monkeypatch.setattr(actions, "run_sweep", fake_run_sweep_factory(calls))

    lows = np.zeros(5)
    highs = np.full(5, 2.0)  # envelope twice as wide as the seed box
    state = MetaState(
        workspace=tmp_path / "ws",
        current_dataset_h5=pool,
        base_sweep_config=_make_sweep_config(),
        envelope_bounds=(lows, highs),
    )
    enrich_active(EnrichActivePayload(n_new=20), state)
    X = calls["X"]
    assert np.all(X >= 0.0) and np.all(X <= 2.0)
    # With a maximin/GP draw over a 2x box and 20 points, some MUST leave the
    # seed [0,1] box (all train data sits inside it).
    assert np.any(X > 1.0)


def test_coerce_enrich_payload_beta():
    """DSPy coercion clamps beta into [0, 3] and defaults junk to 1.0."""
    dspy = pytest.importorskip("dspy")

    from autotokamak.agent.dspy.module import _coerce_to_action_decision

    def _coerce(payload: dict):
        return _coerce_to_action_decision(
            dspy.Prediction(
                action="enrich_active", diagnosis="d", rationale="r",
                payload_json=json.dumps(payload),
            )
        )

    assert _coerce({"n_new": 50, "beta": 0.25}).enrich.beta == 0.25
    assert _coerce({"n_new": 50, "beta": 99}).enrich.beta == 3.0
    assert _coerce({"n_new": 50, "beta": -1}).enrich.beta == 0.0
    assert _coerce({"n_new": 50, "beta": "lots"}).enrich.beta == 1.0
    assert _coerce({"n_new": 50}).enrich.beta == 1.0


def test_coerce_enrich_payload_strategy():
    """DSPy coercion validates strategy; unknown/omitted -> 'auto'."""
    dspy = pytest.importorskip("dspy")

    from autotokamak.agent.dspy.module import _coerce_to_action_decision

    def _coerce(payload: dict):
        return _coerce_to_action_decision(
            dspy.Prediction(
                action="enrich_active", diagnosis="d", rationale="r",
                payload_json=json.dumps(payload),
            )
        )

    assert _coerce({"n_new": 50, "strategy": "space_filling"}).enrich.strategy == "space_filling"
    assert _coerce({"n_new": 50, "strategy": "residual_ucb"}).enrich.strategy == "residual_ucb"
    assert _coerce({"n_new": 50, "strategy": "uncertainty"}).enrich.strategy == "uncertainty"
    assert _coerce({"n_new": 50, "strategy": "nonsense"}).enrich.strategy == "auto"
    assert _coerce({"n_new": 50}).enrich.strategy == "auto"


def test_select_from_dataset_strategy_dispatch(tmp_path: Path):
    """Each strategy routes to the expected acquisition method."""
    from tests.conftest import make_synthetic_h5

    from autotokamak.data.acquire import select_from_dataset
    from autotokamak.surrogate.dataset import load_dataset

    pool = make_synthetic_h5(tmp_path / "pool.h5", n=24, seed=0)
    bundle = load_dataset(pool)
    winner = _train_winner(bundle.inputs, bundle.psi)
    kw = dict(bounds_low=BOUNDS_LOW, bounds_high=BOUNDS_HIGH, n_points=5,
              n_candidates=128, seed=0)

    # space_filling -> maximin regardless of winner.
    r = select_from_dataset(pool, winner_payload=winner, strategy="space_filling", **kw)
    assert r.method == "maximin_fallback"
    assert "space_filling" in r.notes

    # uncertainty -> model-agnostic gp_variance, ignoring the winner.
    r = select_from_dataset(pool, winner_payload=winner, strategy="uncertainty", **kw)
    assert r.method == "gp_variance"

    # residual_ucb -> residual path when a winner is present.
    r = select_from_dataset(pool, winner_payload=winner, strategy="residual_ucb", **kw)
    assert r.method == "residual_ucb"

    # auto -> residual with winner, gp_variance without.
    assert select_from_dataset(pool, winner_payload=winner, strategy="auto", **kw).method == "residual_ucb"
    assert select_from_dataset(pool, winner_payload=None, strategy="auto", **kw).method == "gp_variance"

    with pytest.raises(ValueError, match="strategy must be one of"):
        select_from_dataset(pool, strategy="bogus", **kw)


def test_enrich_active_respects_strategy(tmp_path: Path, monkeypatch):
    """enrich_active threads payload.strategy into acquisition."""
    from tests.conftest import make_synthetic_h5

    from autotokamak.agent.orchestrator import actions
    from autotokamak.agent.orchestrator.actions import MetaState, enrich_active
    from autotokamak.agent.orchestrator.schema import EnrichActivePayload
    from autotokamak.surrogate.dataset import load_dataset

    pool = make_synthetic_h5(tmp_path / "pool.h5", n=24, seed=0)
    calls: dict = {}
    monkeypatch.setattr(actions, "run_sweep", fake_run_sweep_factory(calls))

    bundle = load_dataset(pool)
    winner = _train_winner(bundle.inputs, bundle.psi)
    state = MetaState(
        workspace=tmp_path / "ws",
        current_dataset_h5=pool,
        base_sweep_config=_make_sweep_config(),
        best_winner_payload=winner,
    )
    # Force space_filling even though a winner exists.
    result = enrich_active(EnrichActivePayload(n_new=5, strategy="space_filling"), state)
    assert result["acquisition"]["method"] == "maximin_fallback"


def test_sweep_sobol_sampling():
    """SamplingConfig accepts 'sobol'; the draw stays in-bounds and is seeded."""
    from autotokamak.data.sweep import _sample_inputs

    cfg = _make_sweep_config(sampling={"method": "sobol", "n_samples": 32, "seed": 3})
    X1 = _sample_inputs(cfg)
    X2 = _sample_inputs(cfg)
    assert X1.shape == (32, 5)
    assert np.all(X1 >= 0.0) and np.all(X1 <= 1.0)
    np.testing.assert_array_equal(X1, X2)  # seeded
    # Sobol covers the box more evenly than the seed corner: every dimension
    # should span most of [0, 1].
    assert X1.min(axis=0).max() < 0.15 and X1.max(axis=0).min() > 0.85


# ------------------------- meta-loop envelope mode ------------------------


def test_meta_loop_envelope_mode_replaces_shard(tmp_path: Path):
    """With eval_envelope.h5 set: no seed-shard split; whole dataset = pool."""
    from tests.conftest import make_synthetic_h5

    initial = make_synthetic_h5(tmp_path / "initial.h5", n=20, seed=0)
    envelope = make_synthetic_h5(tmp_path / "envelope.h5", n=8, seed=99)

    cfg_path = tmp_path / "meta.yaml"
    cfg_path.write_text(
        "max_iterations: 1\n"
        "seed: 0\n"
        f"initial_dataset_h5: {initial}\n"
        f"workspace: {tmp_path / 'ws'}\n"
        "model: openai:gpt-5.2\n"
        "eval_envelope:\n"
        f"  h5: {envelope}\n"
    )

    sys.path.insert(0, str(REPO_ROOT / "src" / "autotokamak"))
    try:
        from autotokamak.agent.runners.meta_loop import run
    finally:
        sys.path.pop(0)

    from autotokamak.agent.orchestrator.schema import ActionDecision

    def picker(meta_config, state, diagnostics, history):
        # The eval set must be the envelope file and the pool the untouched
        # initial dataset (no carve).
        assert Path(state.test_shard_h5) == envelope
        assert Path(state.current_dataset_h5) == initial
        return ActionDecision.model_validate(
            {
                "action": "terminate",
                "terminate": {"reason": "smoke", "confidence": "high"},
                "diagnosis": "envelope-mode smoke test",
            }
        )

    report = run(config_path=str(cfg_path), pick_action=picker, trace_enabled=False)
    assert report.terminated_by == "agent"
    extras = report.model_dump()
    assert extras.get("eval_mode") == "envelope"
    # Legacy shard artifacts must NOT exist in envelope mode.
    assert not (tmp_path / "ws" / "datasets" / "train_pool.h5").exists()
    assert not (tmp_path / "ws" / "datasets" / "test_shard.h5").exists()
    split_info = json.loads((tmp_path / "ws" / "datasets" / "split_info.json").read_text())
    assert split_info["mode"] == "envelope"
    assert split_info["n_test"] == 8


def test_worst_cell_accuracy_stop_requires_envelope(tmp_path: Path):
    """target_worst_cell_accuracy_pct without eval_envelope fails fast."""
    from tests.conftest import make_synthetic_h5

    initial = make_synthetic_h5(tmp_path / "initial.h5", n=20, seed=0)
    cfg_path = tmp_path / "meta.yaml"
    cfg_path.write_text(
        "max_iterations: 1\n"
        "seed: 0\n"
        f"initial_dataset_h5: {initial}\n"
        f"base_sweep_config: {REPO_ROOT}/examples/dataset_generation/dataset_config.yaml\n"
        f"workspace: {tmp_path / 'ws'}\n"
        "model: openai:gpt-5.2\n"
        "target_worst_cell_accuracy_pct: 80\n"
    )

    sys.path.insert(0, str(REPO_ROOT / "src" / "autotokamak"))
    try:
        from autotokamak.agent.runners.meta_loop import run
    finally:
        sys.path.pop(0)

    from autotokamak.agent.orchestrator.schema import ActionDecision

    def picker(mc, st, d, h):
        return ActionDecision.model_validate(
            {"action": "terminate", "terminate": {"reason": "x", "confidence": "high"},
             "diagnosis": "x"}
        )

    with pytest.raises(ValueError, match="target_worst_cell_accuracy_pct requires eval_envelope"):
        run(config_path=str(cfg_path), pick_action=picker, trace_enabled=False)


def test_worst_cell_accuracy_stops_loop_early(tmp_path: Path, monkeypatch):
    """Envelope mode + a strong winner clears the per-region bar and stops."""
    from tests.conftest import make_synthetic_h5

    from autotokamak.agent.orchestrator import actions
    from autotokamak.surrogate.dataset import load_dataset

    # A large initial pool + envelope eval on the SAME synthetic generator, so
    # a well-fit winner has low per-cell error everywhere.
    initial = make_synthetic_h5(tmp_path / "initial.h5", n=120, seed=0)
    envelope = make_synthetic_h5(tmp_path / "envelope.h5", n=40, seed=7)

    cfg_path = tmp_path / "meta.yaml"
    cfg_path.write_text(
        "max_iterations: 3\n"
        "seed: 0\n"
        f"initial_dataset_h5: {initial}\n"
        f"base_sweep_config: {REPO_ROOT}/examples/dataset_generation/dataset_config.yaml\n"
        f"workspace: {tmp_path / 'ws'}\n"
        "model: openai:gpt-5.2\n"
        "target_worst_cell_accuracy_pct: 30\n"
        "eval_envelope:\n"
        f"  h5: {envelope}\n"
        "  n_bins: 1\n"   # one joint cell -> worst-cell == aggregate, easy to clear
    )

    # Inject a winner via a fake extend_search (no Optuna / LLM).
    bundle = load_dataset(initial)
    winner = _train_winner(bundle.inputs, bundle.psi, n_comp=2)

    def fake_extend(payload, state):
        import joblib
        state.best_winner_payload = winner
        wp = tmp_path / "ws" / "winner_stub.pkl"
        wp.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(winner, wp)
        state.best_winner_path = wp
        # Register on the frozen shard so measure_test_rmse works too.
        from autotokamak.agent.orchestrator.actions import _maybe_update_best
        _maybe_update_best(winner, wp, {"winner_model_name": "poly_ridge"}, state)
        return {"kind": "extend_search", "mode": "structured"}

    monkeypatch.setitem(actions.DISPATCH, "extend_search", fake_extend)

    sys.path.insert(0, str(REPO_ROOT / "src" / "autotokamak"))
    try:
        from autotokamak.agent.runners.meta_loop import run
    finally:
        sys.path.pop(0)

    from autotokamak.agent.orchestrator.schema import ActionDecision

    picker = make_picker(
        [
            {"action": "extend_search", "extend": {"rationale": "train"}, "diagnosis": "cold start"},
            {"action": "extend_search", "extend": {"rationale": "again"}, "diagnosis": "more"},
        ]
    )

    report = run(config_path=str(cfg_path), pick_action=picker, trace_enabled=False)
    extras = report.model_dump()
    assert extras["target_worst_cell_accuracy_pct"] == 30.0
    assert extras["final_worst_cell_accuracy_pct"] is not None
    # A winner that fit the same generator clears 30% error reduction, so the
    # loop stops on the target after the first iteration (not the 3-cap).
    assert report.terminated_by == "target_reached"
    assert report.n_iterations == 1

