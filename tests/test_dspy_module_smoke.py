# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Smoke tests for MetaActionPickerModule.

Uses dspy's DummyLM to avoid any real LM call. Verifies the ActionDecision
coercion handles each action type plus the graceful-fallback paths.
"""

from __future__ import annotations

import json

import pytest

dspy = pytest.importorskip("dspy", reason="dspy-ai not installed")


@pytest.fixture
def configured_lm():
    """Configure dspy with a DummyLM that returns a canned response.

    Tests parametrize what the response should be via lm._answer.
    """
    from dspy.utils.dummies import DummyLM

    yield DummyLM


def _set_lm(DummyLM, responses):
    """Configure DSPy to return the given list of response dicts in order."""
    lm = DummyLM(responses)
    dspy.configure(lm=lm)
    return lm


def test_module_predicts_terminate(configured_lm):
    from autotokamak.agent.dspy.module import MetaActionPickerModule

    _set_lm(
        configured_lm,
        [
            {
                "reasoning": "RMSE plateaued; further work is wasteful.",
                "action": "terminate",
                "diagnosis": "Plateau reached.",
                "rationale": "No further improvement expected.",
                "payload_json": '{"reason": "plateau", "confidence": "high"}',
            }
        ],
    )

    module = MetaActionPickerModule()
    decision = module.predict_action_decision(
        diagnostics_json='{"learning_curve": {"plateau_detected": true}}',
        history_summary="[]",
        state_summary='{"iteration": 0}',
    )
    assert decision.action == "terminate"
    assert decision.terminate is not None
    assert decision.terminate.reason == "plateau"
    assert decision.terminate.confidence == "high"


def test_module_predicts_regen_dataset(configured_lm):
    from autotokamak.agent.dspy.module import MetaActionPickerModule

    _set_lm(
        configured_lm,
        [
            {
                "reasoning": "Learning curve still falling.",
                "action": "regen_dataset",
                "diagnosis": "Sample-bottlenecked.",
                "rationale": "More data should help.",
                "payload_json": '{"overrides": {"sampling.n_samples": 64}}',
            }
        ],
    )
    module = MetaActionPickerModule()
    decision = module.predict_action_decision(
        diagnostics_json='{"learning_curve": {"slope_log_log": -0.3}}',
        history_summary="[]",
        state_summary='{"iteration": 0}',
    )
    assert decision.action == "regen_dataset"
    assert decision.regen is not None
    assert decision.regen.overrides == {"sampling.n_samples": 64}


def test_coerce_falls_back_on_invalid_action():
    """Direct test of the coercion logic — bypasses DSPy's Literal-type
    retry so we can deterministically exercise the invalid-action branch."""
    from autotokamak.agent.dspy.module import _coerce_to_action_decision

    pred = dspy.Prediction(
        action="do_something_wild",  # not one of our 3 actions
        diagnosis="confused",
        rationale="confused",
        payload_json="{}",
    )
    decision = _coerce_to_action_decision(pred)
    assert decision.action == "terminate"
    assert decision.terminate is not None
    assert "invalid action" in decision.terminate.reason


def test_coerce_falls_back_on_invalid_payload_json():
    """Malformed payload_json should not crash the coercion."""
    from autotokamak.agent.dspy.module import _coerce_to_action_decision

    pred = dspy.Prediction(
        action="regen_dataset",
        diagnosis="x",
        rationale="x",
        payload_json="not even valid json",
    )
    decision = _coerce_to_action_decision(pred)
    # Invalid JSON -> empty overrides dict -> still parses as valid regen
    assert decision.action == "regen_dataset"
    assert decision.regen is not None
    assert decision.regen.overrides == {}


def test_coerce_handles_extend_action():
    from autotokamak.agent.dspy.module import _coerce_to_action_decision

    pred = dspy.Prediction(
        action="extend_search",
        diagnosis="edge-hits in mlp",
        rationale="widen layer_width",
        payload_json='{"models_to_emphasize": ["mlp"], "widen_params": ["mlp.layer_width"], "n_trials_hint": 15}',
    )
    decision = _coerce_to_action_decision(pred)
    assert decision.action == "extend_search"
    assert decision.extend is not None
    assert decision.extend.models_to_emphasize == ["mlp"]
    assert decision.extend.widen_params == ["mlp.layer_width"]
    assert decision.extend.n_trials_hint == 15


def test_load_module_returns_baseline_when_no_optimized_file(tmp_path):
    from autotokamak.agent.dspy.module import MetaActionPickerModule, load_module

    module = load_module(tmp_path / "does_not_exist.json")
    assert isinstance(module, MetaActionPickerModule)


# ---------------- SearchRoundPicker coercion ----------------


def test_search_coerce_valid_run_round_fills_missing_space_entries():
    from autotokamak.agent.dspy.module import _coerce_to_round_decision
    from autotokamak.surrogate.zoo import DEFAULT_SEARCH_SPACES

    models = [
        {
            "name": "poly_ridge",
            "n_trials": 12,
            # Only override alpha; degree should be filled from defaults.
            "search_space": {"alpha": {"type": "loguniform", "low": 1e-4, "high": 10.0}},
        }
    ]
    pred = dspy.Prediction(
        action="run_round",
        models_json=json.dumps(models),
        n_pca_components=6,
        rationale="widen alpha",
    )
    decision = _coerce_to_round_decision(pred)
    assert decision.action == "run_round"
    assert len(decision.models) == 1
    m = decision.models[0]
    assert m.name == "poly_ridge"
    assert m.n_trials == 12
    assert m.search_space["alpha"].low == 1e-4
    # degree inherited from defaults
    assert m.search_space["degree"].model_dump(exclude_none=True) == {
        k: v for k, v in DEFAULT_SEARCH_SPACES["poly_ridge"]["degree"].items()
    }
    assert decision.n_pca_components == 6


def test_search_coerce_drops_unknown_model_and_clamps_trials():
    from autotokamak.agent.dspy.module import _coerce_to_round_decision

    models = [
        {"name": "transformer_xl", "n_trials": 10},  # not in the zoo -> dropped
        {"name": "kernel_ridge", "n_trials": 9999},  # clamped to 200
    ]
    pred = dspy.Prediction(
        action="run_round",
        models_json=json.dumps(models),
        n_pca_components=1000,  # clamped to 64
        rationale="",
    )
    decision = _coerce_to_round_decision(pred)
    assert [m.name for m in decision.models] == ["kernel_ridge"]
    assert decision.models[0].n_trials == 200
    assert decision.n_pca_components == 64


def test_search_coerce_malformed_models_json_falls_back_to_default_round():
    from autotokamak.agent.dspy.module import _coerce_to_round_decision

    pred = dspy.Prediction(
        action="run_round",
        models_json="not json at all",
        n_pca_components="whatever",
        rationale="hmm",
    )
    decision = _coerce_to_round_decision(pred)
    assert decision.action == "run_round"
    assert {m.name for m in decision.models} == {"poly_ridge", "kernel_ridge"}
    assert "default round" in decision.rationale


def test_search_coerce_invalid_action_terminates():
    from autotokamak.agent.dspy.module import _coerce_to_round_decision

    pred = dspy.Prediction(action="explode", models_json="[]", rationale="")
    decision = _coerce_to_round_decision(pred)
    assert decision.action == "terminate"
    assert "invalid action" in decision.rationale


def test_search_module_predicts_round_via_dummy_lm(configured_lm):
    from autotokamak.agent.dspy.module import SearchRoundPickerModule

    _set_lm(
        configured_lm,
        [
            {
                "reasoning": "Round 1: start with two cheap models.",
                "action": "run_round",
                "models_json": json.dumps(
                    [{"name": "poly_ridge", "n_trials": 5, "search_space": {}}]
                ),
                "n_pca_components": 4,
                "rationale": "initial round",
            }
        ],
    )
    module = SearchRoundPickerModule()
    decision = module.predict_round_decision(round_context_json='{"round": 1}')
    assert decision.action == "run_round"
    assert decision.models[0].name == "poly_ridge"
    assert decision.models[0].n_trials == 5
    # Empty search_space -> defaults filled in for every param.
    assert set(decision.models[0].search_space) == {"alpha", "degree"}


def test_load_search_module_baseline_when_no_optimized_file(tmp_path):
    from autotokamak.agent.dspy.module import SearchRoundPickerModule, load_search_module

    module = load_search_module(tmp_path / "missing.json")
    assert isinstance(module, SearchRoundPickerModule)
