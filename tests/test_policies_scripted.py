# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""L0 scripted policies: rule coverage + determinism (same inputs → same decision)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from autotokamak.policies import get_meta_policy, get_search_policy
from autotokamak.policies.scripted import (
    make_scripted_meta_policy,
    make_scripted_search_policy,
)


def _ctx(history=None, seconds_remaining=500.0, time_budget=600):
    from autotokamak.surrogate.zoo import DEFAULT_SEARCH_SPACES

    return {
        "round": (len(history or []) + 1),
        "max_rounds": 4,
        "rounds_remaining": 4 - len(history or []),
        "budget": {
            "time_budget_seconds": time_budget,
            "elapsed_seconds": time_budget - seconds_remaining,
            "seconds_remaining": seconds_remaining,
            "trials_used": 0,
        },
        "focus": None,
        "dataset": {"n_samples": 100, "grid_shape": [96, 64], "path": "x.h5"},
        "n_pca_components_default": 8,
        "default_search_spaces": DEFAULT_SEARCH_SPACES,
        "history": history or [],
    }


def _round_entry(best_values: dict[str, float], *, edge_hit=None, n_pca=8):
    from autotokamak.surrogate.zoo import DEFAULT_SEARCH_SPACES

    best_model = min(best_values, key=best_values.get)
    per_model = {}
    for name, val in best_values.items():
        per_model[name] = {
            "best_value": val,
            "best_params": {p: spec["low"] for p, spec in DEFAULT_SEARCH_SPACES[name].items()
                            if spec["type"] != "categorical"},
            "edge_hit": (edge_hit or {}).get(name, {}),
            "n_trials": 20,
            "best_value_at_25pct_trials": val * 1.5,
        }
    return {
        "round": 1,
        "spec": {
            "round": 1,
            "models": [
                {"name": n, "n_trials": 20, "search_space": DEFAULT_SEARCH_SPACES[n]}
                for n in best_values
            ],
            "n_pca_components": n_pca,
            "val_metric": "psi_rmse",
            "action": "initial",
            "rationale": "",
        },
        "summary": {
            "overall_best": {"model": best_model, "value": best_values[best_model], "params": {}},
            "per_model": per_model,
        },
    }


# ------------------------------ search policy ------------------------------ #

def test_search_round1_runs_all_models():
    fn = make_scripted_search_policy()
    d = fn(_ctx())
    assert d.action == "run_round"
    assert {m.name for m in d.models} == {"gp", "kernel_ridge", "poly_ridge", "mlp"}
    assert d.n_pca_components == 8
    assert "R1" in d.rationale


def test_search_terminates_on_low_budget():
    fn = make_scripted_search_policy()
    h = [_round_entry({"gp": 1.0, "kernel_ridge": 2.0})]
    d = fn(_ctx(history=h, seconds_remaining=30.0))
    assert d.action == "terminate"
    assert "R2" in d.rationale


def test_search_terminates_on_plateau():
    fn = make_scripted_search_policy()
    h = [
        _round_entry({"gp": 1.00, "kernel_ridge": 2.0}),
        _round_entry({"gp": 0.995, "kernel_ridge": 2.0}),  # 0.5% improvement
    ]
    d = fn(_ctx(history=h))
    assert d.action == "terminate"
    assert "R3" in d.rationale


def test_search_depth_round_keeps_top2_and_widens_edge_hits():
    fn = make_scripted_search_policy()
    h = [_round_entry(
        {"gp": 1.0, "kernel_ridge": 0.5, "poly_ridge": 3.0, "mlp": 4.0},
        edge_hit={"kernel_ridge": {"alpha": True}},
    )]
    d = fn(_ctx(history=h))
    assert d.action == "run_round"
    assert {m.name for m in d.models} == {"kernel_ridge", "gp"}
    kr = next(m for m in d.models if m.name == "kernel_ridge")
    from autotokamak.surrogate.zoo import DEFAULT_SEARCH_SPACES

    # alpha best sat at the low edge → low bound widened downward
    assert kr.search_space["alpha"].low < DEFAULT_SEARCH_SPACES["kernel_ridge"]["alpha"]["low"]
    assert "widen kernel_ridge.alpha" in d.rationale


def test_search_policy_is_deterministic():
    h = [_round_entry({"gp": 1.0, "kernel_ridge": 0.5, "poly_ridge": 3.0, "mlp": 4.0})]
    d1 = make_scripted_search_policy(seed=0)(_ctx(history=h))
    d2 = make_scripted_search_policy(seed=0)(_ctx(history=h))
    assert d1.model_dump() == d2.model_dump()


# ------------------------------- meta policy ------------------------------- #

def _state(winner=None, actions=None):
    return SimpleNamespace(
        best_winner_payload=winner,
        actions_taken=actions or [],
    )


def _meta_config(enrich_n_new=None):
    return SimpleNamespace(enrich_n_new=enrich_n_new)


def _record(rmse_after):
    return SimpleNamespace(rmse_after=rmse_after)


def test_meta_no_winner_extends_search():
    pick = make_scripted_meta_policy()
    d = pick(_meta_config(), _state(winner=None), {}, [])
    assert d.action == "extend_search"
    assert d.diagnosis.startswith("M1")


def test_meta_terminates_on_double_plateau():
    pick = make_scripted_meta_policy()
    history = [_record(1.0), _record(0.99), _record(0.985)]  # 1% then 0.5%
    d = pick(_meta_config(), _state(winner={"x": 1}), {}, history)
    assert d.action == "terminate"
    assert d.terminate is not None


def test_meta_edge_hits_trigger_extend_search():
    pick = make_scripted_meta_policy()
    diag = {"edge_hit_summary": {"models_with_edge_hits": {"gp": ["length_scale"]}}}
    d = pick(_meta_config(), _state(winner={"x": 1}), diag, [_record(1.0)])
    assert d.action == "extend_search"
    assert "gp.length_scale" in d.extend.widen_params


def test_meta_data_limited_triggers_enrich_active():
    pick = make_scripted_meta_policy()
    diag = {"learning_curve": {"plateau_detected": False}}
    d = pick(_meta_config(enrich_n_new=250), _state(winner={"x": 1}), diag, [_record(1.0)])
    assert d.action == "enrich_active"
    assert d.enrich.n_new == 250


def test_meta_alternates_away_from_previous_action():
    pick = make_scripted_meta_policy()
    d = pick(
        _meta_config(),
        _state(winner={"x": 1}, actions=["enrich_active"]),
        {},
        [_record(1.0)],
    )
    assert d.action == "extend_search"


def test_meta_policy_is_deterministic():
    diag = {"edge_hit_summary": {"models_with_edge_hits": {"gp": ["length_scale"]}}}
    args = (_meta_config(), _state(winner={"x": 1}), diag, [_record(1.0)])
    d1 = make_scripted_meta_policy(seed=0)(*args)
    d2 = make_scripted_meta_policy(seed=0)(*args)
    assert d1.model_dump() == d2.model_dump()


# ------------------------------- factories -------------------------------- #

def test_get_policy_factories():
    assert callable(get_search_policy("scripted"))
    assert callable(get_meta_policy("scripted"))
    with pytest.raises(ValueError):
        get_search_policy("bogus")
    with pytest.raises(ValueError):
        get_meta_policy("bogus")
