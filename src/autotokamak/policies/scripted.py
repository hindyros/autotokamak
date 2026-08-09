# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""L0 scripted decision policies — seeded heuristics, zero LLM calls.

These are the golden-baseline decision providers: pure functions of their
round/iteration context, so two runs with the same seed and the same solver
outputs make byte-identical decisions. Every emitted decision carries a
``rationale`` naming the rule that fired, which makes L0 traces
self-explaining and directly comparable to LLM rationales.

The heuristics mirror the guidance the prompts give the LLM pickers
(widen on edge-hit, tighten when Optuna converged early, act on the
learning-curve/edge-hit bottleneck signals from ``surrogate.diagnostics``).
"""
from __future__ import annotations

from typing import Any, Optional

from autotokamak.surrogate.schema import ModelSpec, RoundDecision

# ------------------------- search policy tuning ------------------------- #

N_TRIALS_ROUND1 = 20        # per model, breadth round
N_TRIALS_LATER = 30         # per model, depth rounds (top-2 only)
KEEP_TOP_MODELS = 2
MIN_RESERVE_FRACTION = 0.15  # terminate when < this fraction of budget remains
PLATEAU_REL = 0.02           # <2% round-over-round improvement → terminate
EARLY_CONVERGED_REL = 0.01   # 25%-trials best within 1% of final → tighten
PCA_BUMP_REL = 0.05          # <5% improvement → try more PCA components once
PCA_BUMP = 4
PCA_CAP = 64
WIDEN_FACTOR = 4.0           # log-scale ranges widen 4x toward the hit edge
TIGHTEN_FACTOR = 2.0         # tighten to best/2 .. best*2

# Hard caps the zoo factories enforce; never widen past these.
_PARAM_CAPS: dict[tuple[str, str], tuple[float, float]] = {
    ("mlp", "n_layers"): (1, 2),
    ("mlp", "layer_width"): (8, 256),
    ("poly_ridge", "degree"): (1, 5),
}


def _overall_best(summary: dict) -> Optional[float]:
    val = (summary or {}).get("overall_best", {}).get("value")
    return float(val) if val is not None else None


def _widened_range(model: str, name: str, rng: dict, best: Any) -> dict:
    """Widen ``rng`` toward whichever edge ``best`` sits on."""
    out = dict(rng)
    lo, hi = float(rng["low"]), float(rng["high"])
    v = float(best)
    near_low = (v - lo) <= (hi - v)
    if rng.get("type") == "loguniform":
        if near_low:
            out["low"] = lo / WIDEN_FACTOR
        else:
            out["high"] = hi * WIDEN_FACTOR
    else:  # int / uniform: extend by the current span
        span = max(hi - lo, 1.0)
        if near_low:
            out["low"] = lo - span
        else:
            out["high"] = hi + span
    cap = _PARAM_CAPS.get((model, name))
    if cap is not None:
        out["low"] = max(float(out["low"]), cap[0])
        out["high"] = min(float(out["high"]), cap[1])
    if rng.get("type") == "int":
        out["low"] = int(round(out["low"]))
        out["high"] = int(round(out["high"]))
    return out


def _tightened_range(rng: dict, best: Any) -> dict:
    """Tighten ``rng`` around the best value found so far."""
    out = dict(rng)
    v = float(best)
    if rng.get("type") == "loguniform":
        out["low"] = max(float(rng["low"]), v / TIGHTEN_FACTOR)
        out["high"] = min(float(rng["high"]), v * TIGHTEN_FACTOR)
    elif rng.get("type") == "int":
        out["low"] = max(int(rng["low"]), int(round(v)) - 1)
        out["high"] = min(int(rng["high"]), int(round(v)) + 1)
    else:
        span = (float(rng["high"]) - float(rng["low"])) / (2 * TIGHTEN_FACTOR)
        out["low"] = max(float(rng["low"]), v - span)
        out["high"] = min(float(rng["high"]), v + span)
    if float(out["low"]) >= float(out["high"]):  # degenerate → keep original
        return dict(rng)
    return out


def make_scripted_search_policy(seed: int = 0):
    """Return a ``DecisionFn`` for ``run_automl_loop`` (L0, no LLM).

    Rules, in firing order (rationale strings quote these):
      R1 first round → breadth: every zoo model at its default search space.
      R2 budget reserve → terminate when seconds_remaining is below
         max(60, 15% of the budget).
      R3 plateau → terminate when round-over-round best improved < 2%.
      R4 otherwise run a depth round: keep the top-2 models by best_value;
         widen any edge-hit param toward its hit edge; tighten all ranges
         around best_params when Optuna had already converged by 25% of its
         trials; bump n_pca_components once (+4, cap 64) when improvement
         fell below 5%.
    """
    del seed  # decisions are ctx-deterministic; kept for signature parity

    def decision_fn(ctx: dict) -> RoundDecision:
        history = ctx.get("history") or []
        defaults = ctx["default_search_spaces"]
        n_pca_default = int(ctx["n_pca_components_default"])

        if not history:  # R1 — breadth round
            models = [
                ModelSpec(name=name, n_trials=N_TRIALS_ROUND1, search_space=space)
                for name, space in defaults.items()
            ]
            return RoundDecision(
                action="run_round",
                models=models,
                n_pca_components=n_pca_default,
                rationale="R1 breadth: first round runs every zoo model at defaults",
            )

        budget = ctx.get("budget", {})
        remaining = float(budget.get("seconds_remaining", 0.0))
        total = float(budget.get("time_budget_seconds", 0.0))
        if remaining < max(60.0, MIN_RESERVE_FRACTION * total):  # R2
            return RoundDecision(
                action="terminate",
                rationale=(
                    f"R2 budget reserve: {remaining:.0f}s remaining of {total:.0f}s "
                    "is below the reserve threshold"
                ),
            )

        last = history[-1]
        last_best = _overall_best(last.get("summary"))
        if len(history) >= 2:  # R3
            prev_best = _overall_best(history[-2].get("summary"))
            if prev_best and last_best and prev_best > 0:
                rel = (prev_best - last_best) / prev_best
                if rel < PLATEAU_REL:
                    return RoundDecision(
                        action="terminate",
                        rationale=(
                            f"R3 plateau: best_value improved {rel * 100:.1f}% "
                            f"(< {PLATEAU_REL * 100:.0f}%) over the last round"
                        ),
                    )

        # R4 — depth round on the current leaders.
        per_model = (last.get("summary") or {}).get("per_model", {})
        prev_specs = {m["name"]: m for m in (last.get("spec") or {}).get("models", [])}
        ranked = sorted(
            (name for name in per_model if per_model[name].get("best_value") is not None),
            key=lambda name: per_model[name]["best_value"],
        )[:KEEP_TOP_MODELS]
        if not ranked:  # every study failed → retry breadth
            models = [
                ModelSpec(name=name, n_trials=N_TRIALS_ROUND1, search_space=space)
                for name, space in defaults.items()
            ]
            return RoundDecision(
                action="run_round",
                models=models,
                n_pca_components=n_pca_default,
                rationale="R4 fallback: no model produced a finite best_value; rerun breadth",
            )

        notes: list[str] = []
        models = []
        for name in ranked:
            info = per_model[name]
            space = dict(
                (prev_specs.get(name) or {}).get("search_space") or defaults[name]
            )
            best_params = info.get("best_params") or {}
            edge_hit = info.get("edge_hit") or {}
            early = info.get("best_value_at_25pct_trials")
            best_val = info.get("best_value")
            converged_early = (
                early is not None
                and best_val is not None
                and best_val > 0
                and (early - best_val) / best_val < EARLY_CONVERGED_REL
            )
            for pname, rng in list(space.items()):
                if rng.get("type") == "categorical" or pname not in best_params:
                    continue
                if edge_hit.get(pname):
                    space[pname] = _widened_range(name, pname, rng, best_params[pname])
                    notes.append(f"widen {name}.{pname} (edge hit)")
                elif converged_early:
                    space[pname] = _tightened_range(rng, best_params[pname])
            if converged_early:
                notes.append(f"tighten {name} around best_params (converged by 25% trials)")
            models.append(ModelSpec(name=name, n_trials=N_TRIALS_LATER, search_space=space))

        n_pca = int((last.get("spec") or {}).get("n_pca_components") or n_pca_default)
        pca_never_bumped = all(
            int((h.get("spec") or {}).get("n_pca_components") or n_pca_default)
            <= n_pca_default
            for h in history
        )
        if len(history) >= 2 and pca_never_bumped:
            prev_best = _overall_best(history[-2].get("summary"))
            if prev_best and last_best and prev_best > 0:
                if (prev_best - last_best) / prev_best < PCA_BUMP_REL:
                    n_pca = min(n_pca + PCA_BUMP, PCA_CAP)
                    notes.append(f"bump n_pca_components to {n_pca} (slow improvement)")

        return RoundDecision(
            action="run_round",
            models=models,
            n_pca_components=n_pca,
            rationale="R4 depth on top-" + str(len(ranked)) + f" [{', '.join(ranked)}]"
            + (": " + "; ".join(notes) if notes else ""),
        )

    return decision_fn


# --------------------------- meta policy tuning --------------------------- #

META_PLATEAU_REL = 0.03      # two consecutive <3% improvements → terminate
WORST_CELL_LAG = 2.0         # worst-cell RMSE > 2x mean → data-limited region


def make_scripted_meta_policy(seed: int = 0):
    """Return an ``ActionPicker`` for ``meta_loop.run`` (L0, no LLM).

    Rules, in firing order:
      M1 no winner yet → extend_search (train something before judging data).
      M2 two consecutive iterations each improved shard RMSE < 3% → terminate.
      M3 models persistently edge-hit → extend_search widening those params.
      M4 learning curve still steep (not plateaued) or residuals concentrated
         → enrich_active (residual-UCB targets the weak regions).
      M5 fallback: alternate away from the previous action so a stalled
         action is never repeated twice in a row.
    """
    del seed

    from autotokamak.agent.orchestrator.schema import (
        ActionDecision,
        EnrichActivePayload,
        ExtendSearchFocus,
        TerminateReason,
    )

    def pick_action(meta_config, state, diagnostics: dict, history: list) -> ActionDecision:
        if state.best_winner_payload is None:  # M1
            return ActionDecision(
                action="extend_search",
                extend=ExtendSearchFocus(),
                diagnosis="M1: no trained winner yet — run Phase-2 before judging the data",
            )

        rmse = [r.rmse_after for r in history if r.rmse_after is not None]
        if len(rmse) >= 3:  # M2 — needs two consecutive deltas
            d1 = (rmse[-3] - rmse[-2]) / rmse[-3] if rmse[-3] > 0 else 0.0
            d2 = (rmse[-2] - rmse[-1]) / rmse[-2] if rmse[-2] > 0 else 0.0
            if d1 < META_PLATEAU_REL and d2 < META_PLATEAU_REL:
                return ActionDecision(
                    action="terminate",
                    terminate=TerminateReason(
                        reason=(
                            f"M2: two consecutive iterations improved shard RMSE "
                            f"{d1 * 100:.1f}% and {d2 * 100:.1f}% "
                            f"(< {META_PLATEAU_REL * 100:.0f}%)"
                        ),
                        confidence="high",
                    ),
                    diagnosis="M2: improvement plateaued across two iterations",
                )

        edge_hits: dict = (diagnostics.get("edge_hit_summary") or {}).get(
            "models_with_edge_hits"
        ) or {}
        if edge_hits:  # M3
            widen = [f"{m}.{p}" for m, params in edge_hits.items() for p in params]
            return ActionDecision(
                action="extend_search",
                extend=ExtendSearchFocus(
                    models_to_emphasize=sorted(edge_hits),
                    widen_params=widen,
                ),
                diagnosis=f"M3: model-limited — best params sit on range edges ({', '.join(widen)})",
            )

        curve = diagnostics.get("learning_curve") or {}
        residuals = diagnostics.get("residual_structure") or {}
        input_corrs = residuals.get("input_correlations") or {}
        residuals_concentrated = any(abs(v) > 0.5 for v in input_corrs.values())
        data_limited = curve.get("plateau_detected") is False or residuals_concentrated
        prev_action = state.actions_taken[-1] if state.actions_taken else None
        if data_limited and prev_action != "enrich_active":  # M4
            return ActionDecision(
                action="enrich_active",
                enrich=EnrichActivePayload(n_new=meta_config.enrich_n_new or 500),
                diagnosis=(
                    "M4: data-limited — learning curve still dropping or residuals "
                    "concentrated; acquire where the winner is weak"
                ),
            )

        # M5 — alternate away from whatever ran last.
        if prev_action == "enrich_active":
            return ActionDecision(
                action="extend_search",
                extend=ExtendSearchFocus(),
                diagnosis="M5: alternate — enrich_active just ran; search harder on the grown pool",
            )
        return ActionDecision(
            action="enrich_active",
            enrich=EnrichActivePayload(n_new=meta_config.enrich_n_new or 500),
            diagnosis="M5: alternate — no clear bottleneck signal; grow the dataset",
        )

    return pick_action
