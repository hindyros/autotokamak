# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""DSPy signatures for the meta-agent's optimizable LLM calls.

The signature's **docstring** is the prompt GEPA mutates. The fields are
typed inputs/outputs DSPy uses to build the structured-output request.
At runtime, ``module.MetaActionPickerModule`` invokes this signature and
coerces the outputs back into our pydantic ``ActionDecision``.

The initial docstring intentionally mirrors what
``meta_loop._build_action_prompt`` produced today, so the baseline
behavior with this signature is the same as before the DSPy refactor.
"""

from __future__ import annotations

from typing import Literal

import dspy


class MetaActionPicker(dspy.Signature):
    """You are the META-AGENT orchestrating a Grad-Shafranov surrogate-model improvement loop.

    HARD RULE — COLD START: if ``best_rmse_so_far`` in state_summary is null
    (no surrogate has been trained yet), you MUST choose ``extend_search``.
    A data-only loop cannot produce a winner.pkl / report.json / study.db,
    and the ``learning_curve`` diagnostic you see is computed against an
    untuned poly_ridge proxy — its slope tells you nothing about whether
    a properly tuned surrogate is sample- or tuning-limited. Establish the
    baseline surrogate first, then use the residual_structure / edge_hit_summary
    fields on subsequent iterations to decide whether to enrich data
    (regen_dataset) or re-tune (extend_search again).

    Each iteration you must choose ONE action based on the current diagnostics
    and history:

      - regen_dataset: run an ADDITIONAL Phase-1 sweep whose samples are
        APPENDED to the current dataset — the dataset only grows, never
        shrinks. ``sampling.n_samples`` in the overrides is the number of
        NEW samples to add on top of the existing corpus, not a replacement
        total. Prefer values that meaningfully enrich (e.g. +200 or +500 on
        top of ~500). Samples are placed BLINDLY (Latin hypercube). Choose
        this over enrich_active ONLY when you also need to CHANGE the sweep
        configuration — widen parameter bounds, change mesh_dx/solver knobs,
        or switch the sampling design via "sampling.method"
        ("lhs" | "sobol" | "uniform"; sobol gives the most uniform coverage
        of a wide box, uniform is plain random) — because those overrides are
        this action's real lever.

      - enrich_active: append new samples selected by ACTIVE LEARNING
        instead of blindly. Once a winner exists, acquisition is
        RESIDUAL-DRIVEN: an error model fit on the winner's out-of-fold
        residuals scores a candidate pool by UCB (predicted |error| +
        beta * error-model uncertainty), down-weighted where prior solves
        FAILED — the batch lands where the model is MEASURABLY weak.
        Before any winner exists it degrades to a pure-uncertainty PCA-GP
        acquisition. Payload: {"n_new": int, "beta": float in [0, 3]}.
        beta ~ 0.5 exploits known-weak regions (use when residual_structure
        shows |residual| correlating with specific inputs); beta ~ 1-2
        explores toward unsampled regions (use when the target envelope is
        wider than current data coverage). You may also set "strategy":
        "auto" (default — residual-driven if a winner exists, else
        uncertainty), "residual_ucb" (force weakness-targeting),
        "uncertainty" (model-agnostic field variance — use when the winner is
        fresh and its residuals are noisy), or "space_filling" (pure maximin
        coverage, no error model — use to blanket a newly-widened region
        before targeting within it). PREFER this over regen_dataset
        whenever diagnostics indicate the surrogate is sample-bottlenecked:
        learning_curve slope steeply negative, plateau_detected false, or
        cross_seed_variance high (split-luck dominates -> need more data).
        Targeted samples buy more RMSE reduction per OFT solve than blind
        ones.

      - extend_search: run another Phase-2 surrogate search. Specify
        models_to_emphasize, widen_params, and an optional n_trials_hint.
        Choose this when edge_hit_summary shows persistent edge-hits (widen
        those ranges) OR when learning_curve has plateaued but RMSE is still
        far from the baseline (need better tuning, not more data).

      - terminate: stop the loop. Choose this when the surrogate is clearly
        good enough OR when further actions would not help (e.g. you have
        already tried regen + extend and RMSE is not improving). When
        ``target_rmse`` in state_summary is non-null, that is the run's
        quality bar: the loop stops automatically once best_rmse_so_far
        reaches it, so near the bar prefer the cheapest action likely to
        close the remaining gap.

    Always provide a one-sentence diagnosis explaining which bottleneck you
    identified, and a one-sentence rationale explaining why the chosen
    action addresses that bottleneck. The diagnosis and rationale should be
    consistent.
    """

    diagnostics_json: str = dspy.InputField(
        desc="JSON-encoded diagnostics dict (learning_curve, cross_seed_variance, "
             "pca_spectrum, edge_hit_summary, optionally residual_structure)."
    )
    history_summary: str = dspy.InputField(
        desc="JSON-encoded list of prior decisions with their rmse_after values."
    )
    state_summary: str = dspy.InputField(
        desc="JSON-encoded current iteration index and prior RMSE."
    )

    action: Literal["regen_dataset", "enrich_active", "extend_search", "terminate"] = dspy.OutputField(
        desc="Exactly one of: regen_dataset, enrich_active, extend_search, terminate."
    )
    diagnosis: str = dspy.OutputField(
        desc="One-sentence bottleneck diagnosis."
    )
    rationale: str = dspy.OutputField(
        desc="One-sentence explanation of why the chosen action addresses the bottleneck."
    )
    payload_json: str = dspy.OutputField(
        desc='JSON object with action-specific fields. '
             'For regen_dataset: {"overrides": {<dotted SweepConfig key>: value}} — '
             'valid keys are existing SweepConfig paths only, e.g. '
             '"sampling.n_samples", "sampling.seed", "sampling.method" '
             '("lhs"|"sobol"|"uniform"), "parameters.r0.low", '
             '"parameters.kappa.high", "fixed.mesh_dx". Do NOT invent knobs '
             '(no "sampling.strategy", no "sampling.focus_vars"); unknown keys '
             'are dropped. '
             'For enrich_active: {"n_new": int, "beta": float in [0,3] '
             '(UCB exploration weight, default 1.0), "strategy": '
             '"auto"|"residual_ucb"|"uncertainty"|"space_filling" '
             '(default "auto")} (optionally '
             '"feasibility_weighting": bool, default true). '
             'For extend_search: {"models_to_emphasize": [...], "widen_params": [...], "n_trials_hint": int}. '
             'For terminate: {"reason": "...", "confidence": "low"|"medium"|"high"}.'
    )


class SearchRoundPicker(dspy.Signature):
    """You are the SEARCH PICKER inside a Grad-Shafranov surrogate AutoML loop.

    Each round you receive a JSON round context and must decide whether to
    run another Optuna search round (and over which models / ranges / trial
    counts) or to terminate. Decision rules:

      - Round 1 (empty history): pick at least two models from the zoo
        (gp, kernel_ridge, poly_ridge, mlp), copying their entries from
        ``default_search_spaces`` in the context. A sensible per-model
        n_trials is 8-50 depending on ``budget.seconds_remaining``.
      - If a prior round's summary shows ``edge_hit: true`` for a numeric
        hyperparameter, WIDEN that parameter's range (e.g. extend low/high
        by 2-10x in the direction of the hit) and run again.
      - If ``best_value_at_25pct_trials`` is close to ``best_value`` for a
        model, its search converged early — TIGHTEN its ranges around
        best_params, add a different model, or terminate.
      - Honor the meta-agent ``focus`` directive when present:
        emphasize ``models_to_emphasize``, widen ``widen_params``, and use
        ``n_trials_hint`` as the total trial budget across models.
      - Respect ``budget.seconds_remaining``: do not start a round you
        cannot finish; terminate instead.
      - TERMINATE when no edge hits remain, at least two models have been
        tried, and the marginal improvement between rounds is small.

    Output ``models_json`` must be a JSON list of model entries; ranges you
    do not want to change may be omitted (defaults are filled in).
    """

    round_context_json: str = dspy.InputField(
        desc="JSON round context: round index, rounds/budget remaining, focus "
             "directive, dataset stats, default_search_spaces, and per-round "
             "history with per-model summaries (best_value, best_params, "
             "edge_hit flags, best_value_at_25pct_trials)."
    )

    action: Literal["run_round", "terminate"] = dspy.OutputField(
        desc="Exactly one of: run_round, terminate."
    )
    models_json: str = dspy.OutputField(
        desc='JSON list of {"name": "gp"|"kernel_ridge"|"poly_ridge"|"mlp", '
             '"n_trials": int, "search_space": {param: {"type": "float"|"int"|'
             '"categorical"|"loguniform", "low": ..., "high": ..., "choices": ...}}}. '
             'Omit search_space entries to inherit the defaults. Use [] when '
             'action is terminate.'
    )
    n_pca_components: int = dspy.OutputField(
        desc="PCA components for this round (1-64)."
    )
    rationale: str = dspy.OutputField(
        desc="One-sentence justification for the decision."
    )


__all__ = ["MetaActionPicker", "SearchRoundPicker"]
