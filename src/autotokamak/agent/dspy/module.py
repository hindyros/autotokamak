# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""DSPy module wrapping the meta-action-picker signature.

The module's predict step is what GEPA mutates: the signature docstring
and any bootstrapped few-shot demos. The wrapper also handles converting
the LM's structured output back into our pydantic ``ActionDecision`` so
the rest of the runner stack (which already speaks ActionDecision) is
unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import dspy

from autotokamak.agent.dspy.signatures import MetaActionPicker, SearchRoundPicker
from autotokamak.agent.orchestrator.schema import (
    VALID_ACQUISITION_STRATEGIES,
    VALID_ACTIONS,
    ActionDecision,
    EnrichActivePayload,
    ExtendSearchFocus,
    RegenDatasetOverrides,
    TerminateReason,
)


def _clamp_num(value: Any, lo: float, hi: float, default: float, cast=int):
    """Coerce a numeric LLM payload value into [lo, hi]; junk -> ``default``.

    bools are rejected (isinstance(True, int) is True in Python, and an LM
    emitting ``true`` for a count is junk, not 1).
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return cast(max(lo, min(hi, value)))
    return default


class MetaActionPickerModule(dspy.Module):
    """A ChainOfThought predictor over ``MetaActionPicker`` with safe ActionDecision coercion.

    ``forward`` returns the raw DSPy ``Prediction``; ``predict_action_decision``
    additionally coerces to the runner's pydantic ``ActionDecision`` type.
    """

    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(MetaActionPicker)

    def forward(
        self,
        *,
        diagnostics_json: str,
        history_summary: str,
        state_summary: str,
    ) -> dspy.Prediction:
        return self.predict(
            diagnostics_json=diagnostics_json,
            history_summary=history_summary,
            state_summary=state_summary,
        )

    def predict_action_decision(
        self,
        *,
        diagnostics_json: str,
        history_summary: str,
        state_summary: str,
    ) -> ActionDecision:
        """Run the predictor and coerce to ``ActionDecision``.

        Tolerates messy LM outputs — JSON parse failures fall back to a
        minimal valid ActionDecision so the surrounding meta-loop can keep
        running and the scorer can mark the iteration as low-quality
        rather than blowing up.
        """
        pred = self.forward(
            diagnostics_json=diagnostics_json,
            history_summary=history_summary,
            state_summary=state_summary,
        )
        return _coerce_to_action_decision(pred)


def _coerce_to_action_decision(pred: dspy.Prediction) -> ActionDecision:
    """Coerce a raw prediction into a valid ``ActionDecision``.

    Fallback ladder: out-of-vocabulary action -> terminate; unparseable
    payload_json -> empty payload (schema defaults); per-action payload
    repair (clamp numerics, filter model names to the zoo, default unknown
    strategies to "auto"); any remaining validation failure -> terminate.
    The meta-loop must never crash on a messy LM output — the scorer marks
    the iteration as low-quality instead.
    """
    action = str(getattr(pred, "action", "")).strip()
    diagnosis = str(getattr(pred, "diagnosis", ""))
    rationale = str(getattr(pred, "rationale", ""))
    payload_json = str(getattr(pred, "payload_json", "{}") or "{}")

    if action not in VALID_ACTIONS:
        # LM emitted something out-of-vocabulary; fall back to terminate so
        # the meta-loop ends cleanly rather than dispatching garbage.
        return ActionDecision.model_validate(
            {
                "action": "terminate",
                "terminate": {
                    "reason": f"invalid action from picker: {action!r}",
                    "confidence": "low",
                },
                "diagnosis": diagnosis or "picker produced unknown action",
            }
        )

    payload: dict[str, Any] = {}
    try:
        payload = json.loads(payload_json)
        if not isinstance(payload, dict):
            payload = {}
    except (json.JSONDecodeError, TypeError):
        payload = {}

    raw: dict[str, Any] = {
        "action": action,
        "diagnosis": diagnosis,
        "regen": None,
        "enrich": None,
        "extend": None,
        "terminate": None,
    }

    _ZOO = {"gp", "kernel_ridge", "poly_ridge", "mlp"}

    try:
        if action == "regen_dataset":
            overrides = payload.get("overrides", {})
            if not isinstance(overrides, dict):
                overrides = {}
            raw["regen"] = RegenDatasetOverrides.model_validate(
                {
                    "overrides": overrides,
                    "rationale": rationale or payload.get("rationale", ""),
                }
            ).model_dump()
        elif action == "enrich_active":
            strategy = payload.get("strategy")
            if strategy not in VALID_ACQUISITION_STRATEGIES:
                strategy = "auto"  # unknown/omitted -> let the system decide
            feas = payload.get("feasibility_weighting")
            raw["enrich"] = EnrichActivePayload.model_validate(
                {
                    "n_new": _clamp_num(payload.get("n_new"), 1, 2000, default=100),
                    "beta": _clamp_num(payload.get("beta"), 0.0, 3.0, default=1.0, cast=float),
                    "strategy": strategy,
                    "feasibility_weighting": feas if isinstance(feas, bool) else True,
                    "rationale": rationale or payload.get("rationale", ""),
                }
            ).model_dump()
        elif action == "extend_search":
            models_raw = payload.get("models_to_emphasize", []) or []
            if not isinstance(models_raw, list):
                models_raw = []
            models_clean = [m for m in models_raw if isinstance(m, str) and m in _ZOO]
            widen_raw = payload.get("widen_params", []) or []
            if not isinstance(widen_raw, list):
                widen_raw = []
            widen_clean = [str(w) for w in widen_raw if isinstance(w, (str, int, float))]
            n_hint = _clamp_num(payload.get("n_trials_hint"), 1, 200, default=None)
            raw["extend"] = ExtendSearchFocus.model_validate(
                {
                    "models_to_emphasize": models_clean,
                    "widen_params": widen_clean,
                    "n_trials_hint": n_hint,
                    "rationale": rationale or payload.get("rationale", ""),
                }
            ).model_dump()
        else:  # terminate
            raw["terminate"] = TerminateReason.model_validate(
                {
                    "reason": payload.get("reason", rationale or "agent terminated"),
                    "confidence": payload.get("confidence", "medium"),
                }
            ).model_dump()
        return ActionDecision.model_validate(raw)
    except Exception as exc:  # noqa: BLE001
        return ActionDecision.model_validate(
            {
                "action": "terminate",
                "terminate": {
                    "reason": f"payload validation failed ({type(exc).__name__}: {exc}); terminating safely",
                    "confidence": "low",
                },
                "diagnosis": diagnosis or "picker payload invalid",
            }
        )


# --------- optimized-program loading -----------------------------------

DEFAULT_OPTIMIZED_PATH = (
    Path(__file__).resolve().parent / "optimized" / "meta_picker.json"
)


def load_module(optimized_path: Optional[Path] = None) -> MetaActionPickerModule:
    """Construct a baseline module; load optimized state if available.

    ``optimized_path`` defaults to the package-relative
    ``optimized/meta_picker.json``. Missing file → baseline ChainOfThought
    with the in-code signature. Corrupt file → baseline + stderr warning
    (no crash).
    """
    module = MetaActionPickerModule()
    path = optimized_path or DEFAULT_OPTIMIZED_PATH
    if path.is_file():
        try:
            module.load(str(path))
        except Exception as exc:  # noqa: BLE001
            import sys

            print(
                f"WARNING: failed to load optimized DSPy module from {path} "
                f"({type(exc).__name__}: {exc}); using baseline.",
                file=sys.stderr,
            )
    return module


# --------- Phase-2 search-round picker ----------------------------------

DEFAULT_SEARCH_OPTIMIZED_PATH = (
    Path(__file__).resolve().parent / "optimized" / "search_picker.json"
)


class SearchRoundPickerModule(dspy.Module):
    """A ChainOfThought predictor over ``SearchRoundPicker`` with safe coercion.

    ``predict_round_decision`` returns the runner-facing pydantic
    ``RoundDecision`` — malformed LM output degrades to a safe default
    round (or terminate) instead of crashing the automl loop.
    """

    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(SearchRoundPicker)

    def forward(self, *, round_context_json: str) -> dspy.Prediction:
        return self.predict(round_context_json=round_context_json)

    def predict_round_decision(self, *, round_context_json: str):
        pred = self.forward(round_context_json=round_context_json)
        return _coerce_to_round_decision(pred)


def _default_round_models() -> list[dict]:
    """Safe fallback round: two cheap models over the shipped default spaces."""
    from autotokamak.surrogate.zoo import DEFAULT_SEARCH_SPACES

    return [
        {
            "name": name,
            "n_trials": 8,
            "search_space": DEFAULT_SEARCH_SPACES[name],
        }
        for name in ("poly_ridge", "kernel_ridge")
    ]


def _coerce_to_round_decision(pred: dspy.Prediction):
    """Coerce a raw prediction into a valid ``RoundDecision``.

    Fallback ladder: invalid action -> terminate; malformed/empty models on
    run_round -> default round; per-model repairs (drop unknown names, clamp
    n_trials, fill missing/invalid search-space entries from defaults).
    """
    from autotokamak.surrogate.schema import MODEL_KINDS, ParamRange, RoundDecision
    from autotokamak.surrogate.zoo import DEFAULT_SEARCH_SPACES

    action = str(getattr(pred, "action", "")).strip()
    rationale = str(getattr(pred, "rationale", "") or "")

    if action not in {"run_round", "terminate"}:
        return RoundDecision(
            action="terminate",
            rationale=f"invalid action from search picker: {action!r}",
        )
    if action == "terminate":
        return RoundDecision(action="terminate", rationale=rationale or "picker terminated")

    raw_models: Any = getattr(pred, "models_json", "") or ""
    try:
        parsed = json.loads(str(raw_models))
        if not isinstance(parsed, list):
            parsed = []
    except (json.JSONDecodeError, TypeError):
        parsed = []

    models: list[dict] = []
    for entry in parsed:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if name not in MODEL_KINDS:
            continue
        n_trials = _clamp_num(entry.get("n_trials"), 1, 200, default=8)
        defaults = DEFAULT_SEARCH_SPACES[name]
        raw_space = entry.get("search_space")
        if not isinstance(raw_space, dict):
            raw_space = {}
        space: dict[str, dict] = {}
        for param, default_range in defaults.items():
            candidate = raw_space.get(param, default_range)
            try:
                space[param] = ParamRange.model_validate(candidate).model_dump()
            except Exception:  # noqa: BLE001
                space[param] = dict(default_range)
        models.append({"name": name, "n_trials": n_trials, "search_space": space})

    if not models:
        models = _default_round_models()
        rationale = (rationale + " " if rationale else "") + "(fell back to default round)"

    n_pca = _clamp_num(getattr(pred, "n_pca_components", None), 1, 64, default=None)

    try:
        return RoundDecision.model_validate(
            {
                "action": "run_round",
                "models": models,
                "n_pca_components": n_pca,
                "rationale": rationale,
            }
        )
    except Exception as exc:  # noqa: BLE001
        return RoundDecision(
            action="terminate",
            rationale=f"round payload validation failed ({type(exc).__name__}: {exc})",
        )


def load_search_module(optimized_path: Optional[Path] = None) -> SearchRoundPickerModule:
    """Baseline search-picker module; loads optimized state when available."""
    module = SearchRoundPickerModule()
    path = optimized_path or DEFAULT_SEARCH_OPTIMIZED_PATH
    if path.is_file():
        try:
            module.load(str(path))
        except Exception as exc:  # noqa: BLE001
            import sys

            print(
                f"WARNING: failed to load optimized search picker from {path} "
                f"({type(exc).__name__}: {exc}); using baseline.",
                file=sys.stderr,
            )
    return module


def make_search_decision_fn(model: str):
    """Return a ``decision_fn`` for ``automl_loop`` backed by the DSPy picker.

    ``model`` uses the repo's "provider:name" convention; converted to
    litellm-style for ``dspy.LM`` (same pattern as
    ``meta_loop.pick_action_via_llm``).
    """
    desired = model.replace(":", "/", 1)
    settings_lm = getattr(dspy.settings, "lm", None)
    if settings_lm is None or getattr(settings_lm, "model", None) != desired:
        dspy.configure(lm=dspy.LM(desired))
    module = load_search_module()

    def decision_fn(ctx: dict):
        return module.predict_round_decision(
            round_context_json=json.dumps(ctx, default=str)[:6000]
        )

    return decision_fn


__all__ = [
    "DEFAULT_OPTIMIZED_PATH",
    "DEFAULT_SEARCH_OPTIMIZED_PATH",
    "MetaActionPickerModule",
    "SearchRoundPickerModule",
    "load_module",
    "load_search_module",
    "make_search_decision_fn",
]
