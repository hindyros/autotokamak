# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Best-effort score dispatch for the runners.

Each prompt YAML may declare a ``scorer:`` dotted path to a function with the
signature ``(workspace_path: Path, **kwargs) -> ScoreReport-like | None``.
``try_score`` imports it lazily, invokes it, and swallows any exception so
the runner's happy path is never blocked by a scoring bug.

Back-compat: when ``scorer`` is unset, ``try_score`` falls back to the Phase-1
scorer (``autotokamak.agent.dspy.metric:score_run``) and infers
``requested_n_samples`` from ``dataset_config.yaml`` — exactly the behavior
the now-deleted ``_try_score`` blocks in both runners implemented.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any


DEFAULT_PHASE1_SCORER = "autotokamak.agent.dspy.metric:score_run"


def _import_callable(dotted: str):
    if ":" not in dotted:
        raise ValueError(
            f"Scorer reference {dotted!r} must be '<module>:<func>' (got no colon)."
        )
    module_path, _, func_name = dotted.partition(":")
    mod = importlib.import_module(module_path)
    fn = getattr(mod, func_name, None)
    if fn is None:
        raise AttributeError(f"Module {module_path!r} has no attribute {func_name!r}.")
    if not callable(fn):
        raise TypeError(f"{dotted!r} is not callable.")
    return fn


def _phase1_default_kwargs(workspace_path: Path) -> dict[str, Any]:
    """Mirror the old inline ``_try_score``: infer ``requested_n_samples``."""
    requested_n = 16
    cfg_path = workspace_path / "dataset_config.yaml"
    if cfg_path.is_file():
        try:
            import yaml

            data = yaml.safe_load(cfg_path.read_text()) or {}
            requested_n = int(data.get("sampling", {}).get("n_samples", requested_n))
        except Exception:  # noqa: BLE001
            pass
    return {"requested_n_samples": requested_n}


def try_score(
    workspace_path: Path,
    scorer_dotted: str | None = None,
    scorer_kwargs: dict[str, Any] | None = None,
) -> object | None:
    """Import and run the scorer; return its result or ``None`` on any failure.

    Parameters
    ----------
    workspace_path
        Directory the agent wrote into (passed as first positional arg).
    scorer_dotted
        ``"<module>:<func>"`` taken from the prompt YAML's ``scorer:`` field.
        When ``None``, falls back to the Phase-1 default.
    scorer_kwargs
        Optional kwargs taken from the prompt YAML's ``scorer_kwargs:`` field.
        Merged on top of the back-compat defaults for the Phase-1 scorer.
    """
    dotted = scorer_dotted or DEFAULT_PHASE1_SCORER
    try:
        fn = _import_callable(dotted)
    except Exception:  # noqa: BLE001
        return None

    kwargs: dict[str, Any] = dict(scorer_kwargs or {})
    if dotted == DEFAULT_PHASE1_SCORER:
        # Preserve the old behavior: infer n_samples from dataset_config.yaml
        # unless the prompt overrode it.
        defaults = _phase1_default_kwargs(workspace_path)
        for k, v in defaults.items():
            kwargs.setdefault(k, v)

    try:
        return fn(workspace_path, **kwargs)
    except Exception:  # noqa: BLE001
        return None


__all__ = ["DEFAULT_PHASE1_SCORER", "try_score"]
