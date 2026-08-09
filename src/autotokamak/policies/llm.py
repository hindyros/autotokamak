# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""L1 LLM decision policies — thin wrappers over the existing DSPy pickers.

Same factory signatures as ``policies.scripted`` so the pipelines select a
provider by name and stay otherwise identical across levels.
"""
from __future__ import annotations

from typing import Optional

DEFAULT_PICKER_MODEL = "openai:gpt-5-mini"


def make_llm_search_policy(model: Optional[str] = None):
    """DSPy ``SearchRoundPicker`` as a ``DecisionFn`` for ``run_automl_loop``."""
    from autotokamak.agent.dspy.module import make_search_decision_fn

    return make_search_decision_fn(model or DEFAULT_PICKER_MODEL)


def make_llm_meta_policy():
    """DSPy ``MetaActionPicker`` as an ``ActionPicker`` for ``meta_loop.run``.

    The model string comes from ``meta_config.model`` at call time (that is
    how ``pick_action_via_llm`` already works), so no model arg here.
    """
    from autotokamak.agent.runners.meta_loop import pick_action_via_llm

    return pick_action_via_llm
