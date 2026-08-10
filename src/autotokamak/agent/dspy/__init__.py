# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""DSPy integration for the autotokamak agent stack.

See `docs/dspy_integration_plan.md` for the proposal.

Currently exports only `metric.score_run` — the pure-Python composite scoring
function that can run today (no DSPy dependency required). The actual DSPy
modules (linter, planner, repair) are pending the run-instrumentation patch.
"""

from autotokamak.agent.dspy.metric import ScoreReport, score_run

__all__ = ["ScoreReport", "score_run"]
