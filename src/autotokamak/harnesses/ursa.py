# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""URSA harness — PlanningAgent + ExecutionAgent with the feedback loop.

Thin adapter over ``agent.runners.plan_execute_feedback.run_feedback_loop``;
the engine is unchanged, only the config source differs (TaskSpec instead of
a prompt YAML). Env: OPENAI_API_KEY (or whatever the ``init_chat_model``
string requires).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from autotokamak.bench.taskspec import TaskSpec
from autotokamak.bench.trace import RunTrace, utc_run_id
from autotokamak.harnesses.base import Harness, RunResult

DEFAULT_MODEL = "openai:gpt-5.2"


class UrsaHarness(Harness):
    name = "ursa"

    def run(
        self,
        task: TaskSpec,
        workspace: Path,
        *,
        run_dir: Path,
        model: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
    ) -> RunResult:
        started = time.time()
        run_id = utc_run_id()
        model_name = self.resolve_model(task, model) or DEFAULT_MODEL
        self.prepare_workspace(task, workspace)

        trace = RunTrace.open_at(
            run_dir,
            prompt_path=task.source_path or Path("<inline>"),
            model=model_name,
            workspace=str(workspace),
            harness=self.name,
            run_id=run_id,
            feedback_rounds=task.feedback_rounds,
        )

        status, error = "completed", None
        cost_usd, usage = None, None
        try:
            # Import inside the guard: pulls in langchain + ursa, which are
            # optional for every other harness — a missing dep must still
            # produce a result.json, like the CLI-substrate adapters.
            from autotokamak.agent.runners.plan_execute_feedback import run_feedback_loop

            # OpenAI-model token/cost accounting; harmless no-op otherwise.
            try:
                from langchain_community.callbacks import get_openai_callback
            except ImportError:
                get_openai_callback = None

            def _invoke() -> None:
                run_feedback_loop(
                    problem=task.render_prompt(self.name),
                    workspace_path=workspace,
                    model_name=model_name,
                    feedback_rounds=task.feedback_rounds,
                    trace=trace,
                    expected_artifacts=task.expected_artifacts,
                    scorer_dotted=task.scorer,
                    scorer_kwargs=task.scorer_kwargs,
                )

            if get_openai_callback is not None:
                with get_openai_callback() as cb:
                    _invoke()
                if cb.total_tokens:
                    cost_usd = round(cb.total_cost, 6) if cb.total_cost else None
                    usage = {"prompt_tokens": cb.prompt_tokens,
                             "completion_tokens": cb.completion_tokens,
                             "n_lm_calls": cb.successful_requests}
            else:
                _invoke()
        except KeyboardInterrupt:
            status, error = "interrupted", "KeyboardInterrupt"
        except Exception as exc:  # noqa: BLE001 — trace already marked errored
            status, error = "errored", f"{type(exc).__name__}: {exc}"

        return RunResult(
            status=status,
            run_id=run_id,
            condition=self.condition_for(task),
            harness=self.name,
            model=model_name,
            workspace=workspace,
            trace_path=trace._path,
            wall_seconds=time.time() - started,
            cost_usd=cost_usd,
            error=error,
            extra={"usage": usage} if usage else {},
        )
