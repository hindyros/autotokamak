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

        # Import inside run(): pulls in langchain + ursa, which are optional
        # for every other harness.
        from autotokamak.agent.runners.plan_execute_feedback import run_feedback_loop

        status, error = "completed", None
        try:
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
            error=error,
        )
