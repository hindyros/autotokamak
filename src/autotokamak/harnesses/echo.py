# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Echo harness — no LLM, no network. Exercises the full bench machinery
(run dir layout, trace, contract validation, scoring) in tests and CI.

It writes the rendered prompt to ``prompt.txt`` and touches every expected
artifact (copying canned fixtures instead when a ``fixtures_dir`` is set in
``task.scorer_kwargs`` — useful for contract tests that need real content).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from autotokamak.bench.taskspec import TaskSpec
from autotokamak.bench.trace import RunTrace, utc_run_id
from autotokamak.harnesses.base import Harness, RunResult


class EchoHarness(Harness):
    name = "echo"

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
        self.prepare_workspace(task, workspace)

        trace = RunTrace.open_at(
            run_dir,
            prompt_path=task.source_path or Path("<inline>"),
            model=model or "none",
            workspace=str(workspace),
            harness=self.name,
            run_id=run_id,
        )
        round_rec = trace.start_round(1)
        trace.record_plan_steps(round_rec, ["echo the task into the workspace"])
        step = trace.start_step(round_rec, 1, "materialize artifacts")

        (workspace / "prompt.txt").write_text(
            task.render_prompt(self.name), encoding="utf-8"
        )
        fixtures = task.scorer_kwargs.get("fixtures_dir")
        for rel in task.expected_artifacts:
            dst = workspace / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            src = Path(fixtures) / rel if fixtures else None
            if src is not None and src.is_file():
                dst.write_bytes(src.read_bytes())
            elif not dst.exists():
                dst.write_text("", encoding="utf-8")

        trace.finish_step(
            step, ok=True,
            result_text=f"touched {len(task.expected_artifacts)} artifacts",
        )
        trace.record_artifacts(workspace, expected_artifacts=task.expected_artifacts)
        trace.mark_completed()

        return RunResult(
            status="completed",
            run_id=run_id,
            condition=self.condition_for(task),
            harness=self.name,
            model=model or "none",
            workspace=workspace,
            trace_path=trace._path,
            wall_seconds=time.time() - started,
        )
