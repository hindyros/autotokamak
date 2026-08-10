# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Cursor CLI harness (``cursor-agent`` headless mode).

    cursor-agent -p "<prompt>" --output-format stream-json --trust --force [--model M]

``--trust --force`` are REQUIRED for a guaranteed non-interactive run
(``-p`` alone is known to hang waiting for trust confirmation); a hard
subprocess timeout backstops it. The raw stream-json is kept at
``run_dir/cursor_events.jsonl``.

Known limitations (accepted, documented in benchmarks/README.md): no tool
allowlist, and the agent may index the enclosing git repo — the L3
no-autotokamak rule therefore rests on the prompt plus the contract's
import audit, the same trust model as URSA's unrestricted Bash.

Auth: CURSOR_API_KEY env, or a prior interactive ``cursor-agent login``.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

from autotokamak.bench.taskspec import TaskSpec
from autotokamak.bench.trace import RunTrace, utc_run_id
from autotokamak.harnesses.base import Harness, RunResult

CURSOR_BIN = "cursor-agent"


def _argv(prompt: str, model: Optional[str]) -> list[str]:
    argv = [CURSOR_BIN, "-p", prompt, "--output-format", "stream-json",
            "--trust", "--force"]
    if model:
        argv += ["--model", model]
    return argv


class CursorHarness(Harness):
    name = "cursor"

    def dry_run_info(self, task: TaskSpec, workspace: Path,
                     model: Optional[str] = None) -> dict[str, Any]:
        info = super().dry_run_info(task, workspace, model)
        info.update({
            "argv": _argv("<prompt>", self.resolve_model(task, model)),
            "cwd": str(workspace),
            "env_keys": ["CURSOR_API_KEY (or prior `cursor-agent login`)"],
        })
        return info

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
        model_name = self.resolve_model(task, model) or "(cursor default)"
        timeout = timeout_seconds or task.timeout_seconds
        self.prepare_workspace(task, workspace)

        trace = RunTrace.open_at(
            run_dir,
            prompt_path=task.source_path or Path("<inline>"),
            model=model_name,
            workspace=str(workspace),
            harness=self.name,
            run_id=run_id,
        )
        round_rec = trace.start_round(1)
        trace.record_plan_steps(round_rec, ["cursor-agent session (agent plans internally)"])
        step = trace.start_step(round_rec, 1, "cursor-agent -p session")

        events_path = run_dir / "cursor_events.jsonl"
        status, error, tail = "completed", None, ""
        try:
            proc = subprocess.run(
                _argv(task.render_prompt(self.name),
                      self.resolve_model(task, model)),
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            events_path.write_text(proc.stdout, encoding="utf-8")
            if proc.stderr:
                (run_dir / "cursor_stderr.log").write_text(proc.stderr, encoding="utf-8")
            tail = _events_tail(proc.stdout)
            if proc.returncode != 0:
                status = "errored"
                error = f"cursor-agent exit {proc.returncode}: {proc.stderr[-500:]}"
        except subprocess.TimeoutExpired:
            status, error = "timeout", f"exceeded {timeout}s (known -p hang mode — killed)"
        except FileNotFoundError:
            status, error = "errored", f"'{CURSOR_BIN}' not found on PATH"
        except KeyboardInterrupt:
            status, error = "interrupted", "KeyboardInterrupt"

        trace.finish_step(step, ok=(status == "completed"),
                          result_text=tail or (error or ""))
        trace.record_artifacts(workspace, expected_artifacts=task.expected_artifacts)
        if status == "completed":
            trace.mark_completed()
        elif status == "interrupted":
            trace.mark_interrupted()
        else:
            trace.mark_errored(RuntimeError(error or status))

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


def _events_tail(stdout: str, n: int = 5) -> str:
    lines = [ln for ln in stdout.strip().splitlines() if ln.strip()][-n:]
    out = []
    for ln in lines:
        try:
            evt = json.loads(ln)
            out.append(str(evt.get("type") or "event") + ": " + str(evt)[:300])
        except json.JSONDecodeError:
            out.append(ln[:300])
    return "\n".join(out)
