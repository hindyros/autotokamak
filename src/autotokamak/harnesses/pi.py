# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Pi Code harness (pi.dev — badlogic/pi-mono coding agent).

Subprocess invocation; pi has NO cwd flag, so ``cwd=workspace`` on the
subprocess is load-bearing for the jail:

    pi --mode json -p --no-session [--provider P --model M] "<prompt>"

``--mode json`` streams every agent event as a JSON line on stdout — the raw
stream is kept at ``run_dir/pi_events.jsonl`` (pi is fast-moving; the event
schema may drift, so trace mapping is best-effort and the raw file is the
source of truth). Tools restricted to the filesystem/shell builtins.

Env: provider key (ANTHROPIC_API_KEY / OPENAI_API_KEY, per --provider).
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

PI_BIN = "pi"
PI_TOOLS = "read,bash,edit,write,grep,find,ls"


def _argv(prompt: str, model: Optional[str]) -> list[str]:
    argv = [PI_BIN, "--mode", "json", "-p", "--no-session", "--tools", PI_TOOLS]
    if model:
        # our convention "anthropic:claude-x" → --provider anthropic --model claude-x
        provider, sep, model_id = model.partition(":")
        if sep:
            argv += ["--provider", provider, "--model", model_id]
        else:
            argv += ["--model", model]
    argv.append(prompt)
    return argv


class PiHarness(Harness):
    name = "pi"

    def dry_run_info(self, task: TaskSpec, workspace: Path,
                     model: Optional[str] = None) -> dict[str, Any]:
        info = super().dry_run_info(task, workspace, model)
        resolved = self.resolve_model(task, model)
        info.update({
            "argv": _argv("<prompt>", resolved),
            "cwd": str(workspace),
            "env_keys": ["ANTHROPIC_API_KEY or OPENAI_API_KEY (per provider)"],
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
        model_name = self.resolve_model(task, model) or "(pi default)"
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
        trace.record_plan_steps(round_rec, ["pi session (agent plans internally)"])
        step = trace.start_step(round_rec, 1, "pi -p session")

        events_path = run_dir / "pi_events.jsonl"
        status, error, tail = "completed", None, ""
        cost_usd, usage = None, None
        try:
            proc = subprocess.run(
                _argv(task.render_prompt(self.name) + self.workspace_note(workspace),
                      self.resolve_model(task, model)),
                cwd=workspace,           # load-bearing: pi has no cwd flag
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            events_path.write_text(proc.stdout, encoding="utf-8")
            if proc.stderr:
                (run_dir / "pi_stderr.log").write_text(proc.stderr, encoding="utf-8")
            tail = _events_tail(proc.stdout)
            cost_usd, usage = _usage_from_events(proc.stdout)
            if proc.returncode != 0:
                status = "errored"
                error = f"pi exit {proc.returncode}: {proc.stderr[-500:]}"
        except subprocess.TimeoutExpired as exc:
            status, error = "timeout", f"exceeded {timeout}s"
            # keep whatever the stream produced before the kill — a timed-out
            # run must not lose its diagnostics and usage data
            partial = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            if partial:
                events_path.write_text(partial, encoding="utf-8")
                cost_usd, usage = _usage_from_events(partial)
        except FileNotFoundError:
            status, error = "errored", f"'{PI_BIN}' not found on PATH (npm i -g @mariozechner/pi-coding-agent)"
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
            cost_usd=cost_usd,
            error=error,
            extra={"usage": usage} if usage else {},
        )


def _usage_from_events(stdout: str) -> tuple[Optional[float], Optional[dict]]:
    """Sum pi's per-message usage/cost (``message_end`` events only —
    ``turn_end`` repeats the same message and would double-count)."""
    cost_total = 0.0
    tokens = {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}
    seen = False
    for ln in stdout.splitlines():
        try:
            evt = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if evt.get("type") != "message_end":
            continue
        u = (evt.get("message") or {}).get("usage") or {}
        if not u:
            continue
        seen = True
        cost_total += float((u.get("cost") or {}).get("total") or 0.0)
        for k in tokens:
            tokens[k] += int(u.get(k) or 0)
    if not seen:
        return None, None
    return round(cost_total, 6), tokens


def _events_tail(stdout: str, n: int = 5) -> str:
    """Human-readable digest of the last few JSON events (best-effort)."""
    lines = [ln for ln in stdout.strip().splitlines() if ln.strip()][-n:]
    out = []
    for ln in lines:
        try:
            evt = json.loads(ln)
            out.append(str(evt.get("type") or evt.get("kind") or "event")
                       + ": " + str(evt)[:300])
        except json.JSONDecodeError:
            out.append(ln[:300])
    return "\n".join(out)
