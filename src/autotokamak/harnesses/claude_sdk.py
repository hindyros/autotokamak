# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Claude Agent SDK harness.

Runs the task through ``claude_agent_sdk.query()`` headlessly:
  * ``cwd`` jailed to the workspace, ``permission_mode="bypassPermissions"``
    (no interactive prompts — high-trust, same model as URSA's Bash access;
    the L3 no-autotokamak rule is enforced by the contract audit, not the
    sandbox),
  * ``setting_sources=[]`` so THIS repo's .claude/ skills, CLAUDE.md, and
    settings never leak into the benchmarked agent — without it the L3
    condition would be contaminated by our own tooling,
  * web tools disabled: the task is offline by design.

Requires: ``pip install claude-agent-sdk`` (the ``[harnesses]`` extra) and
the Claude Code CLI on PATH. Env: ANTHROPIC_API_KEY.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Optional

from autotokamak.bench.taskspec import TaskSpec
from autotokamak.bench.trace import RunTrace, utc_run_id
from autotokamak.harnesses.base import Harness, RunResult

DEFAULT_MODEL = "claude-sonnet-4-5"
ALLOWED_TOOLS = ["Read", "Write", "Edit", "Bash", "Glob", "Grep"]
DISALLOWED_TOOLS = ["WebSearch", "WebFetch"]
MAX_TURNS = 300


class ClaudeSdkHarness(Harness):
    name = "claude_sdk"

    def dry_run_info(self, task: TaskSpec, workspace: Path,
                     model: Optional[str] = None) -> dict[str, Any]:
        info = super().dry_run_info(task, workspace, model)
        info.update({
            "model": self.resolve_model(task, model) or DEFAULT_MODEL,
            "invocation": "claude_agent_sdk.query(prompt, ClaudeAgentOptions(...))",
            "options": {
                "cwd": str(workspace),
                "permission_mode": "bypassPermissions",
                "allowed_tools": ALLOWED_TOOLS,
                "disallowed_tools": DISALLOWED_TOOLS,
                "setting_sources": [],
                "max_turns": MAX_TURNS,
            },
            "env_keys": ["ANTHROPIC_API_KEY"],
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
        model_name = self.resolve_model(task, model) or DEFAULT_MODEL
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
        trace.record_plan_steps(round_rec, ["claude_sdk session (agent plans internally)"])

        events_path = run_dir / "claude_events.jsonl"
        status, error, cost_usd, num_turns = "completed", None, None, None

        async def _session() -> None:
            nonlocal cost_usd, num_turns, error, status
            from claude_agent_sdk import ClaudeAgentOptions, query

            options = ClaudeAgentOptions(
                cwd=str(workspace),
                permission_mode="bypassPermissions",
                allowed_tools=ALLOWED_TOOLS,
                disallowed_tools=DISALLOWED_TOOLS,
                setting_sources=[],
                model=model_name,
                max_turns=MAX_TURNS,
            )
            # Runtime note, not task info: the Bash tool's cwd persists across
            # calls, so an agent that cd's away (e.g. through the OFT symlink)
            # silently litters the enclosing repo. Pin it to absolute paths.
            prompt = (
                task.render_prompt(self.name)
                + f"\n\nRUNTIME NOTE: your workspace directory is\n{workspace.resolve()}\n"
                  "Create ALL files under this directory (absolute paths are "
                  "safest). Never write outside it, and if you change "
                  "directory for exploration, change back before writing."
            )
            step_no = 0
            with events_path.open("w", encoding="utf-8") as events:
                async for message in query(prompt=prompt, options=options):
                    kind = type(message).__name__
                    payload = _message_payload(message)
                    events.write(json.dumps({"kind": kind, **payload}, default=str) + "\n")
                    events.flush()

                    if kind == "AssistantMessage":
                        step_no += 1
                        step = trace.start_step(round_rec, step_no, f"assistant turn {step_no}")
                        trace.finish_step(step, ok=True,
                                          result_text=payload.get("text", "")[:2000])
                    elif kind == "ResultMessage":
                        cost_usd = payload.get("total_cost_usd")
                        num_turns = payload.get("num_turns")
                        subtype = payload.get("subtype")
                        if subtype and subtype != "success":
                            status = "errored"
                            error = f"ResultMessage.subtype={subtype}"

        try:
            asyncio.run(asyncio.wait_for(_session(), timeout=timeout))
        except asyncio.TimeoutError:
            status, error = "timeout", f"exceeded {timeout}s"
        except KeyboardInterrupt:
            status, error = "interrupted", "KeyboardInterrupt"
        except Exception as exc:  # noqa: BLE001
            status, error = "errored", f"{type(exc).__name__}: {exc}"

        # An agent that escaped its cwd leaves the workspace empty (symlinks
        # aside) while claiming success — flag it instead of trusting it.
        if status == "completed":
            written = [p for p in workspace.iterdir() if not p.is_symlink()]
            if not written:
                status = "errored"
                error = ("workspace empty after a 'successful' session — the "
                         "agent likely wrote its files outside its cwd")

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
            extra={"num_turns": num_turns} if num_turns is not None else {},
        )


def _message_payload(message: Any) -> dict[str, Any]:
    """Best-effort flatten of SDK message objects (robust across SDK versions)."""
    out: dict[str, Any] = {}
    for attr in ("subtype", "total_cost_usd", "num_turns", "duration_ms", "is_error"):
        val = getattr(message, attr, None)
        if val is not None:
            out[attr] = val
    content = getattr(message, "content", None)
    if isinstance(content, list):
        texts, tools = [], []
        for block in content:
            text = getattr(block, "text", None)
            if text:
                texts.append(str(text))
            tool = getattr(block, "name", None)
            if tool and type(block).__name__ == "ToolUseBlock":
                tools.append(str(tool))
        if texts:
            out["text"] = "\n".join(texts)
        if tools:
            out["tools"] = tools
    result = getattr(message, "result", None)
    if isinstance(result, str):
        out["result"] = result[:2000]
    return out
