#!/usr/bin/env python
# provenance: Human/Claude-authored (DSPy capability-test harness; drives the DSPy agent)
"""just_dspy runner: a DSPy-native plan->execute agent for the single-prompt
surrogate-campaign capability test.

DSPy idioms used deliberately (mirroring src/autotokamak/agent/dspy/):
  * ``dspy.Signature`` subclasses whose DOCSTRING is the prompt and whose
    typed Input/Output fields define the structured-output contract.
  * ``dspy.ChainOfThought`` for the pure-reasoning stages (plan, review).
  * ``dspy.ReAct`` with plain-function tools (docstring + type hints become
    the tool schema) for the acting stage.
  * One ``dspy.Module`` composing them, with control flow in ``forward``.

Architecture is intentionally symmetric with agent/runners/plan_execute.py
(plan -> per-step execute threading a previous-step summary) so the URSA and
DSPy conditions differ in agent substrate, not in scaffolding shape.

Run:
  python just_dspy/run_just_dspy.py --config just_dspy/task.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import dspy
import yaml
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

load_dotenv(REPO_ROOT / ".env")

MAX_TOOL_OUTPUT_CHARS = 8_000
MAX_SHELL_TIMEOUT = 4 * 3600  # the full campaign may run in one command


# ---------------------------------------------------------------------------
# Signatures — the docstring is the prompt (repo convention; GEPA-mutable).
# ---------------------------------------------------------------------------

class PlanCampaign(dspy.Signature):
    """Decompose an autonomous engineering task into 4-8 sequential steps.

    Each step must be concretely executable inside a shell+filesystem
    workspace and produce verifiable progress (files written, code run,
    outputs inspected). Order steps so that risky unknowns (e.g. driving an
    unfamiliar physics solver) are de-risked early with small experiments
    before committing the expensive compute budget. The final step must be
    verification that every required deliverable exists and is valid.
    """

    task: str = dspy.InputField(desc="The full task specification, verbatim.")
    steps: list[str] = dspy.OutputField(
        desc="Ordered, self-contained step descriptions; each 2-5 sentences."
    )


class ExecuteStep(dspy.Signature):
    """Fully execute one step of a multi-step engineering campaign.

    You are inside the campaign workspace. Use the tools to write code, run
    it, and inspect the results. Iterate — run, read errors, fix, re-run —
    until this step is genuinely complete; do not declare success on code
    you have not executed. Respect every CONSTRAINT in the task. Long
    commands are fine: pass a large timeout_seconds to run_shell for the
    data-generation campaign itself.
    """

    task: str = dspy.InputField(desc="The full task specification, verbatim.")
    previous_summary: str = dspy.InputField(
        desc="What earlier steps accomplished: artifacts and their paths, "
             "key results, unresolved issues."
    )
    step: str = dspy.InputField(desc="The current step to complete.")
    summary: str = dspy.OutputField(
        desc="What was accomplished: files created (paths), commands run, "
             "quantitative results, and any open issues for later steps."
    )


class ReviewDeliverables(dspy.Signature):
    """Audit a finished campaign workspace against the task's deliverables.

    Judge only from the evidence given. A deliverable is missing if the file
    is absent, empty, or plainly inconsistent with the task's required
    interface/schema. Do not speculate about file contents you cannot see.
    """

    task: str = dspy.InputField(desc="The full task specification, verbatim.")
    workspace_listing: str = dspy.InputField(desc="Recursive file listing of the workspace.")
    report_json: str = dspy.InputField(desc="Contents of report.json, or the read error.")
    complete: bool = dspy.OutputField(desc="True iff every deliverable is present and plausible.")
    missing: list[str] = dspy.OutputField(
        desc="Each missing/broken deliverable with one sentence on what is wrong; [] if complete."
    )


# ---------------------------------------------------------------------------
# Workspace tools — plain functions; DSPy builds tool schemas from the
# docstrings and type hints. All closures over one workspace root.
# ---------------------------------------------------------------------------

def make_tools(workspace: Path, tool_log: list[dict]):
    workspace = workspace.resolve()

    def _resolve(path: str) -> Path:
        p = (workspace / path).resolve()
        if not p.is_relative_to(workspace):
            raise ValueError(f"path escapes the workspace: {path}")
        return p

    def _clip(text: str, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
        if len(text) <= limit:
            return text
        half = limit // 2
        return f"{text[:half]}\n... [{len(text) - limit} chars truncated] ...\n{text[-half:]}"

    def _record(tool: str, args: dict, result: str) -> str:
        tool_log.append(
            {
                "t": time.time(),
                "tool": tool,
                "args": {k: (v if len(str(v)) < 300 else f"{str(v)[:300]}...") for k, v in args.items()},
                "result_head": result[:300],
            }
        )
        return result

    def write_file(path: str, content: str) -> str:
        """Write (or overwrite) a text file at `path`, relative to the workspace root.

        Parent directories are created automatically. Returns a confirmation
        with the byte count.
        """
        p = _resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return _record("write_file", {"path": path, "content": content},
                       f"wrote {len(content.encode())} bytes to {path}")

    def read_file(path: str, max_chars: int = 20_000) -> str:
        """Read a text file at `path` (relative to the workspace root).

        Output beyond `max_chars` is truncated in the middle.
        """
        p = _resolve(path)
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return _record("read_file", {"path": path}, f"ERROR: {exc}")
        return _record("read_file", {"path": path}, _clip(text, max_chars))

    def list_files(path: str = ".") -> str:
        """Recursively list files under `path` (relative to the workspace root)
        with sizes in bytes. Symlinked directories are not descended into.
        """
        root = _resolve(path)
        lines: list[str] = []
        for p in sorted(root.rglob("*")):
            if any(part.startswith("__pycache__") for part in p.parts):
                continue
            if p.is_symlink():
                lines.append(f"{p.relative_to(workspace)} -> (symlink)")
            elif p.is_file():
                lines.append(f"{p.relative_to(workspace)}  {p.stat().st_size}")
            if len(lines) >= 400:
                lines.append("... [listing truncated at 400 entries]")
                break
        return _record("list_files", {"path": path}, "\n".join(lines) or "(empty)")

    def run_shell(command: str, timeout_seconds: int = 1800) -> str:
        """Run a shell command with the workspace as the working directory.

        `python` resolves to the project venv. stdout and stderr are captured
        (long output is middle-truncated). Use a generous `timeout_seconds`
        (up to 14400) for the data-generation campaign; the command is killed
        on timeout.
        """
        timeout_seconds = min(int(timeout_seconds), MAX_SHELL_TIMEOUT)
        env = os.environ.copy()
        env["PATH"] = f"{REPO_ROOT / 'venv' / 'bin'}:{env.get('PATH', '')}"
        env["PYTHONUNBUFFERED"] = "1"
        try:
            proc = subprocess.run(
                ["bash", "-c", command],
                cwd=workspace,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            out = (
                f"exit_code: {proc.returncode}\n"
                f"--- stdout ---\n{_clip(proc.stdout)}\n"
                f"--- stderr ---\n{_clip(proc.stderr)}"
            )
        except subprocess.TimeoutExpired:
            out = f"ERROR: command timed out after {timeout_seconds}s and was killed."
        return _record("run_shell", {"command": command, "timeout_seconds": timeout_seconds}, out)

    return [write_file, read_file, list_files, run_shell]


# ---------------------------------------------------------------------------
# The program
# ---------------------------------------------------------------------------

class SurrogateCampaign(dspy.Module):
    """Plan -> per-step ReAct execution -> deliverables review -> fix-up rounds."""

    def __init__(self, workspace: Path, tool_log: list[dict],
                 max_iters_per_step: int = 40, fix_rounds: int = 1):
        super().__init__()
        self.workspace = workspace
        self.fix_rounds = fix_rounds
        tools = make_tools(workspace, tool_log)
        self._list_files = tools[2]
        self._read_file = tools[1]
        self.plan = dspy.ChainOfThought(PlanCampaign)
        self.execute = dspy.ReAct(ExecuteStep, tools=tools, max_iters=max_iters_per_step)
        self.review = dspy.ChainOfThought(ReviewDeliverables)

    def forward(self, task: str, trace: dict):
        plan = self.plan(task=task)
        trace["plan"] = list(plan.steps)
        print("\n=== PLAN ===")
        for i, s in enumerate(plan.steps, 1):
            print(f"{i}. {s}\n")

        last_summary = "No previous step."
        print("=== EXECUTION ===")
        for i, step in enumerate(plan.steps, 1):
            t0 = time.time()
            try:
                result = self.execute(task=task, previous_summary=last_summary, step=step)
                last_summary = result.summary
                ok = True
            except Exception as exc:  # noqa: BLE001 — record and continue to next step
                last_summary = f"[execution error] {type(exc).__name__}: {exc}"
                ok = False
            trace["steps"].append(
                {"index": i, "step": step, "ok": ok,
                 "summary": last_summary, "seconds": round(time.time() - t0, 1)}
            )
            print(f"\n--- Step {i}/{len(plan.steps)} ({'ok' if ok else 'ERROR'}) ---\n{last_summary}")

        for round_idx in range(self.fix_rounds + 1):
            listing = self._list_files(".")
            report = self._read_file("report.json")
            verdict = self.review(task=task, workspace_listing=listing, report_json=report)
            trace["reviews"].append(
                {"round": round_idx, "complete": bool(verdict.complete), "missing": list(verdict.missing)}
            )
            print(f"\n=== REVIEW (round {round_idx}) === complete={verdict.complete}")
            if verdict.complete or round_idx == self.fix_rounds:
                break
            missing = "\n".join(f"- {m}" for m in verdict.missing)
            print(f"Missing:\n{missing}\nRunning fix-up round...")
            fix_step = (
                "A deliverables audit found the following missing or broken items. "
                "Fix ONLY these, re-running code where needed, and verify each "
                f"fix:\n{missing}"
            )
            result = self.execute(task=task, previous_summary=last_summary, step=fix_step)
            last_summary = result.summary
            trace["steps"].append(
                {"index": f"fix-{round_idx}", "step": fix_step, "ok": True,
                 "summary": last_summary, "seconds": None}
            )

        return last_summary


# ---------------------------------------------------------------------------
# Harness plumbing (mirrors agent/runners/config.py behavior, standalone)
# ---------------------------------------------------------------------------

def materialize_symlinks(workspace: Path, entries) -> None:
    for entry in entries or []:
        src = Path(entry["source"])
        src = (REPO_ROOT / src).resolve() if not src.is_absolute() else src.resolve()
        dst = workspace / entry["dest"]
        if dst.exists() or dst.is_symlink():
            continue
        if not src.exists():
            print(f"WARNING: symlink source missing, skipping: {src}")
            continue
        dst.symlink_to(src, target_is_directory=src.is_dir())
        print(f"Symlinked: {src} -> {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(HERE / "task.yaml"))
    parser.add_argument("--model", default=None, help="litellm model string override")
    parser.add_argument("--workspace", default=None)
    parser.add_argument("--fix-rounds", type=int, default=None)
    parser.add_argument("--max-iters", type=int, default=None, help="ReAct iters per step")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    problem = cfg["problem"]
    model = args.model or cfg.get("model", "openai/gpt-5.2")
    lm_kwargs = cfg.get("lm") or {}
    fix_rounds = args.fix_rounds if args.fix_rounds is not None else int(cfg.get("fix_rounds", 1))
    max_iters = args.max_iters or int(cfg.get("max_iters_per_step", 40))

    workspace = Path(args.workspace or cfg.get("workspace", "just_dspy/workspace"))
    if not workspace.is_absolute():
        workspace = REPO_ROOT / workspace
    workspace.mkdir(parents=True, exist_ok=True)
    materialize_symlinks(workspace, cfg.get("symlinks"))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    trace_dir = REPO_ROOT / "experiments" / f"just_dspy_{stamp}"
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace: dict = {
        "runner": "just_dspy", "model": model, "config": str(args.config),
        "workspace": str(workspace), "started_utc": stamp,
        "fix_rounds": fix_rounds, "max_iters_per_step": max_iters,
        "plan": [], "steps": [], "reviews": [], "status": "running",
    }
    tool_log: list[dict] = []

    def flush() -> None:
        trace["n_tool_calls"] = len(tool_log)
        (trace_dir / "trace.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")
        with (trace_dir / "tool_log.jsonl").open("w", encoding="utf-8") as f:
            for rec in tool_log:
                f.write(json.dumps(rec) + "\n")

    print(f"Using model: {model}")
    print(f"Workspace: {workspace}")
    print(f"Trace: {trace_dir / 'trace.json'}")
    dspy.configure(lm=dspy.LM(model, temperature=lm_kwargs.get("temperature", 1.0),
                              max_tokens=lm_kwargs.get("max_tokens", 32_000)))

    program = SurrogateCampaign(workspace, tool_log,
                                max_iters_per_step=max_iters, fix_rounds=fix_rounds)
    try:
        final = program(task=problem, trace=trace)
        trace["status"] = "completed"
        print(f"\n=== FINAL ===\n{final}\nWorkspace: {workspace}")
    except KeyboardInterrupt:
        trace["status"] = "interrupted"
        raise
    except Exception as exc:
        trace["status"] = f"errored: {type(exc).__name__}: {exc}"
        raise
    finally:
        flush()

    missing = [a for a in cfg.get("expected_artifacts", []) if not (workspace / a).exists()]
    if missing:
        print(f"\nWARNING: expected artifacts missing: {missing}", file=sys.stderr)


if __name__ == "__main__":
    main()
