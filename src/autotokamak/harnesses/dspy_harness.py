# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""DSPy harness — plan → per-step ReAct execution → review → fix rounds.

Generalized from the original ``just_dspy/run_just_dspy.py`` capability-test
runner: same signatures, same jailed workspace tools, same campaign module,
now parameterized by ``TaskSpec`` so any benchmark task runs on the DSPy
substrate. Architecture stays intentionally symmetric with the URSA harness
(plan → per-step execute threading a previous-step summary) so the two
conditions differ in agent substrate, not scaffolding shape.

DSPy idioms (mirroring src/autotokamak/agent/dspy/):
  * ``dspy.Signature`` subclasses whose DOCSTRING is the prompt.
  * ``dspy.ChainOfThought`` for pure-reasoning stages (plan, review).
  * ``dspy.ReAct`` with plain-function tools for the acting stage.
  * One ``dspy.Module`` composing them, control flow in ``forward``.

Env: OPENAI_API_KEY (litellm model strings, e.g. "openai/gpt-5.2").
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

from autotokamak.agent.runners.config import REPO_ROOT
from autotokamak.bench.taskspec import TaskSpec
from autotokamak.bench.trace import RunTrace, utc_run_id
from autotokamak.harnesses.base import Harness, RunResult

MAX_TOOL_OUTPUT_CHARS = 8_000
MAX_SHELL_TIMEOUT = 4 * 3600  # a full campaign may run in one command
DEFAULT_MODEL = "openai/gpt-5.2"
DEFAULT_MAX_ITERS_PER_STEP = 40


def _signatures():
    """Signature classes, built lazily so importing this module never needs dspy."""
    import dspy

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

    return PlanCampaign, ExecuteStep, ReviewDeliverables


# ---------------------------------------------------------------------------
# Workspace tools — plain functions; DSPy builds tool schemas from the
# docstrings and type hints. All closures over one workspace root.
# ---------------------------------------------------------------------------

def make_tools(workspace: Path, tool_log: list[dict]):
    workspace = workspace.resolve()

    def _resolve(path: str) -> Path:
        p = (workspace / path).resolve()
        if not p.is_relative_to(workspace):
            # Task-provided symlinks (e.g. the OpenFUSIONToolkit reference
            # clone) resolve outside the workspace by construction; paths
            # whose UNRESOLVED form stays inside the jail are legitimate —
            # other substrates can read this material, so must dspy.
            unresolved = Path(os.path.normpath(workspace / path))
            if not unresolved.is_relative_to(workspace):
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


def _build_campaign(workspace: Path, tool_log: list[dict],
                    max_iters_per_step: int, fix_rounds: int):
    """The dspy.Module: plan -> per-step ReAct -> deliverables review -> fix-ups."""
    import dspy

    PlanCampaign, ExecuteStep, ReviewDeliverables = _signatures()
    tools = make_tools(workspace, tool_log)

    class Campaign(dspy.Module):
        def __init__(self):
            super().__init__()
            self._list_files = tools[2]
            self._read_file = tools[1]
            self.fix_rounds = fix_rounds
            self.plan = dspy.ChainOfThought(PlanCampaign)
            self.execute = dspy.ReAct(ExecuteStep, tools=tools, max_iters=max_iters_per_step)
            self.review = dspy.ChainOfThought(ReviewDeliverables)

        def forward(self, task: str, trace: RunTrace):
            plan = self.plan(task=task)
            round_rec = trace.start_round(1)
            trace.record_plan_steps(round_rec, list(plan.steps))
            print("\n=== PLAN ===")
            for i, s in enumerate(plan.steps, 1):
                print(f"{i}. {s}\n")

            last_summary = "No previous step."
            print("=== EXECUTION ===")
            for i, step in enumerate(plan.steps, 1):
                step_rec = trace.start_step(round_rec, i, f"step {i}")
                try:
                    result = self.execute(task=task, previous_summary=last_summary, step=step)
                    last_summary = result.summary
                    trace.finish_step(step_rec, ok=True, result_text=last_summary)
                    ok = True
                except Exception as exc:  # noqa: BLE001 — record and continue
                    last_summary = f"[execution error] {type(exc).__name__}: {exc}"
                    trace.finish_step(step_rec, ok=False, error=exc, result_text=last_summary)
                    ok = False
                print(f"\n--- Step {i}/{len(plan.steps)} ({'ok' if ok else 'ERROR'}) ---\n{last_summary}")

            for round_idx in range(self.fix_rounds + 1):
                listing = self._list_files(".")
                report = self._read_file("report.json")
                verdict = self.review(task=task, workspace_listing=listing, report_json=report)
                print(f"\n=== REVIEW (round {round_idx}) === complete={verdict.complete}")
                if verdict.complete or round_idx == self.fix_rounds:
                    break
                missing = "\n".join(f"- {m}" for m in verdict.missing)
                print(f"Missing:\n{missing}\nRunning fix-up round...")
                fix_round = trace.start_round(round_idx + 2)
                fix_step_text = (
                    "A deliverables audit found the following missing or broken items. "
                    "Fix ONLY these, re-running code where needed, and verify each "
                    f"fix:\n{missing}"
                )
                trace.record_plan_steps(fix_round, [fix_step_text])
                step_rec = trace.start_step(fix_round, 1, f"fix-{round_idx}")
                try:
                    result = self.execute(task=task, previous_summary=last_summary, step=fix_step_text)
                    last_summary = result.summary
                    trace.finish_step(step_rec, ok=True, result_text=last_summary)
                except Exception as exc:  # noqa: BLE001
                    last_summary = f"[execution error] {type(exc).__name__}: {exc}"
                    trace.finish_step(step_rec, ok=False, error=exc, result_text=last_summary)

            return last_summary

    return Campaign()


class DspyHarness(Harness):
    name = "dspy"

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
        # litellm convention: "openai/gpt-5.2"; accept our "openai:..." form too.
        model_name = (self.resolve_model(task, model) or DEFAULT_MODEL).replace(":", "/", 1)
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
        lm = None
        try:
            import dspy

            lm = dspy.LM(model_name, temperature=1.0, max_tokens=32_000)
            dspy.configure(lm=lm)
        except Exception as exc:  # noqa: BLE001 — missing dep/bad model must
            # still yield a result.json, like every other adapter
            trace.mark_errored(exc)
            return RunResult(
                status="errored", run_id=run_id,
                condition=self.condition_for(task), harness=self.name,
                model=model_name, workspace=workspace, trace_path=trace._path,
                wall_seconds=time.time() - started,
                error=f"{type(exc).__name__}: {exc}",
            )

        tool_log: list[dict] = []
        campaign = _build_campaign(
            workspace, tool_log,
            max_iters_per_step=DEFAULT_MAX_ITERS_PER_STEP,
            # fix rounds beyond the first pass = feedback_rounds - 1
            fix_rounds=max(0, task.feedback_rounds - 1),
        )

        try:
            final = campaign(task=task.render_prompt(self.name), trace=trace)
            trace.record_artifacts(workspace, expected_artifacts=task.expected_artifacts)
            from autotokamak.bench.scoring import try_score

            score = try_score(workspace, task.scorer, task.scorer_kwargs)
            if score is not None:
                trace.record_score(score)
            trace.mark_completed()
            print(f"\n=== FINAL ===\n{final}\nWorkspace: {workspace}")
        except KeyboardInterrupt:
            trace.mark_interrupted()
            status, error = "interrupted", "KeyboardInterrupt"
        except Exception as exc:  # noqa: BLE001
            trace.mark_errored(exc)
            status, error = "errored", f"{type(exc).__name__}: {exc}"
        finally:
            with (run_dir / "tool_log.jsonl").open("w", encoding="utf-8") as f:
                for rec in tool_log:
                    f.write(json.dumps(rec) + "\n")

        # litellm computes per-call cost + usage; harvest from the LM history.
        cost_usd, usage = None, None
        try:
            history = getattr(lm, "history", None) or []
            costs = [h.get("cost") for h in history if h.get("cost") is not None]
            if costs:
                cost_usd = round(float(sum(costs)), 6)
            prompt_toks = completion_toks = 0
            for h in history:
                u = h.get("usage") or {}
                prompt_toks += int(u.get("prompt_tokens") or 0)
                completion_toks += int(u.get("completion_tokens") or 0)
            if prompt_toks or completion_toks:
                usage = {"prompt_tokens": prompt_toks,
                         "completion_tokens": completion_toks,
                         "n_lm_calls": len(history)}
        except Exception:  # noqa: BLE001 — usage capture must never fail a run
            pass

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
            extra={"n_tool_calls": len(tool_log),
                   **({"usage": usage} if usage else {})},
        )
