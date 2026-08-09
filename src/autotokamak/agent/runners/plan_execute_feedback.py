#!/usr/bin/env python3
# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Plan–execute runner with a global feedback loop.

Same interface as agent.plan_execute, but after the first plan+execute cycle the planner
is invoked again with the execution history; it can propose follow-up steps (e.g. fix
failures) or confirm completion. Repeats for up to feedback_rounds (config), then
optionally runs a validate_after review step.

Config (YAML) may include:
  problem: ...
  workspace: ...
  model: ...
  symlink: {...} or symlinks: [...]
  feedback_rounds: 2   # max plan+execute cycles (default 2)
  validate_after: true # run a post-execution review step (default false)

Each invocation also writes a structured trace to ``experiments/<run_id>/trace.json``
(unless ``--no-trace`` is given). The trace shape is defined in
``bench.trace`` and is the substrate for the DSPy integration plan
(see ``docs/dspy_integration_plan.md``).
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from autotokamak.agent.runners.config import (
    REPO_ROOT,
    load_config,
    materialize_symlinks,
    resolve_workspace,
)
from autotokamak.bench.scoring import try_score
from autotokamak.bench.trace import RunTrace

load_dotenv(REPO_ROOT / ".env")

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

from ursa.agents import ExecutionAgent, PlanningAgent


DEFAULT_EXPERIMENTS_DIR = REPO_ROOT / "experiments"


def _hygiene_enforce(workspace_path, cfg) -> str:
    """Auto-delete extra root files and return a note for the planner.

    Reads ``allowed_root_files`` and ``infra_root_files`` from the prompt YAML.
    If ``allowed_root_files`` is absent, returns "" (backward-compatible with
    old prompts).  When extras are found they are deleted immediately so the
    agent cannot re-read them; the planner sees a clean workspace and a record
    of what was removed.
    """
    import shutil
    from pathlib import Path as _P

    allowed = set(getattr(cfg, "allowed_root_files", None) or [])
    infra = set(getattr(cfg, "infra_root_files", None) or [])
    if not allowed:
        return ""
    ws = _P(workspace_path)
    present = {p.name for p in ws.iterdir()}
    extras = sorted(present - allowed - infra)
    if not extras:
        return (
            "WORKSPACE HYGIENE: Root layout is clean — only the allowed "
            "deliverables and URSA infra files are present. Do not add any new "
            "files at the root in the next round.\n\n"
        )
    deleted, failed = [], []
    for name in extras:
        p = ws / name
        try:
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
            deleted.append(name)
        except OSError as exc:
            failed.append(f"{name} ({exc})")
    lines = ["WORKSPACE HYGIENE: The following files were AUTOMATICALLY DELETED "
             "because they are not in the allowed deliverables list "
             f"(allowed: {sorted(allowed)}; URSA infra: {sorted(infra)}):"]
    for name in deleted:
        lines.append(f"  - {name}  [deleted]")
    for name in failed:
        lines.append(f"  - {name}  [deletion failed — delete manually]")
    lines.append(
        "Do NOT recreate these. Use any freed budget for additional Optuna trials. "
        "The deliverable list is complete and normative.\n"
    )
    return "\n".join(lines) + "\n"


def _hygiene_warning(workspace_path, cfg) -> str:
    """Alias kept for callers; now delegates to _hygiene_enforce."""
    return _hygiene_enforce(workspace_path, cfg)


def run_feedback_loop(
    *,
    problem: str,
    workspace_path,
    model_name: str,
    feedback_rounds: int = 2,
    validate_after: bool = False,
    trace: RunTrace | None = None,
    expected_artifacts=None,
    scorer_dotted: str | None = None,
    scorer_kwargs: dict | None = None,
) -> str:
    """The plan→execute→re-plan engine, decoupled from YAML/CLI plumbing.

    ``main`` (the YAML-config CLI) and ``harnesses.ursa`` (the benchmark
    adapter) both call this. Returns the final execution summary. The
    workspace must already exist with any symlinks materialized; ``trace``
    (if given) is finalized here — completed/errored/interrupted.
    """
    scorer_kwargs = scorer_kwargs or {}
    workspace = str(workspace_path)

    planner_llm = init_chat_model(model=model_name)
    executor_llm = init_chat_model(model=model_name)

    planner = PlanningAgent(
        llm=planner_llm,
        thread_id="demo_planner",
        workspace=workspace,
    )
    executor = ExecutionAgent(
        llm=executor_llm,
        thread_id="demo_executor",
        workspace=workspace,
    )

    execution_history: list[str] = []
    last_summary = "No previous step."

    try:
        for round_no in range(1, feedback_rounds + 1):
            round_rec = trace.start_round(round_no) if trace else None

            if round_no == 1:
                planning_output = planner.invoke(problem)
            else:
                print("\n=== GLOBAL FEEDBACK: RE-PLAN (round {}) ===".format(round_no))
                replan_prompt = (
                    f"Original problem:\n{problem}\n\n"
                    f"Execution history so far:\n"
                    + "\n---\n".join(execution_history)
                    + "\n\n"
                    f"Based on the above, suggest follow-up steps to fix failures or complete the task. "
                    f"If nothing more is needed, return a plan with a single step: 'Confirm completion'."
                )
                planning_output = planner.invoke(replan_prompt)

            steps = planning_output["plan"].steps
            if trace and round_rec is not None:
                trace.record_plan_steps(round_rec, steps)

            if not steps:
                print("No steps in plan; stopping.")
                break

            print("\n=== PLAN (round {}) ===".format(round_no))
            for i, s in enumerate(steps, 1):
                name = getattr(s, "name", f"Step {i}")
                desc = getattr(s, "description", str(s))
                print(f"  {i}. {name}\n     {desc}\n")

            print("\n=== EXECUTION (round {}) ===".format(round_no))
            last_summary = "No previous step."

            for i, step in enumerate(steps, 1):
                step_name = getattr(step, "name", f"Step {i}")
                step_desc = getattr(step, "description", str(step))
                step_text = f"{step_name}\n{step_desc}"
                prompt = (
                    f"You are executing a multi-step plan.\n\n"
                    f"Overall problem:\n{problem}\n\n"
                    f"Previous-step summary:\n{last_summary}\n\n"
                    f"Current step:\n{step_text}\n\n"
                    f"Execute this step fully. Use tools if helpful. "
                    f"If you write code, save it in the workspace.\n"
                )
                step_rec = trace.start_step(round_rec, i, step_name) if (trace and round_rec) else None
                try:
                    result = executor.invoke(
                        {
                            "messages": [HumanMessage(content=prompt)],
                            "workspace": workspace,
                            "symlinkdir": None,
                        }
                    )
                    last_summary = result["messages"][-1].text
                    if trace and step_rec is not None:
                        trace.finish_step(step_rec, ok=True, result_text=last_summary)
                    print(f"\n--- Step {i} result ---\n{last_summary}")
                except Exception as exc:  # noqa: BLE001
                    last_summary = f"[execution error] {type(exc).__name__}: {exc}"
                    if trace and step_rec is not None:
                        trace.finish_step(step_rec, ok=False, error=exc, result_text=last_summary)
                    print(f"\n--- Step {i} ERROR ---\n{last_summary}", file=sys.stderr)
                    # Re-raise: a hard failure mid-step should bubble up; the
                    # planner's feedback loop only handles soft-failure summaries.
                    raise

            execution_history.append(last_summary)

            if (
                len(steps) == 1
                and "confirm completion" in last_summary.lower()
            ):
                print("\nPlanner confirmed completion; stopping feedback loop.")
                break

        if validate_after:
            print("\n=== VALIDATE (post-execution review) ===")
            validate_prompt = (
                f"Review the workspace and execution results for this task.\n\n"
                f"Problem:\n{problem}\n\n"
                f"Execution summary:\n{last_summary}\n\n"
                f"Did the task succeed? If not, what failed or is missing? Be brief."
            )
            # NOTE: bugfix — used to reference `symlinkdict` which was removed
            # in the materialize_symlinks refactor (code review C1). Now None,
            # consistent with the main loop.
            result = executor.invoke(
                {
                    "messages": [HumanMessage(content=validate_prompt)],
                    "workspace": workspace,
                    "symlinkdir": None,
                }
            )
            print(result["messages"][-1].text)

        if trace:
            trace.record_artifacts(workspace_path, expected_artifacts=expected_artifacts)
            score = try_score(workspace_path, scorer_dotted, scorer_kwargs)
            if score is not None:
                trace.record_score(score)
                print(f"\nScore: {score.total:.3f}")
            trace.mark_completed()

        print("\n=== FINAL ===")
        print(last_summary)
        print(f"\nWorkspace: {Path(workspace_path).resolve()}")
        return last_summary

    except KeyboardInterrupt:
        if trace:
            try:
                trace.record_artifacts(workspace_path, expected_artifacts=expected_artifacts)
            except Exception:  # noqa: BLE001
                pass
            trace.mark_interrupted()
        raise
    except Exception as exc:
        if trace:
            try:
                trace.record_artifacts(workspace_path, expected_artifacts=expected_artifacts)
            except Exception:  # noqa: BLE001
                pass
            trace.mark_errored(exc)
        raise


def main(
    config_path: str,
    cli_model: str | None,
    workspace_override: str | None,
    *,
    trace_enabled: bool = True,
    experiments_dir=None,
):
    """YAML-config CLI wrapper around ``run_feedback_loop``."""
    cfg = load_config(config_path)

    problem = getattr(cfg, "problem", None)
    if not problem:
        raise ValueError("config.yaml must contain a top-level 'problem:' string")

    scorer_dotted = getattr(cfg, "scorer", None)
    scorer_kwargs = getattr(cfg, "scorer_kwargs", None) or {}
    expected_artifacts = getattr(cfg, "expected_artifacts", None)

    model_name = (
        cli_model
        or getattr(cfg, "model", None)
        or "openai:gpt-5-mini"
    )
    print(f"\nUsing model: {model_name}")

    workspace_path = resolve_workspace(
        workspace_override
        or getattr(cfg, "workspace", None)
        or "mini_workspace"
    )
    workspace_path.mkdir(parents=True, exist_ok=True)

    # URSA only supports a single symlinkdir dict; we materialize the YAML's
    # `symlinks:` list ourselves and pass None to URSA to skip its broken path.
    symlink_entries = getattr(cfg, "symlinks", None) or getattr(cfg, "symlink", None)
    if isinstance(symlink_entries, dict):
        symlink_entries = [symlink_entries]
    materialize_symlinks(workspace_path, symlink_entries)

    feedback_rounds = max(1, int(getattr(cfg, "feedback_rounds", 2)))
    validate_after = getattr(cfg, "validate_after", False)

    # Structured trace for DSPy integration. All trace I/O is best-effort
    # inside RunTrace.save() — failures will not abort the run.
    trace: RunTrace | None = None
    if trace_enabled:
        try:
            target_dir = experiments_dir if experiments_dir is not None else DEFAULT_EXPERIMENTS_DIR
            trace = RunTrace.open(
                experiments_dir=target_dir,
                prompt_path=REPO_ROOT / config_path if not str(config_path).startswith("/") else config_path,
                model=model_name,
                feedback_rounds=feedback_rounds,
                workspace=str(workspace_path),
            )
            print(f"Trace: {trace._path}")
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: failed to open trace ({type(exc).__name__}: {exc}); continuing without it", file=sys.stderr)
            trace = None

    return run_feedback_loop(
        problem=problem,
        workspace_path=workspace_path,
        model_name=model_name,
        feedback_rounds=feedback_rounds,
        validate_after=validate_after,
        trace=trace,
        expected_artifacts=expected_artifacts,
        scorer_dotted=scorer_dotted,
        scorer_kwargs=scorer_kwargs,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plan–execute with global feedback loop (re-plan after execution).",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config YAML (problem, workspace, feedback_rounds, validate_after, etc.).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model string for init_chat_model (e.g. openai:gpt-5-mini).",
    )
    parser.add_argument(
        "--workspace",
        default=None,
        help="Override workspace directory.",
    )
    parser.add_argument(
        "--no-trace",
        action="store_true",
        help="Disable writing experiments/<run_id>/trace.json for this run.",
    )
    parser.add_argument(
        "--experiments-dir",
        default=None,
        help="Override the experiments root (default: <repo>/experiments).",
    )
    args = parser.parse_args()
    from pathlib import Path as _P
    main(
        args.config,
        args.model,
        args.workspace,
        trace_enabled=not args.no_trace,
        experiments_dir=_P(args.experiments_dir) if args.experiments_dir else None,
    )
