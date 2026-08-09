# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Plan -> execute runner (single pass, no replan loop).

For the version with re-planning after execution, see
``agent.runners.plan_execute_feedback``.

Each invocation writes ``experiments/<run_id>/trace.json`` (unless
``--no-trace``); shape defined in ``agent.runners.trace``.
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
from autotokamak.agent.runners.scoring import try_score
from autotokamak.agent.runners.trace import RunTrace

load_dotenv(REPO_ROOT / ".env")

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

from ursa.agents import ExecutionAgent, PlanningAgent


DEFAULT_EXPERIMENTS_DIR = REPO_ROOT / "experiments"


def main(
    config_path: str,
    cli_model: str | None,
    workspace_override: str | None,
    *,
    trace_enabled: bool = True,
    experiments_dir=None,
):
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
    workspace = str(workspace_path)

    # URSA only supports a single symlinkdir dict; we materialize the YAML's
    # `symlinks:` list ourselves and pass None to URSA to skip its broken path.
    symlink_entries = getattr(cfg, "symlinks", None) or getattr(cfg, "symlink", None)
    if isinstance(symlink_entries, dict):
        symlink_entries = [symlink_entries]
    materialize_symlinks(workspace_path, symlink_entries)

    trace: RunTrace | None = None
    if trace_enabled:
        try:
            target_dir = experiments_dir if experiments_dir is not None else DEFAULT_EXPERIMENTS_DIR
            trace = RunTrace.open(
                experiments_dir=target_dir,
                prompt_path=REPO_ROOT / config_path if not str(config_path).startswith("/") else config_path,
                model=model_name,
                feedback_rounds=1,
                workspace=workspace,
            )
            print(f"Trace: {trace._path}")
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: failed to open trace ({type(exc).__name__}: {exc}); continuing without it", file=sys.stderr)
            trace = None

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

    try:
        round_rec = trace.start_round(1) if trace else None

        planning_output = planner.invoke(problem)
        steps = planning_output["plan"].steps
        if trace and round_rec is not None:
            trace.record_plan_steps(round_rec, steps)

        print("\n=== PLAN ===")
        for i, s in enumerate(steps, 1):
            name = getattr(s, "name", f"Step {i}")
            desc = getattr(s, "description", str(s))
            print(f"{i}. {name}\n   {desc}\n")

        last_summary = "No previous step."
        print("\n=== EXECUTION ===")

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
            except Exception as exc:
                last_summary = f"[execution error] {type(exc).__name__}: {exc}"
                if trace and step_rec is not None:
                    trace.finish_step(step_rec, ok=False, error=exc, result_text=last_summary)
                print(f"\n--- Step {i} ERROR ---\n{last_summary}", file=sys.stderr)
                raise

        if trace:
            trace.record_artifacts(workspace_path, expected_artifacts=expected_artifacts)
            score = try_score(workspace_path, scorer_dotted, scorer_kwargs)
            if score is not None:
                trace.record_score(score)
                print(f"\nScore: {score.total:.3f}")
            trace.mark_completed()

        print("\n=== FINAL ===")
        print(last_summary)
        print(f"\nWorkspace: {workspace_path.resolve()}")

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument(
        "--model",
        default=None,
        help="Model string for init_chat_model (e.g. openai:gpt-5-mini)",
    )
    parser.add_argument(
        "--workspace",
        default=None,
        help="Override workspace directory (optional)",
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

    main(
        args.config,
        args.model,
        args.workspace,
        trace_enabled=not args.no_trace,
        experiments_dir=Path(args.experiments_dir) if args.experiments_dir else None,
    )
