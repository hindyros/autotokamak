#!/usr/bin/env python3
# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Dispatch a named prompt to the correct URSA runner.

Only 5 canonical prompt names are accepted — keeps stdout contract stable
and enforces the prompt → runner mapping documented in
references/agent-runners.md.

Usage:
    python run_agent.py --prompt {dataset_generation | oft_discretization_example |
                                  oft_example_generation | surrogate_automl |
                                  surrogate_meta}
                        [--model M] [--workspace W] [--no-trace]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _locate import (  # type: ignore[import-not-found]
    agent_env,
    locate_root,
    print_env_header,
    print_json_summary,
    read_only_advisory,
    repo_python,
)


PROMPT_TO_RUNNER = {
    "dataset_generation":         "autotokamak.agent.runners.plan_execute_feedback",
    "oft_discretization_example": "autotokamak.agent.runners.plan_execute_feedback",
    "oft_example_generation":     "autotokamak.agent.runners.plan_execute",
    "surrogate_automl":           "autotokamak.agent.runners.plan_execute_feedback",
    "surrogate_meta":             "autotokamak.agent.runners.meta_loop",
}

META_RUNNER = "autotokamak.agent.runners.meta_loop"

PROMPTS_REL = "src/autotokamak/agent/prompts"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompt", required=True, choices=sorted(PROMPT_TO_RUNNER))
    p.add_argument("--model", default=None, help="Override model (e.g. openai:gpt-5-mini)")
    p.add_argument("--workspace", default=None, help="Override workspace dir (plan_execute* only)")
    p.add_argument("--no-trace", action="store_true", help="Disable trace.json")
    p.add_argument("--max-iterations", type=int, default=None, help="meta_loop only")
    p.add_argument("--n-samples", type=int, default=None,
                   help="meta_loop only: override sampling.n_samples in the base sweep config")
    p.add_argument("--time-budget-seconds", type=int, default=None,
                   help="meta_loop only: HARD BUDGET injected into every Phase-2 overlay prompt")
    args = p.parse_args()

    root = locate_root()
    print_env_header(root)
    if root is None:
        read_only_advisory()

    prompt_path = root / PROMPTS_REL / f"{args.prompt}.yaml"
    if not prompt_path.is_file():
        print(f"ERROR: prompt YAML not found: {prompt_path}", file=sys.stderr)
        print_json_summary({"ok": False, "error": "prompt_missing", "prompt": str(prompt_path)})
        sys.exit(2)

    runner_module = PROMPT_TO_RUNNER[args.prompt]
    py = repo_python(root)
    env = agent_env(root)

    cmd = [py, "-u", "-m", runner_module, "--config", str(prompt_path.relative_to(root))]
    if args.model:
        cmd += ["--model", args.model]
    if args.workspace and runner_module != META_RUNNER:
        cmd += ["--workspace", args.workspace]
    if args.no_trace:
        cmd.append("--no-trace")
    if runner_module == META_RUNNER:
        if args.max_iterations is not None:
            cmd += ["--max-iterations", str(args.max_iterations)]
        if args.n_samples is not None:
            cmd += ["--n-samples", str(args.n_samples)]
        if args.time_budget_seconds is not None:
            cmd += ["--time-budget-seconds", str(args.time_budget_seconds)]

    t0 = time.time()
    print(f"→ prompt={args.prompt}  runner={runner_module}", flush=True)
    print(f"→ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=str(root), env=env)
    elapsed = time.time() - t0

    print_json_summary({
        "ok": r.returncode == 0,
        "returncode": r.returncode,
        "prompt": args.prompt,
        "runner": runner_module,
        "prompt_path": str(prompt_path),
        "elapsed_seconds": round(elapsed, 2),
        "root": str(root),
    })
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
