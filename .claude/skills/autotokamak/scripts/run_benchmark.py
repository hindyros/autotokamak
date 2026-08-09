#!/usr/bin/env python3
# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Run one cell of the agent-capability benchmark matrix.

Thin wrapper over `python -m autotokamak.bench run` — an agent harness
(echo | ursa | dspy | claude_sdk | pi | cursor) executes a task spec
(benchmarks/tasks/*.yaml; L2 = may import autotokamak, L3 = from scratch).
Results land in experiments/<tag>/<condition>/<run_id>/{workspace/,
trace.json, result.json}. See benchmarks/README.md for the matrix framing.

Usage:
    python run_benchmark.py --task benchmarks/tasks/smoke.yaml --harness echo
    python run_benchmark.py --task benchmarks/tasks/L3_from_scratch.yaml \
                            --harness claude_sdk [--model M] [--tag T] [--dry-run]
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


HARNESSES = ("echo", "ursa", "dspy", "claude_sdk", "pi", "cursor")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True,
                   help="Task spec YAML (e.g. benchmarks/tasks/L3_from_scratch.yaml)")
    p.add_argument("--harness", required=True, choices=HARNESSES)
    p.add_argument("--model", default=None, help="Model override for this run")
    p.add_argument("--tag", default=None, help="experiments/<tag>/ bucket (default: UTC date)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would run; no agent call, no cost")
    args = p.parse_args()

    root = locate_root()
    print_env_header(root)
    if root is None:
        read_only_advisory()

    task_path = (root / args.task) if not Path(args.task).is_absolute() else Path(args.task)
    if not task_path.is_file():
        print(f"ERROR: task spec not found: {task_path}", file=sys.stderr)
        print_json_summary({"ok": False, "error": "task_missing", "task": str(task_path)})
        sys.exit(2)

    py = repo_python(root)
    env = agent_env(root)
    cmd = [py, "-u", "-m", "autotokamak.bench", "run",
           "--task", str(task_path.relative_to(root)) if task_path.is_relative_to(root) else str(task_path),
           "--harness", args.harness]
    if args.model:
        cmd += ["--model", args.model]
    if args.tag:
        cmd += ["--tag", args.tag]
    if args.dry_run:
        cmd.append("--dry-run")

    t0 = time.time()
    print(f"→ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=str(root), env=env)
    elapsed = time.time() - t0

    print_json_summary({
        "ok": r.returncode == 0,
        "returncode": r.returncode,
        "task": str(task_path),
        "harness": args.harness,
        "model": args.model,
        "tag": args.tag,
        "dry_run": args.dry_run,
        "elapsed_seconds": round(elapsed, 2),
        "root": str(root),
    })
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
