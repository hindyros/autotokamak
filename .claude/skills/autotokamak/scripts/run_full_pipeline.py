#!/usr/bin/env python3
# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Full pipeline: Phase-1 dataset → Phase-2 surrogate → eval → HTML report.

Wraps tools/run_full_pipeline.py. Passes through all its flags.

Usage:
    python run_full_pipeline.py [--model M] [--regen-dataset]
                                [--skip-phase2] [--skip-eval] [--skip-report]
                                [--enable-meta]
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


RUNNER_REL = "tools/run_full_pipeline.py"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=None)
    p.add_argument("--regen-dataset", action="store_true")
    p.add_argument("--skip-phase2", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--skip-report", action="store_true")
    p.add_argument("--enable-meta", action="store_true")
    args = p.parse_args()

    root = locate_root()
    print_env_header(root)
    if root is None:
        read_only_advisory()

    runner = root / RUNNER_REL
    if not runner.is_file():
        print(f"ERROR: runner not found: {runner}", file=sys.stderr)
        print_json_summary({"ok": False, "error": "runner_missing", "runner": str(runner)})
        sys.exit(2)

    py = repo_python(root)
    env = agent_env(root)
    cmd = [py, str(runner)]
    if args.model:          cmd += ["--model", args.model]
    if args.regen_dataset:  cmd.append("--regen-dataset")
    if args.skip_phase2:    cmd.append("--skip-phase2")
    if args.skip_eval:      cmd.append("--skip-eval")
    if args.skip_report:    cmd.append("--skip-report")
    if args.enable_meta:    cmd.append("--enable-meta")

    t0 = time.time()
    print(f"→ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=str(root), env=env)
    elapsed = time.time() - t0

    print_json_summary({
        "ok": r.returncode == 0,
        "returncode": r.returncode,
        "elapsed_seconds": round(elapsed, 2),
        "options": {
            "model": args.model,
            "regen_dataset": args.regen_dataset,
            "skip_phase2": args.skip_phase2,
            "skip_eval": args.skip_eval,
            "skip_report": args.skip_report,
            "enable_meta": args.enable_meta,
        },
        "root": str(root),
    })
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
