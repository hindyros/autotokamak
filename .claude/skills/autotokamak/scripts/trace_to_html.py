#!/usr/bin/env python3
# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Render experiments/*/trace.json into a browsable static HTML report.

Wraps tools/trace_to_html.py.

Usage:
    python trace_to_html.py [--experiments DIR] [--logs DIR] [--out DIR]
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


RUNNER_REL = "tools/trace_to_html.py"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiments", default=None, help="Experiments root (default: <root>/experiments)")
    p.add_argument("--logs", default=None, help="Logs dir (default: <root>/logs)")
    p.add_argument("--out", default=None, help="Output dir (default: <experiments>/_report)")
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
    if args.experiments:  cmd += ["--experiments", str(Path(args.experiments).resolve())]
    if args.logs:         cmd += ["--logs", str(Path(args.logs).resolve())]
    if args.out:          cmd += ["--out", str(Path(args.out).resolve())]

    t0 = time.time()
    print(f"→ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=str(root), env=env)
    elapsed = time.time() - t0

    print_json_summary({
        "ok": r.returncode == 0,
        "returncode": r.returncode,
        "elapsed_seconds": round(elapsed, 2),
        "root": str(root),
    })
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
