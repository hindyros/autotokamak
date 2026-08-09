#!/usr/bin/env python3
# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Feasibility scan for equilibrium shaping bounds.

Wraps tools/probe_feasible_box.py. Reports the isoflux-used fraction across
built-in candidate boxes so you can pick sweep bounds where the fallback rate
is low.

Usage:
    python probe_feasible.py [--out JSON]
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


RUNNER_REL = "tools/probe_feasible_box.py"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=None, help="Advisory — the underlying probe currently prints to stdout only.")
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

    t0 = time.time()
    print(f"→ {py} {runner}", flush=True)
    r = subprocess.run([py, str(runner)], cwd=str(root), env=env)
    elapsed = time.time() - t0

    print_json_summary({
        "ok": r.returncode == 0,
        "returncode": r.returncode,
        "elapsed_seconds": round(elapsed, 2),
        "note": "Human-readable summary printed above. Boxes with high isoflux_used/N are safe bounds.",
        "root": str(root),
    })
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
