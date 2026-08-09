#!/usr/bin/env python3
# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Warn-only environment probe for the autotokamak Skill.

Reports Python version, repo location, OFT availability, and autotokamak
importability. Never exits non-zero — a missing solver is fine as long as
Claude knows about it.

Usage:
    python check_env.py [--verbose]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _locate import (  # type: ignore[import-not-found]
    agent_env,
    locate_root,
    print_env_header,
    print_json_summary,
    python_version,
    repo_python,
)


def _probe_import(mod: str, py: str, env: dict | None = None) -> tuple[bool, str]:
    """Try `python -c 'import <mod>'` in a subprocess. Returns (ok, detail)."""
    try:
        r = subprocess.run(
            [py, "-c", f"import {mod}; import sys; sys.stdout.write(getattr({mod}, '__version__', '?'))"],
            capture_output=True, text=True, timeout=15, env=env,
        )
        if r.returncode == 0:
            return True, (r.stdout.strip() or "?")
        return False, (r.stderr.splitlines()[-1] if r.stderr else f"exit {r.returncode}")
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except FileNotFoundError:
        return False, "interpreter not found"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    root = locate_root()
    print_env_header(root)

    py_major_minor = f"{sys.version_info.major}.{sys.version_info.minor}"
    py_ok = sys.version_info[:2] in {(3, 11), (3, 12)}
    print(f"  python: {python_version()}  {'OK' if py_ok else 'WARN (need 3.11 or 3.12)'}")

    summary: dict = {
        "mode": "probe",
        "python_version": python_version(),
        "python_supported": py_ok,
        "root": str(root) if root else None,
    }

    if root is None:
        print("  root: NOT FOUND — advisory mode only.")
        print("        set AUTOTOKAMAK_ROOT or cd into the repo before running actions.")
        summary["repo_detected"] = False
        print_json_summary(summary)
        sys.exit(0)

    summary["repo_detected"] = True
    py = repo_python(root)
    print(f"  interpreter: {py}  {'(repo venv)' if py != sys.executable else '(current)'}")
    summary["interpreter"] = py

    env = agent_env(root)
    for mod in ("OpenFUSIONToolkit", "autotokamak", "ursa", "langchain"):
        ok, detail = _probe_import(mod, py, env=env if mod != "OpenFUSIONToolkit" else None)
        status = "OK" if ok else "MISSING"
        print(f"  import {mod:20s} {status:8s} {detail}")
        summary[f"import_{mod}"] = {"ok": ok, "detail": detail}

    # OFT_NTHREADS is respected by get_oft_env(); surface the current value.
    import os
    nth = os.environ.get("OFT_NTHREADS", "(unset — get_oft_env will default to 2)")
    print(f"  OFT_NTHREADS: {nth}")
    summary["OFT_NTHREADS"] = nth

    if args.verbose:
        print(f"  disk root: {shutil.disk_usage(root).free // (1024**3)} GB free at {root}")

    print_json_summary(summary)
    sys.exit(0)


if __name__ == "__main__":
    main()
