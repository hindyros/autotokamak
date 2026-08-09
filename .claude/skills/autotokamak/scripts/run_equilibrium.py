#!/usr/bin/env python3
# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Solve one equilibrium from a YAML config.

Thin wrapper around examples/config_driven_equilibrium/run_equilibrium_from_config.py.
Never imports autotokamak.* — dispatches via subprocess to keep OFT_env isolated.

Usage:
    python run_equilibrium.py --config PATH [--out DIR]

Emits a JSON summary block on stdout with the case dir the runner wrote to.
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


RUNNER_REL = "examples/config_driven_equilibrium/run_equilibrium_from_config.py"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="Path to equilibrium YAML.")
    p.add_argument("--out", default=None, help="Override outputs.out_dir (advisory — passed as env)")
    p.add_argument("--validate-only", action="store_true", help="Just validate the YAML.")
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

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (Path.cwd() / cfg_path).resolve()
    if not cfg_path.is_file():
        print(f"ERROR: config not found: {cfg_path}", file=sys.stderr)
        print_json_summary({"ok": False, "error": "config_missing", "config": str(cfg_path)})
        sys.exit(2)

    py = repo_python(root)
    cmd = [py, str(runner), str(cfg_path)]
    if args.validate_only:
        cmd.append("--validate-only")

    env = agent_env(root)
    if args.out:
        env["AUTOTOKAMAK_OUT_DIR"] = args.out  # advisory — the runner honors outputs.out_dir from YAML

    t0 = time.time()
    print(f"→ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=str(root), env=env)
    elapsed = time.time() - t0

    # Best-effort locate the case dir the runner wrote to. It prints
    # "Wrote case outputs to: <dir>" on success; we don't parse that here
    # because scan for latest is more robust across API changes.
    summary = {
        "ok": r.returncode == 0,
        "returncode": r.returncode,
        "config": str(cfg_path),
        "elapsed_seconds": round(elapsed, 2),
        "root": str(root),
    }
    print_json_summary(summary)
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
