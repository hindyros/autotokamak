#!/usr/bin/env python3
# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Fan out a discretization sweep over N worker processes.

The OFT_env singleton forbids threading and forbids concurrent solves inside
one Python interpreter. This wrapper:

  1. Reads a SweepConfig YAML (base_config + cases[]).
  2. Splits cases[] into N shards.
  3. Writes each shard as a temporary sweep YAML in a scratch dir.
  4. Launches N subprocess.Popen children, each running the base runner
     against its shard YAML.
  5. Waits, collects returncodes, prints a JSON summary.

Merging outputs is not needed — the underlying runner writes each case to its
own directory under `outputs/`.

Usage:
    python sweep_parallel.py --config PATH --workers N [--out DIR]
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import tempfile
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


BASE_RUNNER_REL = "examples/config_driven_equilibrium/run_discretization_sweep.py"


def _shard(items: list, n: int) -> list[list]:
    """Balanced shard: shard k gets items[k::n]. Preserves original ordering per shard."""
    n = max(1, min(n, len(items)))
    return [items[i::n] for i in range(n)]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="SweepConfig YAML (base_config + cases[])")
    p.add_argument("--workers", type=int, default=4, help="Number of parallel worker processes.")
    p.add_argument("--out", default=None, help="Advisory — passed to children via env.")
    args = p.parse_args()

    root = locate_root()
    print_env_header(root)
    if root is None:
        read_only_advisory()

    try:
        import yaml  # noqa: F401
    except ImportError:
        print("ERROR: PyYAML not importable. `pip install pyyaml`.", file=sys.stderr)
        print_json_summary({"ok": False, "error": "pyyaml_missing"})
        sys.exit(2)
    import yaml

    cfg_path = Path(args.config).expanduser()
    if not cfg_path.is_absolute():
        cfg_path = (Path.cwd() / cfg_path).resolve()
    if not cfg_path.is_file():
        print(f"ERROR: config not found: {cfg_path}", file=sys.stderr)
        print_json_summary({"ok": False, "error": "config_missing", "config": str(cfg_path)})
        sys.exit(2)

    raw = yaml.safe_load(cfg_path.read_text())
    if not isinstance(raw, dict) or "cases" not in raw:
        print(f"ERROR: {cfg_path} is not a sweep YAML (missing top-level `cases`).", file=sys.stderr)
        print_json_summary({"ok": False, "error": "not_a_sweep_yaml"})
        sys.exit(2)

    cases = list(raw["cases"])
    if not cases:
        print("ERROR: cases[] is empty.", file=sys.stderr)
        print_json_summary({"ok": False, "error": "no_cases"})
        sys.exit(2)

    n_workers = max(1, min(args.workers, len(cases)))
    shards = _shard(cases, n_workers)
    print(f"  sweep: {len(cases)} cases → {n_workers} shards", flush=True)

    runner = root / BASE_RUNNER_REL
    if not runner.is_file():
        print(f"ERROR: runner not found: {runner}", file=sys.stderr)
        print_json_summary({"ok": False, "error": "runner_missing", "runner": str(runner)})
        sys.exit(2)

    py = repo_python(root)
    env = agent_env(root)

    procs: list[tuple[int, subprocess.Popen, Path]] = []
    tmpdir = Path(tempfile.mkdtemp(prefix="autokamak_sweep_"))

    try:
        for i, shard in enumerate(shards):
            shard_cfg = copy.deepcopy(raw)
            shard_cfg["cases"] = shard
            shard_path = tmpdir / f"shard_{i:02d}.yaml"
            shard_path.write_text(yaml.safe_dump(shard_cfg, sort_keys=False))
            cmd = [py, str(runner), str(shard_path)]
            print(f"  → shard {i}: {len(shard)} cases → {shard_path.name}", flush=True)
            proc = subprocess.Popen(cmd, cwd=str(root), env=env)
            procs.append((i, proc, shard_path))

        t0 = time.time()
        results = []
        for i, proc, shard_path in procs:
            rc = proc.wait()
            results.append({"shard": i, "returncode": rc, "shard_config": str(shard_path)})
            print(f"  ← shard {i} exited rc={rc}", flush=True)
        elapsed = time.time() - t0
    finally:
        # Leave tmpdir for post-mortem; it's small.
        pass

    all_ok = all(r["returncode"] == 0 for r in results)
    print_json_summary({
        "ok": all_ok,
        "n_cases": len(cases),
        "n_shards": n_workers,
        "elapsed_seconds": round(elapsed, 2),
        "results": results,
        "tmpdir": str(tmpdir),
        "root": str(root),
    })
    sys.exit(0 if all_ok else 3)


if __name__ == "__main__":
    main()
