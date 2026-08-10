#!/usr/bin/env python3
# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Run a Phase-1 dataset generation sweep via the platform library.

Calls ``autotokamak.data.sweep.run_sweep`` directly — NOT the agent-generated
``examples/dataset_generation/run_dataset_sweep.py``. This guarantees the
canonical platform behaviour: physical ψ via ``gs.get_psi(False)``, atomic
HDF5 writes, and strict SweepConfig schema validation.

Usage:
    python run_dataset_sweep.py --config PATH [--out-dir DIR]
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


PLATFORM_SCRIPT = """
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[1]) / 'src'))
from autotokamak.data.schema import SweepConfig
from autotokamak.data.sweep import run_sweep

root   = Path(sys.argv[1])
cfg_path = Path(sys.argv[2])
out_dir  = Path(sys.argv[3]) if len(sys.argv) > 3 else None

cfg = SweepConfig.from_yaml(cfg_path)

# run_sweep(cfg, output_dir) writes to output_dir / cfg.output_path.
# Strip any leading directory component from cfg.output_path so we control
# exactly where the file lands via output_dir alone.
cfg_out_rel = Path(cfg.output_path).name  # e.g. "dataset.h5"
base = out_dir if out_dir else cfg_path.parent
output_dir = base.resolve()
output_dir.mkdir(parents=True, exist_ok=True)
cfg = cfg.model_copy(update={"output_path": cfg_out_rel})
cfg_out = output_dir / cfg_out_rel

print(f"Sweep: {cfg.sampling.n_samples} samples → {cfg_out}", flush=True)
result = run_sweep(cfg, output_dir)
print(f"Done: {result.n_succeeded}/{result.n_requested} succeeded "
      f"(isoflux_used: {result.n_isoflux_used}/{result.n_succeeded})", flush=True)
print(json.dumps({
    "ok": True,
    "dataset_path": result.dataset_path,
    "n_requested": result.n_requested,
    "n_succeeded": result.n_succeeded,
    "n_isoflux_used": result.n_isoflux_used,
    "config_hash": result.config_hash,
}))
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="dataset_config.yaml path")
    p.add_argument("--out-dir", default=None,
                   help="Override output directory for dataset.h5")
    args = p.parse_args()

    root = locate_root()
    print_env_header(root)
    if root is None:
        read_only_advisory()

    cfg = Path(args.config)
    if not cfg.is_absolute():
        cfg = (Path.cwd() / cfg).resolve()
    if not cfg.is_file():
        print(f"ERROR: config not found: {cfg}", file=sys.stderr)
        print_json_summary({"ok": False, "error": "config_missing"})
        sys.exit(2)

    py  = repo_python(root)
    env = agent_env(root)
    out_dir = str(Path(args.out_dir).resolve()) if args.out_dir else str(cfg.parent)

    cmd = [py, "-c", PLATFORM_SCRIPT, str(root), str(cfg), out_dir]
    t0 = time.time()
    print(f"→ platform run_sweep  config={cfg}  out_dir={out_dir}", flush=True)
    r = subprocess.run(cmd, cwd=str(root), env=env)
    elapsed = time.time() - t0

    print_json_summary({
        "ok": r.returncode == 0,
        "returncode": r.returncode,
        "config": str(cfg),
        "elapsed_seconds": round(elapsed, 2),
        "root": str(root),
    })
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
