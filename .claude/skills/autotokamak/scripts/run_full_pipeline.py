#!/usr/bin/env python3
# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Full pipeline: Phase-1 dataset → Phase-2 surrogate → eval → HTML report.

Chains the unified pipelines CLI (`python -m autotokamak.pipelines`) plus the
post-run analysis scripts in tools/:

    1. `pipelines phase1`  — skipped if the L0 dataset.h5 exists, unless --regen-dataset.
    2. `pipelines phase2 --level L` (or `pipelines meta --level L` with --enable-meta).
    3. `tools/eval_surrogate.py` diagnostic plots (unless --skip-eval).
    4. `tools/trace_to_html.py` report regen (unless --skip-report).

Agent-written pipeline code (L2/L3 conditions) is a benchmark, not a pipeline
run — use `run_benchmark.py` / `python -m autotokamak.bench run` instead.

Usage:
    python run_full_pipeline.py [--level {L0,L1}] [--model M] [--regen-dataset]
                                [--n-samples N] [--time-budget S]
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


DATASET_REL = "examples/dataset_generation/L0/outputs/dataset.h5"
AUTOML_WORKSPACE_REL = "examples/surrogate_automl"
META_WORKSPACE_REL = "examples/surrogate_meta"


def _run(py: str, root: Path, env: dict, argv: list[str]) -> int:
    cmd = [py, "-u", *argv]
    print(f"→ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=str(root), env=env).returncode


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--level", choices=("L0", "L1"), default="L0",
                   help="L0=scripted decisions (no LLM), L1=DSPy LLM-typed decisions")
    p.add_argument("--model", default=None,
                   help="LLM model for L1 decisions (e.g. openai:gpt-5-mini)")
    p.add_argument("--regen-dataset", action="store_true",
                   help="Force Phase-1 even if the L0 dataset.h5 exists")
    p.add_argument("--n-samples", type=int, default=None, help="Phase-1 sweep size override")
    p.add_argument("--time-budget", type=int, default=None,
                   help="Phase-2 Optuna search budget in seconds")
    p.add_argument("--skip-phase2", action="store_true", help="Stop after Phase-1")
    p.add_argument("--skip-eval", action="store_true", help="Skip eval_surrogate.py plots")
    p.add_argument("--skip-report", action="store_true", help="Skip trace_to_html.py regen")
    p.add_argument("--enable-meta", action="store_true",
                   help="Run the self-improving meta-loop instead of vanilla Phase-2")
    args = p.parse_args()

    root = locate_root()
    print_env_header(root)
    if root is None:
        read_only_advisory()

    py = repo_python(root)
    env = agent_env(root)
    t0 = time.time()
    steps: dict[str, str] = {}
    rc = 0

    # 1. Phase-1 dataset
    dataset = root / DATASET_REL
    if dataset.is_file() and not args.regen_dataset:
        print(f"✓ Phase-1 skipped — dataset exists at {DATASET_REL} (--regen-dataset to force)")
        steps["phase1"] = "skipped"
    else:
        argv = ["-m", "autotokamak.pipelines", "phase1"]
        if args.n_samples is not None:
            argv += ["--n-samples", str(args.n_samples)]
        rc = _run(py, root, env, argv)
        steps["phase1"] = "ok" if rc == 0 else f"failed({rc})"

    # 2. Phase-2 / meta
    if rc == 0 and not args.skip_phase2:
        sub = "meta" if args.enable_meta else "phase2"
        argv = ["-m", "autotokamak.pipelines", sub, "--level", args.level]
        if args.model:
            argv += ["--model", args.model]
        if args.time_budget is not None:
            argv += ["--time-budget", str(args.time_budget)]
        rc = _run(py, root, env, argv)
        steps[sub] = "ok" if rc == 0 else f"failed({rc})"
    elif args.skip_phase2:
        steps["phase2"] = "skipped"

    # 3. Eval plots + 4. HTML report (best-effort; don't fail the pipeline)
    if rc == 0 and not args.skip_phase2:
        workspace_rel = META_WORKSPACE_REL if args.enable_meta else AUTOML_WORKSPACE_REL
        if args.skip_eval:
            steps["eval"] = "skipped"
        else:
            r = _run(py, root, env,
                     [str(root / "tools/eval_surrogate.py"), "--workspace", str(root / workspace_rel)])
            steps["eval"] = "ok" if r == 0 else f"failed({r})"
        if args.skip_report:
            steps["report"] = "skipped"
        else:
            r = _run(py, root, env, [str(root / "tools/trace_to_html.py")])
            steps["report"] = "ok" if r == 0 else f"failed({r})"

    print_json_summary({
        "ok": rc == 0,
        "returncode": rc,
        "elapsed_seconds": round(time.time() - t0, 2),
        "steps": steps,
        "options": {
            "level": args.level,
            "model": args.model,
            "regen_dataset": args.regen_dataset,
            "n_samples": args.n_samples,
            "time_budget": args.time_budget,
            "skip_phase2": args.skip_phase2,
            "skip_eval": args.skip_eval,
            "skip_report": args.skip_report,
            "enable_meta": args.enable_meta,
        },
        "root": str(root),
    })
    sys.exit(rc)


if __name__ == "__main__":
    main()
