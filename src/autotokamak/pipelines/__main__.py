# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Entry point: python -m autotokamak.pipelines <phase1|phase2|meta> [--level <L0|L1>]

The pre-written pipeline at two access levels:
    L0 — scripted heuristic decisions, zero LLM calls (reproducible baseline)
    L1 — LLM-typed decisions via the DSPy pickers

Examples:
    python -m autotokamak.pipelines phase1 --n-samples 500
    python -m autotokamak.pipelines phase2 --level L0 --time-budget 600
    python -m autotokamak.pipelines phase2 --level L1 --model openai:gpt-5-mini
    python -m autotokamak.pipelines meta   --level L0 --max-iterations 3

Outputs land under examples/<workspace>/<level>/ and a manifest.json is written.

Agent-written pipeline code (levels L2/L3, any harness) is benchmarked via:
    python -m autotokamak.bench run --task benchmarks/tasks/<task>.yaml --harness <name>
"""
from __future__ import annotations

import argparse
import sys
import time

_MODE_REMOVED = (
    "--mode was removed. Use --level L0|L1 for the pre-written pipeline, or run "
    "agent codegen conditions via: python -m autotokamak.bench run "
    "--task benchmarks/tasks/L2_library.yaml --harness ursa"
)


def _fmt_duration(seconds: float) -> str:
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {(s % 3600) // 60:02d}m"


def _add_level(p, *, choices=("L0", "L1")):
    p.add_argument("--level", choices=choices, default="L0",
                   help="L0=scripted decisions (no LLM), L1=DSPy LLM-typed decisions")


def _phase1_parser(sub):
    p = sub.add_parser("phase1", help="Phase-1: generate a Grad-Shafranov parameter-sweep dataset")
    _add_level(p, choices=("L0",))  # no decision points → always scripted
    p.add_argument("--config", default=None, help="dataset_config.yaml path (default: examples/dataset_generation/dataset_config.yaml)")
    p.add_argument("--n-samples", type=int, default=None, help="Override n_samples in the sweep config")
    return p


def _phase2_parser(sub):
    p = sub.add_parser("phase2", help="Phase-2: AutoML surrogate training")
    _add_level(p)
    p.add_argument("--dataset", default=None, help="Path to dataset.h5 (defaults to phase1 L0 output)")
    p.add_argument("--time-budget", type=int, default=600, help="Optuna search time in seconds")
    p.add_argument("--max-rounds", type=int, default=4, help="Max outer search rounds")
    p.add_argument("--model", default=None, help="LLM model for round decisions (L1 only; e.g. openai:gpt-5-mini)")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    return p


def _meta_parser(sub):
    p = sub.add_parser("meta", help="Meta-loop: autonomous Phase-1+2 improvement loop")
    _add_level(p)
    p.add_argument("--max-iterations", type=int, default=3, help="Max meta-loop iterations")
    p.add_argument("--n-samples", type=int, default=None, help="Samples per regen_dataset action")
    p.add_argument("--enrich-n-new", type=int, default=None,
                   help="Force every enrich_active action to acquire exactly this many samples")
    p.add_argument("--time-budget", type=int, default=600, help="Phase-2 Optuna budget per iteration")
    p.add_argument("--model", default=None, help="LLM model for decisions (L1 only; e.g. openai:gpt-5-mini)")
    p.add_argument("--dataset", default=None, help="Override initial dataset.h5 path")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--target-rmse", type=float, default=None, help="Early-stop when frozen-shard RMSE drops below this")
    p.add_argument("--target-accuracy-pct", type=float, default=None,
                   help="Performance stop: stop once the winner is >= this %% better than "
                        "the baseline mean-predictor (error reduction; e.g. 90). Cap = --max-iterations")
    p.add_argument("--target-worst-cell-accuracy-pct", type=float, default=None,
                   help="Performance stop on the weakest geometry region: stop once EVERY eval "
                        "cell is >= this %% better than its own baseline. Requires eval_envelope.")
    return p


def main():
    if "--mode" in sys.argv:
        sys.exit(_MODE_REMOVED)

    parser = argparse.ArgumentParser(
        prog="python -m autotokamak.pipelines",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="pipeline", metavar="<phase1|phase2|meta>")
    sub.required = True
    _phase1_parser(sub)
    _phase2_parser(sub)
    _meta_parser(sub)

    args = parser.parse_args()
    wall_start = time.time()

    if args.pipeline == "phase1":
        from autotokamak.pipelines.phase1 import run_phase1
        run_phase1(config=args.config, n_samples=args.n_samples)

    elif args.pipeline == "phase2":
        from autotokamak.pipelines.phase2 import run_phase2
        run_phase2(
            level=args.level,
            dataset=args.dataset,
            time_budget=args.time_budget,
            model=args.model,
            max_rounds=args.max_rounds,
            seed=args.seed,
        )

    elif args.pipeline == "meta":
        from autotokamak.pipelines.meta import run_meta
        run_meta(
            level=args.level,
            max_iterations=args.max_iterations,
            n_samples=args.n_samples,
            enrich_n_new=args.enrich_n_new,
            time_budget=args.time_budget,
            model=args.model,
            dataset=args.dataset,
            seed=args.seed,
            target_rmse=args.target_rmse,
            target_accuracy_pct=args.target_accuracy_pct,
            target_worst_cell_accuracy_pct=args.target_worst_cell_accuracy_pct,
        )

    print(f"[{args.pipeline}/{args.level}] total wall time: "
          f"{_fmt_duration(time.time() - wall_start)}")


if __name__ == "__main__":
    main()
