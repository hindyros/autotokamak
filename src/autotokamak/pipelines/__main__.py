# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Entry point: python -m autotokamak.pipelines <phase1|phase2|meta> --mode <fast|ursa>

Examples:
    python -m autotokamak.pipelines phase1 --mode fast --n-samples 500
    python -m autotokamak.pipelines phase1 --mode ursa --model openai:gpt-5-mini
    python -m autotokamak.pipelines phase2 --mode fast --time-budget 600
    python -m autotokamak.pipelines phase2 --mode ursa --dataset PATH
    python -m autotokamak.pipelines meta   --mode fast --max-iterations 3 --time-budget 600
    python -m autotokamak.pipelines meta   --mode ursa --max-iterations 1

Outputs land under examples/<workspace>/<mode>/ and a manifest.json is written.
"""
from __future__ import annotations

import argparse
import sys
import time


def _fmt_duration(seconds: float) -> str:
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {(s % 3600) // 60:02d}m"


def _phase1_parser(sub):
    p = sub.add_parser("phase1", help="Phase-1: generate a Grad-Shafranov parameter-sweep dataset")
    p.add_argument("--mode", choices=("fast", "ursa"), required=True,
                   help="fast=library, ursa=URSA code-generation agent")
    p.add_argument("--config", default=None, help="dataset_config.yaml path (fast only; default: examples/dataset_generation/dataset_config.yaml)")
    p.add_argument("--n-samples", type=int, default=None, help="Override n_samples in the sweep config (fast only)")
    p.add_argument("--model", default=None, help="LLM model string (ursa only; e.g. openai:gpt-5-mini)")
    return p


def _phase2_parser(sub):
    p = sub.add_parser("phase2", help="Phase-2: AutoML surrogate training")
    p.add_argument("--mode", choices=("fast", "ursa"), required=True,
                   help="fast=Optuna+DSPy library, ursa=URSA code-generation agent")
    p.add_argument("--dataset", default=None, help="Path to dataset.h5 (defaults to phase1/fast output)")
    p.add_argument("--time-budget", type=int, default=600, help="Optuna search time in seconds (fast only)")
    p.add_argument("--max-rounds", type=int, default=4, help="Max outer search rounds (fast only)")
    p.add_argument("--model", default="openai:gpt-5-mini", help="LLM model for round decisions (both modes)")
    p.add_argument("--seed", type=int, default=0, help="Random seed (fast only)")
    return p


def _meta_parser(sub):
    p = sub.add_parser("meta", help="Meta-loop: autonomous Phase-1+2 improvement loop")
    p.add_argument("--mode", choices=("fast", "ursa"), required=True,
                   help="fast=structured library loop, ursa=hybrid (meta-agent via DSPy, nested Phase-2 via URSA)")
    p.add_argument("--max-iterations", type=int, default=3, help="Max meta-loop iterations")
    p.add_argument("--n-samples", type=int, default=None, help="Samples per regen_dataset action")
    p.add_argument("--enrich-n-new", type=int, default=None,
                   help="Force every enrich_active action to acquire exactly this many samples")
    p.add_argument("--time-budget", type=int, default=600, help="Phase-2 Optuna budget per iteration (fast only)")
    p.add_argument("--model", default=None, help="LLM model override (e.g. openai:gpt-5-mini)")
    p.add_argument("--dataset", default=None, help="Override initial dataset.h5 path")
    p.add_argument("--target-rmse", type=float, default=None, help="Early-stop when frozen-shard RMSE drops below this")
    p.add_argument("--target-accuracy-pct", type=float, default=None,
                   help="Performance stop: stop once the winner is >= this %% better than "
                        "the baseline mean-predictor (error reduction; e.g. 90). Cap = --max-iterations")
    p.add_argument("--target-worst-cell-accuracy-pct", type=float, default=None,
                   help="Performance stop on the weakest geometry region: stop once EVERY eval "
                        "cell is >= this %% better than its own baseline. Requires eval_envelope.")
    return p


def main():
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
        from autotokamak.pipelines.phase1 import run_phase1_fast, run_phase1_ursa
        if args.mode == "fast":
            run_phase1_fast(config=args.config, n_samples=args.n_samples)
        else:
            run_phase1_ursa(model=args.model)

    elif args.pipeline == "phase2":
        from autotokamak.pipelines.phase2 import run_phase2_fast, run_phase2_ursa
        if args.mode == "fast":
            run_phase2_fast(
                dataset=args.dataset,
                time_budget=args.time_budget,
                model=args.model,
                max_rounds=args.max_rounds,
                seed=args.seed,
            )
        else:
            run_phase2_ursa(dataset=args.dataset, model=args.model)

    elif args.pipeline == "meta":
        from autotokamak.pipelines.meta import run_meta
        run_meta(
            mode=args.mode,
            max_iterations=args.max_iterations,
            n_samples=args.n_samples,
            enrich_n_new=args.enrich_n_new,
            time_budget=args.time_budget,
            model=args.model,
            dataset=args.dataset,
            target_rmse=args.target_rmse,
            target_accuracy_pct=args.target_accuracy_pct,
            target_worst_cell_accuracy_pct=args.target_worst_cell_accuracy_pct,
        )

    print(f"[{args.pipeline}/{args.mode}] total wall time: "
          f"{_fmt_duration(time.time() - wall_start)}")


if __name__ == "__main__":
    main()
