# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""GEPA optimization CLI for the meta-action-picker prompt.

Usage::

    python -m autotokamak.agent.dspy.optimize_meta \
        --experiments-dir experiments/ \
        --output src/autotokamak/agent/dspy/optimized/meta_picker.json \
        --auto medium

Loads completed meta-agent traces (one ``dspy.Example`` per iteration),
runs ``dspy.GEPA`` over the ``MetaActionPickerModule``, and saves the
optimized module state to disk. The runner at
``agent/runners/meta_loop.py`` picks it up automatically next time it
runs.

The script also prints a baseline-vs-optimized comparison on the held-out
val split (mean score across val examples).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import dspy

from autotokamak.agent.dspy.metric_adapter import gepa_metric
from autotokamak.agent.dspy.module import MetaActionPickerModule
from autotokamak.agent.dspy.trace_loader import load_meta_traces, split_trainset


def _eval_mean_score(
    module: dspy.Module,
    examples: list[dspy.Example],
) -> float:
    """Run module on each example, return mean metric score.

    Used for baseline vs optimized comparison. Uses ``gepa_metric`` so the
    comparison is on the same scale as the optimization objective.
    """
    if not examples:
        return 0.0
    total = 0.0
    n = 0
    for ex in examples:
        try:
            pred = module(
                diagnostics_json=ex["diagnostics_json"],
                history_summary=ex["history_summary"],
                state_summary=ex["state_summary"],
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"WARNING: prediction failed on run_id={ex.get('run_id')}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            continue
        feedback = gepa_metric(ex, pred)
        total += float(getattr(feedback, "score", 0.0))
        n += 1
    return total / n if n else 0.0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path("experiments"),
        help="Directory containing <run_id>/trace.json subdirectories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/autotokamak/agent/dspy/optimized/meta_picker.json"),
        help="Where to write the optimized DSPy module JSON.",
    )
    parser.add_argument(
        "--task-lm",
        default="openai:gpt-5-mini",
        help="LM the action picker uses during eval/optimization.",
    )
    parser.add_argument(
        "--reflection-lm",
        default="openai:gpt-5.2",
        help="LM GEPA uses for reflective mutation (stronger is better here).",
    )
    parser.add_argument(
        "--auto",
        choices=["light", "medium", "heavy"],
        default="light",
        help="GEPA budget preset. light = cheap, heavy = thorough.",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.25,
        help="Fraction of RUNS held out for val (deterministic split).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    args = parser.parse_args(argv)

    # Load DSPy-compatible API keys from .env if present.
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    examples = load_meta_traces(args.experiments_dir)
    if not examples:
        print(
            f"ERROR: no usable meta-agent traces found under {args.experiments_dir}. "
            f"Run tools/collect_traces.sh first.",
            file=sys.stderr,
        )
        return 2
    train, val = split_trainset(examples, val_frac=args.val_frac, seed=args.seed)
    print(f"Loaded {len(examples)} examples from {args.experiments_dir} "
          f"({len(train)} train, {len(val)} val).")

    # Configure DSPy LMs. dspy.LM uses litellm-style model strings.
    # "openai:gpt-5-mini" -> "openai/gpt-5-mini" for litellm.
    def _to_litellm(model_str: str) -> str:
        if ":" in model_str:
            provider, name = model_str.split(":", 1)
            return f"{provider}/{name}"
        return model_str

    task_lm = dspy.LM(_to_litellm(args.task_lm))
    reflection_lm = dspy.LM(_to_litellm(args.reflection_lm))
    dspy.configure(lm=task_lm)

    student = MetaActionPickerModule()
    baseline_score = _eval_mean_score(student, val) if val else float("nan")
    print(f"Baseline mean val score: {baseline_score:.3f}")

    from dspy.teleprompt import GEPA

    optimizer = GEPA(
        metric=gepa_metric,
        auto=args.auto,
        reflection_lm=reflection_lm,
        seed=args.seed,
        track_stats=False,
    )
    print(f"Running GEPA(auto={args.auto}) on {len(train)} train examples...")
    optimized = optimizer.compile(student, trainset=train, valset=val or None)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    optimized.save(str(args.output))
    print(f"Saved optimized module to {args.output}")

    optimized_score = _eval_mean_score(optimized, val) if val else float("nan")
    print()
    print("=== Comparison on val ===")
    print(f"  baseline:  {baseline_score:.3f}")
    print(f"  optimized: {optimized_score:.3f}")
    if val:
        delta = optimized_score - baseline_score
        print(f"  delta:     {delta:+.3f}")
        if delta > 0:
            print("Optimized prompt beats baseline. Consider shipping.")
        else:
            print("Optimized prompt does NOT beat baseline; do not ship.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
