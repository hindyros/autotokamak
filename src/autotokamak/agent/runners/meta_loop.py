"""Meta-loop runner — the autonomous outer loop above Phase-2.

Per iteration:
  1. Compute diagnostics (deterministic).
  2. Ask the LLM for an ``ActionDecision`` (structured output).
  3. Dispatch the chosen action.
  4. Measure and record into the meta-trace.

The action-picker LLM call is factored into ``pick_action`` so tests can
inject a hand-written decision sequence without standing up langchain.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from dotenv import load_dotenv

from autotokamak.agent.orchestrator.actions import MetaState, dispatch
from autotokamak.agent.orchestrator.schema import (
    VALID_ACTIONS,
    ActionDecision,
    MetaConfig,
    MetaIterationRecord,
    MetaReport,
)
from autotokamak.data.schema import SweepConfig
from autotokamak.eval.data import kfold, load_dataset
from autotokamak.eval.metrics import (
    baseline_mean_prediction,
    baseline_mean_predictor_rmse,
    per_cell_errors,
)

from agent.runners.config import REPO_ROOT, resolve_workspace
from agent.runners.trace import RunTrace

load_dotenv(REPO_ROOT / ".env")


DEFAULT_EXPERIMENTS_DIR = REPO_ROOT / "experiments"
ActionPicker = Callable[[MetaConfig, MetaState, dict, list[MetaIterationRecord]], ActionDecision]


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


# ----------------------- terminal formatting helpers -----------------------
#
# The meta-loop is watched live from a terminal; every iteration prints its
# RMSE with % improvement vs the previous iteration and vs baseline, and the
# run ends with an aligned per-iteration summary table. Pure stdlib — no
# rich/tabulate dependency.


def _pct_improvement(new: Optional[float], old: Optional[float]) -> Optional[float]:
    """Error-reduction percentage: positive means ``new`` is BETTER (lower)."""
    if new is None or old is None or not np.isfinite(old) or old <= 0:
        return None
    return 100.0 * (old - new) / old


def _fmt_pct(pct: Optional[float]) -> str:
    if pct is None:
        return "—"
    return f"{pct:+.1f}%"


def _fmt_rmse(rmse: Optional[float]) -> str:
    return f"{rmse:.5g}" if rmse is not None else "—"


def _fmt_duration(seconds: Optional[float]) -> str:
    if seconds is None or seconds < 0:
        return "—"
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {(s % 3600) // 60:02d}m"


def _record_duration(rec: MetaIterationRecord) -> Optional[float]:
    """Iteration wall time from the record's ISO timestamps."""
    try:
        start = _dt.datetime.fromisoformat(rec.started_utc)
        end = _dt.datetime.fromisoformat(rec.finished_utc)
        return (end - start).total_seconds()
    except (TypeError, ValueError):
        return None


def _truncate(text: str, width: int) -> str:
    text = " ".join(str(text).split())  # collapse newlines/runs of whitespace
    return text if len(text) <= width else text[: width - 1] + "…"


def _action_outcome(result: Optional[dict]) -> str:
    """One-line human summary of an action's result dict."""
    if not result:
        return "—"
    if result.get("error"):
        return f"ERROR: {result['error']}"
    kind = result.get("kind", "")
    if kind == "regen_dataset":
        return (
            f"+{result.get('n_new_succeeded', '?')}/{result.get('n_new_requested', '?')}"
            f" samples → pool {result.get('n_total_succeeded', '?')}"
        )
    if kind == "enrich_active":
        return (
            f"+{result.get('n_new_succeeded', '?')}/{result.get('n_new_requested', '?')}"
            f" targeted samples → pool {result.get('n_total_succeeded', '?')}"
        )
    if kind == "extend_search":
        parts = []
        if result.get("n_rounds") is not None:
            parts.append(f"{result['n_rounds']} rounds")
        if result.get("elapsed_seconds") is not None:
            parts.append(f"{result['elapsed_seconds']:.0f}s")
        if result.get("shard_rmse") is not None:
            parts.append(f"shard RMSE {result['shard_rmse']:.5g}")
        models = result.get("models_emphasized") or []
        if models:
            parts.append("models: " + ",".join(models))
        return "search: " + ", ".join(parts) if parts else "search"
    if kind == "terminate":
        reason = result.get("reason") or "(no reason given)"
        return f"reason: {reason}"
    return kind or "—"


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    """Aligned plain-text table (left-aligned text, right-aligned numbers-ish)."""
    widths = [
        max(len(headers[c]), *(len(r[c]) for r in rows)) if rows else len(headers[c])
        for c in range(len(headers))
    ]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    print("  " + line)
    print("  " + "  ".join("-" * w for w in widths))
    for r in rows:
        print("  " + "  ".join(cell.ljust(w) for cell, w in zip(r, widths)))


def _print_iteration_metrics(
    rmse_after: Optional[float],
    prev_rmse: Optional[float],
    baseline_rmse: float,
) -> None:
    """Per-iteration RMSE line with Δ vs previous loop and vs baseline."""
    if rmse_after is None:
        print("  shard RMSE: — (no winner yet)")
        return
    d_prev = _pct_improvement(rmse_after, prev_rmse)
    d_base = _pct_improvement(rmse_after, baseline_rmse)
    print(
        f"  shard RMSE: {_fmt_rmse(rmse_after)}"
        f"  |  Δ vs prev iter: {_fmt_pct(d_prev)}"
        f"  |  vs baseline: {_fmt_pct(d_base)}"
    )


def _print_summary_table(
    history: list[MetaIterationRecord], baseline_rmse: float
) -> None:
    """Aligned per-iteration recap: action, outcome, RMSE, Δprev, vs baseline."""
    if not history:
        return
    print("\n=== META SUMMARY (per-iteration) ===")
    rows: list[list[str]] = []
    prev_rmse: Optional[float] = None
    for rec in history:
        rmse = rec.rmse_after
        rows.append(
            [
                str(rec.iteration),
                rec.decision.action if rec.decision else "—",
                _truncate(_action_outcome(rec.result), 52),
                _fmt_rmse(rmse),
                _fmt_pct(_pct_improvement(rmse, prev_rmse)),
                _fmt_pct(_pct_improvement(rmse, baseline_rmse)),
                _fmt_duration(_record_duration(rec)),
            ]
        )
        if rmse is not None:
            prev_rmse = rmse
    _print_table(
        ["iter", "action", "outcome", "RMSE", "Δ prev", "vs baseline", "time"], rows
    )
    print(
        "  (Δ prev / vs baseline = % error reduction; positive = improvement; "
        "RMSE on frozen eval shard)"
    )


def pick_action_via_llm(
    meta_config: MetaConfig,
    state: MetaState,
    diagnostics: dict,
    history: list[MetaIterationRecord],
) -> ActionDecision:
    """Default action-picker. Delegates to the DSPy MetaActionPickerModule.

    Behavior:
      - Loads optimized prompt state from
        ``agent/dspy/optimized/meta_picker.json`` if present (post-GEPA).
      - Falls back to the in-code baseline (signature docstring at
        ``agent/dspy/signatures.py``) when no optimized state exists.
      - Honors ``state.use_baseline_picker`` (set via the runner's
        ``--use-baseline`` flag) to force the in-code baseline for A/B
        comparison even when an optimized JSON exists.

    The LM is configured via ``dspy.configure(lm=...)`` lazily on first call.
    """
    import dspy

    from autotokamak.agent.dspy.module import (
        DEFAULT_OPTIMIZED_PATH,
        MetaActionPickerModule,
        load_module,
    )

    # Configure DSPy's LM once per process. dspy.LM uses litellm-style strings
    # ("openai/gpt-5-mini"); convert from our "openai:gpt-5-mini" convention.
    settings_lm = getattr(dspy.settings, "lm", None)
    desired_lm_string = meta_config.model.replace(":", "/", 1)
    if settings_lm is None or getattr(settings_lm, "model", None) != desired_lm_string:
        dspy.configure(lm=dspy.LM(desired_lm_string))

    use_baseline = bool(getattr(state, "use_baseline_picker", False))
    if use_baseline or not DEFAULT_OPTIMIZED_PATH.is_file():
        module = MetaActionPickerModule()
    else:
        module = load_module(DEFAULT_OPTIMIZED_PATH)

    # Single source of truth for the LM inputs — the runner records the same
    # strings into the trace, so GEPA trains on byte-identical inputs.
    from autotokamak.agent.dspy.picker_inputs import picker_inputs_from_runtime

    inputs = picker_inputs_from_runtime(meta_config, state, diagnostics, history)
    return module.predict_action_decision(**inputs)


def measure_test_rmse(state: MetaState) -> Optional[float]:
    """Evaluate the current best winner on the FROZEN eval set.

    Returns None if no winner or no eval set is available. The eval set is
    frozen at meta-loop start (a shard carved from the initial dataset, or
    the full-envelope eval set in envelope mode) and never grows or
    reshuffles, so RMSEs are comparable across iterations regardless of how
    the train pool has changed — and no winner-training sample can leak in.

    Delegates to ``actions._frozen_shard_rmse`` — the loop's single most
    important measurement must have exactly one implementation.
    """
    from autotokamak.agent.orchestrator.actions import _frozen_shard_rmse

    if state.best_winner_payload is None:
        return None
    return _frozen_shard_rmse(state.best_winner_payload, state)


# --------------------- evaluation-set setup and scoring ---------------------


@dataclass
class EvalSetup:
    """Everything the loop needs to know about the frozen evaluation set.

    Built once by ``_setup_eval_sets``; ``envelope_bounds is None`` means
    legacy seed-shard mode (no per-cell scoring). This is the single seam
    between the two eval modes — downstream code only ever checks
    ``per_cell_enabled``, never re-derives the mode from the config.
    """

    train_pool: Path
    eval_h5: Path
    split_info: dict
    envelope_bounds: Optional[tuple] = None  # (lows, highs) in PARAM_ORDER
    n_bins: int = 2

    @property
    def per_cell_enabled(self) -> bool:
        return self.envelope_bounds is not None


def _count_success(h5_path: Path) -> int:
    """Number of successful rows, reading ONLY the success column (not psi)."""
    import h5py

    with h5py.File(h5_path, "r") as f:
        return int(np.asarray(f["outputs/success"][...], dtype=bool).sum())


def _setup_eval_sets(
    meta_config: MetaConfig,
    initial_dataset: Path,
    datasets_dir: Path,
    base_sweep: Optional[SweepConfig],
) -> EvalSetup:
    """Freeze the evaluation set BEFORE anything else touches the data.

    Every RMSE the loop reports (baseline, per-iteration, final) is measured
    on it. Two modes (design doc docs/active_learning_design.md §D2):
      - eval_envelope configured: the eval set covers the full TARGET
        envelope (pre-solved h5 or generated once by LHS + OFT); the WHOLE
        initial dataset becomes train pool, and enrich_active acquires over
        the envelope bounds.
      - legacy: carve a shard from the initial (seed-box) dataset.
    Writes ``split_info.json`` into ``datasets_dir`` either way.
    """
    if meta_config.eval_envelope is not None:
        from autotokamak.data.envelope import (
            generate_envelope_eval,
            resolve_envelope_bounds,
        )

        env = meta_config.eval_envelope
        if env.h5 is None and base_sweep is None:
            raise ValueError(
                "eval_envelope without 'h5' requires base_sweep_config "
                "(it supplies the solver knobs and output grid for generation)"
            )
        eval_h5 = generate_envelope_eval(env, base_sweep, datasets_dir)
        bounds = resolve_envelope_bounds(env, base_sweep) if base_sweep is not None else None
        split_info = {
            "mode": "envelope",
            "eval_envelope_path": str(eval_h5),
            "n_test": _count_success(eval_h5),
            "n_train_success": _count_success(initial_dataset),
        }
        setup = EvalSetup(
            train_pool=initial_dataset,
            eval_h5=eval_h5,
            split_info=split_info,
            envelope_bounds=bounds,
            n_bins=env.n_bins,
        )
        print(
            f"Envelope eval set: {eval_h5} ({split_info['n_test']} samples); "
            f"train pool (full seed dataset): {split_info['n_train_success']} "
            f"successful samples"
        )
    else:
        from autotokamak.data.h5io import split_h5

        train_pool = datasets_dir / "train_pool.h5"
        test_shard = datasets_dir / "test_shard.h5"
        split_info = split_h5(
            initial_dataset,
            train_path=train_pool,
            test_path=test_shard,
            test_frac=meta_config.holdout_test_frac,
            min_test=meta_config.holdout_min_test,
            seed=meta_config.seed,
        )
        setup = EvalSetup(train_pool=train_pool, eval_h5=test_shard, split_info=split_info)
        print(
            f"Frozen test shard: {test_shard} ({split_info['n_test']} samples); "
            f"train pool: {split_info['n_train_success']} successful samples"
        )

    (datasets_dir / "split_info.json").write_text(
        json.dumps(split_info, indent=2, default=str)
    )
    return setup


def _resolve_target_rmse_abs(
    meta_config: MetaConfig, baseline_rmse: float
) -> Optional[float]:
    """Collapse the three aggregate stopping targets into one absolute RMSE.

    ``target_rmse`` (absolute), ``target_rmse_ratio`` (× baseline), and
    ``target_accuracy_pct`` (error reduction: rmse <= (1 - pct/100) ×
    baseline) are the same quantity in three unit systems; the strictest
    (lowest) wins. Returns None when no target is set. The worst-cell target
    is deliberately separate — it is a min-over-cells quantity, not a scalar
    threshold on the aggregate RMSE.
    """
    candidates = []
    if meta_config.target_rmse is not None:
        candidates.append(float(meta_config.target_rmse))
    if meta_config.target_rmse_ratio is not None:
        candidates.append(float(meta_config.target_rmse_ratio) * baseline_rmse)
    if meta_config.target_accuracy_pct is not None:
        candidates.append(
            (1.0 - float(meta_config.target_accuracy_pct) / 100.0) * baseline_rmse
        )
    return min(candidates) if candidates else None


def _per_cell_breakdown(
    winner_payload: dict, shard_bundle, setup: EvalSetup
) -> Optional[dict]:
    """Winner's per-geometry-cell error breakdown on the eval set.

    Never raises — a failed predict degrades to None so the loop keeps
    running. Only meaningful when ``setup.per_cell_enabled``.
    """
    from autotokamak.surrogate.automl import predict_with_winner

    try:
        pred = predict_with_winner(winner_payload, shard_bundle.inputs)
        return per_cell_errors(
            shard_bundle.inputs,
            shard_bundle.psi,
            pred,
            bounds_low=setup.envelope_bounds[0],
            bounds_high=setup.envelope_bounds[1],
            n_bins=setup.n_bins,
        )
    except Exception:  # noqa: BLE001
        return None


def _worst_cell_accuracy(
    per_cell: Optional[dict], baseline_cell_rmse: Optional[dict]
) -> Optional[float]:
    """Accuracy of the WEAKEST occupied eval cell, from precomputed breakdowns.

    Per cell: ``100*(1 - winner_cell_rmse / baseline_cell_rmse)`` using that
    cell's OWN baseline (so a large-ψ region is not penalized for scale);
    returns the minimum over cells present in both maps, or None when
    nothing is comparable. Pure function — callers compute the (expensive)
    per-cell breakdown once and share it.
    """
    if per_cell is None or baseline_cell_rmse is None:
        return None
    accs = []
    for key, cell in per_cell["cells"].items():
        b = baseline_cell_rmse.get(key)
        if b is None or not np.isfinite(b) or b <= 0 or not np.isfinite(cell["rmse"]):
            continue
        accs.append(100.0 * (1.0 - cell["rmse"] / b))
    return float(min(accs)) if accs else None


def _per_cell_baseline(
    train_bundle, shard_bundle, setup: EvalSetup
) -> Optional[dict]:
    """Per-cell RMSE of the trivial mean-predictor — the worst-cell denominator."""
    try:
        base_pred = baseline_mean_prediction(train_bundle.psi, shard_bundle.n_samples)
        bpc = per_cell_errors(
            shard_bundle.inputs,
            shard_bundle.psi,
            base_pred,
            bounds_low=setup.envelope_bounds[0],
            bounds_high=setup.envelope_bounds[1],
            n_bins=setup.n_bins,
        )
        return {k: v["rmse"] for k, v in bpc["cells"].items()}
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: per-cell baseline failed: {exc}", file=sys.stderr)
        return None


def _initial_diagnostics(state: MetaState) -> dict:
    """Cheap diagnostics that don't require an existing winner."""
    from autotokamak.eval import diagnostics as diag_mod
    from autotokamak.surrogate.zoo import make_poly_ridge

    bundle = load_dataset(state.current_dataset_h5)
    factory = lambda: make_poly_ridge(alpha=0.1, degree=2)
    return diag_mod.run_all(bundle, model_factory=factory)


def _diagnostics_with_winner(state: MetaState) -> dict:
    """Full diagnostics including residual structure (requires a winner)."""
    from autotokamak.eval import diagnostics as diag_mod
    from autotokamak.surrogate.zoo import make_poly_ridge

    bundle = load_dataset(state.current_dataset_h5)
    factory = lambda: make_poly_ridge(alpha=0.1, degree=2)
    splits = None
    if state.best_winner_payload is not None:
        splits = kfold(bundle, k=4, test_frac=2 / bundle.n_samples, seed=state.seed)
    return diag_mod.run_all(
        bundle,
        model_factory=factory,
        winner_payload=state.best_winner_payload,
        splits=splits,
    )


def run(
    config_path: str,
    *,
    pick_action: ActionPicker = pick_action_via_llm,
    trace_enabled: bool = True,
    experiments_dir: Optional[Path] = None,
    model_override: Optional[str] = None,
    max_iterations_override: Optional[int] = None,
    n_samples_override: Optional[int] = None,
    phase2_time_budget_override: Optional[int] = None,
    use_baseline_picker: bool = False,
    workspace_override: Optional[str] = None,
    phase2_mode_override: Optional[str] = None,
    target_rmse_override: Optional[float] = None,
    target_rmse_ratio_override: Optional[float] = None,
    target_accuracy_pct_override: Optional[float] = None,
    target_worst_cell_accuracy_pct_override: Optional[float] = None,
) -> MetaReport:
    """Run the meta-loop. Returns the final ``MetaReport``.

    ``model_override`` and ``max_iterations_override``, if set, win over the
    values in the meta YAML — convenient for cheap test runs.

    ``phase2_mode_override`` selects the Phase-2 execution strategy:
      "structured" (default) — Optuna + DSPy library (fast, no code-gen)
      "codegen"             — URSA PlanningAgent + ExecutionAgent writes the runner

    ``use_baseline_picker=True`` forces the in-code baseline DSPy module
    (ignoring any saved optimized prompt). Used for A/B comparison after
    GEPA optimization.
    """
    if phase2_mode_override is not None and phase2_mode_override not in ("structured", "codegen"):
        raise ValueError(
            f"phase2_mode_override must be 'structured' or 'codegen', got {phase2_mode_override!r}"
        )
    # CLI/kwarg overrides win over the YAML; merged and re-validated in one go.
    overrides = {
        "model": model_override,
        "max_iterations": max_iterations_override,
        "phase2_mode": phase2_mode_override,
        "target_rmse": target_rmse_override,
        "target_rmse_ratio": target_rmse_ratio_override,
        "target_accuracy_pct": target_accuracy_pct_override,
        "target_worst_cell_accuracy_pct": target_worst_cell_accuracy_pct_override,
    }
    meta_config = MetaConfig.from_yaml(config_path)
    meta_config = MetaConfig.model_validate(
        {
            **meta_config.model_dump(),
            **{k: v for k, v in overrides.items() if v is not None},
        }
    )

    wall_start = time.time()
    workspace_path = resolve_workspace(workspace_override or meta_config.workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)
    (workspace_path / "iterations").mkdir(exist_ok=True)
    (workspace_path / "datasets").mkdir(exist_ok=True)
    (workspace_path / "surrogate_runs").mkdir(exist_ok=True)

    # Resolve initial dataset path (relative to repo root if not absolute).
    initial_dataset = Path(meta_config.initial_dataset_h5)
    if not initial_dataset.is_absolute():
        initial_dataset = (REPO_ROOT / initial_dataset).resolve()

    base_sweep = None
    if meta_config.base_sweep_config:
        base_path = Path(meta_config.base_sweep_config)
        if not base_path.is_absolute():
            base_path = (REPO_ROOT / base_path).resolve()
        base_sweep = SweepConfig.from_yaml(base_path)
        if n_samples_override is not None:
            bumped = base_sweep.sampling.model_copy(
                update={"n_samples": int(n_samples_override)}
            )
            base_sweep = base_sweep.model_copy(update={"sampling": bumped})

    setup = _setup_eval_sets(
        meta_config, initial_dataset, workspace_path / "datasets", base_sweep
    )
    split_info = setup.split_info

    state = MetaState(
        workspace=workspace_path,
        current_dataset_h5=setup.train_pool,
        base_sweep_config=base_sweep,
        phase2_prompt=(REPO_ROOT / meta_config.phase2_prompt),
        seed=meta_config.seed,
        test_shard_h5=setup.eval_h5,
        envelope_bounds=setup.envelope_bounds,
        phase2_mode=meta_config.phase2_mode,
        phase2_model=meta_config.model,
        phase2_max_rounds=meta_config.phase2_max_rounds,
        phase2_time_budget_seconds=(
            int(phase2_time_budget_override)
            if phase2_time_budget_override is not None
            else None
        ),
        use_baseline_picker=bool(use_baseline_picker),
    )

    trace = None
    if trace_enabled:
        try:
            target_dir = experiments_dir or DEFAULT_EXPERIMENTS_DIR
            trace = RunTrace.open(
                experiments_dir=target_dir,
                prompt_path=(REPO_ROOT / config_path
                             if not str(config_path).startswith("/") else Path(config_path)),
                model=meta_config.model,
                feedback_rounds=meta_config.max_iterations,
                workspace=str(workspace_path),
            )
            print(f"Meta-trace: {trace._path}")
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: failed to open meta-trace ({exc})", file=sys.stderr)
            trace = None

    # Baseline: mean-predictor of the train pool evaluated on the frozen
    # eval set — the same samples every winner is measured on, so
    # final_rmse / baseline_rmse is apples-to-apples. The shard bundle is
    # also cached on state so per-iteration RMSE checks don't reload it.
    train_bundle = load_dataset(state.current_dataset_h5)
    shard_bundle = load_dataset(state.test_shard_h5)
    state.shard_bundle_cache = shard_bundle
    baseline_rmse = float(
        baseline_mean_predictor_rmse(train_bundle.psi, shard_bundle.psi)
    )

    # The loop stops as soon as the frozen-shard RMSE meets the bar — no
    # point spending further iterations once the model is good enough.
    target_rmse_abs = _resolve_target_rmse_abs(meta_config, baseline_rmse)
    state.target_rmse_abs = target_rmse_abs
    if target_rmse_abs is not None:
        acc_note = ""
        if meta_config.target_accuracy_pct is not None:
            acc_note = f" [>= {meta_config.target_accuracy_pct:.1f}% error reduction]"
        print(
            f"Early-stop target: shard RMSE <= {target_rmse_abs:.6g} "
            f"(baseline {baseline_rmse:.6g}){acc_note}"
        )

    # Per-cell baseline for the worst-cell performance stop / reporting. Only
    # meaningful in envelope mode (per-cell scoring needs the envelope box).
    if meta_config.target_worst_cell_accuracy_pct is not None and not setup.per_cell_enabled:
        raise ValueError(
            "target_worst_cell_accuracy_pct requires eval_envelope with "
            "resolvable bounds (set base_sweep_config) — per-cell scoring "
            "needs the target-envelope box."
        )
    baseline_cell_rmse: Optional[dict] = None
    if setup.per_cell_enabled:
        baseline_cell_rmse = _per_cell_baseline(train_bundle, shard_bundle, setup)

    history: list[MetaIterationRecord] = []
    terminated_by: str = "iterations_cap"
    # Worst-cell check cache: (winner identity, computed accuracy). The
    # breakdown is a shard-sized predict; recompute only when the best
    # winner actually changed.
    wca_cache: tuple = (None, None)

    try:
        for i in range(meta_config.max_iterations):
            iter_dir = workspace_path / "iterations" / f"{i:03d}"
            iter_dir.mkdir(exist_ok=True)

            diag = (
                _diagnostics_with_winner(state)
                if state.best_winner_payload is not None
                else _initial_diagnostics(state)
            )
            (iter_dir / "diagnostics.json").write_text(
                json.dumps(diag, indent=2, default=str)
            )

            print(f"\n=== META iteration {i} (of {meta_config.max_iterations}) ===")
            # Record the EXACT LM inputs for this decision (regardless of which
            # picker is plugged in) so GEPA later trains on byte-identical
            # inputs — see agent/dspy/picker_inputs.py (train/serve skew fix).
            from autotokamak.agent.dspy.picker_inputs import picker_inputs_from_runtime

            picker_inputs = picker_inputs_from_runtime(meta_config, state, diag, history)
            decision = pick_action(meta_config, state, diag, history)
            (iter_dir / "action.json").write_text(decision.model_dump_json(indent=2))
            print(f"  action   : {decision.action}")
            print(f"  diagnosis: {_truncate(decision.diagnosis, 120)}")

            record = MetaIterationRecord(
                iteration=i,
                started_utc=_now(),
                diagnostics=diag,
                decision=decision,
                picker_inputs=picker_inputs,
            )

            if decision.action == "terminate":
                record.finished_utc = _now()
                term = decision.terminate
                record.result = {
                    "kind": "terminate",
                    "reason": term.reason if term else "",
                    "confidence": term.confidence if term else None,
                }
                history.append(record)
                terminated_by = "agent"
                break

            try:
                result = dispatch(decision, state)
            except Exception as exc:  # noqa: BLE001
                result = {"kind": "error", "error": f"{type(exc).__name__}: {exc}"}
                print(f"  action dispatch failed: {result['error']}", file=sys.stderr)

            (iter_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
            record.result = result
            print(f"  outcome  : {_action_outcome(result)}")

            state.actions_taken.append(decision.action)
            prev_rmse = state.rmse_history[-1] if state.rmse_history else None
            rmse_after = measure_test_rmse(state)
            record.rmse_after = rmse_after
            _print_iteration_metrics(rmse_after, prev_rmse, baseline_rmse)
            if rmse_after is not None:
                state.rmse_history.append(rmse_after)

            record.finished_utc = _now()
            history.append(record)
            print(f"  iter time : {_fmt_duration(_record_duration(record))}")

            stop_for_target = False
            if (
                target_rmse_abs is not None
                and rmse_after is not None
                and rmse_after <= target_rmse_abs
            ):
                stop_for_target = True
                print(
                    f"  target reached: shard RMSE {rmse_after:.6g} <= "
                    f"{target_rmse_abs:.6g}; stopping."
                )
            if (
                not stop_for_target
                and meta_config.target_worst_cell_accuracy_pct is not None
                and baseline_cell_rmse is not None
                and state.best_winner_payload is not None
            ):
                if wca_cache[0] is not state.best_winner_payload:
                    pc = _per_cell_breakdown(state.best_winner_payload, shard_bundle, setup)
                    wca_cache = (
                        state.best_winner_payload,
                        _worst_cell_accuracy(pc, baseline_cell_rmse),
                    )
                wca = wca_cache[1]
                if wca is not None and wca >= meta_config.target_worst_cell_accuracy_pct:
                    stop_for_target = True
                    print(
                        f"  worst-cell target reached: weakest cell "
                        f"{wca:.1f}% >= {meta_config.target_worst_cell_accuracy_pct:.1f}%; "
                        f"stopping."
                    )
            if stop_for_target:
                terminated_by = "target_reached"
                break

        # Final refit on the latest dataset state, if we have any winner.
        winner_path = workspace_path / "winner.pkl"
        if state.best_winner_payload is not None and state.best_winner_path is not None:
            import shutil

            shutil.copy(state.best_winner_path, winner_path)

        # None (not baseline) when no winner was ever produced — falling back
        # to baseline_rmse here made a winnerless run read as "matched
        # baseline" in the report.
        final_rmse = state.best_rmse if state.best_rmse != float("inf") else None
        # Intuitive "accuracy" the performance stop is phrased against: error
        # reduction vs the trivial baseline mean-predictor on the frozen shard.
        final_accuracy_pct = (
            100.0 * (1.0 - final_rmse / baseline_rmse)
            if (final_rmse is not None and baseline_rmse > 0)
            else None
        )

        # Per-geometry-cell breakdown on the envelope eval set: shows WHERE
        # the surrogate is weak (the headline for active sampling). Computed
        # once; the worst-cell accuracy is derived from the same breakdown.
        eval_per_cell = None
        if setup.per_cell_enabled and state.best_winner_payload is not None:
            eval_per_cell = _per_cell_breakdown(
                state.best_winner_payload, shard_bundle, setup
            )
        final_worst_cell_accuracy_pct = _worst_cell_accuracy(
            eval_per_cell, baseline_cell_rmse
        )

        actions_taken_typed = [a for a in state.actions_taken if a in VALID_ACTIONS]
        report = MetaReport(
            n_iterations=len(history),
            terminated_by=terminated_by,
            target_rmse=target_rmse_abs,
            final_rmse=final_rmse,
            baseline_rmse=float(baseline_rmse),
            test_shard_path=str(setup.eval_h5),
            n_test_samples=int(split_info["n_test"]),
            n_train_pool_samples=int(split_info["n_train_success"]),
            initial_rmse=state.rmse_history[0] if state.rmse_history else None,
            winner_model_name=(
                state.best_surrogate_report.get("winner_model_name", "none")
                if state.best_surrogate_report else "none"
            ),
            winner_hyperparams=(
                state.best_surrogate_report.get("winner_hyperparams", {})
                if state.best_surrogate_report else {}
            ),
            rmse_history=list(state.rmse_history),
            actions_taken=actions_taken_typed,
            eval_mode=("envelope" if meta_config.eval_envelope is not None else "seed_shard"),
            eval_per_cell=eval_per_cell,
            target_accuracy_pct=meta_config.target_accuracy_pct,
            final_accuracy_pct=final_accuracy_pct,
            target_worst_cell_accuracy_pct=meta_config.target_worst_cell_accuracy_pct,
            final_worst_cell_accuracy_pct=final_worst_cell_accuracy_pct,
        )
        report_path = workspace_path / "report.json"
        report_path.write_text(report.model_dump_json(indent=2))

        # Write a meta_trace.json with the full per-iteration log alongside the
        # RunTrace.json (which only records the outer skeleton).
        (workspace_path / "meta_trace.json").write_text(
            json.dumps(
                {
                    "iterations": [r.model_dump(mode="json") for r in history],
                    "report": report.model_dump(mode="json"),
                },
                indent=2,
                default=str,
            )
        )

        if trace:
            trace.record_artifacts(
                workspace_path,
                expected_artifacts=["winner.pkl", "report.json", "meta_trace.json"],
            )
            try:
                from autotokamak.agent.dspy.metric_meta import score_meta_run

                score = score_meta_run(workspace_path)
                trace.record_score(score)
                print(f"\nMeta score: {score.total:.3f}")
            except Exception as exc:  # noqa: BLE001
                print(f"WARNING: meta scoring failed: {exc}", file=sys.stderr)
            trace.mark_completed()

        _print_summary_table(history, baseline_rmse)

        print(f"\n=== META FINAL ===")
        print(f"  total wall time: {_fmt_duration(time.time() - wall_start)}")
        print(f"  iterations: {len(history)}; terminated_by: {terminated_by}")
        final_str = f"{final_rmse:.4f}" if final_rmse is not None else "n/a (no winner)"
        print(f"  final RMSE: {final_str}; baseline: {baseline_rmse:.4f} "
              f"(both on frozen shard, n={split_info['n_test']})")
        if len(state.rmse_history) >= 2:
            net = _pct_improvement(state.rmse_history[-1], state.rmse_history[0])
            print(
                f"  net improvement across iterations: {_fmt_pct(net)} "
                f"(first measured → last measured)"
            )
        if final_accuracy_pct is not None:
            bar = (
                f" (target {meta_config.target_accuracy_pct:.1f}%)"
                if meta_config.target_accuracy_pct is not None else ""
            )
            print(f"  accuracy: {final_accuracy_pct:.1f}% error reduction vs baseline{bar}")
        if eval_per_cell is not None and eval_per_cell.get("worst_cell"):
            wc = eval_per_cell["worst_cell"]
            print(
                f"  per-cell: worst cell {wc['key']} RMSE {wc['rmse']:.4f} "
                f"(n={wc['n']}); mean-cell {eval_per_cell['mean_cell_rmse']:.4f}; "
                f"{eval_per_cell['n_cells_occupied']}/{eval_per_cell['n_cells_total']} "
                f"cells covered"
            )
        if final_worst_cell_accuracy_pct is not None:
            bar = (
                f" (target {meta_config.target_worst_cell_accuracy_pct:.1f}%)"
                if meta_config.target_worst_cell_accuracy_pct is not None else ""
            )
            print(
                f"  worst-cell accuracy: {final_worst_cell_accuracy_pct:.1f}% "
                f"(weakest region vs its own baseline){bar}"
            )
        return report

    except KeyboardInterrupt:
        if trace:
            try:
                trace.record_artifacts(workspace_path)
            except Exception:  # noqa: BLE001
                pass
            trace.mark_interrupted()
        raise
    except Exception as exc:
        if trace:
            try:
                trace.record_artifacts(workspace_path)
            except Exception:  # noqa: BLE001
                pass
            trace.mark_errored(exc)
        raise


def main():
    parser = argparse.ArgumentParser(description="Run the autotokamak meta-agent loop.")
    parser.add_argument("--config", required=True, help="Path to surrogate_meta.yaml")
    parser.add_argument("--no-trace", action="store_true")
    parser.add_argument("--experiments-dir", default=None)
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Override max_iterations from the config (1 = cheapest real test).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the LLM model (e.g. 'openai:gpt-5-mini' for cheap testing).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Override sampling.n_samples in the loaded base_sweep_config "
             "(effective on the next regen_dataset action).",
    )
    parser.add_argument(
        "--time-budget-seconds",
        type=int,
        default=None,
        help="Injected as a HARD BUDGET directive into every extend_search "
             "overlay prompt so the Phase-2 agent writes it into "
             "surrogate_config.yaml.",
    )
    parser.add_argument(
        "--use-baseline",
        action="store_true",
        help="Force the in-code baseline action-picker prompt (ignore any optimized JSON). "
             "Used for A/B comparison after GEPA optimization.",
    )
    parser.add_argument(
        "--workspace",
        default=None,
        help="Override the workspace from the config. Required for trace "
             "collection: each run needs its OWN workspace or successive runs "
             "clobber each other's meta_trace.json.",
    )
    parser.add_argument(
        "--target-rmse",
        type=float,
        default=None,
        help="Early-stop when the frozen-shard RMSE reaches this ABSOLUTE "
             "value (dataset psi units).",
    )
    parser.add_argument(
        "--target-rmse-ratio",
        type=float,
        default=None,
        help="Early-stop when shard RMSE <= ratio * baseline RMSE "
             "(scale-free; e.g. 0.3).",
    )
    parser.add_argument(
        "--target-accuracy-pct",
        type=float,
        default=None,
        help="Performance stop: stop once the winner is at least this percent "
             "better than the baseline mean-predictor (error reduction; e.g. "
             "90 = stop at 90%%). Scale-free; max-iterations is the cap.",
    )
    parser.add_argument(
        "--target-worst-cell-accuracy-pct",
        type=float,
        default=None,
        help="Performance stop on the WEAKEST geometry region: stop once EVERY "
             "eval cell is >= this %% better than its own baseline. Requires "
             "eval_envelope. Stricter than --target-accuracy-pct.",
    )
    parser.add_argument(
        "--mode",
        choices=("fast", "ursa"),
        default=None,
        help="Phase-2 execution mode: 'fast' (Optuna+DSPy library, default) or "
             "'ursa' (hybrid: URSA code-gen for nested Phase-2 searches). "
             "Equivalent to --phase2-mode structured|codegen.",
    )
    args = parser.parse_args()

    phase2_mode = None
    if args.mode == "fast":
        phase2_mode = "structured"
    elif args.mode == "ursa":
        phase2_mode = "codegen"

    run(
        config_path=args.config,
        trace_enabled=not args.no_trace,
        experiments_dir=Path(args.experiments_dir) if args.experiments_dir else None,
        model_override=args.model,
        max_iterations_override=args.max_iterations,
        n_samples_override=args.n_samples,
        phase2_time_budget_override=args.time_budget_seconds,
        use_baseline_picker=args.use_baseline,
        workspace_override=args.workspace,
        phase2_mode_override=phase2_mode,
        target_rmse_override=args.target_rmse,
        target_rmse_ratio_override=args.target_rmse_ratio,
        target_accuracy_pct_override=args.target_accuracy_pct,
        target_worst_cell_accuracy_pct_override=args.target_worst_cell_accuracy_pct,
    )


if __name__ == "__main__":
    main()
