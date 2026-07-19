"""Composite scorer for Phase-2 surrogate AutoML runs.

Mirrors the shape of ``metric.score_run`` (same ``ScoreReport`` dataclass,
same ``hard_gates × quality`` composite). Lives in its own module so the
prompt-side ``scorer:`` dotted-path dispatch can pick it up without
case-branching inside the Phase-1 scorer.

Score shape
-----------
Hard gates (boolean, ALL must pass for a nonzero total):
    deliverables_present : the four agent-authored files exist
    winner_loads         : joblib.load works and the payload has the expected keys
    report_parseable     : outputs/report.json validates against SurrogateReport
    winner_predicts      : loaded model predicts the test split without error

Quality terms (each in [0, 1]; weighted sum becomes the total):
    val_rmse_vs_baseline (0.30) : 1 - val_rmse / baseline_mean_rmse, clipped
    pca_efficiency       (0.10) : explained variance vs 0.95 target
    generalization_gap   (0.20) : 1 - max(0, test - val) / val
    search_efficiency    (0.10) : best_at_25pct_trials / final_best
    zoo_coverage         (0.10) : fraction of 4 zoo models tried
    agent_decisiveness   (0.10) : 1 if terminated_by="agent" else 0
    runner_cleanliness   (0.10) : imports from eval/surrogate, no torch

The baseline RMSE is RE-COMPUTED here from the dataset to avoid trusting
agent-reported numbers; the scorer's claim is independent of the agent.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

EXPECTED_DELIVERABLES = (
    "surrogate_config.yaml",
    "run_surrogate_automl.py",
    "outputs/winner.pkl",
    "outputs/report.json",
    "README.md",
)
# Structured (automl_loop) workspaces have no agent-authored runner/README —
# the loop is library code; the per-round decisions live in search_history.
EXPECTED_DELIVERABLES_STRUCTURED = (
    "surrogate_config.yaml",
    "outputs/winner.pkl",
    "outputs/report.json",
    "outputs/search_history.jsonl",
)
ZOO_MODELS = ("gp", "kernel_ridge", "poly_ridge", "mlp")

WEIGHTS = {
    "val_rmse_vs_baseline": 0.30,
    "pca_efficiency": 0.10,
    "generalization_gap": 0.20,
    "search_efficiency": 0.10,
    "zoo_coverage": 0.10,
    "agent_decisiveness": 0.10,
    "runner_cleanliness": 0.10,
}


@dataclass
class ScoreReport:
    """Same shape as ``metric.ScoreReport`` so the runner-side recorder is reusable."""

    workspace: Path
    hard_gates: dict[str, bool] = field(default_factory=dict)
    quality: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def all_gates_pass(self) -> bool:
        return bool(self.hard_gates) and all(self.hard_gates.values())

    @property
    def total(self) -> float:
        if not self.all_gates_pass:
            return 0.0
        return float(sum(WEIGHTS[k] * self.quality.get(k, 0.0) for k in WEIGHTS))

    def summary(self) -> str:
        lines = [f"ScoreReport[{self.workspace}]"]
        lines.append("  hard gates:")
        for k, v in self.hard_gates.items():
            lines.append(f"    [{'PASS' if v else 'FAIL'}] {k}")
        if self.all_gates_pass:
            lines.append("  quality:")
            for k, w in WEIGHTS.items():
                q = self.quality.get(k, 0.0)
                lines.append(f"    {q:.3f}  (weight {w:.2f})  {k}")
            lines.append(f"  --> total = {self.total:.3f}")
        else:
            lines.append("  --> total = 0.000 (hard gate failed)")
        return "\n".join(lines)


def _clip01(x: float) -> float:
    if not np.isfinite(x):
        return 0.0
    return float(max(0.0, min(1.0, x)))


def _read_report(report_path: Path) -> tuple[Any | None, str | None]:
    """Validate ``outputs/report.json`` against ``SurrogateReport``."""
    try:
        from autotokamak.surrogate.schema import SurrogateReport
    except Exception as exc:  # noqa: BLE001
        return None, f"schema import failed: {type(exc).__name__}: {exc}"
    try:
        raw = json.loads(report_path.read_text(encoding="utf-8"))
        return SurrogateReport.model_validate(raw), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def _load_winner(winner_path: Path) -> tuple[dict | None, str | None]:
    try:
        import joblib
        payload = joblib.load(winner_path)
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"
    required = {"estimator", "pca", "model_name", "hyperparams"}
    missing = required - set(payload)
    if missing:
        return None, f"winner.pkl missing keys: {sorted(missing)}"
    return payload, None


def _resolve_dataset_path(ws: Path) -> Path | None:
    """Find the dataset HDF5 the agent trained against.

    Priority:
      1. surrogate_config.yaml -> dataset_h5 (relative to ws)
      2. ws/dataset.h5 (symlink the prompt suggests)
      3. None (skip baseline-aware quality terms)
    """
    cfg = ws / "surrogate_config.yaml"
    if cfg.is_file():
        try:
            import yaml
            data = yaml.safe_load(cfg.read_text()) or {}
            rel = str(data.get("dataset_h5", "")).strip()
            if rel:
                candidate = (ws / rel).resolve()
                if candidate.is_file():
                    return candidate
        except Exception:  # noqa: BLE001
            pass
    fallback = ws / "dataset.h5"
    if fallback.is_file():
        return fallback.resolve()
    return None


def score_surrogate_run(workspace: str | Path, *, mode: str = "codegen") -> ScoreReport:
    """Score a Phase-2 workspace.

    ``mode="codegen"`` (default): workspace produced by the surrogate_automl
    agent run (runner script + README expected).
    ``mode="structured"``: workspace produced by ``surrogate.automl_loop``
    (no runner script; cleanliness is judged from search_history.jsonl, and
    the test evaluation uses the frozen shard recorded in report.json when
    available — the structured winner trains on every train-pool row, so an
    internal re-split would test on training data).
    """
    ws = Path(workspace)
    report = ScoreReport(workspace=ws)
    report.details["mode"] = mode

    # -- Hard gate 1: deliverables present --
    expected = EXPECTED_DELIVERABLES_STRUCTURED if mode == "structured" else EXPECTED_DELIVERABLES
    missing = [f for f in expected if not (ws / f).is_file()]
    report.hard_gates["deliverables_present"] = not missing
    report.details["missing_deliverables"] = missing

    # -- Hard gate 2: winner.pkl loads --
    winner_payload, winner_err = _load_winner(ws / "outputs/winner.pkl")
    report.hard_gates["winner_loads"] = winner_payload is not None
    if winner_err:
        report.details["winner_load_error"] = winner_err

    # -- Hard gate 3: report.json parses against schema --
    parsed_report, report_err = _read_report(ws / "outputs/report.json")
    report.hard_gates["report_parseable"] = parsed_report is not None
    if report_err:
        report.details["report_parse_error"] = report_err

    # -- Hard gate 4: winner predicts on the test split --
    dataset_path = _resolve_dataset_path(ws)
    test_psi_pred = None
    test_psi_true = None
    shard_path = _resolve_shard_path(parsed_report) if mode == "structured" else None
    if winner_payload is not None and dataset_path is not None:
        try:
            from autotokamak.surrogate.dataset import load_dataset, kfold
            from autotokamak.surrogate.optuna_search import predict_with_winner

            bundle = load_dataset(dataset_path)
            if shard_path is not None:
                # Structured mode: evaluate on the frozen shard the loop
                # recorded — the winner trained on every train-pool row.
                shard = load_dataset(shard_path)
                X_test = shard.inputs
                test_psi_true = shard.psi
                report.details["test_set"] = f"frozen shard ({shard_path})"
            else:
                # Reproduce the same split the runner used (default seed=0).
                seed = (
                    int(parsed_report.model_extra.get("seed", 0))
                    if (parsed_report is not None and parsed_report.model_extra)
                    else 0
                )
                splits = kfold(bundle, k=4, test_frac=2 / bundle.n_samples, seed=seed)
                X_test = bundle.inputs[splits.test_idx]
                test_psi_true = bundle.psi[splits.test_idx]
            test_psi_pred = predict_with_winner(winner_payload, X_test)
            expected_shape = (len(X_test), bundle.psi.shape[1], bundle.psi.shape[2])
            ok_predict = (
                isinstance(test_psi_pred, np.ndarray) and test_psi_pred.shape == expected_shape
            )
            report.hard_gates["winner_predicts"] = bool(ok_predict)
            if not ok_predict:
                report.details["winner_predict_shape"] = str(getattr(test_psi_pred, "shape", None))
        except Exception as exc:  # noqa: BLE001
            report.hard_gates["winner_predicts"] = False
            report.details["winner_predict_error"] = f"{type(exc).__name__}: {exc}"
    else:
        report.hard_gates["winner_predicts"] = False
        if dataset_path is None:
            report.details["dataset_resolve_error"] = (
                "Could not resolve dataset.h5 from surrogate_config.yaml or ws/dataset.h5"
            )

    # All hard gates must pass before we score quality terms.
    if not report.all_gates_pass:
        return report

    assert parsed_report is not None  # mypy hint
    assert winner_payload is not None
    assert test_psi_pred is not None and test_psi_true is not None

    # -- val_rmse_vs_baseline --
    # Recompute the baseline from the data; do NOT trust the report's number.
    from autotokamak.surrogate.metrics import baseline_mean_predictor_rmse, psi_rmse

    from autotokamak.surrogate.dataset import load_dataset, kfold  # re-import for clarity

    bundle = load_dataset(dataset_path)
    splits = kfold(bundle, k=4, test_frac=2 / bundle.n_samples, seed=0)
    # Baseline RMSE: average across folds.
    baseline_vals = []
    for _, tr_idx, va_idx in splits.iter_folds():
        baseline_vals.append(
            baseline_mean_predictor_rmse(bundle.psi[tr_idx], bundle.psi[va_idx])
        )
    baseline = float(np.mean(baseline_vals))
    report.details["recomputed_baseline_rmse"] = baseline

    val_rmse = float(parsed_report.val_psi_rmse)
    report.quality["val_rmse_vs_baseline"] = _clip01(1.0 - val_rmse / max(baseline, 1e-12))

    # -- pca_efficiency --
    target_var = 0.95
    pev = float(parsed_report.pca_explained_var)
    report.quality["pca_efficiency"] = _clip01(pev / target_var)

    # -- generalization_gap --
    test_rmse = float(psi_rmse(test_psi_true, test_psi_pred))
    report.details["recomputed_test_rmse"] = test_rmse
    gap = max(0.0, test_rmse - val_rmse)
    report.quality["generalization_gap"] = _clip01(1.0 - gap / max(val_rmse, 1e-12))

    # -- search_efficiency --
    # Read the study DB if present and compute best_at_25%/best_final.
    eff = _search_efficiency(ws / "outputs/study.db", parsed_report.winner_model_name)
    report.quality["search_efficiency"] = _clip01(eff if eff is not None else 0.0)

    # -- zoo_coverage --
    tried = set(parsed_report.models_tried or [])
    report.quality["zoo_coverage"] = len(tried & set(ZOO_MODELS)) / len(ZOO_MODELS)

    # -- agent_decisiveness --
    report.quality["agent_decisiveness"] = 1.0 if parsed_report.terminated_by == "agent" else 0.0

    # -- runner_cleanliness --
    if mode == "structured":
        report.quality["runner_cleanliness"] = _history_cleanliness(
            ws / "outputs/search_history.jsonl"
        )
    else:
        report.quality["runner_cleanliness"] = _runner_cleanliness(
            ws / "run_surrogate_automl.py"
        )

    return report


def score_structured_surrogate_run(workspace: str | Path) -> ScoreReport:
    """Dotted-path-friendly wrapper: ``score_surrogate_run(mode="structured")``."""
    return score_surrogate_run(workspace, mode="structured")


def _resolve_shard_path(parsed_report) -> Path | None:
    """Frozen-shard path recorded by ``automl_loop`` in report.json extras."""
    if parsed_report is None or not parsed_report.model_extra:
        return None
    raw = parsed_report.model_extra.get("test_shard_path")
    if not raw:
        return None
    p = Path(str(raw))
    return p if p.is_file() else None


def _history_cleanliness(history_path: Path) -> float:
    """Structured-mode cleanliness: fraction of history lines with a valid spec."""
    if not history_path.is_file():
        return 0.0
    try:
        from autotokamak.surrogate.schema import SearchSpec

        lines = [ln for ln in history_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines:
            return 0.0
        ok = 0
        for ln in lines:
            try:
                SearchSpec.model_validate(json.loads(ln).get("spec", {}))
                ok += 1
            except Exception:  # noqa: BLE001
                pass
        return ok / len(lines)
    except Exception:  # noqa: BLE001
        return 0.0


def _search_efficiency(study_db_path: Path, winner_name: str) -> float | None:
    """Compute best_at_25%_of_trials / final_best for the winner's study.

    1.0 means "the winner was found very early"; lower means later trials
    were needed. Returns None if the DB can't be read, which the caller
    converts to a quality score of 0.
    """
    if not study_db_path.is_file():
        return None
    try:
        import optuna

        storage = f"sqlite:///{study_db_path}"
        study = optuna.load_study(study_name=winner_name, storage=storage)
        vals = sorted(
            float(t.value) for t in study.trials if t.value is not None and np.isfinite(t.value)
        )
        if not vals:
            return None
        early = vals[: max(1, len(vals) // 4)]
        return min(vals) / min(early)
    except Exception:  # noqa: BLE001
        return None


ALLOWED_ROOT_FILES = {
    "surrogate_config.yaml",
    "run_surrogate_automl.py",
    "README.md",
}

PLATFORM_ROOT_INFRA = {
    "graph_store.sqlite",   # URSA state
    "ursa_metrics",         # URSA per-step logs
    "dataset.h5",           # symlinked in by the runner
    "overlay_prompt.yaml",  # meta-loop's overlay for extend_search
    "outputs",              # the deliverable outputs dir itself
    "__pycache__",          # Python bytecode cache
    ".DS_Store",            # macOS junk (ignore, don't penalize)
}


def _workspace_root_hygiene(workspace: Path) -> float:
    """Fraction of workspace-root entries that are legitimate.

    1.0 when workspace root has ONLY the 3 deliverables + platform infra.
    Penalizes each extra file/dir linearly; hits 0.0 at 10+ extras. Catches
    the sprawl the agent invents when it panics — env_capture.py, preflight.py,
    warmup.py, iteration_logs/, backups/, outputs_backup_*, run_*_iter_N.py,
    etc. — even when those exact names aren't in the CONSTRAINT text.
    """
    if not workspace.is_dir():
        return 0.0
    try:
        seen = {p.name for p in workspace.iterdir()}
    except OSError:
        return 0.0
    extras = seen - ALLOWED_ROOT_FILES - PLATFORM_ROOT_INFRA
    return float(max(0.0, 1.0 - len(extras) / 10.0))


def _runner_cleanliness(runner_path: Path) -> float:
    """Codegen-mode cleanliness: library-usage check + workspace-root hygiene.

    (Structured-mode workspaces have no agent-authored runner; they are graded
    by ``_history_cleanliness`` instead.)
    """
    if not runner_path.is_file():
        return 0.0
    src = runner_path.read_text(encoding="utf-8", errors="replace")
    clean = [
        r"from autotokamak\.eval",
        r"from autotokamak\.surrogate",
    ]
    dirty = [
        r"\bimport torch\b",
        r"\bimport tensorflow\b",
        r"\bfrom sklearn\.neural_network import MLPRegressor\b",  # use zoo.make_mlp
    ]
    n_clean = sum(1 for pat in clean if re.search(pat, src))
    n_dirty = sum(1 for pat in dirty if re.search(pat, src))
    clean_frac = n_clean / max(len(clean), 1)
    dirty_penalty = min(n_dirty / max(len(dirty), 1), 1.0)
    import_score = float(max(0.0, clean_frac - dirty_penalty))
    hygiene_score = _workspace_root_hygiene(runner_path.parent)
    return 0.5 * import_score + 0.5 * hygiene_score


__all__ = [
    "EXPECTED_DELIVERABLES",
    "EXPECTED_DELIVERABLES_STRUCTURED",
    "ScoreReport",
    "WEIGHTS",
    "ZOO_MODELS",
    "score_structured_surrogate_run",
    "score_surrogate_run",
]
