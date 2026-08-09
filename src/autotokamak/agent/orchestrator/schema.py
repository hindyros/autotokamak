# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Pydantic schemas for the meta-agent: config, decisions, per-iteration records.

The LLM's structured output schema is ``ActionDecision`` — the meta-loop forces
the model to emit one of these per iteration, so the action space is small,
typed, and validated before dispatch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field
from typing import get_args

from autotokamak.data.envelope import EnvelopeConfig


ActionKind = Literal["regen_dataset", "enrich_active", "extend_search", "terminate"]
# Single source of truth for the action vocabulary — the dispatcher table,
# DSPy coercion, meta-metric, and meta-loop all derive from this instead of
# repeating the string set (adding an action = editing ActionKind + DISPATCH).
VALID_ACTIONS: tuple[str, ...] = get_args(ActionKind)

AcquisitionStrategy = Literal["auto", "residual_ucb", "uncertainty", "space_filling"]
VALID_ACQUISITION_STRATEGIES: tuple[str, ...] = get_args(AcquisitionStrategy)


class RegenDatasetOverrides(BaseModel):
    """Flat overrides applied on top of the current sweep config.

    Use dotted keys (``"sampling.n_samples"``, ``"fixed.mesh_dx"``) so the
    LLM can target any field without us hard-coding which knobs are tweakable.
    Validation against ``SweepConfig`` happens in the dispatcher after the
    merge.
    """

    model_config = ConfigDict(extra="allow")
    overrides: Dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""


class EnrichActivePayload(BaseModel):
    """Payload for the active-learning data-enrichment action.

    Unlike ``regen_dataset`` (blind LHS append), this action picks the NEW
    sample locations on purpose. With a trained winner available, the batch
    is selected by RESIDUAL-DRIVEN UCB: an error model fit on the winner's
    out-of-fold residuals targets the geometries where the model is
    measurably weak (``beta`` trades exploitation of measured weakness
    against exploration of unvisited envelope regions). Before any winner
    exists, a PCA-GP variance acquisition is used instead. Both are
    feasibility-weighted. See ``autotokamak.data.acquire``.
    """

    model_config = ConfigDict(extra="forbid")
    n_new: int = Field(
        default=500,
        ge=1,
        le=2000,
        description="How many acquisition-selected samples to solve and append.",
    )
    strategy: AcquisitionStrategy = Field(
        default="auto",
        description="Which acquisition strategy to use. 'auto' = residual-driven "
        "when a winner exists, else model-agnostic uncertainty. "
        "'residual_ucb' = force targeting the winner's measured weakness. "
        "'uncertainty' = model-agnostic PCA-GP field variance (use when the "
        "winner's residuals are not yet trustworthy). 'space_filling' = pure "
        "maximin coverage, no error model (use to blanket a new region).",
    )
    beta: float = Field(
        default=1.0,
        ge=0.0,
        le=3.0,
        description="UCB exploration weight for the residual-driven path: "
        "0 = pure exploitation of measured weakness, larger = more "
        "exploration of unsampled regions. Ignored unless strategy resolves "
        "to residual_ucb.",
    )
    feasibility_weighting: bool = Field(
        default=True,
        description="Down-weight candidates near previously FAILED solves.",
    )
    rationale: str = ""


class ExtendSearchFocus(BaseModel):
    """What the meta-agent wants the nested Phase-2 sub-run to emphasize.

    The dispatcher writes a thin overlay prompt that prepends this focus
    block to the existing ``surrogate_automl.yaml`` problem text. The nested
    LLM is otherwise unchanged.
    """

    model_config = ConfigDict(extra="forbid")
    models_to_emphasize: List[Literal["gp", "kernel_ridge", "poly_ridge", "mlp"]] = Field(
        default_factory=list,
        description="Subset of the zoo to focus on; empty = let Phase-2 pick.",
    )
    widen_params: List[str] = Field(
        default_factory=list,
        description="Specific hyperparameters to widen (e.g. 'gp.length_scale').",
    )
    n_trials_hint: Optional[int] = Field(
        default=None,
        ge=1,
        le=200,
        description="Total trials suggestion for the nested Phase-2 search.",
    )
    rationale: str = ""


class TerminateReason(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reason: str
    confidence: Literal["low", "medium", "high"] = "medium"


class ActionDecision(BaseModel):
    """The typed JSON the meta-agent's LLM emits per iteration.

    Exactly one of (``regen``, ``enrich``, ``extend``, ``terminate``) is
    populated; the ``action`` discriminator selects which.
    """

    model_config = ConfigDict(extra="forbid")
    action: ActionKind
    regen: Optional[RegenDatasetOverrides] = None
    enrich: Optional[EnrichActivePayload] = None
    extend: Optional[ExtendSearchFocus] = None
    terminate: Optional[TerminateReason] = None
    diagnosis: str = Field(
        default="",
        description="Free-text statement of what the agent thinks the bottleneck is.",
    )

    def selected_payload(self) -> BaseModel | None:
        return {"regen_dataset": self.regen,
                "enrich_active": self.enrich,
                "extend_search": self.extend,
                "terminate": self.terminate}[self.action]


class MetaConfig(BaseModel):
    """Top-level ``surrogate_meta.yaml`` schema.

    Differs from ``SurrogateConfig`` because the meta-agent's job is
    cross-phase orchestration, not surrogate training per se.
    """

    model_config = ConfigDict(extra="allow")
    max_iterations: int = Field(default=3, ge=1, le=20)
    initial_dataset_h5: str = Field(description="Path to the starting dataset.")
    base_sweep_config: Optional[str] = Field(
        default=None,
        description="Path to a SweepConfig YAML the regen_dataset action overrides.",
    )
    phase2_prompt: str = Field(
        default="src/autotokamak/agent/prompts/surrogate_automl.yaml",
        description="The prompt the extend_search action invokes (codegen mode only).",
    )
    phase2_mode: Literal["structured", "codegen"] = Field(
        default="structured",
        description="How extend_search runs Phase-2: 'structured' = deterministic "
        "automl_loop with one typed LLM decision per round; 'codegen' = the "
        "legacy nested plan_execute_feedback agent.",
    )
    phase2_max_rounds: int = Field(default=3, ge=1, le=10)
    enrich_n_new: Optional[int] = Field(
        default=None,
        ge=1,
        le=2000,
        description="When set, force every enrich_active action to acquire "
        "exactly this many samples (experiment control; overrides the "
        "picker's chosen n_new).",
    )
    holdout_test_frac: float = Field(
        default=0.15,
        gt=0.0,
        lt=0.5,
        description="Fraction of the initial dataset's successful samples frozen "
        "into the held-out test shard at meta-loop start. IGNORED when "
        "``eval_envelope`` is set (the envelope eval set replaces the shard "
        "and the whole initial dataset becomes train pool).",
    )
    holdout_min_test: int = Field(default=2, ge=1)
    # Full-envelope evaluation set (design doc §D2). When set, the frozen
    # eval samples cover the TARGET envelope (possibly wider than the seed
    # sweep box) instead of being carved from the seed dataset — required
    # for honest measurement once enrich_active samples new geometries.
    # Acquisition bounds for enrich_active follow the envelope too.
    eval_envelope: Optional["EnvelopeConfig"] = Field(
        default=None,
        description="Envelope eval-set spec: either {'h5': path} for a "
        "pre-solved set, or generation knobs (n_eval, parameters, seed). "
        "None = legacy seed-shard behavior.",
    )
    # Early-stopping quality bar (both optional; loop stops when EITHER is
    # met by the frozen-shard RMSE; max_iterations remains the safety net
    # for unreachable targets):
    target_rmse: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Absolute shard-RMSE target in the dataset's psi units.",
    )
    target_rmse_ratio: Optional[float] = Field(
        default=None,
        gt=0.0,
        le=1.0,
        description="Relative target: stop when shard RMSE <= ratio * baseline "
        "(mean-predictor) RMSE. Scale-free, so it survives dataset changes.",
    )
    target_accuracy_pct: Optional[float] = Field(
        default=None,
        ge=0.0,
        lt=100.0,
        description="Performance stop, phrased as accuracy: stop as soon as the "
        "winner is at least this percent better than the trivial baseline "
        "mean-predictor on the frozen eval set, i.e. error reduction "
        "100*(1 - rmse/baseline) >= target. 90 => stop at 90% error reduction "
        "(equivalently target_rmse_ratio = 0.10). Scale-free. When several "
        "targets are set the STRICTEST (lowest-RMSE) one wins; max_iterations "
        "remains the safety cap for unreachable targets.",
    )
    target_worst_cell_accuracy_pct: Optional[float] = Field(
        default=None,
        ge=0.0,
        lt=100.0,
        description="Performance stop on the WEAKEST geometry region: stop only "
        "once EVERY occupied eval cell is at least this percent better than the "
        "baseline mean-predictor IN THAT CELL, i.e. "
        "min_cells 100*(1 - cell_rmse/cell_baseline) >= target. Uses each "
        "cell's own baseline so a large-psi region is not penalized for its "
        "scale. Stricter than target_accuracy_pct (which gates only the "
        "aggregate) and requires eval_envelope for per-cell scoring. "
        "max_iterations remains the cap.",
    )
    seed: int = 0
    workspace: str = "examples/surrogate_meta"
    model: str = "openai:gpt-5.2"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MetaConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise ValueError("Top-level YAML must be a mapping/object.")
        return cls.model_validate(raw)


class MetaIterationRecord(BaseModel):
    """One row in the meta-trace's iteration log."""

    model_config = ConfigDict(extra="allow")
    iteration: int
    started_utc: str
    finished_utc: str = ""
    diagnostics: Dict[str, Any] = Field(default_factory=dict)
    decision: ActionDecision
    result: Dict[str, Any] = Field(default_factory=dict)
    rmse_after: Optional[float] = None
    parent_run_id: Optional[str] = None  # for extend_search actions only


class MetaReport(BaseModel):
    """Final report.json written into the meta workspace."""

    model_config = ConfigDict(extra="allow")
    n_iterations: int
    terminated_by: Literal["agent", "iterations_cap", "target_reached"]
    # The resolved absolute target the run stopped against (None = no target).
    target_rmse: Optional[float] = None
    initial_rmse: Optional[float] = None
    # None = no winner was ever produced. All RMSEs (final, baseline, and the
    # per-iteration rmse_history) are measured on the SAME frozen test shard.
    final_rmse: Optional[float] = Field(default=None, ge=0.0)
    baseline_rmse: float = Field(ge=0.0)
    # Performance stop, phrased as accuracy (error reduction vs baseline
    # mean-predictor). target_accuracy_pct is the requested bar (None if
    # unset); final_accuracy_pct is what the winner actually achieved.
    target_accuracy_pct: Optional[float] = None
    final_accuracy_pct: Optional[float] = None
    # Same, but for the WEAKEST geometry cell (envelope mode only).
    target_worst_cell_accuracy_pct: Optional[float] = None
    final_worst_cell_accuracy_pct: Optional[float] = None
    # Which eval-set mode the run used, and (envelope mode) the winner's
    # per-geometry-cell breakdown from surrogate.metrics.per_cell_errors.
    eval_mode: Optional[Literal["envelope", "seed_shard"]] = None
    eval_per_cell: Optional[Dict[str, Any]] = None
    test_shard_path: Optional[str] = None
    n_test_samples: Optional[int] = None
    n_train_pool_samples: Optional[int] = None
    winner_model_name: str
    winner_hyperparams: Dict[str, Any]
    rmse_history: List[float] = Field(default_factory=list)
    actions_taken: List[ActionKind] = Field(default_factory=list)


__all__ = [
    "AcquisitionStrategy",
    "ActionDecision",
    "ActionKind",
    "VALID_ACQUISITION_STRATEGIES",
    "VALID_ACTIONS",
    "EnrichActivePayload",
    "ExtendSearchFocus",
    "MetaConfig",
    "MetaIterationRecord",
    "MetaReport",
    "RegenDatasetOverrides",
    "TerminateReason",
]
