# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""TaskSpec — the harness-agnostic definition of one benchmark task.

A task YAML under ``benchmarks/tasks/`` declares WHAT the agent must do
(the problem text, the deliverable contract, the access level); a harness
adapter decides HOW an agent substrate executes it. One task therefore runs
unchanged across every harness, which is what makes conditions comparable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field


class TaskSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    # L2 = may import autotokamak; L3 = from scratch, import audited & forbidden.
    access_level: Literal["L2", "L3"]
    problem: str
    expected_artifacts: List[str] = Field(default_factory=list)
    # Dotted "<module>:<func>" consumed by bench.scoring.try_score.
    scorer: Optional[str] = None
    scorer_kwargs: Dict[str, Any] = Field(default_factory=dict)
    # [{"source": "./OpenFUSIONToolkit", "dest": "OpenFUSIONToolkit"}, ...]
    symlinks: List[Dict[str, str]] = Field(default_factory=list)
    # Per-harness default model, e.g. {"ursa": "openai:gpt-5.2",
    # "claude_sdk": "claude-sonnet-4-5"}. "default" is the fallback key.
    model: Dict[str, str] = Field(default_factory=dict)
    # Per-harness prompt appendix (e.g. URSA's graph_store.sqlite note).
    harness_notes: Dict[str, str] = Field(default_factory=dict)
    timeout_seconds: int = Field(default=21600, ge=60)
    # Consumed by harnesses that support outer feedback/fix loops.
    feedback_rounds: int = Field(default=1, ge=1, le=5)

    # Set by from_yaml; lets harnesses resolve relative paths.
    source_path: Optional[Path] = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TaskSpec":
        path = Path(path)
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Task YAML must be a mapping: {path}")
        raw.setdefault("task_id", path.stem)
        spec = cls.model_validate(raw)
        spec.source_path = path.resolve()
        return spec

    def render_prompt(self, harness: str) -> str:
        """Problem text plus the harness-specific appendix, if any."""
        note = self.harness_notes.get(harness, "").strip()
        return self.problem if not note else f"{self.problem.rstrip()}\n\n{note}\n"

    def model_for(self, harness: str) -> Optional[str]:
        return self.model.get(harness) or self.model.get("default")
