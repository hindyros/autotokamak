# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Harness ABC + RunResult — the contract every agent-substrate adapter meets."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal, Optional

from autotokamak.bench.taskspec import TaskSpec


@dataclass
class RunResult:
    status: Literal["completed", "errored", "interrupted", "timeout"]
    run_id: str
    condition: str          # "<level>-<harness>", e.g. "L3-claude_sdk"
    harness: str
    model: str
    workspace: Path
    trace_path: Optional[Path]
    wall_seconds: float
    cost_usd: Optional[float] = None   # filled where the substrate reports it
    error: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "run_id": self.run_id,
            "condition": self.condition,
            "harness": self.harness,
            "model": self.model,
            "workspace": str(self.workspace),
            "trace_path": str(self.trace_path) if self.trace_path else None,
            "wall_seconds": round(self.wall_seconds, 1),
            "cost_usd": self.cost_usd,
            "error": self.error,
            **({"extra": self.extra} if self.extra else {}),
        }


class Harness(ABC):
    """One agent substrate.

    Responsibilities of every adapter:
      * materialize ``task.symlinks`` in the workspace,
      * write a ``bench.trace`` RunTrace under ``run_dir`` (schema v1 plus
        the additive ``harness`` prompt field),
      * never write outside ``workspace`` and ``run_dir``,
      * map substrate-native events into trace rounds/steps best-effort and
        keep any raw event stream as ``run_dir/<name>_events.jsonl``.
    """

    name: ClassVar[str]

    @abstractmethod
    def run(
        self,
        task: TaskSpec,
        workspace: Path,
        *,
        run_dir: Path,
        model: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
    ) -> RunResult:
        """Execute ``task`` in ``workspace``; artifacts/trace under ``run_dir``."""

    # ---- shared helpers -------------------------------------------------

    def condition_for(self, task: TaskSpec) -> str:
        return f"{task.access_level}-{self.name}"

    def resolve_model(self, task: TaskSpec, model: Optional[str]) -> Optional[str]:
        return model or task.model_for(self.name)

    def prepare_workspace(self, task: TaskSpec, workspace: Path) -> None:
        from autotokamak.agent.runners.config import materialize_symlinks

        workspace.mkdir(parents=True, exist_ok=True)
        materialize_symlinks(workspace, task.symlinks)

    def workspace_note(self, workspace: Path) -> str:
        """Cwd-jail warning appended to the prompt by every subprocess-jailed
        adapter (claude_sdk/pi/cursor) — identical wording across harnesses so
        no cell gets private help against the shared cwd-escape failure mode."""
        return (
            f"\n\nRUNTIME NOTE: your workspace directory is\n{Path(workspace).resolve()}\n"
            "Create ALL files under this directory (absolute paths are "
            "safest). Never write outside it, and if you change "
            "directory for exploration, change back before writing."
        )

    def dry_run_info(self, task: TaskSpec, workspace: Path,
                     model: Optional[str] = None) -> dict[str, Any]:
        """What would run — subclasses override to expose exact argv/env keys."""
        return {
            "harness": self.name,
            "condition": self.condition_for(task),
            "model": self.resolve_model(task, model),
            "workspace": str(workspace),
            "timeout_seconds": task.timeout_seconds,
        }
