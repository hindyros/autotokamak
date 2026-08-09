# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Structured per-run tracing for the agent runners.

Why this exists
---------------
The DSPy integration plan (`docs/dspy_integration_plan.md` §3) is gated on
having structured `(prompt, plan, execution, outcome)` records. Plain stdout
logs are lossy and unparseable. Every agent run now writes
`experiments/<run_id>/trace.json` with the same shape so the trace corpus is
useful as soon as it exists.

Schema (v1)
-----------
::

    {
      "schema_version": 1,
      "run_id": "20260616T120000Z",
      "started_utc": "...",
      "finished_utc": "..." | null,
      "status": "running" | "completed" | "errored" | "interrupted",
      "prompt": {
        "path": "src/autotokamak/agent/prompts/...",
        "sha256": "abc123...",
        "model": "openai:gpt-5.2",
        "feedback_rounds": 2,
        "workspace": "examples/dataset_generation/"
      },
      "rounds": [
        {
          "round": 1,
          "plan_steps": [{"name": "...", "description": "..."}, ...],
          "execution": [
            {
              "step": 1,
              "name": "...",
              "ok": true | false,
              "started_utc": "...",
              "finished_utc": "...",
              "result_excerpt": "first ~500 chars of result text",
              "error": null | "ExceptionType: msg"
            }
          ]
        }
      ],
      "artifacts": {
        "workspace_path": "/abs/path",
        "files_written": ["dataset_config.yaml", "run_dataset_sweep.py", ...],
        "dataset_h5": "examples/.../outputs/dataset.h5" | null
      },
      "score": null | {"total": 0.798, "hard_gates": {...}, "quality": {...}},
      "error": null | "fatal exception message"
    }

Guarantees
----------
* Trace I/O is wrapped in try/except inside `RunTrace.save()`. A trace write
  failure NEVER aborts the agent run — we log a warning to stderr and continue.
* `save()` does an atomic write (temp file + os.replace) so a SIGTERM mid-write
  leaves either the previous state or the new state, never a partial JSON.
* Strings are truncated to `EXCERPT_MAX` chars on insertion to keep the file
  small and avoid runaway memory for long agent outputs.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import sys
import tempfile
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
EXCERPT_MAX = 500
TRACE_FILENAME = "trace.json"


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _truncate(text: Any, *, max_chars: int = EXCERPT_MAX) -> str:
    if text is None:
        return ""
    s = str(text)
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + f"... [truncated, total {len(s)} chars]"


def _sha256_file(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def utc_run_id() -> str:
    """Sortable run id, ``20260616T120000Z``."""
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


@dataclass
class StepRecord:
    step: int
    name: str
    ok: bool = False
    started_utc: str = ""
    finished_utc: str = ""
    result_excerpt: str = ""
    error: str | None = None


@dataclass
class RoundRecord:
    round: int
    plan_steps: list[dict[str, str]] = field(default_factory=list)
    execution: list[StepRecord] = field(default_factory=list)


@dataclass
class RunTrace:
    """Per-run trace document. Mutable; call `save()` after every meaningful update.

    Construct via :py:meth:`open` or directly. Use the recording methods
    (`record_plan_steps`, `start_step`, `finish_step`, `record_artifacts`,
    `record_score`, `mark_*`) rather than poking the fields directly — they
    update timestamps and call `save()` for you.
    """

    schema_version: int = SCHEMA_VERSION
    run_id: str = ""
    started_utc: str = ""
    finished_utc: str | None = None
    status: str = "running"
    prompt: dict[str, Any] = field(default_factory=dict)
    rounds: list[RoundRecord] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    score: dict[str, Any] | None = None
    error: str | None = None
    # Optional cross-phase linkage for nested runs invoked by the meta-agent.
    parent_run_id: str | None = None
    meta_iteration: int | None = None

    # Set by `open()`; not serialized.
    _path: Path | None = field(default=None, repr=False)

    # ---- construction -----------------------------------------------

    @classmethod
    def open(
        cls,
        *,
        experiments_dir: Path,
        prompt_path: Path,
        model: str,
        feedback_rounds: int,
        workspace: str,
    ) -> "RunTrace":
        """Create a fresh run dir and initial trace, then write it to disk."""
        run_id = utc_run_id()
        run_dir = experiments_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        trace = cls(
            run_id=run_id,
            started_utc=_now_iso(),
            prompt={
                "path": str(prompt_path),
                "sha256": _sha256_file(prompt_path),
                "model": model,
                "feedback_rounds": int(feedback_rounds),
                "workspace": str(workspace),
            },
        )
        trace._path = run_dir / TRACE_FILENAME
        trace.save()
        return trace

    # ---- recording methods ------------------------------------------

    def start_round(self, round_no: int) -> RoundRecord:
        rec = RoundRecord(round=int(round_no))
        self.rounds.append(rec)
        self.save()
        return rec

    def record_plan_steps(self, round_rec: RoundRecord, steps: list[Any]) -> None:
        round_rec.plan_steps = [
            {
                "name": _truncate(getattr(s, "name", f"Step {i+1}"), max_chars=120),
                "description": _truncate(getattr(s, "description", str(s))),
            }
            for i, s in enumerate(steps)
        ]
        self.save()

    def start_step(self, round_rec: RoundRecord, step_no: int, name: str) -> StepRecord:
        rec = StepRecord(
            step=int(step_no),
            name=_truncate(name, max_chars=120),
            started_utc=_now_iso(),
        )
        round_rec.execution.append(rec)
        self.save()
        return rec

    def finish_step(
        self,
        step_rec: StepRecord,
        *,
        ok: bool,
        result_text: str | None = None,
        error: BaseException | str | None = None,
    ) -> None:
        step_rec.finished_utc = _now_iso()
        step_rec.ok = bool(ok)
        if result_text is not None:
            step_rec.result_excerpt = _truncate(result_text)
        if error is not None:
            if isinstance(error, BaseException):
                step_rec.error = f"{type(error).__name__}: {error}"
            else:
                step_rec.error = _truncate(error)
        self.save()

    def record_artifacts(
        self,
        workspace_path: Path,
        *,
        expected_artifacts: list[str] | None = None,
    ) -> None:
        """Record the workspace's top-level files and any expected artifacts.

        ``expected_artifacts`` is a list of workspace-relative paths the prompt
        commits the agent to produce (e.g. ``["outputs/dataset.h5"]`` for
        Phase 1, ``["outputs/winner.pkl", "outputs/report.json", "outputs/study.db"]``
        for Phase 2). For each, the trace stores whether it exists and its
        absolute path. The legacy ``dataset_h5`` key is kept for back-compat
        with downstream consumers reading older traces.
        """
        if expected_artifacts is None:
            expected_artifacts = ["outputs/dataset.h5"]
        files: list[str] = []
        artifact_status: dict[str, dict[str, Any]] = {}
        dataset_h5_legacy: str | None = None
        try:
            if workspace_path.is_dir():
                for child in sorted(workspace_path.iterdir()):
                    if child.is_file():
                        files.append(child.name)
                for rel in expected_artifacts:
                    candidate = workspace_path / rel
                    exists = candidate.is_file()
                    artifact_status[rel] = {
                        "exists": exists,
                        "path": str(candidate) if exists else None,
                    }
                    if rel == "outputs/dataset.h5" and exists:
                        dataset_h5_legacy = str(candidate)
        except OSError:
            pass
        self.artifacts = {
            "workspace_path": str(workspace_path),
            "files_written": files,
            "expected": artifact_status,
            "dataset_h5": dataset_h5_legacy,
        }
        self.save()

    def record_score(self, score: Any) -> None:
        """Record a `ScoreReport` (from `autotokamak.agent.dspy.metric`) or a dict."""
        if score is None:
            self.score = None
        elif hasattr(score, "total") and hasattr(score, "hard_gates"):
            self.score = {
                "total": float(score.total),
                "hard_gates": dict(score.hard_gates),
                "quality": dict(score.quality),
                "details": dict(getattr(score, "details", {})),
            }
        else:
            self.score = dict(score)
        self.save()

    def mark_completed(self) -> None:
        self.status = "completed"
        self.finished_utc = _now_iso()
        self.save()

    def mark_errored(self, exc: BaseException) -> None:
        self.status = "errored"
        self.finished_utc = _now_iso()
        self.error = f"{type(exc).__name__}: {exc}\n{_truncate(traceback.format_exc(), max_chars=2000)}"
        self.save()

    def mark_interrupted(self) -> None:
        self.status = "interrupted"
        self.finished_utc = _now_iso()
        self.save()

    # ---- serialization ----------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("_path", None)
        return d

    def save(self) -> None:
        """Atomic write to ``self._path``. Failures are warned, never raised."""
        if self._path is None:
            return
        try:
            payload = json.dumps(self.to_dict(), indent=2, default=str)
            parent = self._path.parent
            parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                "w", delete=False, dir=str(parent), encoding="utf-8"
            ) as tf:
                tf.write(payload)
                tf.flush()
                os.fsync(tf.fileno())
                tmp_name = tf.name
            os.replace(tmp_name, self._path)
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: trace save failed ({type(exc).__name__}: {exc})", file=sys.stderr)


__all__ = [
    "EXCERPT_MAX",
    "RoundRecord",
    "RunTrace",
    "SCHEMA_VERSION",
    "StepRecord",
    "TRACE_FILENAME",
    "utc_run_id",
]
