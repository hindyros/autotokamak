# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Shared utilities for all pipeline dispatchers."""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any

# Walk up from this file to find the repo root (contains pyproject.toml with name=autotokamak)
def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        candidate = parent / "pyproject.toml"
        if candidate.is_file():
            try:
                text = candidate.read_text()
                if 'name = "autotokamak"' in text or "name = 'autotokamak'" in text:
                    return parent
            except OSError:
                pass
    raise RuntimeError("Could not locate autotokamak repo root from pipelines/_common.py")


REPO_ROOT: Path = _find_repo_root()

WORKSPACE: dict[str, str] = {
    "phase1": "dataset_generation",
    "phase2": "surrogate_automl",
    "meta":   "surrogate_meta",
}

# Access levels the pre-written pipeline supports. L0 = scripted decisions
# (no LLM, reproducible); L1 = LLM-typed decisions via the DSPy pickers.
# L2/L3 (agent-written code) live in `python -m autotokamak.bench`, not here.
LEVELS = ("L0", "L1")

# Condition name recorded in the manifest: <level>-<decision substrate>.
CONDITION = {"L0": "L0-none", "L1": "L1-dspy"}


def resolve_output_dir(pipeline: str, level: str) -> Path:
    """Return examples/<workspace>/<level>/ — created if absent."""
    if pipeline not in WORKSPACE:
        raise ValueError(f"Unknown pipeline {pipeline!r}. Choose from {list(WORKSPACE)}")
    if level not in LEVELS:
        raise ValueError(f"Unknown level {level!r}. Choose from {LEVELS}")
    d = REPO_ROOT / "examples" / WORKSPACE[pipeline] / level
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_manifest(out_dir: Path, *, pipeline: str, level: str, **kwargs: Any) -> Path:
    """Write out_dir/manifest.json and return its path."""
    manifest = {
        "pipeline": pipeline,
        "level": level,
        "condition": CONDITION.get(level, level),
        "run_id": _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "output_dir": str(out_dir),
        **kwargs,
    }
    p = out_dir / "manifest.json"
    p.write_text(json.dumps(manifest, indent=2, default=str))
    return p


def default_dataset_path(level: str = "L0") -> Path:
    """Canonical dataset.h5 produced by phase1 --level <level>."""
    return REPO_ROOT / "examples" / "dataset_generation" / level / "outputs" / "dataset.h5"


def default_dataset_config() -> Path:
    return REPO_ROOT / "examples" / "dataset_generation" / "dataset_config.yaml"
