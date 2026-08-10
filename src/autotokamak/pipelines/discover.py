# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Locate a run's artifacts (dataset, eval set, winner, report) across the two
workspace layouts the platform produces.

Direct phase workspaces (``phase1``/``phase2``) put artifacts under ``outputs/``.
Meta-loop workspaces put ``winner.pkl``/``report.json`` at the workspace root and
datasets under ``datasets/`` (``train_pool.h5``, ``test_shard.h5``,
``iterN_dataset.h5`` grown pools, plus an optional ``eval_envelope.h5``). The
frozen split is recorded authoritatively in ``datasets/split_info.json``.

Every plotting/report tool routes through here, so a layout change is a one-file
fix instead of a scavenger hunt duplicated in each script. Without this, the plot
generators looked only for ``dataset.h5`` / ``outputs/winner.pkl`` and silently
produced nothing for meta runs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional


def _first_existing(paths) -> Optional[Path]:
    for p in paths:
        if p is not None and Path(p).exists():
            return Path(p)
    return None


def _split_info(workspace: Path) -> dict:
    p = workspace / "datasets" / "split_info.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            return {}
    return {}


def _latest_iter_dataset(datasets_dir: Path) -> Optional[Path]:
    """The grown pool from the highest meta iteration (largest, best coverage)."""
    if not datasets_dir.is_dir():
        return None
    best, best_n = None, -1
    for p in datasets_dir.glob("iter*_dataset.h5"):
        m = re.search(r"iter(\d+)_dataset\.h5$", p.name)
        n = int(m.group(1)) if m else 0
        if n > best_n:
            best_n, best = n, p
    return best


def find_training_dataset(workspace: Path | str) -> Optional[Path]:
    """Dataset to visualize physics from — the largest / most representative pool."""
    workspace = Path(workspace)
    # Meta: prefer the latest grown pool, else the initial train pool.
    meta = _latest_iter_dataset(workspace / "datasets")
    if meta is None:
        info = _split_info(workspace)
        meta = _first_existing(
            [info.get("train_path"), workspace / "datasets" / "train_pool.h5"]
        )
    if meta:
        return meta
    # Direct phase workspaces.
    direct = _first_existing(
        [workspace / "dataset.h5", workspace / "outputs" / "dataset.h5"]
    )
    if direct:
        return direct
    # Meta codegen sub-runs.
    sur = workspace / "surrogate_runs"
    if sur.is_dir():
        for it in sorted(sur.iterdir()):
            got = _first_existing([it / "dataset.h5", it / "outputs" / "dataset.h5"])
            if got:
                return got
    return None


def find_eval_dataset(workspace: Path | str) -> Optional[Path]:
    """Held-out set the winner is scored on — for honest true-vs-predicted plots.

    Falls back to the training dataset for direct phase workspaces, which have no
    separate frozen shard on disk.
    """
    workspace = Path(workspace)
    info = _split_info(workspace)
    got = _first_existing(
        [
            info.get("eval_envelope_path"),
            info.get("test_path"),
            workspace / "datasets" / "eval_envelope.h5",
            workspace / "datasets" / "test_shard.h5",
        ]
    )
    return got or find_training_dataset(workspace)


def find_winner(workspace: Path | str) -> Optional[Path]:
    """The trained surrogate payload (``winner.pkl``)."""
    workspace = Path(workspace)
    got = _first_existing(
        [workspace / "winner.pkl", workspace / "outputs" / "winner.pkl"]
    )
    if got:
        return got
    sur = workspace / "surrogate_runs"
    if sur.is_dir():
        for it in sorted(sur.iterdir(), reverse=True):  # latest sub-run first
            w = it / "outputs" / "winner.pkl"
            if w.exists():
                return w
    return None


def find_report(workspace: Path | str) -> Optional[Path]:
    """The run's ``report.json`` (meta root or direct ``outputs/``)."""
    workspace = Path(workspace)
    return _first_existing(
        [workspace / "report.json", workspace / "outputs" / "report.json"]
    )


__all__ = [
    "find_eval_dataset",
    "find_report",
    "find_training_dataset",
    "find_winner",
]
