"""Regression tests for autotokamak.pipelines.discover.

These lock the behavior that broke the HTML report's plots: the meta-loop
workspace layout (winner.pkl at root, datasets/ subdir, grown iterN pools) must
be discoverable, not just the direct phase-2 layout (everything under outputs/).
"""

from __future__ import annotations

import json

from autotokamak.pipelines.discover import (
    find_eval_dataset,
    find_report,
    find_training_dataset,
    find_winner,
)


def _touch(p):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x")
    return p


def test_meta_layout_discovery(tmp_path):
    ws = tmp_path / "meta_ws"
    _touch(ws / "winner.pkl")
    _touch(ws / "report.json")
    _touch(ws / "datasets" / "train_pool.h5")
    _touch(ws / "datasets" / "test_shard.h5")
    _touch(ws / "datasets" / "iter1_dataset.h5")
    _touch(ws / "datasets" / "iter12_dataset.h5")  # highest iter — should win

    assert find_winner(ws) == ws / "winner.pkl"
    assert find_report(ws) == ws / "report.json"
    # Training viz should use the largest grown pool, not the seed train_pool.
    assert find_training_dataset(ws) == ws / "datasets" / "iter12_dataset.h5"
    # Eval should be the frozen held-out shard.
    assert find_eval_dataset(ws) == ws / "datasets" / "test_shard.h5"


def test_split_info_overrides_eval_path(tmp_path):
    ws = tmp_path / "meta_env"
    _touch(ws / "datasets" / "train_pool.h5")
    envelope = _touch(ws / "datasets" / "eval_envelope.h5")
    (ws / "datasets" / "split_info.json").write_text(
        json.dumps({"mode": "envelope", "eval_envelope_path": str(envelope)})
    )
    assert find_eval_dataset(ws) == envelope


def test_direct_phase_layout_discovery(tmp_path):
    ws = tmp_path / "phase2_ws"
    _touch(ws / "outputs" / "winner.pkl")
    _touch(ws / "outputs" / "report.json")
    _touch(ws / "outputs" / "dataset.h5")

    assert find_winner(ws) == ws / "outputs" / "winner.pkl"
    assert find_report(ws) == ws / "outputs" / "report.json"
    assert find_training_dataset(ws) == ws / "outputs" / "dataset.h5"
    # No frozen shard on disk → eval falls back to the training dataset.
    assert find_eval_dataset(ws) == ws / "outputs" / "dataset.h5"


def test_missing_artifacts_return_none(tmp_path):
    ws = tmp_path / "empty"
    ws.mkdir()
    assert find_winner(ws) is None
    assert find_report(ws) is None
    assert find_training_dataset(ws) is None
    # eval falls back to training discovery, which is also None here.
    assert find_eval_dataset(ws) is None
