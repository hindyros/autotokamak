"""Tests for ``autotokamak.data.h5io`` — read/write/merge/split helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from autotokamak.data.h5io import merge_h5, read_h5_arrays, split_h5, write_h5_arrays
from autotokamak.surrogate.dataset import load_dataset

from tests.conftest import make_synthetic_h5


def _input_rows(arrays) -> set[tuple]:
    """Hashable per-row input tuples for disjointness checks."""
    cols = [arrays.inputs[p] for p in ("r0", "a", "kappa", "delta", "Ip")]
    return {tuple(round(float(c[i]), 12) for c in cols) for i in range(arrays.n_rows)}


def test_write_read_roundtrip_and_load_dataset(tmp_path: Path):
    src = make_synthetic_h5(tmp_path / "src.h5", n=16, n_failures=3)
    arrays = read_h5_arrays(src)
    copy = tmp_path / "copy.h5"
    write_h5_arrays(copy, arrays)

    back = read_h5_arrays(copy)
    assert np.array_equal(back.R, arrays.R)
    assert np.array_equal(back.success, arrays.success)
    assert np.allclose(back.psi, arrays.psi, equal_nan=True)

    # load_dataset works on files written by write_h5_arrays and
    # success-filters as usual.
    bundle = load_dataset(copy)
    assert bundle.n_samples == 13  # 16 - 3 failures


def test_split_h5_disjoint_and_shard_all_successful(tmp_path: Path):
    src = make_synthetic_h5(tmp_path / "src.h5", n=20, n_failures=4)
    train_p, test_p = tmp_path / "train.h5", tmp_path / "test.h5"
    info = split_h5(src, train_path=train_p, test_path=test_p, test_frac=0.15, seed=0)

    train = read_h5_arrays(train_p)
    test = read_h5_arrays(test_p)

    # Shard rows are all successful; failures stay in the train pool.
    assert test.success.all()
    assert int((~train.success).sum()) == 4

    # Row-level disjointness and completeness.
    assert _input_rows(train).isdisjoint(_input_rows(test))
    assert train.n_rows + test.n_rows == 20

    # Grid preserved on both sides.
    src_arrays = read_h5_arrays(src)
    for part in (train, test):
        assert np.array_equal(part.R, src_arrays.R)
        assert np.array_equal(part.Z, src_arrays.Z)

    # Provenance counts consistent.
    assert info["n_test"] == test.n_rows
    assert info["n_train_success"] == int(train.success.sum())


def test_split_h5_min_test_floor(tmp_path: Path):
    # 16 successful samples at test_frac=0.15 rounds to 2 -> min_test floor holds.
    src = make_synthetic_h5(tmp_path / "src.h5", n=16)
    info = split_h5(
        src,
        train_path=tmp_path / "train.h5",
        test_path=tmp_path / "test.h5",
        test_frac=0.05,  # rounds to 1; floor lifts it to 2
        min_test=2,
        seed=0,
    )
    assert info["n_test"] == 2


def test_split_h5_too_small_raises(tmp_path: Path):
    src = make_synthetic_h5(tmp_path / "src.h5", n=7)
    with pytest.raises(ValueError, match="too small"):
        split_h5(src, train_path=tmp_path / "t.h5", test_path=tmp_path / "s.h5")


def test_split_h5_deterministic(tmp_path: Path):
    src = make_synthetic_h5(tmp_path / "src.h5", n=20)
    info_a = split_h5(
        src, train_path=tmp_path / "a_tr.h5", test_path=tmp_path / "a_te.h5", seed=3
    )
    info_b = split_h5(
        src, train_path=tmp_path / "b_tr.h5", test_path=tmp_path / "b_te.h5", seed=3
    )
    assert info_a["test_row_indices"] == info_b["test_row_indices"]


def test_merge_h5_counts_and_grid_mismatch(tmp_path: Path):
    a = make_synthetic_h5(tmp_path / "a.h5", n=10, n_failures=1, seed=0)
    b = make_synthetic_h5(tmp_path / "b.h5", n=6, n_failures=2, seed=1)
    counts = merge_h5(a, b, tmp_path / "merged.h5")
    assert counts["n_total"] == 16
    assert counts["n_succeeded"] == 13

    # Different grid shape -> ValueError.
    c = make_synthetic_h5(tmp_path / "c.h5", n=6, nz=10, nr=5)
    with pytest.raises(ValueError, match="grid"):
        merge_h5(a, c, tmp_path / "bad.h5")


def test_kfold_zero_test_frac():
    from autotokamak.surrogate.dataset import kfold

    bundle = load_dataset_from_synth()
    splits = kfold(bundle, k=4, test_frac=0.0, seed=0)
    assert splits.test_idx.size == 0
    # Folds partition ALL samples.
    all_val = np.concatenate([va for _, va in splits.folds])
    assert np.array_equal(np.sort(all_val), np.arange(bundle.n_samples))


def load_dataset_from_synth():
    """Small in-memory bundle helper (module-local, not a fixture)."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = make_synthetic_h5(Path(d) / "ds.h5", n=16)
        return load_dataset(p)
