# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Canonical HDF5 read/write/merge/split helpers for Phase-1 datasets.

The dataset layout (written by ``data.sweep.run_sweep`` and the agent-authored
Phase-1 runner) is:

    grid/R, grid/Z                        : (nr,), (nz,) float64
    inputs/{r0,a,kappa,delta,Ip}          : (N,) float64
    outputs/psi                           : (N, nz, nr) float64
    outputs/success                       : (N,) bool
    outputs/isoflux_used                  : (N,) bool

This module is the single owner of that layout for row-level operations:
merging shards (meta-loop ``regen_dataset``) and carving the frozen held-out
test shard the meta-loop evaluates every winner against. Writes are atomic
(temp file in the same dir → fsync → ``os.replace``), matching
``core.io.atomic_write_text`` — a crashed merge must not leave a truncated
dataset behind.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np

from autotokamak.data.schema import PARAM_ORDER


@dataclass
class DatasetArrays:
    """In-memory image of one dataset HDF5 file (all rows, incl. failures)."""

    R: np.ndarray
    Z: np.ndarray
    inputs: Dict[str, np.ndarray]
    psi: np.ndarray
    success: np.ndarray
    isoflux_used: np.ndarray

    @property
    def n_rows(self) -> int:
        return int(self.psi.shape[0])

    def take(self, idx: np.ndarray) -> "DatasetArrays":
        """Row-subset (grid arrays are shared, row arrays are copied views)."""
        idx = np.asarray(idx, dtype=int)
        return DatasetArrays(
            R=self.R,
            Z=self.Z,
            inputs={p: self.inputs[p][idx] for p in PARAM_ORDER},
            psi=self.psi[idx],
            success=self.success[idx],
            isoflux_used=self.isoflux_used[idx],
        )


def read_h5_arrays(path: Path | str) -> DatasetArrays:
    import h5py

    with h5py.File(path, "r") as f:
        return DatasetArrays(
            R=np.asarray(f["grid/R"][...], dtype=np.float64),
            Z=np.asarray(f["grid/Z"][...], dtype=np.float64),
            inputs={
                p: np.asarray(f[f"inputs/{p}"][...], dtype=np.float64)
                for p in PARAM_ORDER
            },
            psi=np.asarray(f["outputs/psi"][...], dtype=np.float64),
            success=np.asarray(f["outputs/success"][...], dtype=bool),
            isoflux_used=np.asarray(f["outputs/isoflux_used"][...], dtype=bool),
        )


def write_h5_arrays(path: Path | str, arrays: DatasetArrays) -> None:
    """Atomic write: temp file in the destination dir → fsync → replace."""
    import h5py

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".h5.tmp")
    os.close(fd)
    try:
        with h5py.File(tmp_name, "w") as f:
            g_grid = f.create_group("grid")
            g_grid.create_dataset("R", data=arrays.R, dtype="f8")
            g_grid.create_dataset("Z", data=arrays.Z, dtype="f8")
            g_in = f.create_group("inputs")
            for p in PARAM_ORDER:
                g_in.create_dataset(p, data=arrays.inputs[p].astype(np.float64), dtype="f8")
            g_out = f.create_group("outputs")
            g_out.create_dataset("psi", data=arrays.psi.astype(np.float64), dtype="f8")
            g_out.create_dataset("success", data=arrays.success.astype(bool), dtype=np.bool_)
            g_out.create_dataset(
                "isoflux_used", data=arrays.isoflux_used.astype(bool), dtype=np.bool_
            )
        with open(tmp_name, "rb") as fh:
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


def merge_h5(old_path: Path | str, new_path: Path | str, merged_path: Path | str) -> Dict[str, int]:
    """Concatenate old + new datasets into ``merged_path``.

    Requires matching ``grid/R`` and ``grid/Z``. Only the canonical fields
    are carried over; agent-runner extras (``error_msgs``, ``solve_info``)
    are dropped.
    """
    old = read_h5_arrays(old_path)
    new = read_h5_arrays(new_path)
    if not (np.array_equal(old.R, new.R) and np.array_equal(old.Z, new.Z)):
        raise ValueError(
            "Cannot merge datasets: output_grid changed between shards "
            f"(old R:{old.R.shape}/Z:{old.Z.shape} vs new R:{new.R.shape}/Z:{new.Z.shape})"
        )
    merged = DatasetArrays(
        R=old.R,
        Z=old.Z,
        inputs={
            p: np.concatenate([old.inputs[p], new.inputs[p]]).astype(np.float64)
            for p in PARAM_ORDER
        },
        psi=np.concatenate([old.psi, new.psi], axis=0).astype(np.float64),
        success=np.concatenate([old.success, new.success]).astype(bool),
        isoflux_used=np.concatenate([old.isoflux_used, new.isoflux_used]).astype(bool),
    )
    write_h5_arrays(merged_path, merged)
    return {
        "n_total": merged.n_rows,
        "n_succeeded": int(merged.success.sum()),
        "n_isoflux_used": int(merged.isoflux_used.sum()),
    }


def split_h5(
    src_path: Path | str,
    *,
    train_path: Path | str,
    test_path: Path | str,
    test_frac: float = 0.15,
    min_test: int = 2,
    min_train_success: int = 6,
    seed: int = 0,
) -> Dict[str, Any]:
    """Carve a FROZEN held-out test shard from ``src_path``.

    The test shard is sampled from ``success == True`` rows only (a test
    sample must be evaluable); every other row — including failures — stays
    in the train pool, where ``autotokamak.surrogate.dataset.load_dataset`` filters them as
    usual. Both output files use the canonical layout, so ``load_dataset``
    round-trips on either.

    Raises ``ValueError`` when the source has too few successful samples to
    support both a ``min_test``-sample shard and a ``min_train_success``
    train pool.
    """
    src_path = Path(src_path)
    src = read_h5_arrays(src_path)
    success_idx = np.flatnonzero(src.success)
    n_success = int(success_idx.size)
    if n_success < min_test + min_train_success:
        raise ValueError(
            f"Dataset too small to freeze a test shard: need >= "
            f"{min_test + min_train_success} successful samples "
            f"(min_test={min_test} + min_train_success={min_train_success}), "
            f"got {n_success} in {src_path}"
        )
    n_test = int(np.clip(round(test_frac * n_success), min_test, n_success - min_train_success))

    rng = np.random.default_rng(int(seed))
    test_rows = np.sort(rng.choice(success_idx, size=n_test, replace=False))
    train_mask = np.ones(src.n_rows, dtype=bool)
    train_mask[test_rows] = False
    train_rows = np.flatnonzero(train_mask)

    test_arrays = src.take(test_rows)
    train_arrays = src.take(train_rows)
    write_h5_arrays(test_path, test_arrays)
    write_h5_arrays(train_path, train_arrays)

    return {
        "source_path": str(src_path),
        "n_source_rows": src.n_rows,
        "n_source_success": n_success,
        "n_test": n_test,
        "n_train_rows": int(train_rows.size),
        "n_train_success": int(train_arrays.success.sum()),
        "test_row_indices": [int(i) for i in test_rows],
        "test_frac": float(test_frac),
        "seed": int(seed),
        "train_path": str(train_path),
        "test_path": str(test_path),
    }


__all__ = [
    "DatasetArrays",
    "merge_h5",
    "read_h5_arrays",
    "split_h5",
    "write_h5_arrays",
]
