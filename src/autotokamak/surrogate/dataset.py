"""Dataset loading and splitting for surrogate training.

The Phase-1 prompt produces ``examples/dataset_generation/outputs/dataset.h5``
with:

* ``grid/R``  (nr,) float64
* ``grid/Z``  (nz,) float64
* ``inputs/{r0, a, kappa, delta, Ip}``  (N,) float64
* ``outputs/psi``  (N, nz, nr) float64
* ``outputs/success``  (N,) bool

This module is the single home for opening that file, filtering failed solves,
and producing reproducible CV folds. The surrogate runner the agent writes
imports from here so it never has to reinvent the split logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


PARAM_ORDER = ("r0", "a", "kappa", "delta", "Ip")


@dataclass
class DatasetBundle:
    """All of ``dataset.h5`` loaded into memory, success-filtered.

    ``inputs`` is shape ``(N, 5)`` with columns in ``PARAM_ORDER``.
    ``psi`` is shape ``(N, nz, nr)``; values outside the LCFS interpolation
    domain are NaN (Phase-1 convention; the agent's runner uses
    ``LinearTriInterpolator`` with NaN-fill).
    """

    inputs: np.ndarray
    psi: np.ndarray
    R: np.ndarray
    Z: np.ndarray
    source_path: str

    @property
    def n_samples(self) -> int:
        return int(self.inputs.shape[0])

    @property
    def grid_shape(self) -> tuple[int, int]:
        return int(self.psi.shape[1]), int(self.psi.shape[2])


@dataclass
class Splits:
    """k-fold CV splits with a held-out test set.

    ``test_idx`` and ``folds`` index into the source ``DatasetBundle`` (which
    has already been success-filtered). Each fold is ``(train_idx, val_idx)``.
    ``folds`` partitions the non-test indices; the union of all val sets
    equals the non-test indices, with no overlap.
    """

    test_idx: np.ndarray
    folds: list[tuple[np.ndarray, np.ndarray]]
    seed: int

    def iter_folds(self) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
        for k, (tr, va) in enumerate(self.folds):
            yield k, tr, va


def load_dataset(h5_path: str | Path) -> DatasetBundle:
    """Open ``dataset.h5`` and return a success-filtered ``DatasetBundle``.

    Drops any row where ``outputs/success == False`` so downstream code never
    has to special-case failed solves. Raises ``FileNotFoundError`` if the
    file is missing; lets h5py raise on schema mismatches (we want loud
    failure if the schema drifts from the Phase-1 contract).
    """
    import h5py

    p = Path(h5_path)
    if not p.is_file():
        raise FileNotFoundError(f"Dataset HDF5 not found: {p}")

    with h5py.File(p, "r") as f:
        R = np.asarray(f["grid/R"][...], dtype=np.float64)
        Z = np.asarray(f["grid/Z"][...], dtype=np.float64)
        success = np.asarray(f["outputs/success"][...], dtype=bool)
        psi_all = np.asarray(f["outputs/psi"][...], dtype=np.float64)
        cols = []
        for name in PARAM_ORDER:
            cols.append(np.asarray(f[f"inputs/{name}"][...], dtype=np.float64))
        inputs_all = np.column_stack(cols)

    keep = np.where(success)[0]
    if keep.size == 0:
        raise ValueError(f"Dataset {p} has no successful samples; cannot train surrogates.")

    return DatasetBundle(
        inputs=inputs_all[keep],
        psi=psi_all[keep],
        R=R,
        Z=Z,
        source_path=str(p),
    )


def kfold(
    bundle: DatasetBundle,
    *,
    k: int = 4,
    test_frac: float = 2 / 16,
    seed: int = 0,
) -> Splits:
    """Shuffle, hold out ``test_frac`` of samples, k-fold the rest.

    For N=16, the default ``k=4, test_frac=2/16`` yields a 2-sample held-out
    test set and 4 folds of (~10 train, ~4 val) each. Same seed -> same
    partitioning, so Optuna trials and the final score share the splits.

    The test set is intended to be evaluated EXACTLY ONCE, by the scorer,
    after the agent picks a winner. The folds are what the inner Optuna
    objective averages over.

    ``test_frac=0.0`` yields an empty ``test_idx`` — used when an EXTERNAL
    frozen test shard exists (meta-loop) so no samples are wasted on a
    second internal holdout.
    """
    n = bundle.n_samples
    n_test = 0 if test_frac == 0.0 else max(1, int(round(test_frac * n)))
    if n < k + n_test:
        raise ValueError(
            f"Need at least k+n_test={k + n_test} samples for k={k}-fold "
            f"+ {n_test} test sample(s); got {n}"
        )

    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n)
    test_idx = np.sort(perm[:n_test])
    rest = perm[n_test:]

    # Split `rest` into k roughly-equal contiguous chunks (already shuffled).
    fold_bounds = np.linspace(0, rest.size, k + 1, dtype=int)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for j in range(k):
        val = np.sort(rest[fold_bounds[j] : fold_bounds[j + 1]])
        train = np.sort(np.setdiff1d(rest, val, assume_unique=True))
        if train.size == 0 or val.size == 0:
            raise ValueError(
                f"k={k} too large for N={n} after holding out {n_test}: fold {j} empty."
            )
        folds.append((train, val))

    return Splits(test_idx=test_idx, folds=folds, seed=int(seed))


__all__ = ["DatasetBundle", "PARAM_ORDER", "Splits", "kfold", "load_dataset"]
