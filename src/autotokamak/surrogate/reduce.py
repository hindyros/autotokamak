"""PCA reduction for ψ(R,Z) fields.

ψ is 96×64 ≈ 6k outputs per sample. Direct multi-output regression on 6k
targets is wasteful and noisy; in practice equilibrium ψ datasets are very
low-rank (95% variance in ≲20 components). The surrogate runner fits PCA on
the TRAIN fold only, regresses in reduced space, and reconstructs at scoring
time.

Outside-LCFS pixels are NaN per the Phase-1 convention. We replace them with
the train column-mean before fitting and remember the mask so the inverse can
re-NaN those cells if a caller wants honest "no prediction outside domain"
output. The default ``inverse_transform`` just returns the dense reconstruction
because downstream RMSE is computed with the same mask treatment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PCAModel:
    """Wraps sklearn PCA with NaN-aware flatten/unflatten.

    ``mean_fill`` is the per-pixel train mean used to fill NaNs before PCA
    (same value is added back during ``inverse_transform``). ``nan_mask`` is
    True where the training data had NaN (typically constant across samples —
    outside the LCFS bounding box).
    """

    pca: object  # sklearn.decomposition.PCA
    grid_shape: tuple[int, int]
    mean_fill: np.ndarray  # shape (nz*nr,)
    nan_mask: np.ndarray  # shape (nz*nr,), bool, True where train was NaN

    @property
    def n_components(self) -> int:
        return int(self.pca.n_components_)

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        return np.asarray(self.pca.explained_variance_ratio_, dtype=np.float64)

    @property
    def total_explained_variance(self) -> float:
        return float(np.sum(self.explained_variance_ratio))


def _flatten(psi: np.ndarray) -> np.ndarray:
    if psi.ndim != 3:
        raise ValueError(f"Expected psi of shape (N, nz, nr); got {psi.shape}")
    return psi.reshape(psi.shape[0], -1)


def fit_pca(psi_train: np.ndarray, n_components: int) -> PCAModel:
    """Fit PCA on ``psi_train`` (shape (N, nz, nr)).

    NaN cells in training data are replaced with the per-pixel sample mean
    (computed using ``np.nanmean`` across the N axis). If a pixel is NaN in
    EVERY training sample, it falls back to 0 (the PCA then sees a constant
    zero column there).
    """
    if psi_train.ndim != 3:
        raise ValueError(f"Expected psi_train of shape (N, nz, nr); got {psi_train.shape}")
    nz, nr = psi_train.shape[1], psi_train.shape[2]
    flat = _flatten(psi_train).astype(np.float64, copy=True)

    # Per-pixel mean ignoring NaN; fall back to 0 where the column is all NaN.
    # np.nanmean raises a RuntimeWarning on all-NaN slices; we expect those
    # (outside-LCFS pixels) so silence them locally.
    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_fill = np.nanmean(flat, axis=0)
    mean_fill = np.where(np.isfinite(mean_fill), mean_fill, 0.0)

    nan_mask_per_pixel = np.all(np.isnan(flat), axis=0)
    # Fill: replace any NaN with the column mean.
    flat = np.where(np.isnan(flat), mean_fill[None, :], flat)

    from sklearn.decomposition import PCA

    n_components = int(min(n_components, min(flat.shape) - 1, flat.shape[0] - 1))
    n_components = max(1, n_components)
    pca = PCA(n_components=n_components, svd_solver="full")
    pca.fit(flat)

    return PCAModel(
        pca=pca,
        grid_shape=(int(nz), int(nr)),
        mean_fill=mean_fill,
        nan_mask=nan_mask_per_pixel,
    )


def transform(model: PCAModel, psi: np.ndarray) -> np.ndarray:
    """Project (N, nz, nr) → (N, n_components). NaN handled like fit."""
    flat = _flatten(psi).astype(np.float64, copy=True)
    flat = np.where(np.isnan(flat), model.mean_fill[None, :], flat)
    return np.asarray(model.pca.transform(flat), dtype=np.float64)


def inverse_transform(model: PCAModel, coeffs: np.ndarray) -> np.ndarray:
    """Project (N, n_components) → (N, nz, nr) dense reconstruction.

    Does NOT re-mask NaN cells; the caller can apply ``model.nan_mask`` if
    they want honest no-prediction-outside-domain output. Keeping the dense
    reconstruction means RMSE callers can choose the masking convention.
    """
    flat = np.asarray(model.pca.inverse_transform(coeffs), dtype=np.float64)
    nz, nr = model.grid_shape
    return flat.reshape(-1, nz, nr)


__all__ = ["PCAModel", "fit_pca", "inverse_transform", "transform"]
