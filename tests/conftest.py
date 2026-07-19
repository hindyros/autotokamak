"""Shared test fixtures/helpers for the autotokamak suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def smooth_psi(inputs: np.ndarray, nz: int = 8, nr: int = 6, seed: int = 0) -> np.ndarray:
    """Low-rank synthetic psi driven by the first two input coords.

    Two Gaussian basis fields weighted by inputs[:, 0] and inputs[:, 1] plus
    tiny noise — smooth enough for GPs/PCA to model, cheap enough for unit
    tests. Shared by ``make_synthetic_h5`` and the acquisition tests.
    """
    rng = np.random.default_rng(seed)
    R = np.linspace(0.2, 0.7, nr)
    Z = np.linspace(-0.3, 0.3, nz)
    RR, ZZ = np.meshgrid(R, Z, indexing="xy")
    basis0 = np.exp(-((RR - 0.4) ** 2 + ZZ**2) / 0.05)
    basis1 = np.exp(-((RR - 0.55) ** 2 + ZZ**2) / 0.03)
    psi = (
        inputs[:, 0][:, None, None] * basis0[None, :, :]
        + inputs[:, 1][:, None, None] * basis1[None, :, :]
    )
    return psi + rng.normal(scale=1e-4, size=psi.shape)


def make_sweep_config(**overrides):
    """A minimal valid SweepConfig over the unit box (grid matches make_synthetic_h5)."""
    from autotokamak.data.schema import PARAM_ORDER, SweepConfig

    raw = {
        "sampling": {"method": "lhs", "n_samples": 10, "seed": 0},
        "parameters": {p: {"low": 0.0, "high": 1.0} for p in PARAM_ORDER},
        "output_grid": {
            "R": {"min": 0.2, "max": 0.7, "n": 6},
            "Z": {"min": -0.3, "max": 0.3, "n": 8},
        },
    }
    raw.update(overrides)
    return SweepConfig.model_validate(raw)


def fake_run_sweep_factory(calls: dict, *, expect_X: bool = True):
    """A ``run_sweep`` stand-in: records its arguments, writes a canonical H5.

    ``expect_X=True`` asserts the caller passed explicit acquisition points
    (the enrich_active contract); ``expect_X=False`` lets the fake draw from
    ``cfg.sampling`` size (the envelope-generation contract). The written
    grid matches ``make_synthetic_h5`` so merges succeed.
    """
    from autotokamak.data.h5io import DatasetArrays, write_h5_arrays
    from autotokamak.data.schema import PARAM_ORDER, SweepResult

    def fake_run_sweep(cfg, output_dir, X=None):
        if expect_X:
            assert X is not None, "expected explicit acquisition points"
        if X is None:
            rng = np.random.default_rng(cfg.sampling.seed)
            X = rng.uniform(size=(cfg.sampling.n_samples, 5))
        X = np.asarray(X)
        calls["X"] = X
        calls["cfg"] = cfg
        n = X.shape[0]
        out_path = Path(output_dir) / cfg.output_path
        write_h5_arrays(
            out_path,
            DatasetArrays(
                R=np.linspace(0.2, 0.7, 6),
                Z=np.linspace(-0.3, 0.3, 8),
                inputs={p: X[:, j].astype(np.float64) for j, p in enumerate(PARAM_ORDER)},
                psi=np.random.default_rng(0).normal(size=(n, 8, 6)),
                success=np.ones(n, dtype=bool),
                isoflux_used=np.zeros(n, dtype=bool),
            ),
        )
        return SweepResult(
            dataset_path=str(out_path), n_requested=n, n_succeeded=n,
            n_isoflux_used=0, config_hash="fake",
        )

    return fake_run_sweep


def train_winner(X: np.ndarray, psi: np.ndarray, n_comp: int = 2) -> dict:
    """Fit a tiny poly_ridge winner payload on (X, psi) — the automl output shape."""
    from autotokamak.surrogate.reduce import fit_pca, transform
    from autotokamak.surrogate.zoo import make_model

    pca = fit_pca(psi, n_components=n_comp)
    est = make_model("poly_ridge", alpha=0.1, degree=1)
    est.fit(X, transform(pca, psi))
    return {
        "estimator": est,
        "pca": pca,
        "model_name": "poly_ridge",
        "hyperparams": {"alpha": 0.1, "degree": 1},
        "n_pca_components": n_comp,
    }


def make_picker(decisions: list[dict]):
    """An ActionPicker yielding the given decisions in order (last one repeats)."""
    from autotokamak.agent.orchestrator.schema import ActionDecision

    materialized = [ActionDecision.model_validate(d) for d in decisions]
    counter = {"i": 0}

    def picker(meta_config, state, diagnostics, history):
        d = materialized[min(counter["i"], len(materialized) - 1)]
        counter["i"] += 1
        return d

    return picker


def make_synthetic_h5(
    path: Path,
    *,
    n: int = 16,
    nz: int = 8,
    nr: int = 6,
    seed: int = 0,
    n_failures: int = 0,
) -> Path:
    """Write a tiny synthetic dataset in the canonical Phase-1 HDF5 layout.

    Low-rank psi (rank 2 + noise) with a NaN border, mirroring
    ``test_surrogate_smoke._synthetic_bundle`` but persisted to disk so
    ``h5io`` / ``load_dataset`` / ``automl_loop`` round-trips can run on it.
    The last ``n_failures`` rows are marked ``success=False`` with NaN psi.
    """
    import h5py

    rng = np.random.default_rng(seed)
    R = np.linspace(0.2, 0.7, nr)
    Z = np.linspace(-0.3, 0.3, nz)
    RR, ZZ = np.meshgrid(R, Z, indexing="xy")

    basis0 = np.exp(-((RR - 0.4) ** 2 + ZZ**2) / 0.05)
    basis1 = np.exp(-((RR - 0.55) ** 2 + ZZ**2) / 0.03)

    inputs = rng.uniform(size=(n, 5))
    psi = (
        inputs[:, 0][:, None, None] * basis0[None, :, :]
        + inputs[:, 1][:, None, None] * basis1[None, :, :]
    )
    psi += rng.normal(scale=1e-3, size=psi.shape)
    psi[:, 0, :] = np.nan
    psi[:, -1, :] = np.nan
    psi[:, :, 0] = np.nan
    psi[:, :, -1] = np.nan

    success = np.ones(n, dtype=bool)
    if n_failures:
        success[n - n_failures :] = False
        psi[n - n_failures :] = np.nan
    isoflux_used = np.zeros(n, dtype=bool)

    param_names = ("r0", "a", "kappa", "delta", "Ip")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        g_grid = f.create_group("grid")
        g_grid.create_dataset("R", data=R, dtype="f8")
        g_grid.create_dataset("Z", data=Z, dtype="f8")
        g_in = f.create_group("inputs")
        for j, p in enumerate(param_names):
            g_in.create_dataset(p, data=inputs[:, j], dtype="f8")
        g_out = f.create_group("outputs")
        g_out.create_dataset("psi", data=psi, dtype="f8")
        g_out.create_dataset("success", data=success, dtype=np.bool_)
        g_out.create_dataset("isoflux_used", data=isoflux_used, dtype=np.bool_)
    return path
