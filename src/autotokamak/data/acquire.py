# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Active-learning acquisition: pick the next OFT solve points on purpose.

Phase-1 sweeps sample the (r0, a, kappa, delta, Ip) space blindly (LHS /
uniform). This module replaces "blind" with "targeted". Two acquisition
strategies, in preference order:

**residual_ucb** (``select_acquisition_points_residual``) — the "based on
observed model performance" path. Requires a trained Phase-2 winner. Computes
the winner's OUT-OF-FOLD residuals on the train pool (refit per CV fold —
train-set residuals of interpolators like GP/KRR are ~0 and carry no signal),
fits an ARD-RBF GP error model ``g: x -> log|residual|``, and scores a Sobol
candidate pool with the UCB ``mu_g + beta*sigma_g + log P(feasible)``:
exploit measured weakness, explore where the error model has no data (which
automatically covers a wider-than-seed target envelope). Batch selection
conditions the GP variance exactly per pick (kriging believer — the posterior
MEAN is unchanged by a hallucinated observation at its own mean, so only
sigma shrinks near picked points, spreading the batch).

**gp_variance** (``select_acquisition_points``) — model-agnostic fallback
used before any winner exists, or when the residual path fails:

Method (each piece is standard; the composition is fitted to this repo):

1. **PCA-GP emulator** (Higdon et al. 2008, JASA — high-dimensional-output
   computer-model emulation): project ψ(R,Z) onto its leading PCA
   components (reusing ``surrogate.reduce.fit_pca``) and fit one Gaussian
   Process with an ARD RBF kernel per retained coefficient. This is an
   *acquisition* model, deliberately independent of the Phase-2 winner —
   the winner may be a kernel-ridge/MLP with no uncertainty at all.

2. **Variance acquisition** (ALM — MacKay 1992, Neural Computation):
   score(x) = total predictive variance of the reconstructed ψ field.
   Because the PCA basis is orthonormal, the pixel-summed field variance
   collapses EXACTLY to a weighted sum of per-component GP variances:
   ``sum_pixels Var[psi_hat] = sum_j s_j^2 * sigma_j^2(x)`` where ``s_j``
   is the train-set std of coefficient j (GPs are fit on standardized
   coefficients). No sampling, no approximation.

3. **Exact greedy batch** ("kriging believer", Ginsbourger et al. 2010):
   GP posterior variance depends only on input LOCATIONS, never on y —
   so conditioning on a selected point with a hallucinated y is exact for
   a variance acquisition. After each pick we append the point to every
   component GP via an incremental Cholesky border update (O(n·m) per
   pick) and re-score the remaining pool: the batch spreads out instead
   of piling onto one variance peak.

4. **Feasibility weighting** (probability-of-feasibility, Gardner et al.
   2014, ICML — constrained BO): OFT solves FAIL in parts of the shape
   space, and a failed solve costs the same wall-clock as a success. A
   k-NN estimate of P(success) over ALL previously attempted inputs
   (successes and failures) multiplies the acquisition, with an
   exploration floor so an unlucky region is down-weighted, never
   written off. The floor is deliberately harsh (default 0.02): with a
   soft floor, once the greedy batch has covered the feasible region the
   prior-level variance inside an all-failure region starts winning the
   argmax and the batch drifts into known-infeasible space (observed in
   testing at floor=0.1 — half the batch landed in a 0/20-success slab).

Fallback: with too few successful solves to fit a trustworthy GP (or on
any numerical failure), degrade to greedy maximin space-filling in
standardized input space — still feasibility-weighted, still better than
another blind LHS batch — and say so in the result's ``method`` field.

Everything is seeded; two calls with identical inputs select identical
points.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

from autotokamak.data.schema import PARAM_ORDER


# Successful solves needed before the GP path is trusted; below this we
# fall back to maximin space-filling (a GP on 5-D inputs with fewer points
# than ~1.5x its ARD dimensions is hyperparameter noise, not signal).
MIN_SUCCESS_FOR_GP = 8


@dataclass
class AcquisitionResult:
    """Selected batch + full provenance for the meta-trace / LLM."""

    X_new: np.ndarray  # (n_points, 5), columns in PARAM_ORDER
    method: str  # "residual_ucb" | "gp_variance" | "maximin_fallback"
    scores: np.ndarray  # acquisition score of each point AT SELECTION TIME
    feasibility: np.ndarray  # P(success) estimate per selected point
    n_pca_components: int
    pca_explained_variance: float
    kernel_summaries: List[str] = field(default_factory=list)
    n_candidates: int = 0
    seed: int = 0
    notes: str = ""
    beta: float = 0.0  # UCB exploration weight (residual_ucb only)
    residual_summary: dict = field(default_factory=dict)  # OOF-residual stats

    def summary_dict(self) -> dict:
        """JSON-serializable summary (goes into the action's result.json)."""
        return {
            "method": self.method,
            "n_selected": int(self.X_new.shape[0]),
            "n_candidates": int(self.n_candidates),
            "n_pca_components": int(self.n_pca_components),
            "pca_explained_variance": float(self.pca_explained_variance),
            "score_first": float(self.scores[0]) if self.scores.size else None,
            "score_last": float(self.scores[-1]) if self.scores.size else None,
            "mean_feasibility": float(self.feasibility.mean()) if self.feasibility.size else None,
            "kernel_summaries": list(self.kernel_summaries),
            "seed": int(self.seed),
            "notes": self.notes,
            "beta": float(self.beta),
            "residual_summary": dict(self.residual_summary),
        }


# --------------------------- internals ----------------------------------


def _ard_rbf(d: int, *, noise_level: Optional[float] = None, noise_bounds=(1e-8, 1e0)):
    """The module's standard ARD-RBF kernel; optional additive WhiteKernel.

    One factory for the three GP fits in this module (feasibility classifier,
    per-PCA-component regressors, residual error model) so their kernel
    bounds cannot drift apart.
    """
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

    kern = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
        length_scale=np.ones(d), length_scale_bounds=(1e-2, 1e2)
    )
    if noise_level is not None:
        kern = kern + WhiteKernel(noise_level=noise_level, noise_level_bounds=noise_bounds)
    return kern


@dataclass
class _Pool:
    """Shared acquisition prelude: candidate pool + standardization + feasibility.

    Built once by ``_prepare_pool`` and consumed by both acquisition paths
    (proxy GP-variance and residual-UCB) so the Sobol draw, box
    standardization, and feasibility weighting are identical across them.
    """

    cand: np.ndarray  # (n_candidates, 5) raw coordinates
    cand_std: np.ndarray  # box-standardized candidates
    attempted_std: np.ndarray  # box-standardized attempted inputs
    feas: np.ndarray  # feasibility weight per candidate
    n_candidates: int


def _prepare_pool(
    attempted_inputs: np.ndarray,
    success: np.ndarray,
    lows: np.ndarray,
    highs: np.ndarray,
    *,
    n_points: int,
    n_candidates: int,
    seed: int,
    feasibility_floor: float,
) -> _Pool:
    from autotokamak.data.sweep import sobol_sample

    n_candidates = int(max(n_candidates, 4 * n_points))
    # Standardize inputs with box-derived scale — stable even when the
    # attempted set is tiny or degenerate, and identical for train/cands.
    span = np.maximum(highs - lows, 1e-12)
    cand = sobol_sample(lows, highs, n_candidates, seed=seed)
    cand_std = (cand - lows[None, :]) / span[None, :]
    attempted_std = (attempted_inputs - lows[None, :]) / span[None, :]
    feas = _feasibility_prob(
        attempted_std, success, cand_std, floor=feasibility_floor, seed=seed
    )
    return _Pool(cand, cand_std, attempted_std, feas, n_candidates)


# Cap on classifier training size: GPC's Laplace approximation is O(n^3);
# beyond this we subsample (class-stratified, seeded).
_FEAS_MAX_TRAIN = 800


def _feasibility_prob(
    attempted_std: np.ndarray,
    success: np.ndarray,
    query_std: np.ndarray,
    *,
    floor: float,
    seed: int = 0,
) -> np.ndarray:
    """Feasibility weight P(success)^2 at ``query_std``; ones when no failure.

    P comes from an ARD-RBF Gaussian-Process classifier (the standard
    probability-of-feasibility model in constrained BO — Gardner et al.
    2014); ARD matters because solve failures typically align with a few
    input directions (extreme shaping), and an isotropic-distance model
    (k-NN) lets the irrelevant dimensions dilute exactly that signal —
    measured on a synthetic failure slab, k-NN(k=10) assigned P≈0.5 deep
    inside a 0%-success region. The probability is SQUARED so that a
    known-bad region cannot out-bid a merely well-covered feasible one on
    prior variance alone, then floored so it is never written off entirely.
    Falls back to distance-weighted k-NN if the GPC fit fails.
    """
    n_query = query_std.shape[0]
    if bool(success.all()):
        return np.ones(n_query, dtype=np.float64)
    if not bool(success.any()):
        # Nothing ever succeeded; no gradient to learn — weight uniformly.
        return np.full(n_query, float(floor), dtype=np.float64)

    X_fit, y_fit = attempted_std, success.astype(int)
    if X_fit.shape[0] > _FEAS_MAX_TRAIN:
        rng = np.random.default_rng(seed)
        keep_parts = []
        for cls in (0, 1):
            cls_idx = np.flatnonzero(y_fit == cls)
            quota = max(1, int(round(_FEAS_MAX_TRAIN * cls_idx.size / X_fit.shape[0])))
            keep_parts.append(rng.choice(cls_idx, size=min(quota, cls_idx.size), replace=False))
        keep = np.concatenate(keep_parts)
        X_fit, y_fit = X_fit[keep], y_fit[keep]

    try:
        from sklearn.gaussian_process import GaussianProcessClassifier

        kern = _ard_rbf(X_fit.shape[1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gpc = GaussianProcessClassifier(kernel=kern, random_state=seed).fit(X_fit, y_fit)
            p = gpc.predict_proba(query_std)[:, list(gpc.classes_).index(1)]
    except Exception:  # noqa: BLE001 — degrade, don't die
        from sklearn.neighbors import NearestNeighbors

        k = int(min(5, X_fit.shape[0]))
        nn = NearestNeighbors(n_neighbors=k).fit(X_fit)
        dist, idx = nn.kneighbors(query_std)
        w = 1.0 / np.maximum(dist, 1e-9)
        p = (y_fit[idx] * w).sum(axis=1) / w.sum(axis=1)

    return np.maximum(p**2, float(floor))


class _GPBatch:
    """Greedy exact-variance batch selection over a shared candidate pool.

    Holds one frozen-kernel GP per PCA component (all sharing training and
    candidate LOCATIONS) and their per-component Cholesky state. Total
    field variance = sum_j weight_j * var_j.
    """

    def __init__(
        self,
        kernels: List,
        weights: np.ndarray,
        X_train: np.ndarray,
        X_cand: np.ndarray,
        jitter: float = 1e-8,
    ):
        from scipy.linalg import solve_triangular

        self._st = solve_triangular
        self.X_cand = X_cand
        self.weights = np.asarray(weights, dtype=np.float64)
        self.jitter = float(jitter)
        self.state = []
        for kern in kernels:
            K = kern(X_train) + self.jitter * np.eye(X_train.shape[0])
            L = np.linalg.cholesky(K)
            V = solve_triangular(L, kern(X_train, X_cand), lower=True)
            self.state.append(
                {"kernel": kern, "L": L, "V": V, "prior": kern.diag(X_cand)}
            )
        self.alive = np.ones(X_cand.shape[0], dtype=bool)

    def total_variance(self) -> np.ndarray:
        """Weighted field variance for every candidate (dead ones = -inf)."""
        tot = np.zeros(self.X_cand.shape[0], dtype=np.float64)
        for w, s in zip(self.weights, self.state):
            var = s["prior"] - np.einsum("ij,ij->j", s["V"], s["V"])
            tot += w * np.maximum(var, 0.0)
        tot[~self.alive] = -np.inf
        return tot

    def consume(self, idx: int) -> None:
        """Move candidate ``idx`` into every component's training set."""
        x_new = self.X_cand[idx]
        self.alive[idx] = False
        for s in self.state:
            kern, L, V = s["kernel"], s["L"], s["V"]
            b = V[:, idx]  # L^{-1} k(train, x_new)
            d2 = float(kern.diag(x_new[None, :])[0]) + self.jitter - float(b @ b)
            d = np.sqrt(max(d2, 1e-12))
            # Border-update the Cholesky factor: L' = [[L, 0], [b^T, d]]
            n = L.shape[0]
            L_new = np.zeros((n + 1, n + 1), dtype=np.float64)
            L_new[:n, :n] = L
            L_new[n, :n] = b
            L_new[n, n] = d
            # New V row: (k(x_new, cand) - b^T V) / d
            k_row = kern(x_new[None, :], self.X_cand)[0]
            v_row = (k_row - b @ V) / d
            s["L"] = L_new
            s["V"] = np.vstack([V, v_row[None, :]])


def _fit_component_kernels(
    X_train_std: np.ndarray,
    coeffs_std: np.ndarray,
    seed: int,
) -> List:
    """One ARD-RBF GP fit per standardized PCA coefficient; returns kernels.

    Hyperparameters are optimized ONCE here (marginal likelihood, one
    restart) and then FROZEN for the batch-selection loop — re-optimizing
    per greedy step is both expensive and unstable at these sample sizes.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor

    d = X_train_std.shape[1]
    kernels = []
    for j in range(coeffs_std.shape[1]):
        kern = _ard_rbf(d, noise_level=1e-4)
        gpr = GaussianProcessRegressor(
            kernel=kern,
            alpha=1e-10,
            n_restarts_optimizer=1,
            random_state=seed + j,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # ConvergenceWarning on lbfgs bound hits
            gpr.fit(X_train_std, coeffs_std[:, j])
        kernels.append(gpr.kernel_)
    return kernels


def _maximin_select(
    X_ref_std: np.ndarray,
    cand_std: np.ndarray,
    feas: np.ndarray,
    n_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy feasibility-weighted maximin: fill the emptiest regions first.

    Returns (selected candidate indices, score at selection time).
    """
    ref = X_ref_std.copy()
    chosen: list[int] = []
    scores: list[float] = []
    alive = np.ones(cand_std.shape[0], dtype=bool)
    # min distance from each candidate to any reference point, maintained
    # incrementally.
    if ref.shape[0]:
        d2 = ((cand_std[:, None, :] - ref[None, :, :]) ** 2).sum(axis=2)
        min_d = np.sqrt(d2.min(axis=1))
    else:
        min_d = np.full(cand_std.shape[0], np.inf)
    for _ in range(n_points):
        score = np.where(alive, min_d * feas, -np.inf)
        idx = int(np.argmax(score))
        chosen.append(idx)
        scores.append(float(score[idx]))
        alive[idx] = False
        d_new = np.sqrt(((cand_std - cand_std[idx]) ** 2).sum(axis=1))
        min_d = np.minimum(min_d, d_new)
    return np.asarray(chosen, dtype=int), np.asarray(scores, dtype=np.float64)


# --------------------------- public API ----------------------------------


def select_acquisition_points(
    *,
    attempted_inputs: np.ndarray,
    success: np.ndarray,
    psi: np.ndarray,
    bounds_low: np.ndarray,
    bounds_high: np.ndarray,
    n_points: int,
    n_candidates: int = 4096,
    seed: int = 0,
    max_pca_components: int = 8,
    target_explained_variance: float = 0.995,
    feasibility_floor: float = 0.02,
    force_maximin: bool = False,
) -> AcquisitionResult:
    """Pick the ``n_points`` most informative next inputs to solve with OFT.

    ``force_maximin=True`` skips the GP-variance path entirely and returns a
    pure feasibility-weighted maximin space-filling batch — the explicit
    "just spread out, ignore any error model" strategy the agent can request.

    Parameters
    ----------
    attempted_inputs : (M, 5) float array, columns in ``PARAM_ORDER``
        EVERY input ever handed to the solver — successes AND failures.
        Failures carry the feasibility signal.
    success : (M,) bool
        Which of those solves succeeded.
    psi : (M, nz, nr) float array
        The ψ fields, aligned with ``attempted_inputs`` (failed rows may be
        all-NaN; only successful rows are used for the GP).
    bounds_low, bounds_high : (5,) float arrays
        Sampling box, in ``PARAM_ORDER``.
    n_points : int
        Batch size to select.

    Returns
    -------
    AcquisitionResult
        Selected inputs plus provenance (method, scores, feasibility,
        kernel summaries, human-readable notes).
    """
    attempted_inputs = np.asarray(attempted_inputs, dtype=np.float64)
    success = np.asarray(success, dtype=bool)
    lows = np.asarray(bounds_low, dtype=np.float64)
    highs = np.asarray(bounds_high, dtype=np.float64)
    if attempted_inputs.ndim != 2 or attempted_inputs.shape[1] != len(PARAM_ORDER):
        raise ValueError(
            f"attempted_inputs must be (M, {len(PARAM_ORDER)}); got {attempted_inputs.shape}"
        )
    if n_points < 1:
        raise ValueError(f"n_points must be >= 1; got {n_points}")

    pool = _prepare_pool(
        attempted_inputs, success, lows, highs,
        n_points=n_points, n_candidates=n_candidates, seed=seed,
        feasibility_floor=feasibility_floor,
    )
    cand, cand_std, feas = pool.cand, pool.cand_std, pool.feas
    attempted_std, n_candidates = pool.attempted_std, pool.n_candidates

    n_success = int(success.sum())
    X_succ_std = attempted_std[success]
    psi_succ = np.asarray(psi, dtype=np.float64)[success]

    # ---------------- GP-variance path ----------------
    if force_maximin:
        fallback_reason = "space_filling strategy requested (maximin, no error model)"
    elif n_success >= MIN_SUCCESS_FOR_GP:
        try:
            from autotokamak.surrogate.reduce import fit_pca, transform

            n_comp = int(min(max_pca_components, n_success - 1))
            pca = fit_pca(psi_succ, n_components=n_comp)
            evr = pca.explained_variance_ratio
            # Keep the smallest leading set reaching the EV target (>= 2).
            cum = np.cumsum(evr)
            keep = int(np.searchsorted(cum, target_explained_variance) + 1)
            keep = int(np.clip(keep, min(2, evr.size), evr.size))

            coeffs = transform(pca, psi_succ)[:, :keep]
            c_mean = coeffs.mean(axis=0)
            c_std = np.maximum(coeffs.std(axis=0), 1e-12)
            coeffs_std = (coeffs - c_mean[None, :]) / c_std[None, :]

            kernels = _fit_component_kernels(X_succ_std, coeffs_std, seed)
            # Orthonormal PCA basis => field variance = sum_j s_j^2 sigma_j^2.
            batch = _GPBatch(
                kernels, weights=c_std**2, X_train=X_succ_std, X_cand=cand_std
            )

            chosen: list[int] = []
            scores: list[float] = []
            for _ in range(n_points):
                acq = batch.total_variance() * feas
                idx = int(np.argmax(acq))
                if not np.isfinite(acq[idx]):
                    break  # pool exhausted
                chosen.append(idx)
                scores.append(float(acq[idx]))
                batch.consume(idx)

            sel = np.asarray(chosen, dtype=int)
            ev_kept = float(cum[keep - 1])
            notes = (
                f"GP-variance acquisition: {len(sel)} points from a "
                f"{n_candidates}-point Sobol pool; {keep} PCA components "
                f"({ev_kept:.4f} explained variance), per-component ARD-RBF GPs "
                f"fit on {n_success} successful solves; greedy batch with exact "
                f"posterior-variance updates; feasibility weighting "
                f"{'ACTIVE' if not success.all() else 'inactive (no failures)'}"
                f" (floor={feasibility_floor})."
            )
            return AcquisitionResult(
                X_new=cand[sel],
                method="gp_variance",
                scores=np.asarray(scores, dtype=np.float64),
                feasibility=feas[sel],
                n_pca_components=keep,
                pca_explained_variance=ev_kept,
                kernel_summaries=[str(k) for k in kernels],
                n_candidates=n_candidates,
                seed=int(seed),
                notes=notes,
            )
        except Exception as exc:  # noqa: BLE001
            fallback_reason = f"GP path failed ({type(exc).__name__}: {exc})"
    else:
        fallback_reason = (
            f"only {n_success} successful solves (< {MIN_SUCCESS_FOR_GP} needed for GP)"
        )

    # ---------------- maximin fallback ----------------
    sel, scores = _maximin_select(attempted_std, cand_std, feas, n_points)
    notes = (
        f"Maximin space-filling fallback ({fallback_reason}): {len(sel)} points "
        f"maximizing feasibility-weighted distance to all previously attempted "
        f"inputs, from a {n_candidates}-point Sobol pool."
    )
    return AcquisitionResult(
        X_new=cand[sel],
        method="maximin_fallback",
        scores=scores,
        feasibility=feas[sel],
        n_pca_components=0,
        pca_explained_variance=0.0,
        kernel_summaries=[],
        n_candidates=n_candidates,
        seed=int(seed),
        notes=notes,
    )


def _oof_residuals(
    winner_payload: dict,
    X_succ: np.ndarray,
    psi_succ: np.ndarray,
    *,
    k_folds: int,
    seed: int,
) -> np.ndarray:
    """Per-sample OUT-OF-FOLD mean-|ψ error| of the winner's architecture.

    Refits the winner's model + hyperparameters + PCA size per fold (train
    rows only) and records each sample's mean absolute residual when it sat
    in the val fold. This is the honest "observed model performance" signal:
    train-set residuals of GP/KRR interpolators are ~0 by construction.
    """
    from autotokamak.surrogate.reduce import fit_pca, inverse_transform, transform
    from autotokamak.surrogate.zoo import make_model

    n = X_succ.shape[0]
    n_comp = int(
        winner_payload.get("n_pca_components") or winner_payload["pca"].n_components
    )
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(n), k_folds)
    resid = np.full(n, np.nan, dtype=np.float64)
    for val_idx in folds:
        if val_idx.size == 0:
            continue
        tr_mask = np.ones(n, dtype=bool)
        tr_mask[val_idx] = False
        tr_idx = np.flatnonzero(tr_mask)
        pca = fit_pca(psi_succ[tr_idx], n_components=min(n_comp, tr_idx.size - 1))
        est = make_model(winner_payload["model_name"], **winner_payload["hyperparams"])
        est.fit(X_succ[tr_idx], transform(pca, psi_succ[tr_idx]))
        Y = est.predict(X_succ[val_idx])
        if Y.ndim == 1:
            Y = Y.reshape(len(val_idx), -1)
        pred = inverse_transform(pca, Y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            resid[val_idx] = np.nanmean(np.abs(psi_succ[val_idx] - pred), axis=(1, 2))
    return resid


def select_acquisition_points_residual(
    *,
    winner_payload: dict,
    attempted_inputs: np.ndarray,
    success: np.ndarray,
    psi: np.ndarray,
    bounds_low: np.ndarray,
    bounds_high: np.ndarray,
    n_points: int,
    beta: float = 1.0,
    k_folds: int = 4,
    n_candidates: int = 4096,
    seed: int = 0,
    feasibility_floor: float = 0.02,
) -> AcquisitionResult:
    """Residual-driven UCB acquisition keyed to the CURRENT winner's weakness.

    Score (log space, so the feasibility weight composes additively):
    ``mu_g(x) + beta * sigma_g(x) + log P(feasible)`` where ``g`` is an
    ARD-RBF GP fit on ``log`` OOF residuals of the winner. ``beta=0`` is pure
    exploitation (batch may cluster on one weak pocket); larger beta trades
    toward exploration of regions the error model has never seen — which is
    what covers a target envelope wider than the seed box.

    Falls back to ``select_acquisition_points`` (proxy GP-variance, which
    itself can fall back to maximin) on too little data or any numerical
    failure, recording why in ``notes``.
    """
    attempted_inputs = np.asarray(attempted_inputs, dtype=np.float64)
    success = np.asarray(success, dtype=bool)
    lows = np.asarray(bounds_low, dtype=np.float64)
    highs = np.asarray(bounds_high, dtype=np.float64)
    n_success = int(success.sum())
    beta = float(np.clip(beta, 0.0, 3.0))

    min_needed = max(MIN_SUCCESS_FOR_GP, 2 * k_folds)
    fallback_reason: Optional[str] = None
    if winner_payload is None:
        fallback_reason = "no winner payload"
    elif n_success < min_needed:
        fallback_reason = (
            f"only {n_success} successful solves (< {min_needed} needed for OOF residuals)"
        )

    if fallback_reason is None:
        try:
            pool = _prepare_pool(
                attempted_inputs, success, lows, highs,
                n_points=n_points, n_candidates=n_candidates, seed=seed,
                feasibility_floor=feasibility_floor,
            )
            cand, cand_std, feas = pool.cand, pool.cand_std, pool.feas
            n_candidates_eff = pool.n_candidates

            X_succ_std = pool.attempted_std[success]
            psi_succ = np.asarray(psi, dtype=np.float64)[success]
            # OOF refits use RAW inputs — the winner's hyperparameters were
            # tuned on raw inputs (zoo pipelines scale internally). Only the
            # error-model GP below works in box-standardized coordinates.
            resid = _oof_residuals(
                winner_payload,
                attempted_inputs[success],
                psi_succ,
                k_folds=k_folds,
                seed=seed,
            )
            ok = np.isfinite(resid) & (resid >= 0)
            if ok.sum() < min_needed:
                raise RuntimeError(
                    f"only {int(ok.sum())} finite OOF residuals (< {min_needed})"
                )
            # Log-residual target: multiplicative error structure, and keeps
            # the GP from being dominated by a few catastrophic samples.
            log_r = np.log(np.maximum(resid[ok], 1e-30))

            from sklearn.gaussian_process import GaussianProcessRegressor

            kern = _ard_rbf(
                X_succ_std.shape[1], noise_level=1e-2, noise_bounds=(1e-8, 1e1)
            )
            gpr = GaussianProcessRegressor(
                kernel=kern, alpha=1e-10, n_restarts_optimizer=1, random_state=seed
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gpr.fit(X_succ_std[ok], log_r)
            mu = gpr.predict(cand_std)

            # Frozen-kernel variance state for exact greedy conditioning
            # (kriging believer: mean unchanged, sigma shrinks near picks).
            batch = _GPBatch(
                [gpr.kernel_],
                weights=np.ones(1),
                X_train=X_succ_std[ok],
                X_cand=cand_std,
            )
            log_feas = np.log(np.maximum(feas, 1e-30))

            chosen: list[int] = []
            scores: list[float] = []
            for _ in range(n_points):
                var = batch.total_variance()  # -inf on consumed candidates
                sigma = np.sqrt(np.maximum(var, 0.0))  # maximum(-inf, 0) == 0
                acq = np.where(np.isfinite(var), mu + beta * sigma + log_feas, -np.inf)
                idx = int(np.argmax(acq))
                if not np.isfinite(acq[idx]):
                    break
                chosen.append(idx)
                scores.append(float(acq[idx]))
                batch.consume(idx)

            sel = np.asarray(chosen, dtype=int)
            residual_summary = {
                "n_oof": int(ok.sum()),
                "k_folds": int(k_folds),
                "resid_mean": float(resid[ok].mean()),
                "resid_max": float(resid[ok].max()),
                "resid_p90": float(np.quantile(resid[ok], 0.9)),
            }
            notes = (
                f"Residual-UCB acquisition: {len(sel)} points from a "
                f"{n_candidates_eff}-point Sobol pool; error model = ARD-RBF GP on "
                f"log OOF residuals of winner '{winner_payload.get('model_name')}' "
                f"({int(ok.sum())} samples, {k_folds}-fold); beta={beta}; "
                f"feasibility weighting "
                f"{'ACTIVE' if not success.all() else 'inactive (no failures)'}"
                f" (floor={feasibility_floor})."
            )
            return AcquisitionResult(
                X_new=cand[sel],
                method="residual_ucb",
                scores=np.asarray(scores, dtype=np.float64),
                feasibility=feas[sel],
                n_pca_components=0,
                pca_explained_variance=0.0,
                kernel_summaries=[str(gpr.kernel_)],
                n_candidates=n_candidates_eff,
                seed=int(seed),
                notes=notes,
                beta=beta,
                residual_summary=residual_summary,
            )
        except Exception as exc:  # noqa: BLE001 — degrade, don't die
            fallback_reason = f"residual path failed ({type(exc).__name__}: {exc})"

    res = select_acquisition_points(
        attempted_inputs=attempted_inputs,
        success=success,
        psi=psi,
        bounds_low=lows,
        bounds_high=highs,
        n_points=n_points,
        n_candidates=n_candidates,
        seed=seed,
        feasibility_floor=feasibility_floor,
    )
    res.notes = f"[residual_ucb unavailable: {fallback_reason}] " + res.notes
    return res


ACQUISITION_STRATEGIES = ("auto", "residual_ucb", "uncertainty", "space_filling")


def select_from_dataset(
    dataset_h5: str | Path,
    *,
    bounds_low: np.ndarray,
    bounds_high: np.ndarray,
    n_points: int,
    winner_payload: Optional[dict] = None,
    beta: float = 1.0,
    strategy: str = "auto",
    **kwargs,
) -> AcquisitionResult:
    """Convenience wrapper: run acquisition straight off a Phase-1 HDF5 file.

    Reads ALL rows (h5io layout) — failures included, they feed the
    feasibility model — then dispatches by ``strategy``:

      - ``auto`` (default): residual-driven UCB when a ``winner_payload`` is
        given, else the model-agnostic PCA-GP variance path.
      - ``residual_ucb``: force the residual-driven path (it degrades to
        gp_variance/maximin internally when there is no usable winner).
      - ``uncertainty``: force the model-agnostic PCA-GP field-variance path
        (ignore the winner) — useful when the winner's residuals are not yet
        trustworthy (early, tiny pool).
      - ``space_filling``: force pure feasibility-weighted maximin — blanket
        a region uniformly, no error model at all.
    """
    from autotokamak.data.h5io import read_h5_arrays

    if strategy not in ACQUISITION_STRATEGIES:
        raise ValueError(
            f"strategy must be one of {ACQUISITION_STRATEGIES}; got {strategy!r}"
        )

    arrays = read_h5_arrays(dataset_h5)
    X = np.column_stack([arrays.inputs[p] for p in PARAM_ORDER])
    common = dict(
        attempted_inputs=X,
        success=arrays.success,
        psi=arrays.psi,
        bounds_low=bounds_low,
        bounds_high=bounds_high,
        n_points=n_points,
        **kwargs,
    )

    if strategy == "space_filling":
        return select_acquisition_points(force_maximin=True, **common)
    if strategy == "uncertainty":
        return select_acquisition_points(**common)
    if strategy == "residual_ucb" or (strategy == "auto" and winner_payload is not None):
        return select_acquisition_points_residual(
            winner_payload=winner_payload, beta=beta, **common
        )
    return select_acquisition_points(**common)


__all__ = [
    "MIN_SUCCESS_FOR_GP",
    "AcquisitionResult",
    "select_acquisition_points",
    "select_acquisition_points_residual",
    "select_from_dataset",
]
