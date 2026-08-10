# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Classical surrogate model factories — the Phase-2 PoC zoo.

Per advisor scope (``docs/project_agenda.md`` §3): four sklearn-only models.
PINN / DeepONet / FNO are EXPLICITLY OUT OF SCOPE for the PoC.

The agent does NOT instantiate sklearn classes directly; it calls these
factories with hyperparameter dicts sampled by Optuna. That gives us one
place to enforce caps (e.g. MLP layers) and one place for the prompt's
``API REFERENCE`` to point at.

``DEFAULT_SEARCH_SPACES`` is quoted verbatim into the prompt as the
"starting suggestion"; the agent is free to widen/tighten/replace it per
outer-loop round.
"""

from __future__ import annotations

from typing import Any, Dict


# Hyperparameter-range presets the prompt advertises. Values are ``ParamRange``
# dicts (see ``schema.ParamRange``) so the agent can copy them into a
# ``ModelSpec`` with no shape adjustment.
DEFAULT_SEARCH_SPACES: Dict[str, Dict[str, Dict[str, Any]]] = {
    "gp": {
        "length_scale": {"type": "loguniform", "low": 1e-2, "high": 1e2},
        "noise_level": {"type": "loguniform", "low": 1e-6, "high": 1e-1},
        "alpha": {"type": "loguniform", "low": 1e-10, "high": 1e-3},
    },
    "kernel_ridge": {
        "alpha": {"type": "loguniform", "low": 1e-6, "high": 1e2},
        "gamma": {"type": "loguniform", "low": 1e-3, "high": 1e2},
        "kernel": {"type": "categorical", "choices": ["rbf", "laplacian"]},
    },
    "poly_ridge": {
        "alpha": {"type": "loguniform", "low": 1e-6, "high": 1e2},
        "degree": {"type": "int", "low": 1, "high": 3},
    },
    "mlp": {
        "n_layers": {"type": "int", "low": 1, "high": 2},
        "layer_width": {"type": "int", "low": 16, "high": 256},
        "alpha": {"type": "loguniform", "low": 1e-6, "high": 1e-1},
        "learning_rate_init": {"type": "loguniform", "low": 1e-4, "high": 1e-1},
    },
}

MLP_MAX_LAYERS = 2
MLP_MAX_WIDTH = 256


def _with_input_scaling(estimator):
    """Wrap an estimator so it sees standardized inputs.

    The raw inputs span wildly different scales (Ip ~ 1e5 A vs r0 ~ 0.4 m).
    Without standardization every kernel distance / MLP activation is
    dominated by Ip alone, and the model collapses to predicting the mean —
    diagnosed as the winner≈baseline plateau (RMSE 0.204 vs 0.203). The
    DEFAULT_SEARCH_SPACES ranges for length_scale/gamma (1e-2..1e2) assume
    O(1) features, so scaling also makes those ranges meaningful.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([("scale", StandardScaler()), ("model", estimator)])


def make_gp(**hp: Any):
    """Gaussian-Process regressor (RBF + WhiteKernel) on standardized inputs.

    Hyperparameters: ``length_scale``, ``noise_level``, ``alpha``.
    Multi-output is native in sklearn's GP — no MultiOutputRegressor needed.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

    length_scale = float(hp.get("length_scale", 1.0))
    noise_level = float(hp.get("noise_level", 1e-3))
    alpha = float(hp.get("alpha", 1e-6))

    kernel = ConstantKernel(1.0) * RBF(length_scale=length_scale) + WhiteKernel(
        noise_level=noise_level
    )
    return _with_input_scaling(
        GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=True)
    )


def make_kernel_ridge(**hp: Any):
    """Kernel ridge regression on standardized inputs. Native multi-output."""
    from sklearn.kernel_ridge import KernelRidge

    return _with_input_scaling(
        KernelRidge(
            alpha=float(hp.get("alpha", 1.0)),
            gamma=float(hp.get("gamma", 1.0)),
            kernel=str(hp.get("kernel", "rbf")),
        )
    )


def make_poly_ridge(**hp: Any):
    """Polynomial features → ridge regression. Native multi-output via Ridge."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    degree = int(hp.get("degree", 2))
    alpha = float(hp.get("alpha", 1.0))
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )


def make_mlp(**hp: Any):
    """sklearn MLPRegressor with hard caps on depth + width.

    Caps enforced HERE (not in the prompt) so a hallucinated
    ``hidden_layer_sizes=(512, 512, 512)`` is rejected at construction
    rather than silently exceeding scope. ``ValueError`` propagates to the
    Optuna trial which records it as a failed trial.
    """
    from sklearn.neural_network import MLPRegressor

    n_layers = int(hp.get("n_layers", 1))
    layer_width = int(hp.get("layer_width", 64))
    alpha = float(hp.get("alpha", 1e-4))
    lr_init = float(hp.get("learning_rate_init", 1e-3))

    if n_layers < 1 or n_layers > MLP_MAX_LAYERS:
        raise ValueError(f"mlp.n_layers must be in [1, {MLP_MAX_LAYERS}]; got {n_layers}")
    if layer_width < 1 or layer_width > MLP_MAX_WIDTH:
        raise ValueError(f"mlp.layer_width must be in [1, {MLP_MAX_WIDTH}]; got {layer_width}")

    hidden = tuple([layer_width] * n_layers)
    return _with_input_scaling(
        MLPRegressor(
            hidden_layer_sizes=hidden,
            alpha=alpha,
            learning_rate_init=lr_init,
            max_iter=2000,
            random_state=0,
        )
    )


FACTORIES = {
    "gp": make_gp,
    "kernel_ridge": make_kernel_ridge,
    "poly_ridge": make_poly_ridge,
    "mlp": make_mlp,
}


def make_model(model_name: str, **hp: Any):
    """Dispatch by model name. Raises KeyError on unknown names."""
    if model_name not in FACTORIES:
        raise KeyError(
            f"Unknown surrogate model {model_name!r}; choose from {list(FACTORIES)}"
        )
    return FACTORIES[model_name](**hp)


__all__ = [
    "DEFAULT_SEARCH_SPACES",
    "FACTORIES",
    "MLP_MAX_LAYERS",
    "MLP_MAX_WIDTH",
    "make_gp",
    "make_kernel_ridge",
    "make_mlp",
    "make_model",
    "make_poly_ridge",
]
