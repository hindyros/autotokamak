"""Structured Phase-2 surrogate AutoML: sklearn model zoo + Optuna search.

Submodules:
- :mod:`autotokamak.surrogate.dataset`       — HDF5 dataset loading + k-fold splits
- :mod:`autotokamak.surrogate.reduce`        — PCA reduction of psi maps
- :mod:`autotokamak.surrogate.metrics`       — psi-RMSE and baseline metrics
- :mod:`autotokamak.surrogate.zoo`           — sklearn model factories (``make_model``)
- :mod:`autotokamak.surrogate.schema`        — ``SurrogateConfig``/``SearchSpec``/``StudyResult``
- :mod:`autotokamak.surrogate.optuna_search` — Optuna study harness (inner loop)
- :mod:`autotokamak.surrogate.automl_loop`   — LLM-driven outer search loop (Phase-2)
- :mod:`autotokamak.surrogate.diagnostics`   — bottleneck diagnostics for the meta-agent
"""
