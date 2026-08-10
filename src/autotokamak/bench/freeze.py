# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Solve the frozen head-to-head test set (bench freeze-testset).

Ground truth every benchmark condition is scored against. Uses the platform
sweep machinery (``data.sweep.run_sweep`` with an explicit sample matrix) on
the canonical dataset config, whose output grid IS the frozen evaluation
grid. The resulting .h5 is gitignored (heavy) but reproducible from the
committed ``benchmarks/assets/test_params.json``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from autotokamak.agent.runners.config import REPO_ROOT

CANONICAL_SWEEP_CONFIG = REPO_ROOT / "examples" / "dataset_generation" / "dataset_config.yaml"


def solve_testset(
    records: list[dict],
    out_h5: Path,
    *,
    grid_json: Optional[Path] = None,
    sweep_config: Optional[Path] = None,
) -> int:
    """Solve ``records`` (list of param dicts) and write ``out_h5``.

    Returns the number of successful solves. Asserts the sweep config's
    output grid matches the frozen eval grid so head-to-head scoring is
    guaranteed consistent.
    """
    from autotokamak.bench.contract import grid_axes, load_grid
    from autotokamak.data.schema import SweepConfig
    from autotokamak.data.sweep import run_sweep
    from autotokamak.surrogate.dataset import PARAM_ORDER

    cfg = SweepConfig.from_yaml(sweep_config or CANONICAL_SWEEP_CONFIG)
    out_h5 = Path(out_h5)
    cfg = cfg.model_copy(update={"output_path": out_h5.name})

    R_ref, Z_ref = grid_axes(load_grid(grid_json))
    R_cfg = np.linspace(cfg.output_grid.R.min, cfg.output_grid.R.max, cfg.output_grid.R.n)
    Z_cfg = np.linspace(cfg.output_grid.Z.min, cfg.output_grid.Z.max, cfg.output_grid.Z.n)
    if not (np.allclose(R_cfg, R_ref) and np.allclose(Z_cfg, Z_ref)):
        raise ValueError(
            "Sweep config output_grid differs from the frozen eval grid — "
            "refusing to build an incomparable test set."
        )

    X = np.array([[float(r[p]) for p in PARAM_ORDER] for r in records], dtype=np.float64)
    result = run_sweep(cfg, out_h5.parent, X=X)
    return int(result.n_succeeded)


def load_test_params(path: Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
