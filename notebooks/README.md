# Manual surrogate notebooks

| Path | Role |
|---|---|
| `manual_surrogate.ipynb` | Generate equilibria → plot → train PCA + sklearn surrogates |
| `scaling_laws.ipynb` | Learning curves vs $N$, input coverage, PCA rank, distribution shift |
| `manual_surrogate_run/` | Artifacts from the manual notebook (`dataset.h5`, winners) |
| `scaling_runs/` | Plots/JSON from the scaling notebook (gitignored) |

## Setup

Use the project venv as the notebook kernel (a generic system Python will raise `ModuleNotFoundError: autotokamak`).

```bash
cd /path/to/autokamak
source venv/bin/activate
pip install -e ".[ml,dev]"
python -m ipykernel install --user --name=autotokamak --display-name="Python (autotokamak)"
```

In Cursor/VS Code: open the notebook → kernel picker → **Python (autotokamak)** or `venv/bin/python`.

## Knobs

**`manual_surrogate.ipynb`**
- `N_SAMPLES` — start at 40; raise to 200–500+ for a serious model (cap 10000 in schema)
- `FORCE_REGENERATE` — set `True` to rebuild HDF5
- `RUN_AUTOML` — optional Optuna pass

**`scaling_laws.ipynb`**
- Reuses `manual_surrogate_run/dataset.h5` (run the manual notebook first)
- Shrink `N_GRID` for a quick smoke pass
- Artifacts → `scaling_runs/scaling_report.json`

**Next models to try** (priority order): [`docs/surrogate_model_candidates.md`](../docs/surrogate_model_candidates.md).

**OFT:** one `OFT_env` per process. Generate data in a single kernel; restart the kernel before a second full sweep if you hit the singleton error.
