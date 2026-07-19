# `tools/` — post-run analysis and diagnostics scripts

Standalone scripts that read the artifacts a pipeline run leaves behind.
None of them are needed to *run* the pipelines — the front door for that is:

```bash
python -m autotokamak.pipelines <phase1|phase2|meta> --mode <fast|ursa>
```

| Script | What it does |
|---|---|
| `eval_surrogate.py` | 7 diagnostic PNGs + JSON metrics for a trained surrogate (default workspace: `examples/surrogate_meta`). |
| `render_meta_plots.py` | Meta-loop convergence + per-cell RMSE plots from `meta_trace.json` / `report.json`. |
| `render_physics.py` | ψ(R,Z) contour samples + parameter histograms from a `dataset.h5`. |
| `trace_to_html.py` | Renders `experiments/*/trace.json` into a browsable static HTML report (stdlib-only). |
| `probe_feasible_box.py` | LHS-samples candidate shaping-parameter boxes and reports the clean-isoflux success rate per box. |
| `collect_traces.sh` | Runs the meta-loop N times to collect traces for offline GEPA prompt optimization (`agent/dspy/optimize_meta.py`). |

Each script's module docstring documents its exact CLI; run with `--help` for flags.
