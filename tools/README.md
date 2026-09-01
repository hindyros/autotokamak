# `tools/` — post-run analysis and diagnostics scripts

Standalone scripts that read the artifacts a pipeline run leaves behind.
None of them are needed to *run* the pipelines — the front door for that is:

```bash
python -m autotokamak.pipelines <phase1|phase2|meta> [--level L0|L1]
```

(Agent-codegen benchmark conditions run via
`python -m autotokamak.bench run --task benchmarks/tasks/<task>.yaml --harness <name>`;
see `benchmarks/README.md`.)

| Script | What it does |
|---|---|
| `eval_surrogate.py` | 7 diagnostic PNGs + JSON metrics for a trained surrogate (default workspace: `examples/surrogate_meta`). |
| `render_meta_plots.py` | Meta-loop convergence + per-cell RMSE plots from `meta_trace.json` / `report.json`. |
| `render_physics.py` | ψ(R,Z) contour samples + parameter histograms from a `dataset.h5`. |
| `trace_to_html.py` | Renders `experiments/*/trace.json` into a browsable static HTML report (stdlib-only). |
| `probe_feasible_box.py` | LHS-samples candidate shaping-parameter boxes and reports the clean-isoflux success rate per box. |
| `collect_traces.sh` | Runs the meta-loop N times to collect traces for offline GEPA prompt optimization (`agent/dspy/optimize_meta.py`). |
| `matrix_report.py` | Cross-condition matrix report: scores every cell (bench `predict.py` or pipeline `winner.pkl`) on the frozen benchmark test set; renders `experiments/<tag>/index.html`. |
| `eval_code_metrics.py` | Static (zero-LLM) metrics over every agent workspace in a tag: SLOC, structure, imports (library leverage / L3-violation cross-check / LLM-in-loop detection), seeds, ruff counts → `<run_dir>/eval/code_metrics.json` + `experiments/<tag>/code_metrics.csv`. |
| `judge_code.py` | Blind, outcome-blind LLM-as-judge over agent workspaces: 7-dimension rubric (1–5, evidence-required) per run (`score`), plus a cross-cell synthesis (`compare` → `judge_report.md`). Anonymized bundles, brand tokens redacted; `--samples N` for median-of-N. |
| `cost_report.py` | Per-run cost/efficiency table for a tag: measured $ (claude_sdk/pi/dspy/ursa), token-derived $ (cursor + price table), wall-clock/turn proxies, solver-call counts. Harvests old runs' raw event streams retroactively. |

Each script's module docstring documents its exact CLI; run with `--help` for flags.
