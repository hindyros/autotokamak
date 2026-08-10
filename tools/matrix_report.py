#!/usr/bin/env python
# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Render the cross-condition matrix report: one index.html for a full run
of the experiment matrix (pipeline L0/L1 cells + bench L2/L3 cells).

Every cell is scored on the SAME yardstick — relative-L2 against the frozen
benchmark test set (``benchmarks/assets/test_set.h5``) — regardless of how
it ran: bench cells through their workspace ``predict.py``, pipeline cells
through the saved ``winner.pkl``. Accuracy = 100 * (1 - err/baseline_err)
where the baseline predicts the frozen set's mean psi map.

Usage:
    python tools/matrix_report.py --tag matrix-20260810 \
        [--meta-workspaces examples/surrogate_meta/L0 examples/surrogate_meta/L1] \
        [--out experiments/matrix-20260810/index.html]
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

FROZEN_TESTSET = REPO_ROOT / "benchmarks" / "assets" / "test_set.h5"
N_PLOT_SAMPLES = 3


# ---------------------------------------------------------------------------
# Frozen-set scoring (shared yardstick)
# ---------------------------------------------------------------------------

def load_frozen():
    from autotokamak.data.h5io import read_h5_arrays
    from autotokamak.surrogate.dataset import PARAM_ORDER

    a = read_h5_arrays(FROZEN_TESTSET)
    ok = np.asarray(a.success, dtype=bool)
    X = np.stack([a.inputs[p][ok] for p in PARAM_ORDER], axis=1)
    records = [{p: float(a.inputs[p][i]) for p in PARAM_ORDER} for i in np.flatnonzero(ok)]
    return {"X": X, "records": records, "psi": a.psi[ok], "R": a.R, "Z": a.Z}


def rel_l2(psi_pred, psi_true):
    from autotokamak.bench.contract import rel_l2_errors

    return rel_l2_errors(psi_pred, psi_true)


def baseline_errors(frozen):
    import warnings

    with warnings.catch_warnings():
        # all-NaN grid cells (outside every plasma) are expected
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_map = np.nanmean(frozen["psi"], axis=0)
    pred = np.repeat(mean_map[None, :, :], frozen["psi"].shape[0], axis=0)
    return rel_l2(pred, frozen["psi"])


def predict_bench_cell(workspace: Path, frozen) -> np.ndarray:
    from autotokamak.bench.contract import run_predict

    arrays = run_predict(workspace, frozen["records"], timeout_seconds=3600)
    return np.asarray(arrays["psi"], dtype=np.float64)


def predict_meta_cell(workspace: Path, frozen) -> np.ndarray:
    import joblib

    from autotokamak.pipelines.discover import find_winner
    from autotokamak.surrogate.optuna_search import predict_with_winner

    winner_path = find_winner(workspace)
    if winner_path is None:
        raise FileNotFoundError(f"no winner.pkl under {workspace}")
    payload = joblib.load(winner_path)
    return predict_with_winner(payload, frozen["X"])


# ---------------------------------------------------------------------------
# Cell collection
# ---------------------------------------------------------------------------

def collect_bench_cells(tag_dir: Path) -> list[dict]:
    cells = []
    for result_path in sorted(tag_dir.glob("*/*/result.json")):
        try:
            r = json.loads(result_path.read_text())
        except Exception:  # noqa: BLE001
            continue
        r["_workspace"] = result_path.parent / "workspace"
        r["_kind"] = "bench"
        cells.append(r)
    return cells


def collect_meta_cells(workspaces: list[Path]) -> list[dict]:
    cells = []
    for ws in workspaces:
        manifest = ws / "manifest.json"
        if not manifest.is_file():
            continue
        m = json.loads(manifest.read_text())
        report = {}
        if (ws / "report.json").is_file():
            try:
                report = json.loads((ws / "report.json").read_text())
            except Exception:  # noqa: BLE001
                pass
        cells.append({
            "_kind": "meta",
            "_workspace": ws,
            "condition": m.get("condition", ws.name),
            "run_id": m.get("run_id"),
            "status": "completed" if m.get("n_iterations") is not None else "unknown",
            "harness": "none" if m.get("condition") == "L0-none" else "dspy",
            "model": m.get("model"),
            "wall_seconds": m.get("elapsed_seconds"),
            "cost_usd": None,
            "meta": {
                "n_iterations": m.get("n_iterations"),
                "terminated_by": m.get("terminated_by"),
                "self_accuracy_pct": m.get("final_accuracy_pct"),
                "winner_model": m.get("winner_model_name") or report.get("winner_model_name"),
            },
        })
    return cells


# ---------------------------------------------------------------------------
# Plots (matplotlib → base64 PNG)
# ---------------------------------------------------------------------------

def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    import matplotlib.pyplot as plt

    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def plot_accuracy_bars(rows: list[dict]) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scored = [r for r in rows if r.get("accuracy_pct") is not None]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    if scored:
        names = [r["condition"] for r in scored]
        vals = [r["accuracy_pct"] for r in scored]
        colors = ["#1a7f37" if v >= 70 else "#d4a72c" if v >= 0 else "#cf222e" for v in vals]
        ax.bar(names, vals, color=colors)
        ax.axhline(70, color="#cf222e", lw=1, ls="--", label="70% target")
        ax.legend(facecolor="white", labelcolor="#1f2328", edgecolor="#d0d7de")
    ax.set_ylabel("accuracy vs baseline [%]", color="#1f2328")
    ax.tick_params(colors="#1f2328", labelrotation=25)
    for s in ax.spines.values():
        s.set_color("#d0d7de")
    ax.set_title("Frozen-test-set accuracy = 100·(1 − relL2 / baseline relL2)",
                 color="#1f2328", fontsize=10)
    return _fig_to_b64(fig)


def plot_psi_panels(psi_pred, frozen, title: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    R, Z = frozen["R"], frozen["Z"]
    idx = np.linspace(0, len(frozen["psi"]) - 1, N_PLOT_SAMPLES).astype(int)
    fig, axes = plt.subplots(len(idx), 3, figsize=(9.5, 2.9 * len(idx)))
    fig.patch.set_facecolor("white")
    axes = np.atleast_2d(axes)
    for row, i in enumerate(idx):
        true, pred = frozen["psi"][i], psi_pred[i]
        err = np.abs(pred - true)
        for col, (fld, name, cmap) in enumerate(
            [(true, "ψ true [Wb]", "viridis"), (pred, "ψ pred [Wb]", "viridis"),
             (err, "|error| [Wb]", "magma")]
        ):
            ax = axes[row, col]
            ax.set_facecolor("white")
            pc = ax.pcolormesh(R, Z, fld, cmap=cmap, shading="auto")
            cb = fig.colorbar(pc, ax=ax, shrink=0.85)
            cb.ax.tick_params(colors="#1f2328", labelsize=7)
            ax.set_title(f"#{i}  {name}", color="#1f2328", fontsize=8)
            ax.tick_params(colors="#1f2328", labelsize=7)
            ax.set_aspect("equal")
            for s in ax.spines.values():
                s.set_color("#d0d7de")
    fig.suptitle(title, color="#1f2328", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _fig_to_b64(fig)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

CSS = """
body { background:#ffffff; color:#1f2328; font-family:-apple-system,'Segoe UI',sans-serif;
       margin:0 auto; max-width:1080px; padding:24px; }
h1,h2,h3 { color:#111418; } a { color:#0969da; }
table { border-collapse:collapse; width:100%; font-size:13px; margin:12px 0; }
th,td { border:1px solid #d0d7de; padding:6px 9px; text-align:left; }
th { background:#f6f8fa; } tr:nth-child(even) { background:#fafbfc; }
.ok { color:#1a7f37; font-weight:600; } .warn { color:#9a6700; font-weight:600; }
.bad { color:#cf222e; font-weight:600; }
.cell { background:#fafbfc; border:1px solid #d0d7de; border-radius:8px;
        padding:14px 18px; margin:14px 0; }
img { max-width:100%; border-radius:6px; }
code { background:#f6f8fa; padding:1px 5px; border-radius:4px; font-size:12px; }
.small { color:#57606a; font-size:12px; }
"""


HARNESS_NAMES = {
    "none": "no agent",
    "ursa": "the URSA agent (LangChain plan→execute)",
    "dspy": "the DSPy agent (plan → ReAct steps → review)",
    "claude_sdk": "the Claude Agent SDK",
    "pi": "the Pi Code CLI agent",
    "cursor": "the Cursor CLI agent",
    "echo": "a no-LLM mock agent",
}

LEVEL_DESCRIPTIONS = {
    "L0": "Golden baseline: the pre-written pipeline ran with every decision "
          "(which models to search, which meta-action to take next) made by "
          "seeded heuristic rules. No LLM is involved anywhere, so the run is "
          "exactly reproducible.",
    "L1": "Pre-written pipeline, LLM judgment only: the same library code ran, "
          "but an LLM made the typed decisions at the two choice points "
          "(search rounds and meta-actions). Measures the value of LLM "
          "decision-making with all code held fixed.",
    "L2": "Library-assisted agent: {h} wrote and ran the pipeline code itself, "
          "and was allowed (and encouraged) to build on the autotokamak "
          "library for solving, sampling, and training. Measures agent skill "
          "when proven scaffolding is available.",
    "L3": "From-scratch agent: {h} built the entire pipeline — solver driving, "
          "adaptive sampling campaign, surrogate training, evaluation — with "
          "only the raw TokaMaker solver available. Importing autotokamak is "
          "forbidden and audited by a hard contract gate.",
}


def condition_description(condition: str, harness) -> str:
    level = str(condition)[:2]
    h = HARNESS_NAMES.get(str(harness), str(harness))
    return LEVEL_DESCRIPTIONS.get(level, "").format(h=h)


def _fmt(v, nd=4):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}g}"
    return html.escape(str(v))


def status_span(status, accuracy):
    if status != "completed":
        return f'<span class="bad">{html.escape(str(status))}</span>'
    if accuracy is not None and accuracy >= 70:
        return '<span class="ok">completed · ≥70% ✓</span>'
    return '<span class="warn">completed</span>'


def build_html(tag: str, rows: list[dict], bars_b64: str, baseline_mean: float) -> str:
    head = [
        f"<title>autotokamak matrix — {html.escape(tag)}</title>",
        f"<style>{CSS}</style>",
        f"<h1>Experiment matrix — <code>{html.escape(tag)}</code></h1>",
        "<p class='small'>Stopping criterion: 70% error reduction vs the "
        "mean-map baseline. All conditions scored on the frozen benchmark "
        f"test set (n={rows[0]['n_scored'] if rows and rows[0].get('n_scored') else '—'}, "
        f"baseline mean relL2 = {baseline_mean:.4g}).</p>",
    ]
    tbl = ["<table><tr><th>condition</th><th>what this condition is</th>"
           "<th>status</th><th>gates</th>"
           "<th>relL2 mean</th><th>accuracy</th><th>self-reported</th>"
           "<th>wall</th><th>cost</th><th>notes</th></tr>"]
    for r in rows:
        gates = r.get("contract", {}).get("gates", {})
        gates_str = f"{sum(map(bool, gates.values()))}/{len(gates)}" if gates else "—"
        acc = r.get("accuracy_pct")
        acc_cls = "ok" if (acc is not None and acc >= 70) else "warn" if acc is not None else ""
        wall = r.get("wall_seconds")
        tbl.append(
            "<tr>"
            f"<td><b>{html.escape(r['condition'])}</b></td>"
            f"<td class='small' style='max-width:260px'>"
            f"{html.escape(condition_description(r['condition'], r.get('harness')))}</td>"
            f"<td>{status_span(r.get('status'), acc)}</td>"
            f"<td>{gates_str}</td>"
            f"<td>{_fmt(r.get('rel_l2_mean'))}</td>"
            f"<td class='{acc_cls}'>{_fmt(acc, 3)}{'%' if acc is not None else ''}</td>"
            f"<td>{_fmt(r.get('self_reported'))}</td>"
            f"<td>{_fmt(round(wall) if wall else None)}s</td>"
            f"<td>{('$' + _fmt(r.get('cost_usd'), 3)) if r.get('cost_usd') else '—'}</td>"
            f"<td class='small'>{html.escape(str(r.get('note') or ''))[:160]}</td>"
            "</tr>"
        )
    tbl.append("</table>")
    body = head + tbl + [f'<img src="data:image/png;base64,{bars_b64}">']

    for r in rows:
        body.append(f'<div class="cell"><h2>{html.escape(r["condition"])}</h2>')
        body.append(f"<p>{html.escape(condition_description(r['condition'], r.get('harness')))}</p>")
        body.append(f"<p class='small'>harness=<code>{_fmt(r.get('harness'))}</code> "
                    f"model=<code>{_fmt(r.get('model'))}</code> "
                    f"run_id=<code>{_fmt(r.get('run_id'))}</code></p>")
        if r.get("meta"):
            m = r["meta"]
            body.append(f"<p>meta: {m.get('n_iterations')} iteration(s), "
                        f"terminated_by=<code>{_fmt(m.get('terminated_by'))}</code>, "
                        f"winner=<code>{_fmt(m.get('winner_model'))}</code>, "
                        f"self-reported accuracy {_fmt(m.get('self_accuracy_pct'), 3)}%</p>")
        if r.get("error"):
            body.append(f"<p class='bad'>error: {html.escape(str(r['error'])[:400])}</p>")
        gates = r.get("contract", {}).get("gates")
        if gates:
            failed = [k for k, v in gates.items() if not v]
            body.append("<p>contract gates: " + (
                '<span class="ok">all passed</span>' if not failed
                else f'<span class="bad">failed: {html.escape(", ".join(failed))}</span>') + "</p>")
        if r.get("psi_b64"):
            body.append(f'<img src="data:image/png;base64,{r["psi_b64"]}">')
        body.append("</div>")

    body.append("<p class='small'>Generated by tools/matrix_report.py</p>")
    return "\n".join(body)


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--meta-workspaces", nargs="*", default=[
        "examples/surrogate_meta/L0", "examples/surrogate_meta/L1"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tag_dir = REPO_ROOT / "experiments" / args.tag
    out_path = Path(args.out) if args.out else tag_dir / "index.html"

    frozen = load_frozen()
    base_errs = baseline_errors(frozen)
    baseline_mean = float(np.mean(base_errs))

    cells = collect_bench_cells(tag_dir) + collect_meta_cells(
        [REPO_ROOT / w for w in args.meta_workspaces])

    rows = []
    for c in cells:
        row = {
            "condition": c.get("condition", "?"),
            "status": c.get("status"),
            "harness": c.get("harness"),
            "model": c.get("model"),
            "run_id": c.get("run_id"),
            "wall_seconds": c.get("wall_seconds"),
            "cost_usd": c.get("cost_usd"),
            "contract": c.get("contract", {}),
            "error": c.get("error"),
            "meta": c.get("meta"),
            "n_scored": len(frozen["records"]),
        }
        if c["_kind"] == "bench":
            ws = c["_workspace"]
            rep = ws / "report.json"
            if rep.is_file():
                try:
                    sr = json.loads(rep.read_text()).get("metrics", {})
                    row["self_reported"] = (sr.get("test_rel_l2") or {}).get("mean")
                except Exception:  # noqa: BLE001
                    pass
        else:
            row["self_reported"] = (c.get("meta") or {}).get("self_accuracy_pct")

        # Head-to-head on the frozen set.
        try:
            if c["_kind"] == "bench":
                if c.get("status") == "completed" and c.get("contract", {}).get("passed"):
                    pred = predict_bench_cell(c["_workspace"], frozen)
                else:
                    pred = None
            else:
                pred = predict_meta_cell(c["_workspace"], frozen)
            if pred is not None:
                errs = rel_l2(pred, frozen["psi"])
                finite = errs[np.isfinite(errs)]
                row["rel_l2_mean"] = float(np.mean(finite))
                row["accuracy_pct"] = 100.0 * (1.0 - row["rel_l2_mean"] / baseline_mean)
                row["psi_b64"] = plot_psi_panels(
                    pred, frozen,
                    f"{row['condition']} — frozen-set relL2 mean {row['rel_l2_mean']:.4g} "
                    f"({row['accuracy_pct']:.1f}% vs baseline)")
        except Exception as exc:  # noqa: BLE001
            row["note"] = f"frozen-set scoring failed: {type(exc).__name__}: {exc}"
        rows.append(row)

    order = {"L0": 0, "L1": 1, "L2": 2, "L3": 3}
    rows.sort(key=lambda r: (order.get(str(r["condition"])[:2], 9), str(r["condition"])))

    bars = plot_accuracy_bars(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_html(args.tag, rows, bars, baseline_mean), encoding="utf-8")
    print(f"Wrote {out_path} ({len(rows)} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
