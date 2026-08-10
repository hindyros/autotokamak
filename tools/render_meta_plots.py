# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Render meta-loop visualizations from a workspace's meta_trace.json + report.json.

Writes into ``<workspace>/outputs/meta_plots/``:

  - ``convergence.png`` — frozen-shard RMSE vs meta iteration, with the baseline
    mean-predictor line and (if set) the early-stop target line. Each point is
    annotated with the action that produced it (regen_dataset / extend_search /
    enrich_active / …). This is the single most interpretable view of the run:
    did the loop actually improve the surrogate, and through which actions?

  - ``per_cell_rmse.png`` — envelope runs only: per-geometry-cell RMSE of the
    winner on the eval set, worst cell highlighted. Shows WHERE in shape space
    the surrogate is weak — i.e. where active sampling should focus. Skipped
    when the run has no per-cell breakdown.

Best-effort: a missing input skips that plot rather than raising, so the caller
(the HTML report) always renders.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from autotokamak.pipelines.discover import find_report  # noqa: E402

# Short, stable colors per action so the same action reads the same across runs.
_ACTION_COLOR = {
    "regen_dataset": "#0a7d3f",
    "enrich_active": "#7b2fbf",
    "extend_search": "#0a4a7a",
    "terminate": "#8a4b00",
}


def _load_json(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None


def _iterations_and_report(workspace: Path) -> tuple[list, dict]:
    mt = _load_json(workspace / "meta_trace.json")
    if isinstance(mt, dict) and mt.get("iterations"):
        report = mt.get("report") or (_load_json(find_report(workspace)) if find_report(workspace) else {}) or {}
        return list(mt["iterations"]), report
    rp = find_report(workspace)
    report = (_load_json(rp) if rp else {}) or {}
    return [], report


def plot_convergence(workspace: Path, out_dir: Path) -> Path | None:
    iterations, report = _iterations_and_report(workspace)

    # Prefer per-iteration rmse_after (carries the action); fall back to history.
    xs, ys, actions = [], [], []
    for rec in iterations:
        if not isinstance(rec, dict):
            continue
        rmse = rec.get("rmse_after")
        if rmse is None:
            continue
        xs.append(rec.get("iteration", len(xs)))
        ys.append(float(rmse))
        actions.append((rec.get("decision") or {}).get("action", ""))
    if not ys:
        hist = report.get("rmse_history") or []
        xs = list(range(len(hist)))
        ys = [float(v) for v in hist]
        actions = [""] * len(ys)
    if not ys:
        return None

    baseline = report.get("baseline_rmse")
    target = report.get("target_rmse")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(xs, ys, "-o", color="#1a1a1a", zorder=3, label="winner (frozen-shard RMSE)")

    # Color each marker by the action that produced it.
    for x, y, a in zip(xs, ys, actions):
        c = _ACTION_COLOR.get(a, "#1a1a1a")
        ax.plot([x], [y], "o", color=c, markersize=9, zorder=4)
        if a:
            ax.annotate(
                a, (x, y), textcoords="offset points", xytext=(0, 9),
                ha="center", fontsize=8, color=c, rotation=20,
            )

    if isinstance(baseline, (int, float)) and baseline > 0:
        ax.axhline(baseline, ls="--", color="#a10000", lw=1.3,
                   label=f"baseline mean-predictor ({baseline:.3g})")
    if isinstance(target, (int, float)) and target > 0:
        ax.axhline(target, ls=":", color="#0a7d3f", lw=1.3,
                   label=f"early-stop target ({target:.3g})")

    # Headline improvement in the title.
    subtitle = ""
    if isinstance(baseline, (int, float)) and baseline > 0 and ys:
        acc = 100.0 * (1.0 - ys[-1] / baseline)
        subtitle = f"  ·  final {acc:+.1f}% vs baseline"
    ax.set_xlabel("meta iteration")
    ax.set_ylabel("frozen-shard RMSE (ψ units)")
    ax.set_title(f"Meta-loop convergence{subtitle}")
    ax.set_xticks(xs)
    ax.margins(y=0.15)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    out = out_dir / "convergence.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_per_cell_rmse(workspace: Path, out_dir: Path) -> Path | None:
    _, report = _iterations_and_report(workspace)
    per_cell = report.get("eval_per_cell") or {}
    cells = per_cell.get("cells") or {}
    if not cells:
        return None

    items = sorted(cells.items(), key=lambda kv: kv[1].get("rmse", 0.0), reverse=True)
    keys = [k for k, _ in items]
    rmses = [float(v.get("rmse", 0.0)) for _, v in items]
    ns = [int(v.get("n", 0)) for _, v in items]
    worst_key = (per_cell.get("worst_cell") or {}).get("key")
    colors = ["#a10000" if k == worst_key else "#0a4a7a" for k in keys]

    fig, ax = plt.subplots(figsize=(max(7, 0.4 * len(keys)), 4.5))
    ax.bar(range(len(keys)), rmses, color=colors)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(
        [f"{k}\n(n={n})" for k, n in zip(keys, ns)], rotation=90, fontsize=7
    )
    mean_cell = per_cell.get("mean_cell_rmse")
    if isinstance(mean_cell, (int, float)):
        ax.axhline(mean_cell, ls="--", color="#555",
                   label=f"mean over cells ({mean_cell:.3g})")
        ax.legend(fontsize=8)
    ax.set_ylabel("winner RMSE on cell (ψ units)")
    ax.set_xlabel("geometry cell (bin indices per r0-a-kappa-delta-Ip; worst in red)")
    ax.set_title("Per-geometry-cell surrogate error — where active sampling should focus")
    fig.tight_layout()
    out = out_dir / "per_cell_rmse.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def render_all(workspace: Path) -> list[Path]:
    out_dir = workspace / "outputs" / "meta_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for fn in (plot_convergence, plot_per_cell_rmse):
        try:
            p = fn(workspace, out_dir)
            if p is not None:
                written.append(p)
                print(f"Wrote {p}")
        except Exception as exc:  # noqa: BLE001
            print(f"  (skipping {fn.__name__}: {type(exc).__name__}: {exc})")
    if not written:
        print(f"No meta plots produced for {workspace} (not a meta run, or no data yet).")
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workspace", required=True, help="Meta-loop workspace dir.")
    args = ap.parse_args()
    render_all(Path(args.workspace))


if __name__ == "__main__":
    main()
