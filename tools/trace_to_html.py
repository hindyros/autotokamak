# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Render experiments/*/trace.json into a browsable static HTML report.

Usage:
    python tools/trace_to_html.py                    # scan experiments/, write experiments/_report/
    python tools/trace_to_html.py --out some/dir     # custom output
    python tools/trace_to_html.py --experiments X    # custom experiments root

Zero dependencies beyond the stdlib. Regeneration is idempotent.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import html
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]

# Default report cutoff: runs started before this are excluded from the
# index. Currently set to the start of the 20260718T231755Z run (the
# max_iterations=20 meta run) — earlier same-day shakeout runs and everything
# pre-active-sampling are hidden. Pass --since all to include everything, or
# --since YYYY-MM-DD[THH:MM] for a different cutoff.
DEFAULT_SINCE = "2026-07-18T23:17"

LOG_TS_RE = re.compile(r"_(\d{8}T\d{6}Z)\.log$")
RUN_ID_TS_RE = re.compile(r"^(\d{8}T\d{6}Z)")


@dataclass
class LogFile:
    path: Path
    started_utc: _dt.datetime


def _parse_utc(s: str) -> Optional[_dt.datetime]:
    if not s:
        return None
    try:
        dt = _dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:  # naive stamps (e.g. "20260719_182653") are UTC
        dt = dt.replace(tzinfo=_dt.timezone.utc)
    return dt


def _index_logs(logs_dir: Path) -> list[LogFile]:
    out: list[LogFile] = []
    if not logs_dir.is_dir():
        return out
    for p in logs_dir.glob("*.log"):
        m = LOG_TS_RE.search(p.name)
        if not m:
            continue
        try:
            ts = _dt.datetime.strptime(m.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=_dt.timezone.utc)
        except ValueError:
            continue
        out.append(LogFile(path=p, started_utc=ts))
    return out


def _match_log(run_started: Optional[_dt.datetime], logs: list[LogFile], tol_seconds: int = 120) -> Optional[LogFile]:
    """Pick the log whose start time is closest to (but not after) the run's start."""
    if run_started is None or not logs:
        return None
    best: Optional[LogFile] = None
    best_dt: Optional[_dt.timedelta] = None
    for lf in logs:
        delta = run_started - lf.started_utc
        if delta.total_seconds() < -5:
            continue
        if abs(delta.total_seconds()) > tol_seconds:
            continue
        if best_dt is None or abs(delta.total_seconds()) < abs(best_dt.total_seconds()):
            best = lf
            best_dt = delta
    return best


ROUND_START_RE = re.compile(r"^=== EXECUTION \(round (\d+)\) ===\s*$", re.MULTILINE)
ROUND_END_RE = re.compile(r"^=== (?:GLOBAL FEEDBACK|FINAL|VALIDATE)", re.MULTILINE)
STEP_END_RE = re.compile(r"^--- Step (\d+) (result|ERROR) ---\s*$", re.MULTILINE)


def _parse_log_by_round(log_text: str) -> dict[int, dict[int, str]]:
    """Slice the raw log into per-round, per-step transcripts.

    Uses the runner's own stdout markers (=== EXECUTION (round N) === and
    --- Step N result --- / --- Step N ERROR ---) to segment.
    """
    out: dict[int, dict[int, str]] = {}
    round_starts = [(int(m.group(1)), m.end()) for m in ROUND_START_RE.finditer(log_text)]
    for i, (round_no, r_start) in enumerate(round_starts):
        r_end = round_starts[i + 1][1] if i + 1 < len(round_starts) else len(log_text)
        end_match = ROUND_END_RE.search(log_text, pos=r_start, endpos=r_end)
        if end_match:
            r_end = end_match.start()
        segment = log_text[r_start:r_end]
        # Split by step-end markers; text BEFORE marker N belongs to step N.
        steps: dict[int, str] = {}
        cursor = 0
        for m in STEP_END_RE.finditer(segment):
            step_no = int(m.group(1))
            steps[step_no] = segment[cursor:m.start()].strip("\n")
            cursor = m.end()
        out[round_no] = steps
    return out


WRITE_RE = re.compile(r"^Writing file:\s+(\S.+?)\s*$", re.MULTILINE)
RUNNING_RE = re.compile(r"^RUNNING:\s+(.+?)\s*$", re.MULTILINE)
# Match a well-formed Python traceback:
#   header line + one or more indented stack-frame lines + a terminating error line.
# The (?:  .+\n)+ segment requires every interior line to start with two spaces,
# which excludes Optuna INFO logs like "[I 2026-... Trial N finished ...]" that
# otherwise get slurped when tracebacks and Optuna output are interleaved on stderr.
TRACEBACK_RE = re.compile(
    r"^(Traceback \(most recent call last\):\n(?:  .+\n)+\w+(?:Error|Exception)[^\n]*)",
    re.MULTILINE,
)
STDERR_LINE_RE = re.compile(r"^STDERR:[^\n]*(?:\n(?!STDOUT:|STDERR:|RUNNING:|Command).+)*", re.MULTILINE)


def _extract_step_facts(raw: str) -> dict:
    files = WRITE_RE.findall(raw)
    commands = [c[:200] + ("…" if len(c) > 200 else "") for c in RUNNING_RE.findall(raw)]
    tracebacks = TRACEBACK_RE.findall(raw)
    return {
        "files_written": files,
        "commands": commands,
        "tracebacks": tracebacks,
    }


def _fmt_duration(a: str, b: str) -> str:
    ta, tb = _parse_utc(a), _parse_utc(b)
    if not ta or not tb:
        return ""
    seconds = (tb - ta).total_seconds()
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


def _fmt_score(score: dict) -> tuple[str, str]:
    total = score.get("total", 0.0)
    gates = score.get("hard_gates", {}) or {}
    all_pass = bool(gates) and all(gates.values())
    if not gates:
        cls, label = "score-none", "n/a"
    elif not all_pass:
        cls, label = "score-fail", "GATE FAIL"
    elif total >= 0.75:
        cls, label = "score-good", f"{total:.3f}"
    elif total >= 0.5:
        cls, label = "score-mid", f"{total:.3f}"
    else:
        cls, label = "score-low", f"{total:.3f}"
    return cls, label


CSS = """
:root { color-scheme: light only; }
html, body { background: #ffffff; color: #24292e; }
body { font: 14px/1.5 -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; padding: 24px; max-width: 1200px; }
h1 { margin: 0 0 4px; font-size: 20px; color: #24292e; }
h2 { font-size: 15px; margin: 24px 0 8px; color: #24292e; }
p  { color: #24292e; }
a { color: #0366d6; text-decoration: none; }
a:hover { text-decoration: underline; }
.muted { color: #6a737d; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
table { border-collapse: collapse; width: 100%; margin-top: 12px; background: white; color: #24292e; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid #e1e4e8; color: #24292e; }
th { background: #f6f8fa; font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: 0.03em; }
tr:hover { background: rgba(3, 102, 214, 0.05); }
.badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; }
.score-good { background: #d4f4d4; color: #1a5d1a; }
.score-mid  { background: #fff3c4; color: #7a5c00; }
.score-low  { background: #ffd1d1; color: #7a1a1a; }
.score-fail { background: #4a1a1a; color: #ffd1d1; }
.score-none { background: #eee; color: #666; }
.phase { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; text-transform: capitalize; }
.phase-meta { background: #e6dcff; color: #4b2c91; }
.phase-dataset { background: #d7ecff; color: #0a4a7a; }
.phase-surrogate { background: #ffe6cc; color: #8a4b00; }
.phase-unknown { background: #eee; color: #666; }
.pos { color: #1a5d1a; font-weight: 600; }
.neg { color: #a10000; font-weight: 600; }
.status-completed { color: #1a5d1a; }
.status-errored, .status-failed { color: #a10000; }
.status-interrupted { color: #7a5c00; }
.card { border: 1px solid #e1e4e8; border-radius: 6px; padding: 12px 14px; margin: 10px 0; background: #fafbfc; color: #24292e; }
.step { border-left: 3px solid #ccc; padding: 6px 10px; margin: 6px 0; background: white; color: #24292e; }
.step.ok { border-left-color: #28a745; }
.step.err { border-left-color: #d73a49; background: #fff5f5; }
.step-head { display: flex; justify-content: space-between; align-items: center; font-weight: 600; color: #24292e; }
.step-body { margin-top: 6px; color: #24292e; }
pre { background: #f6f8fa; color: #24292e; padding: 10px; border-radius: 4px; overflow-x: auto; max-height: 400px; font-size: 12px; margin: 6px 0; white-space: pre-wrap; word-break: break-word; }
details { margin: 8px 0; }
details > summary { cursor: pointer; padding: 6px 0; user-select: none; color: #24292e; }
.qbar { display: inline-block; width: 100px; height: 8px; background: #eee; border-radius: 4px; vertical-align: middle; margin: 0 6px; overflow: hidden; }
.qbar > span { display: block; height: 100%; background: linear-gradient(90deg, #d73a49, #ffab00, #28a745); }
.q-row { display: flex; align-items: center; padding: 3px 0; color: #24292e; }
.q-name { flex: 0 0 220px; }
.q-val  { flex: 0 0 60px; text-align: right; font-family: ui-monospace, monospace; }
.q-weight { flex: 0 0 60px; text-align: right; color: #6a737d; font-size: 12px; }
.eval-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 14px; margin: 12px 0; }
.eval-fig { margin: 0; padding: 10px; background: #fafbfc; border: 1px solid #e1e4e8; border-radius: 6px; }
.eval-fig img { display: block; width: 100%; height: auto; border-radius: 4px; }
.eval-fig figcaption { margin-top: 6px; font-size: 12px; color: #24292e; text-align: center; }
.iter-card { border: 1px solid #e1e4e8; border-radius: 6px; margin: 10px 0; background: white; }
.iter-head { display: flex; justify-content: space-between; align-items: baseline; padding: 10px 14px; background: #f6f8fa; border-bottom: 1px solid #e1e4e8; border-radius: 6px 6px 0 0; }
.iter-head b { font-size: 14px; }
.iter-body { padding: 10px 14px; }
.act-badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; margin-left: 8px; }
.act-regen { background: #dbeafe; color: #1e40af; }
.act-extend { background: #fef3c7; color: #7a5c00; }
.act-terminate { background: #dcfce7; color: #14532d; }
.diag-block { margin: 6px 0; padding: 8px 12px; background: #f6f8fa; border-left: 3px solid #0366d6; border-radius: 3px; font-size: 13px; }
.diag-label { font-weight: 600; color: #24292e; text-transform: uppercase; font-size: 11px; letter-spacing: 0.05em; margin-right: 6px; }
.leaderboard { border-collapse: collapse; width: 100%; margin: 8px 0; }
.leaderboard th, .leaderboard td { border-bottom: 1px solid #e1e4e8; padding: 6px 10px; text-align: left; }
.leaderboard th { background: #f6f8fa; font-size: 11px; text-transform: uppercase; letter-spacing: 0.03em; }
.leaderboard tr.winner { background: #dcfce7; }
.leaderboard tr.winner td { font-weight: 600; }
.winner-tag { display: inline-block; background: #14532d; color: white; font-size: 10px; padding: 1px 6px; border-radius: 8px; margin-left: 6px; text-transform: uppercase; }
.kv-list { list-style: none; padding: 0; margin: 4px 0; font-size: 13px; }
.kv-list li { padding: 2px 0; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
.kv-list li b { display: inline-block; min-width: 180px; color: #6a737d; font-weight: 500; }
.physics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; margin: 12px 0; }
.physics-fig { margin: 0; padding: 8px; background: #fafbfc; border: 1px solid #e1e4e8; border-radius: 6px; }
.physics-fig img { display: block; width: 100%; height: auto; border-radius: 3px; }
.physics-fig figcaption { margin-top: 4px; font-size: 11px; color: #24292e; text-align: center; font-family: ui-monospace, monospace; }
th[data-tip] { position: relative; cursor: help; border-bottom: 2px dotted #9aa5b1; }
th[data-tip]::after {
    content: attr(data-tip);
    position: absolute;
    top: calc(100% + 4px);
    left: 0;
    min-width: 300px;
    max-width: 460px;
    padding: 10px 12px;
    background: #24292e;
    color: #f5f5f5;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 400;
    line-height: 1.45;
    white-space: pre-line;
    text-transform: none;
    letter-spacing: 0;
    z-index: 20;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.12s ease-in;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}
th[data-tip]:hover::after { opacity: 1; }
"""


COLUMN_TIPS = {
    "Run ID": "Timestamp-based ID (UTC) of the agent run. Click to see plan, steps, and score breakdown.",
    "Model": "LLM string passed to langchain init_chat_model, used for both the PlanningAgent and the ExecutionAgent.",
    "Rounds": "Number of plan → execute → re-plan feedback rounds actually completed.",
    "Status": "Final trace status: completed = clean exit, errored = uncaught exception, interrupted = SIGINT/timeout.",
    "Started": "UTC start time of the run (trace started_utc).",
    "Duration": "Total wall time from trace start to finish. '—' = the trace was never closed (still running, or the process died).",
    "Phase": (
        "Which stage this run is: dataset = Phase-1 sweep generation · surrogate = Phase-2 AutoML · "
        "meta = the self-improving Phase-1→Phase-2 loop · unknown = could not classify."
    ),
    "Result": (
        "Headline outcome, read per phase:\n\n"
        "meta: % error reduction vs the mean-predictor baseline on the frozen eval shard, "
        "iteration count, and the chain of actions taken (regen_dataset / extend_search / "
        "enrich_active / terminate).\n\n"
        "dataset / surrogate: the scorer's composite ∈ [0, 1] — ≥ 0.75 good · ≥ 0.50 mid · "
        "< 0.50 low · GATE FAIL = a hard gate missed.\n\n"
        "Open the run page for the full per-iteration table and decision timeline."
    ),
    "Workspace": "The workspace/ dir under examples/ where the agent wrote its deliverables.",
    "Parent": "For Phase-3 meta-runs: the outer meta-loop run ID that spawned this Phase-2 child. Blank for standalone runs.",
}


def _phase_of(score: dict, prompt_path: str) -> str:
    """Classify the run as 'dataset' (Phase-1) or 'surrogate' (Phase-2) by inspecting
    the score's hard-gate keys; fall back to the prompt filename."""
    gates = (score or {}).get("hard_gates") or {}
    if "winner_loads" in gates or "winner_predicts" in gates:
        return "surrogate"
    if "dataset_h5_opens" in gates or "at_least_one_success" in gates:
        return "dataset"
    name = Path(prompt_path or "").name
    if "dataset" in name:
        return "dataset"
    if "surrogate" in name or "automl" in name:
        return "surrogate"
    return "unknown"


def _write_index(
    runs: list[dict], out_dir: Path, since: Optional[_dt.datetime] = None
) -> None:
    rows = []
    for r in runs:
        trace = r["trace"]
        prompt = trace.get("prompt", {}) or {}
        score = trace.get("score") or {}
        cls, label = _fmt_score(score)
        workspace_str = str(prompt.get("workspace", ""))
        workspace_path = Path(workspace_str) if workspace_str else None
        meta = _meta_index_summary(workspace_path)
        phase = "meta" if meta else _phase_of(score, str(prompt.get("path", "")))
        n_rounds = len(trace.get("rounds") or [])
        status = trace.get("status") or "unknown"
        parent = trace.get("parent_run_id")
        parent_html = f'<a href="{html.escape(parent)}.html">{html.escape(parent)}</a>' if parent else ""
        phase_cell = f'<span class="phase phase-{phase}">{html.escape(phase)}</span>'
        started = _parse_utc(trace.get("started_utc") or "")
        started_cell = started.strftime("%Y-%m-%d %H:%M") if started else "—"
        duration_cell = _fmt_duration(
            trace.get("started_utc") or "", trace.get("finished_utc") or ""
        ) or "—"

        # One "Result" column that reads correctly for every run type:
        #   meta      → accuracy vs baseline + iteration count + action chain
        #   dataset/surrogate → the scorer's badge
        if meta:
            acc = meta["acc"]
            acc_html = (
                f'{_fmt_meta_pct(acc)} <span class="muted">vs baseline</span>'
                if acc is not None else '<span class="muted">no winner</span>'
            )
            # Compress consecutive repeats: [a, b, b, b] → "a → b ×3".
            groups: list[tuple[str, int]] = []
            for a in meta["actions"]:
                if groups and groups[-1][0] == a:
                    groups[-1] = (a, groups[-1][1] + 1)
                else:
                    groups.append((a, 1))
            actions = " → ".join(
                html.escape(a) + (f" ×{n}" if n > 1 else "") for a, n in groups
            ) or "—"
            result_cell = (
                f'<div>{acc_html} · {html.escape(str(meta["iters"]))} iters</div>'
                f'<div class="mono muted" style="font-size:11px">{actions}</div>'
            )
        elif phase in ("dataset", "surrogate"):
            result_cell = f'<span class="badge {cls}">{label}</span>'
        else:
            result_cell = '<span class="muted">—</span>'

        rows.append(
            f'<tr>'
            f'<td><a href="{html.escape(r["run_id"])}.html">{html.escape(r["run_id"])}</a></td>'
            f'<td>{phase_cell}</td>'
            f'<td class="mono">{html.escape(str(prompt.get("model", "")))}</td>'
            f'<td>{n_rounds}</td>'
            f'<td class="status-{html.escape(status)}">{html.escape(status)}</td>'
            f'<td class="mono muted">{started_cell}</td>'
            f'<td class="mono">{duration_cell}</td>'
            f'<td>{result_cell}</td>'
            f'<td class="mono muted">{html.escape(Path(workspace_str).name)}</td>'
            f'<td>{parent_html}</td>'
            f'</tr>'
        )
    def _th(name: str) -> str:
        tip = COLUMN_TIPS.get(name)
        if tip:
            return f'<th data-tip="{html.escape(tip)}">{html.escape(name)}</th>'
        return f'<th>{html.escape(name)}</th>'

    columns = ["Run ID", "Phase", "Model", "Rounds", "Status", "Started", "Duration", "Result", "Workspace", "Parent"]
    header = "".join(_th(c) for c in columns)
    if since is not None:
        cutoff_str = (
            str(since.date())
            if (since.hour, since.minute, since.second) == (0, 0, 0)
            else since.strftime("%Y-%m-%d %H:%M UTC")
        )
        since_note = (
            f' Showing runs since <span class="mono">{cutoff_str}</span>; '
            f'regenerate with <span class="mono">--since all</span> '
            f'for the full history.'
        )
    else:
        since_note = ""
    body = (
        f'<h1>autotokamak — agent runs</h1>'
        f'<p class="muted">Hover any column header for its definition. Regenerate: '
        f'<span class="mono">python tools/trace_to_html.py</span>.{since_note}</p>'
        f'<table>'
        f'<thead><tr>{header}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody>'
        f'</table>'
    )
    (out_dir / "index.html").write_text(
        f'<!doctype html><html><head><meta charset="utf-8"><title>autotokamak runs</title><style>{CSS}</style></head><body>{body}</body></html>'
    )


def _render_score(score: dict) -> str:
    if not score:
        return '<p class="muted">No score recorded.</p>'
    cls, label = _fmt_score(score)
    parts = [f'<h2>Score <span class="badge {cls}">{label}</span></h2>']
    gates = score.get("hard_gates") or {}
    if gates:
        gate_bits = []
        for k, v in gates.items():
            ok = "✓" if v else "✗"
            color = "#28a745" if v else "#d73a49"
            gate_bits.append(f'<span style="color:{color};margin-right:12px">{ok} {html.escape(k)}</span>')
        parts.append('<div class="card" style="padding:8px 12px">' + "".join(gate_bits) + "</div>")
    quality = score.get("quality") or {}
    details = score.get("details") or {}
    weights = (details.get("weights") if isinstance(details, dict) else None) or {}
    if quality:
        rows = []
        for k, v in quality.items():
            w = weights.get(k)
            frac = max(0.0, min(1.0, float(v)))
            width = int(frac * 100)
            wstr = f'w {float(w):.2f}' if isinstance(w, (int, float)) else ""
            rows.append(
                f'<div class="q-row">'
                f'<span class="q-name mono">{html.escape(k)}</span>'
                f'<span class="qbar"><span style="width:{width}%"></span></span>'
                f'<span class="q-val">{float(v):.3f}</span>'
                f'<span class="q-weight">{wstr}</span>'
                f'</div>'
            )
        parts.append('<div class="card">' + "".join(rows) + "</div>")
    return "".join(parts)


EVAL_PLOTS = [
    ("true_pred_residual.png",   "True vs. predicted ψ (per test sample)"),
    ("per_sample_rmse.png",      "Per-sample test RMSE vs. baseline"),
    ("pred_vs_true_scatter.png", "Predicted vs. true ψ, per pixel"),
    ("residual_histogram.png",   "Distribution of per-pixel residuals"),
    ("pca_variance.png",         "PCA cumulative explained variance"),
    ("pca_reconstruction.png",   "PCA-only vs. full-pipeline error"),
    ("optuna_history.png",       "Optuna best-value convergence"),
]


def _render_eval_plots(workspace: str | None, out_dir: Path) -> str:
    """If <workspace>/outputs/eval_plots/ has PNGs, copy them under out_dir/eval/<run_id-safe>
    and embed. Silent no-op when not present."""
    if not workspace:
        return ""
    ws = Path(workspace)
    src = ws / "outputs/eval_plots"
    if not src.is_dir():
        return ""
    # Namespace under out_dir by workspace name to avoid collisions across runs.
    ns = ws.name
    dst = out_dir / "eval" / ns
    dst.mkdir(parents=True, exist_ok=True)
    import shutil
    embedded: list[tuple[str, str]] = []
    for fname, title in EVAL_PLOTS:
        s = src / fname
        if s.is_file():
            try:
                shutil.copy(s, dst / fname)
                embedded.append((fname, title))
            except Exception:
                pass
    if not embedded:
        return ""
    figs = "".join(
        f'<figure class="eval-fig">'
        f'<a href="eval/{html.escape(ns)}/{html.escape(fname)}"><img src="eval/{html.escape(ns)}/{html.escape(fname)}" alt="{html.escape(title)}"></a>'
        f'<figcaption>{html.escape(title)}</figcaption>'
        f'</figure>'
        for fname, title in embedded
    )
    return (
        f'<h2>Surrogate evaluation plots</h2>'
        f'<p class="muted">Regenerate: '
        f'<span class="mono">python tools/eval_surrogate.py --workspace {html.escape(str(ws))}</span></p>'
        f'<div class="eval-grid">{figs}</div>'
    )


def _fmt_number(v, digits: int = 4) -> str:
    if v is None:
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return html.escape(str(v))
    if not (f == f):  # NaN
        return "—"
    if abs(f) >= 1000 or (0 < abs(f) < 0.001):
        return f"{f:.3e}"
    return f"{f:.{digits}f}"


def _load_json(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None


def _render_diagnostics_block(diag: dict) -> str:
    """Surface the human-readable `interpretation` string from each diagnostic
    section, plus the key numbers backing it."""
    if not diag:
        return ""
    rows: list[str] = []
    lc = diag.get("learning_curve") or {}
    if lc:
        curve = lc.get("curve") or {}
        pts = ", ".join(f"N={k}: {_fmt_number(v)}" for k, v in curve.items())
        slope = lc.get("slope_log_log")
        rows.append(
            f'<div class="diag-block"><span class="diag-label">learning curve</span>'
            f'{html.escape(lc.get("interpretation", ""))}'
            + (f'<div class="mono muted">{html.escape(pts)}'
               + (f' · slope log-log = {_fmt_number(slope, 3)}' if slope is not None else "")
               + '</div>' if pts else '')
            + '</div>'
        )
    cv = diag.get("cross_seed_variance") or {}
    if cv:
        rows.append(
            f'<div class="diag-block"><span class="diag-label">cross-seed variance</span>'
            f'{html.escape(cv.get("interpretation", ""))}'
            f'<div class="mono muted">mean RMSE = {_fmt_number(cv.get("mean_rmse"))} · cv = {_fmt_number(cv.get("cv"), 3)}</div>'
            f'</div>'
        )
    pca = diag.get("pca_spectrum") or {}
    if pca:
        cum = pca.get("cumulative") or []
        cum_txt = ""
        if cum:
            top = ", ".join(_fmt_number(x, 3) for x in cum[:6])
            cum_txt = f'<div class="mono muted">first-6 cumulative var = [{top}] · 95% at n = {pca.get("n_components_for_95pct") or "—"}</div>'
        rows.append(
            f'<div class="diag-block"><span class="diag-label">PCA spectrum</span>'
            f'{html.escape(pca.get("interpretation", ""))}'
            f'{cum_txt}'
            f'</div>'
        )
    return "".join(rows)


def _render_action_block(action: dict) -> str:
    """Render the ActionDecision JSON as: badge + diagnosis + rationale + key knobs."""
    if not action:
        return ""
    kind = action.get("action") or "?"
    badge_cls = {"regen_dataset": "act-regen", "extend_search": "act-extend", "terminate": "act-terminate"}.get(kind, "")
    parts = [f'<div><b>action</b> <span class="act-badge {badge_cls}">{html.escape(kind)}</span></div>']
    diagnosis = action.get("diagnosis") or ""
    if diagnosis:
        parts.append(
            f'<div class="diag-block" style="border-left-color:#a10000">'
            f'<span class="diag-label">diagnosis</span>{html.escape(diagnosis)}</div>'
        )
    payload = action.get(
        {"regen_dataset": "regen", "extend_search": "extend", "terminate": "terminate"}.get(kind, "")
    ) or {}
    rationale = payload.get("rationale") or payload.get("reason") or ""
    if rationale:
        parts.append(
            f'<div class="diag-block" style="border-left-color:#28a745">'
            f'<span class="diag-label">rationale</span>{html.escape(rationale)}</div>'
        )
    knobs: list[str] = []
    if kind == "extend_search":
        emph = payload.get("models_to_emphasize") or []
        widen = payload.get("widen_params") or []
        n_trials = payload.get("n_trials_hint")
        if emph:
            knobs.append(f'<li><b>models to emphasize</b> {", ".join(html.escape(str(m)) for m in emph)}</li>')
        if widen:
            knobs.append(f'<li><b>widen params</b> {", ".join(html.escape(str(w)) for w in widen)}</li>')
        if n_trials is not None:
            knobs.append(f'<li><b>n_trials hint</b> {n_trials}</li>')
    elif kind == "regen_dataset":
        overrides = payload.get("overrides") or {}
        for k, v in overrides.items():
            knobs.append(f'<li><b>{html.escape(str(k))}</b> {html.escape(str(v))}</li>')
    elif kind == "terminate":
        conf = payload.get("confidence")
        if conf:
            knobs.append(f'<li><b>confidence</b> {html.escape(str(conf))}</li>')
    if knobs:
        parts.append(f'<ul class="kv-list">{"".join(knobs)}</ul>')
    return "".join(parts)


def _render_meta_iterations(workspace: Path | None) -> str:
    """Render each meta-loop iteration as a card: diagnosis, action, rationale,
    diagnostics interpretations, RMSE-after. Silent no-op if not a meta run."""
    if workspace is None:
        return ""
    iter_root = workspace / "iterations"
    if not iter_root.is_dir():
        return ""
    iter_dirs = sorted(p for p in iter_root.iterdir() if p.is_dir())
    if not iter_dirs:
        return ""
    cards: list[str] = []
    for iter_dir in iter_dirs:
        action = _load_json(iter_dir / "action.json") or {}
        diag = _load_json(iter_dir / "diagnostics.json") or {}
        result = _load_json(iter_dir / "result.json") or {}
        try:
            idx = int(iter_dir.name)
        except ValueError:
            idx = iter_dir.name
        rmse_after = result.get("rmse_after") if isinstance(result, dict) else None
        header_meta_parts = []
        if rmse_after is not None:
            header_meta_parts.append(f"RMSE after = {_fmt_number(rmse_after)}")
        kind = result.get("kind") if isinstance(result, dict) else None
        if kind and kind != "terminate":
            header_meta_parts.append(f"result: {kind}")
        header_meta = " · ".join(header_meta_parts)
        cards.append(
            f'<div class="iter-card">'
            f'  <div class="iter-head"><b>Iteration {idx}</b>'
            f'    <span class="muted mono">{html.escape(header_meta)}</span></div>'
            f'  <div class="iter-body">'
            f'    {_render_action_block(action)}'
            f'    <details><summary>Diagnostics that led to this decision</summary>{_render_diagnostics_block(diag)}</details>'
            f'  </div>'
            f'</div>'
        )
    return (
        f'<h2>Agent decision timeline</h2>'
        f'<p class="muted">Each iteration: what the deterministic diagnostics reported, what the LLM decided, and why.</p>'
        + "".join(cards)
    )


def _meta_pct(new, old) -> Optional[float]:
    """Error-reduction %: positive means ``new`` is better (lower). None if not computable."""
    try:
        new_f = float(new)
        old_f = float(old)
    except (TypeError, ValueError):
        return None
    if not (old_f > 0) or new_f != new_f or old_f != old_f:  # reject NaN / non-positive base
        return None
    return 100.0 * (old_f - new_f) / old_f


def _fmt_meta_pct(p: Optional[float]) -> str:
    if p is None:
        return '<span class="muted">—</span>'
    cls = "pos" if p >= 0 else "neg"
    return f'<span class="{cls}">{p:+.1f}%</span>'


def _meta_outcome(result) -> str:
    """One-line human summary of a meta action's result dict (mirrors the terminal)."""
    if not isinstance(result, dict) or not result:
        return "—"
    if result.get("error"):
        return f"ERROR: {result['error']}"
    kind = result.get("kind", "")
    if kind in ("regen_dataset", "enrich_active"):
        noun = "targeted samples" if kind == "enrich_active" else "samples"
        return (
            f"+{result.get('n_new_succeeded', '?')}/{result.get('n_new_requested', '?')} "
            f"{noun} → pool {result.get('n_total_succeeded', '?')}"
        )
    if kind == "extend_search":
        parts = []
        if result.get("n_rounds") is not None:
            parts.append(f"{result['n_rounds']} rounds")
        if result.get("elapsed_seconds") is not None:
            try:
                parts.append(f"{float(result['elapsed_seconds']):.0f}s")
            except (TypeError, ValueError):
                pass
        if result.get("shard_rmse") is not None:
            parts.append(f"shard RMSE {_fmt_number(result['shard_rmse'])}")
        models = result.get("models_emphasized") or []
        if models:
            parts.append("models: " + ",".join(map(str, models)))
        return "search: " + ", ".join(parts) if parts else "search"
    if kind == "terminate":
        return f"terminate: {result.get('reason') or '(no reason given)'}"
    return kind or "—"


def _meta_records(workspace: Path) -> tuple[list, dict]:
    """Return (iterations, report) from meta_trace.json; fall back to report.json."""
    mt = _load_json(workspace / "meta_trace.json")
    if isinstance(mt, dict) and mt.get("iterations"):
        report = mt.get("report") or _load_json(workspace / "report.json") or {}
        return list(mt.get("iterations") or []), report
    return [], (_load_json(workspace / "report.json") or {})


def _meta_index_summary(workspace: Path | None) -> Optional[dict]:
    """Headline meta metrics for an index row, or None if this isn't a meta run."""
    if workspace is None:
        return None
    iterations, report = _meta_records(workspace)
    is_meta = bool(iterations) or bool(report.get("actions_taken")) or ("n_iterations" in report)
    if not is_meta:
        return None
    acc = report.get("final_accuracy_pct")
    if acc is None:
        acc = _meta_pct(report.get("final_rmse"), report.get("baseline_rmse"))
    actions = report.get("actions_taken") or [
        (rec.get("decision") or {}).get("action")
        for rec in iterations
        if isinstance(rec, dict)
    ]
    return {
        "iters": report.get("n_iterations", len(iterations)),
        "acc": acc,
        "actions": [a for a in actions if a],
        "terminated_by": report.get("terminated_by"),
    }


def _render_meta_iteration_table(workspace: Path | None) -> str:
    """Compact per-iteration table: action, outcome, RMSE, Δ vs prev, vs baseline.

    Mirrors the terminal ``=== META SUMMARY ===`` table so the HTML report and
    the live terminal tell the same story. Silent no-op if not a meta run.
    """
    if workspace is None:
        return ""
    iterations, report = _meta_records(workspace)
    if not iterations:
        return ""
    baseline = report.get("baseline_rmse")
    rows: list[str] = []
    prev_rmse = None
    for rec in iterations:
        if not isinstance(rec, dict):
            continue
        it = rec.get("iteration", "?")
        action = (rec.get("decision") or {}).get("action", "—")
        outcome = _meta_outcome(rec.get("result"))
        rmse = rec.get("rmse_after")
        rmse_txt = _fmt_number(rmse) if rmse is not None else '<span class="muted">—</span>'
        rows.append(
            f'<tr><td>{html.escape(str(it))}</td>'
            f'<td class="mono">{html.escape(str(action))}</td>'
            f'<td>{html.escape(outcome)}</td>'
            f'<td class="mono">{rmse_txt}</td>'
            f'<td>{_fmt_meta_pct(_meta_pct(rmse, prev_rmse))}</td>'
            f'<td>{_fmt_meta_pct(_meta_pct(rmse, baseline))}</td></tr>'
        )
        if rmse is not None:
            prev_rmse = rmse
    return (
        '<h2>Per-iteration summary</h2>'
        '<p class="muted">Δ vs prev / vs baseline = % error reduction on the frozen eval '
        'shard (positive = improvement). Same numbers as the terminal META SUMMARY table.</p>'
        '<table><thead><tr>'
        '<th>Iter</th><th>Action</th><th>Outcome</th><th>RMSE</th>'
        '<th>Δ vs prev</th><th>vs baseline</th>'
        '</tr></thead><tbody>' + "".join(rows) + '</tbody></table>'
    )


def _render_meta_report(workspace: Path | None) -> str:
    """Render the MetaReport (final RMSE, winner model, actions taken, RMSE history)."""
    if workspace is None:
        return ""
    report = _load_json(workspace / "report.json")
    if not report or "winner_model_name" not in report:
        return ""
    winner = report.get("winner_model_name", "—")
    hp = report.get("winner_hyperparams") or {}
    hp_items = "".join(
        f'<li><b>{html.escape(str(k))}</b> {html.escape(str(v))}</li>'
        for k, v in hp.items()
    )
    hp_block = f'<details><summary>Winner hyperparameters</summary><ul class="kv-list">{hp_items}</ul></details>' if hp_items else ""
    baseline = report.get("baseline_rmse")
    initial = report.get("initial_rmse")
    final = report.get("final_rmse")
    improvement_txt = ""
    if isinstance(baseline, (int, float)) and isinstance(final, (int, float)) and baseline > 0:
        improvement_txt = f' · improvement vs baseline: {(baseline - final) / baseline * 100:+.1f}%'
    rmse_hist = report.get("rmse_history") or []
    rmse_hist_txt = ", ".join(_fmt_number(x) for x in rmse_hist) if rmse_hist else "(none)"
    actions_taken = report.get("actions_taken") or []
    actions_txt = " → ".join(html.escape(a) for a in actions_taken) if actions_taken else "(none)"
    terminated_by = report.get("terminated_by", "?")
    return (
        f'<h2>Meta-loop summary</h2>'
        f'<div class="card">'
        f'<div><b>Winner model:</b> <span class="mono">{html.escape(str(winner))}</span></div>'
        f'<div><b>Final RMSE:</b> {_fmt_number(final)}'
        f' · <b>baseline (mean-predictor):</b> {_fmt_number(baseline)}'
        f' · <b>initial:</b> {_fmt_number(initial)}{improvement_txt}</div>'
        f'<div><b>RMSE history:</b> <span class="mono">[{html.escape(rmse_hist_txt)}]</span></div>'
        f'<div><b>Actions taken:</b> <span class="mono">{actions_txt}</span></div>'
        f'<div><b>Terminated by:</b> <span class="mono">{html.escape(str(terminated_by))}</span></div>'
        f'{hp_block}'
        f'</div>'
    )


def _render_surrogate_leaderboard(workspace: Path | None) -> str:
    """Render the SurrogateReport (winner + per-model best RMSE, if outputs/report.json exists)."""
    if workspace is None:
        return ""
    surro = _load_json(workspace / "outputs" / "report.json")
    if not surro or "winner_model_name" not in surro:
        return ""
    winner = surro.get("winner_model_name", "—")
    models_tried = surro.get("models_tried") or []
    per_model = surro.get("per_model") or {}
    rows_html: list[str] = []
    if per_model:
        rows_sorted = sorted(
            per_model.items(),
            key=lambda kv: (kv[1].get("best_value", float("inf")) if isinstance(kv[1], dict) else float("inf")),
        )
        for name, m in rows_sorted:
            if not isinstance(m, dict):
                continue
            is_winner = (name == winner)
            best = m.get("best_value")
            best25 = m.get("best_value_at_25pct_trials")
            edge = m.get("edge_hit")
            n_trials = m.get("n_trials")
            row_cls = " class=\"winner\"" if is_winner else ""
            winner_tag = ' <span class="winner-tag">winner</span>' if is_winner else ""
            edge_html = ('<span style="color:#a10000">edge-hit</span>' if edge else '<span style="color:#28a745">clean</span>')
            rows_html.append(
                f'<tr{row_cls}><td class="mono">{html.escape(str(name))}{winner_tag}</td>'
                f'<td class="mono">{_fmt_number(best)}</td>'
                f'<td class="mono">{_fmt_number(best25)}</td>'
                f'<td>{n_trials if n_trials is not None else "—"}</td>'
                f'<td>{edge_html}</td></tr>'
            )
    elif models_tried:
        for name in models_tried:
            is_winner = (name == winner)
            row_cls = " class=\"winner\"" if is_winner else ""
            winner_tag = ' <span class="winner-tag">winner</span>' if is_winner else ""
            rows_html.append(
                f'<tr{row_cls}><td class="mono">{html.escape(str(name))}{winner_tag}</td>'
                f'<td class="mono">—</td><td class="mono">—</td><td>—</td><td>—</td></tr>'
            )
    table = ""
    if rows_html:
        table = (
            f'<table class="leaderboard"><thead><tr>'
            f'<th>Model</th><th>Best psi_rmse</th><th>@25% trials</th><th>Trials</th><th>Search edge</th>'
            f'</tr></thead><tbody>{"".join(rows_html)}</tbody></table>'
        )
    hp = surro.get("winner_hyperparams") or {}
    hp_items = "".join(
        f'<li><b>{html.escape(str(k))}</b> {html.escape(str(v))}</li>'
        for k, v in hp.items()
    )
    hp_block = f'<details><summary>Winner hyperparameters</summary><ul class="kv-list">{hp_items}</ul></details>' if hp_items else ""
    val = surro.get("val_psi_rmse")
    test = surro.get("test_psi_rmse")
    baseline = surro.get("baseline_mean_psi_rmse")
    n_pca = surro.get("pca_n_components")
    exp_var = surro.get("pca_explained_var")
    n_rounds = surro.get("n_outer_rounds")
    n_trials = surro.get("n_total_trials")
    return (
        f'<h2>Surrogate leaderboard</h2>'
        f'<div class="card">'
        f'<div><b>Winner:</b> <span class="mono">{html.escape(str(winner))}</span> — '
        f'test RMSE = {_fmt_number(test)} · val RMSE = {_fmt_number(val)} · '
        f'baseline (mean predictor) = {_fmt_number(baseline)}</div>'
        f'<div><b>PCA:</b> {n_pca if n_pca is not None else "—"} components, '
        f'{_fmt_number(exp_var, 3) if exp_var is not None else "—"} cumulative variance</div>'
        f'<div><b>Search:</b> {n_rounds if n_rounds is not None else "?"} outer rounds · '
        f'{n_trials if n_trials is not None else "?"} total trials</div>'
        f'{hp_block}'
        f'</div>'
        f'{table}'
    )


PHYSICS_PLOTS = [
    ("psi_samples.png", "Sample ψ(R,Z) fields with LCFS shapes"),
    ("param_distributions.png", "Input parameter distributions (r0, a, kappa, delta, Ip)"),
]


def _render_physics_plots(workspace: Path | None, out_dir: Path) -> str:
    """Embed physics visualizations (ψ contours + LCFS + parameter distributions).
    Generated by tools/render_physics.py; silent no-op if not present.
    Also probes surrogate_meta workspaces which point to a dataset elsewhere.
    """
    if workspace is None:
        return ""
    candidates: list[Path] = [workspace / "outputs" / "physics_plots"]
    # For meta-loop workspaces, physics viz is generated against the dataset
    # symlinked into surrogate_runs/iterN — check the first one we find.
    sur_runs = workspace / "surrogate_runs"
    if sur_runs.is_dir():
        for iter_dir in sorted(sur_runs.iterdir()):
            candidates.append(iter_dir / "outputs" / "physics_plots")
    src = next((c for c in candidates if c.is_dir()), None)
    if src is None:
        return ""
    ns = workspace.name
    dst = out_dir / "physics" / ns
    dst.mkdir(parents=True, exist_ok=True)
    import shutil
    embedded: list[tuple[str, str]] = []
    for fname, title in PHYSICS_PLOTS:
        s = src / fname
        if s.is_file():
            try:
                shutil.copy(s, dst / fname)
                embedded.append((fname, title))
            except Exception:  # noqa: BLE001
                pass
    if not embedded:
        return ""
    figs = "".join(
        f'<figure class="physics-fig">'
        f'<a href="physics/{html.escape(ns)}/{html.escape(fname)}"><img src="physics/{html.escape(ns)}/{html.escape(fname)}" alt="{html.escape(title)}"></a>'
        f'<figcaption>{html.escape(title)}</figcaption>'
        f'</figure>'
        for fname, title in embedded
    )
    return (
        f'<h2>Physics visualizations</h2>'
        f'<p class="muted">Sample equilibria from the dataset — the ψ(R,Z) fields the surrogate is trying to predict. '
        f'Regenerate: <span class="mono">python tools/render_physics.py --workspace {html.escape(str(workspace))}</span></p>'
        f'<div class="physics-grid">{figs}</div>'
    )


META_PLOTS = [
    ("convergence.png",   "Meta-loop convergence — frozen-shard RMSE vs iteration, by action"),
    ("per_cell_rmse.png", "Per-geometry-cell error — where the surrogate is weakest"),
]


def _render_meta_plots(workspace: Path | None, out_dir: Path) -> str:
    """Embed meta-loop plots (convergence curve + per-cell error).
    Generated by tools/render_meta_plots.py; silent no-op if not a meta run."""
    if workspace is None:
        return ""
    src = workspace / "outputs" / "meta_plots"
    if not src.is_dir():
        return ""
    ns = workspace.name
    dst = out_dir / "meta" / ns
    dst.mkdir(parents=True, exist_ok=True)
    import shutil
    embedded: list[tuple[str, str]] = []
    for fname, title in META_PLOTS:
        s = src / fname
        if s.is_file():
            try:
                shutil.copy(s, dst / fname)
                embedded.append((fname, title))
            except Exception:  # noqa: BLE001
                pass
    if not embedded:
        return ""
    figs = "".join(
        f'<figure class="physics-fig">'
        f'<a href="meta/{html.escape(ns)}/{html.escape(fname)}"><img src="meta/{html.escape(ns)}/{html.escape(fname)}" alt="{html.escape(title)}"></a>'
        f'<figcaption>{html.escape(title)}</figcaption>'
        f'</figure>'
        for fname, title in embedded
    )
    return (
        f'<h2>Meta-loop plots</h2>'
        f'<p class="muted">How the loop improved the surrogate over its iterations. '
        f'Regenerate: <span class="mono">python tools/render_meta_plots.py --workspace {html.escape(str(workspace))}</span></p>'
        f'<div class="physics-grid">{figs}</div>'
    )


def _render_dataset_provenance(workspace: Path | None) -> str:
    """Read (or cache-read) dataset.h5 metadata: N samples, grid, per-input ranges,
    source path. Written to <workspace>/outputs/dataset_provenance.json so we don't
    re-open the h5 on every report regen."""
    if workspace is None:
        return ""
    cache = workspace / "outputs" / "dataset_provenance.json"
    prov = _load_json(cache)
    if not prov:
        prov = _build_dataset_provenance(workspace)
        if prov:
            cache.parent.mkdir(parents=True, exist_ok=True)
            try:
                cache.write_text(json.dumps(prov, indent=2, default=str))
            except Exception:  # noqa: BLE001
                pass
    if not prov:
        return ""
    n = prov.get("n_samples", "?")
    grid = prov.get("grid_shape", [])
    grid_txt = f"{grid[0]}×{grid[1]}" if len(grid) == 2 else "?"
    src = prov.get("source_path", "?")
    ranges = prov.get("param_ranges") or {}
    range_rows = "".join(
        f'<li><b>{html.escape(str(name))}</b> '
        f'[{_fmt_number(vals.get("min"))}, {_fmt_number(vals.get("max"))}] '
        f'mean={_fmt_number(vals.get("mean"))} '
        f'</li>'
        for name, vals in ranges.items()
    )
    return (
        f'<h2>Dataset provenance</h2>'
        f'<div class="card">'
        f'<div><b>Source:</b> <span class="mono">{html.escape(str(src))}</span></div>'
        f'<div><b>N successful samples:</b> {n} · <b>grid (nz × nr):</b> {grid_txt}</div>'
        f'<div><b>Input parameter ranges:</b><ul class="kv-list">{range_rows}</ul></div>'
        f'</div>'
    )


def _build_dataset_provenance(workspace: Path) -> dict | None:
    """Locate dataset.h5 for a workspace and extract metadata. Returns None on failure."""
    import sys as _sys
    _sys.path.insert(0, str(REPO_ROOT / "src"))
    try:
        from autotokamak.surrogate.dataset import PARAM_ORDER, load_dataset
    except Exception:  # noqa: BLE001
        return None
    candidates = [workspace / "dataset.h5", workspace / "outputs" / "dataset.h5"]
    sur_runs = workspace / "surrogate_runs"
    if sur_runs.is_dir():
        for iter_dir in sorted(sur_runs.iterdir()):
            candidates.append(iter_dir / "dataset.h5")
    dataset_path = next((p for p in candidates if p.exists()), None)
    if dataset_path is None:
        return None
    try:
        bundle = load_dataset(dataset_path)
    except Exception:  # noqa: BLE001
        return None
    ranges = {}
    for i, name in enumerate(PARAM_ORDER):
        col = bundle.inputs[:, i]
        ranges[name] = {
            "min": float(col.min()),
            "max": float(col.max()),
            "mean": float(col.mean()),
        }
    return {
        "source_path": str(dataset_path),
        "n_samples": int(bundle.n_samples),
        "grid_shape": list(bundle.grid_shape),
        "param_ranges": ranges,
    }


def _count_optuna_trials(workspace: Path | None) -> int | None:
    """Count total Optuna trials across all surrogate_runs/iter*/outputs/study.db."""
    if workspace is None:
        return None
    total = 0
    found_any = False
    import sqlite3
    db_paths = list((workspace / "surrogate_runs").glob("*/outputs/study.db")) if (workspace / "surrogate_runs").is_dir() else []
    db_paths += list(workspace.glob("outputs/study.db"))
    for db in db_paths:
        try:
            with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as con:
                n = con.execute("SELECT COUNT(*) FROM trials").fetchone()[0]
            total += int(n)
            found_any = True
        except Exception:  # noqa: BLE001
            continue
    return total if found_any else None


def _render_compute_cost(trace: dict, workspace: Path | None) -> str:
    """Wall-clock, trial count, iteration count, sub-run count."""
    started, finished = trace.get("started_utc"), trace.get("finished_utc")
    wall = _fmt_duration(started or "", finished or "") if started and finished else "—"
    if wall == "—" and trace.get("steps"):
        # just_dspy traces: no finished_utc, but per-step timings exist.
        total_s = sum(float(s.get("seconds") or 0.0) for s in trace["steps"] if isinstance(s, dict))
        if total_s:
            wall = f"{total_s/3600:.1f} h (agent step time)"
    iter_root = (workspace / "iterations") if workspace else None
    n_iters = len(list(iter_root.iterdir())) if iter_root and iter_root.is_dir() else 0
    sur_root = (workspace / "surrogate_runs") if workspace else None
    n_sub = len(list(sur_root.iterdir())) if sur_root and sur_root.is_dir() else 0
    n_trials = _count_optuna_trials(workspace)
    if not (wall or n_iters or n_sub or n_trials):
        return ""
    parts = [f'<div><b>Wall-clock:</b> {html.escape(wall)}</div>']
    if n_iters:
        parts.append(f'<div><b>Meta iterations:</b> {n_iters}</div>')
    if n_sub:
        parts.append(f'<div><b>Nested Phase-2 sub-runs:</b> {n_sub}</div>')
    if n_trials is not None:
        parts.append(f'<div><b>Total Optuna trials:</b> {n_trials}</div>')
    model = (trace.get("prompt") or {}).get("model")
    if model:
        parts.append(f'<div><b>LLM:</b> <span class="mono">{html.escape(str(model))}</span></div>')
    return f'<h2>Compute cost</h2><div class="card">{"".join(parts)}</div>'


def _render_cross_run_comparison(current_run_id: str, current_trace: dict, all_runs: list[dict]) -> str:
    """Table of prior runs using the same prompt, so the reader can see whether
    the agent is trending better across attempts."""
    prompt_path = (current_trace.get("prompt") or {}).get("path")
    if not prompt_path:
        return ""
    peers = [
        r for r in all_runs
        if (r["trace"].get("prompt") or {}).get("path") == prompt_path
    ]
    if len(peers) < 2:
        return ""
    # Sort oldest → newest so improvement direction is left-to-right.
    peers = sorted(peers, key=lambda r: r["trace"].get("started_utc", ""))
    rows = []
    for r in peers:
        rid = r["run_id"]
        rt = r["trace"]
        score = (rt.get("score") or {}).get("total")
        # For meta runs, final rmse is in workspace/report.json — read defensively.
        ws = (rt.get("prompt") or {}).get("workspace")
        report = _load_json(Path(ws) / "report.json") if ws else None
        winner = (report or {}).get("winner_model_name") if isinstance(report, dict) else None
        final = (report or {}).get("final_rmse") if isinstance(report, dict) else None
        actions = (report or {}).get("actions_taken") if isinstance(report, dict) else None
        actions_txt = "→".join(a[:1].upper() for a in (actions or []))  # E for extend, R for regen, T for terminate
        status = rt.get("status", "?")
        is_cur = (rid == current_run_id)
        row_cls = ' class="winner"' if is_cur else ''
        cur_tag = ' <span class="winner-tag">this run</span>' if is_cur else ''
        rows.append(
            f'<tr{row_cls}><td class="mono"><a href="{html.escape(rid)}.html">{html.escape(rid)}</a>{cur_tag}</td>'
            f'<td class="mono">{_fmt_number(score, 3) if score is not None else "—"}</td>'
            f'<td class="mono">{_fmt_number(final)}</td>'
            f'<td class="mono">{html.escape(str(winner or "—"))}</td>'
            f'<td class="mono">{html.escape(actions_txt) or "—"}</td>'
            f'<td class="mono">{html.escape(status)}</td></tr>'
        )
    return (
        f'<h2>Cross-run comparison</h2>'
        f'<p class="muted">All prior runs of the same prompt (<span class="mono">{html.escape(Path(prompt_path).name)}</span>), '
        f'oldest → newest. Actions abbreviated: E=extend_search, R=regen_dataset, T=terminate.</p>'
        f'<table class="leaderboard"><thead><tr>'
        f'<th>Run</th><th>Score</th><th>Final RMSE</th><th>Winner</th><th>Actions</th><th>Status</th>'
        f'</tr></thead><tbody>{"".join(rows)}</tbody></table>'
    )


def _try_run(cmd: list[str], cwd: Path) -> None:
    """Best-effort subprocess. Failures are non-fatal — the report page still
    renders, just without the missing plots."""
    import subprocess
    try:
        subprocess.run(cmd, cwd=str(cwd), check=False, timeout=180,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:  # noqa: BLE001
        pass


def _ensure_plots(workspace: Path | None) -> None:
    """Run render_physics + eval_surrogate as needed. Silent on failure."""
    if workspace is None or not workspace.is_dir():
        return
    physics_out = workspace / "outputs" / "physics_plots" / "psi_samples.png"
    sur_runs = workspace / "surrogate_runs"
    if sur_runs.is_dir():
        # Meta workspace: physics viz may also land under the sub-run's outputs
        physics_out_alt = next(
            (d / "outputs" / "physics_plots" / "psi_samples.png"
             for d in sorted(sur_runs.iterdir())),
            None,
        )
    else:
        physics_out_alt = None
    if not physics_out.exists() and not (physics_out_alt and physics_out_alt.exists()):
        _try_run(
            [sys.executable, str(REPO_ROOT / "tools" / "render_physics.py"),
             "--workspace", str(workspace)],
            cwd=REPO_ROOT,
        )
    # eval_surrogate needs a trained winner.pkl + dataset.h5; skip if absent.
    if (workspace / "outputs" / "winner.pkl").exists() or (workspace / "winner.pkl").exists():
        eval_marker = workspace / "outputs" / "eval_plots" / "true_pred_residual.png"
        if not eval_marker.exists():
            _try_run(
                [sys.executable, str(REPO_ROOT / "tools" / "eval_surrogate.py"),
                 "--workspace", str(workspace)],
                cwd=REPO_ROOT,
            )
    # Meta-loop plots (convergence curve + per-cell error) — only meaningful when
    # the workspace has a meta trace; the generator no-ops otherwise.
    if (workspace / "meta_trace.json").exists() or (workspace / "report.json").exists():
        meta_marker = workspace / "outputs" / "meta_plots" / "convergence.png"
        if not meta_marker.exists():
            _try_run(
                [sys.executable, str(REPO_ROOT / "tools" / "render_meta_plots.py"),
                 "--workspace", str(workspace)],
                cwd=REPO_ROOT,
            )


def _render_step_body(step: dict, raw: str | None) -> str:
    parts: list[str] = []
    if raw:
        facts = _extract_step_facts(raw)
        if facts["files_written"]:
            items = "".join(
                f'<li class="mono">{html.escape(f)}</li>' for f in facts["files_written"]
            )
            parts.append(f'<div><b>Files written ({len(facts["files_written"])})</b><ul>{items}</ul></div>')
        if facts["commands"]:
            items = "".join(
                f'<li class="mono">{html.escape(c)}</li>' for c in facts["commands"]
            )
            parts.append(f'<div><b>Commands run ({len(facts["commands"])})</b><ul>{items}</ul></div>')
        if facts["tracebacks"]:
            for tb in facts["tracebacks"][:3]:
                parts.append(
                    f'<div><b style="color:#a10000">Traceback</b>'
                    f'<pre style="background:#fff5f5;color:#7a1a1a">{html.escape(tb[-3000:])}</pre></div>'
                )
        # The full transcript itself, collapsed by default.
        raw_shown = raw if len(raw) < 200_000 else (raw[:100_000] + "\n\n… [truncated, " + str(len(raw)) + " chars total] …\n\n" + raw[-100_000:])
        parts.append(
            f'<details><summary>Raw transcript ({len(raw)} chars)</summary>'
            f'<pre>{html.escape(raw_shown)}</pre></details>'
        )
    excerpt = step.get("result_excerpt") or ""
    if excerpt:
        parts.append(
            f'<details><summary>URSA post-step summary</summary>'
            f'<pre>{html.escape(excerpt[:6000])}</pre></details>'
        )
    err = step.get("error") or ""
    if err:
        parts.append(
            f'<pre style="background:#fff5f5;color:#7a1a1a">{html.escape(str(err)[:2000])}</pre>'
        )
    return "".join(parts) if parts else '<p class="muted">No captured content for this step.</p>'


def _render_round(rnd: dict, log_slices: dict[int, str] | None = None) -> str:
    plan_steps = rnd.get("plan_steps") or []
    execution = rnd.get("execution") or []
    plan_html = "".join(
        f'<li><b>{html.escape(str(ps.get("name", "?")))}</b><br><span class="muted">{html.escape(str(ps.get("description", "")))}</span></li>'
        for ps in plan_steps
    )
    log_slices = log_slices or {}
    step_html = []
    for step in execution:
        ok = step.get("ok")
        klass = "ok" if ok else "err"
        badge = "✓" if ok else "✗"
        dur = _fmt_duration(step.get("started_utc", ""), step.get("finished_utc", ""))
        step_no = step.get("step")
        raw = log_slices.get(step_no)
        body = _render_step_body(step, raw)
        step_html.append(
            f'<div class="step {klass}">'
            f'<div class="step-head"><span>{badge} step {step_no}: {html.escape(str(step.get("name", "")))}</span>'
            f'<span class="muted mono">{dur}</span></div>'
            f'<div class="step-body">{body}</div>'
            f'</div>'
        )
    return (
        f'<details open><summary><b>Round {rnd.get("round", "?")}</b> — '
        f'{len(plan_steps)} planned steps, {len(execution)} executed'
        f'</summary>'
        f'<details><summary>Plan</summary><ol>{plan_html}</ol></details>'
        f'<h2>Execution</h2>{"".join(step_html)}'
        f'</details>'
    )


def _render_capability_report(workspace: Path | None, trace: dict | None = None) -> str:
    """Summary card for a capability-test run's workspace/report.json
    (just_ursa / just_dspy schema: solve counts + rel-L2 metrics)."""
    if workspace is None:
        return ""
    rpt_path = workspace / "report.json"
    rpt = _load_json(rpt_path)
    if not rpt or "n_solves_attempted" not in rpt:
        return ""
    # Workspaces are reused across rounds: only show the report if it was
    # written during THIS run's window, else we'd attribute a later round's
    # results to an earlier trace.
    started = _parse_utc((trace or {}).get("started_utc", "") or "")
    if started is not None:
        mtime = _dt.datetime.fromtimestamp(rpt_path.stat().st_mtime, tz=_dt.timezone.utc)
        finished = _parse_utc((trace or {}).get("finished_utc", "") or "")
        window_end = (finished or mtime) + _dt.timedelta(minutes=5)
        if not (started <= mtime <= window_end):
            return ""
    rows = [
        f'<div><b>Solves:</b> {rpt.get("n_solves_succeeded", "?")} succeeded / '
        f'{rpt.get("n_solves_attempted", "?")} attempted</div>',
        f'<div><b>Train / test samples:</b> {rpt.get("n_train", "?")} / {rpt.get("n_test", "?")}</div>',
    ]
    metrics = rpt.get("metrics") or {}
    model_m = (metrics.get("test_rel_l2") or {}).get("mean")
    base_m = (metrics.get("baseline_rel_l2") or {}).get("mean")
    if isinstance(model_m, (int, float)) and isinstance(base_m, (int, float)) and base_m:
        reduction = 100.0 * (1.0 - model_m / base_m)
        rows.append(
            f'<div><b>Test rel-L2:</b> {model_m:.4f} vs baseline {base_m:.4f} '
            f'→ <b>{reduction:.1f}% error reduction</b></div>'
        )
    strat = rpt.get("sampling_strategy")
    if strat:
        rows.append(f'<div><b>Sampling:</b> {html.escape(str(strat)[:300])}</div>')
    return f'<h2>Campaign report</h2><div class="card">{"".join(rows)}</div>'


def _render_dspy_run(t: dict) -> str:
    """Render a just_dspy trace (plan/steps/reviews schema, no URSA rounds)."""
    parts: list[str] = []
    plan = t.get("plan") or []
    if plan:
        items = "".join(f"<li>{html.escape(str(p))}</li>" for p in plan)
        parts.append(f"<h2>Plan ({len(plan)} steps)</h2><ol>{items}</ol>")
    steps = t.get("steps") or []
    if steps:
        rows = []
        total_s = 0.0
        for s in steps:
            ok = s.get("ok")
            mark, color = ("✓", "#28a745") if ok else ("✗", "#d73a49")
            secs = float(s.get("seconds") or 0.0)
            total_s += secs
            step_text = html.escape(str(s.get("step", "")))
            summary = html.escape(str(s.get("summary", "")))
            rows.append(
                f'<details><summary><span style="color:{color}">{mark}</span> '
                f'<b>Step {s.get("index", "?")}</b> '
                f'<span class="muted">({secs:.0f}s)</span> '
                f'{step_text[:120]}…</summary>'
                f'<p class="muted"><b>Task:</b> {step_text}</p>'
                f'<p><b>Agent summary:</b> {summary}</p></details>'
            )
        parts.append(
            f"<h2>Execution ({len(steps)} steps, {total_s/60:.0f} min agent time, "
            f"{t.get('n_tool_calls', '?')} tool calls)</h2>" + "".join(rows)
        )
    reviews = t.get("reviews") or []
    if reviews:
        rows = []
        for r in reviews:
            complete = r.get("complete")
            mark, color = ("✓ complete", "#28a745") if complete else ("✗ incomplete", "#d73a49")
            missing = r.get("missing") or []
            missing_html = (
                "<ul>" + "".join(f"<li>{html.escape(str(m))}</li>" for m in missing) + "</ul>"
                if missing else ""
            )
            rows.append(
                f'<div class="card"><b>Review round {r.get("round", "?")}:</b> '
                f'<span style="color:{color}">{mark}</span>{missing_html}</div>'
            )
        parts.append(f"<h2>Deliverable reviews</h2>" + "".join(rows))
    return "".join(parts)


def _write_detail(run: dict, out_dir: Path, all_runs: list[dict] | None = None) -> None:
    t = run["trace"]
    prompt = t.get("prompt") or {}
    if not prompt and t.get("runner") == "just_dspy":
        # just_dspy traces carry these fields at top level, not under prompt.
        prompt = {
            "model": t.get("model", ""),
            "path": t.get("config", ""),
            "workspace": t.get("workspace", ""),
            "feedback_rounds": t.get("fix_rounds", "?"),
        }
    run_id = run["run_id"]
    log_link = ""
    if run["log_path"]:
        rel = Path("../..") / run["log_path"].relative_to(REPO_ROOT)
        log_link = f'<p><a href="{html.escape(str(rel))}">Open raw log ({run["log_path"].stat().st_size // 1024} KB)</a></p>'
    parent = t.get("parent_run_id")
    parent_html = f'<p>Parent run: <a href="{html.escape(parent)}.html">{html.escape(parent)}</a> (meta-iter {t.get("meta_iteration")})</p>' if parent else ""
    header = (
        f'<h1>{html.escape(run_id)} <span class="muted mono" style="font-size:12px">{html.escape(t.get("status", ""))}</span></h1>'
        f'<p class="muted">Started {html.escape(t.get("started_utc", ""))} · Finished {html.escape(t.get("finished_utc", "") or "—")}</p>'
        f'<div class="card">'
        f'<div><b>Model:</b> <span class="mono">{html.escape(str(prompt.get("model", "")))}</span></div>'
        f'<div><b>Prompt:</b> <span class="mono">{html.escape(str(prompt.get("path", "")))}</span></div>'
        f'<div><b>Workspace:</b> <span class="mono">{html.escape(str(prompt.get("workspace", "")))}</span></div>'
        f'<div><b>Feedback rounds:</b> {prompt.get("feedback_rounds", "?")}</div>'
        f'</div>'
        f'{parent_html}'
        f'{log_link}'
    )
    score_html = _render_score(t.get("score") or {})
    workspace_path = Path(prompt.get("workspace")) if prompt.get("workspace") else None
    _ensure_plots(workspace_path)
    capability_html = _render_capability_report(workspace_path, t)
    meta_report_html = _render_meta_report(workspace_path)
    meta_table_html = _render_meta_iteration_table(workspace_path)
    meta_plots_html = _render_meta_plots(workspace_path, out_dir)
    meta_iters_html = _render_meta_iterations(workspace_path)
    surrogate_html = _render_surrogate_leaderboard(workspace_path)
    physics_html = _render_physics_plots(workspace_path, out_dir)
    eval_html = _render_eval_plots(prompt.get("workspace"), out_dir)
    provenance_html = _render_dataset_provenance(workspace_path)
    compute_html = _render_compute_cost(t, workspace_path)
    cross_run_html = _render_cross_run_comparison(run["run_id"], t, all_runs or [])
    log_by_round: dict[int, dict[int, str]] = {}
    if run["log_path"]:
        try:
            log_text = run["log_path"].read_text(errors="replace")
            log_by_round = _parse_log_by_round(log_text)
        except Exception:
            log_by_round = {}
    rounds_html = "".join(
        _render_round(r, log_by_round.get(r.get("round"), {})) for r in t.get("rounds") or []
    )
    if t.get("runner") == "just_dspy":
        rounds_html = _render_dspy_run(t) + rounds_html
    art = t.get("artifacts") or {}
    files = art.get("files_written") or []
    files_html = ""
    if files:
        files_html = (
            f'<h2>Files written by agent</h2><ul class="mono">'
            + "".join(f'<li>{html.escape(f)}</li>' for f in files)
            + "</ul>"
        )
    expected = art.get("expected") or {}
    expected_html = ""
    if expected:
        rows = []
        for path, meta in expected.items():
            ok = meta.get("exists") if isinstance(meta, dict) else False
            mark = "✓" if ok else "✗"
            color = "#28a745" if ok else "#d73a49"
            rows.append(f'<li><span style="color:{color}">{mark}</span> <span class="mono">{html.escape(path)}</span></li>')
        expected_html = f'<h2>Expected artifacts</h2><ul>{"".join(rows)}</ul>'

    # Section order: Results → Narrative → Data → Cost → URSA rounds → files.
    # Results = what won and how well it does (winner + per-model + eval plots).
    # Narrative = what the agent decided and why + how this run compares to prior ones.
    # Data = the physics inputs it was trained on + dataset provenance.
    body = (
        header
        + score_html
        # ── Results ──
        + capability_html
        + meta_report_html
        + meta_table_html
        + meta_plots_html
        + surrogate_html
        + eval_html
        # ── Narrative ──
        + meta_iters_html
        + cross_run_html
        # ── Data ──
        + physics_html
        + provenance_html
        # ── Cost ──
        + compute_html
        # ── URSA plan-execute rounds (Phase-1/2 direct runs only) ──
        + rounds_html
        + files_html
        + expected_html
        + '<p><a href="index.html">← Back to index</a></p>'
    )
    (out_dir / f"{run_id}.html").write_text(
        f'<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(run_id)}</title><style>{CSS}</style></head><body>{body}</body></html>'
    )


def _run_started(trace: dict, run_id: str) -> Optional[_dt.datetime]:
    """Run start time from the trace, falling back to the run-id timestamp."""
    started = _parse_utc(trace.get("started_utc", ""))
    if started is not None:
        return started
    m = RUN_ID_TS_RE.match(run_id)
    if m:
        try:
            return _dt.datetime.strptime(m.group(1), "%Y%m%dT%H%M%SZ").replace(
                tzinfo=_dt.timezone.utc
            )
        except ValueError:
            pass
    return None


def _parse_since(value: str) -> Optional[_dt.datetime]:
    """'all' → no cutoff; else an ISO date/datetime (assumed UTC if naive)."""
    if not value or value.lower() == "all":
        return None
    cutoff = _parse_utc(value if "T" in value else value + "T00:00:00")
    if cutoff is None:
        raise SystemExit(f"--since: cannot parse {value!r} (use YYYY-MM-DD or 'all')")
    if cutoff.tzinfo is None:
        cutoff = cutoff.replace(tzinfo=_dt.timezone.utc)
    return cutoff


def _prune_stale_pages(out_dir: Path, kept_run_ids: set[str]) -> int:
    """Delete detail pages for runs no longer in the index (e.g. pre-cutoff)."""
    n = 0
    for f in out_dir.glob("*.html"):
        if f.name == "index.html":
            continue
        m = RUN_ID_TS_RE.match(f.stem)
        if m and m.group(1) not in kept_run_ids:
            f.unlink()
            n += 1
    return n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiments", default=str(REPO_ROOT / "experiments"))
    p.add_argument("--logs", default=str(REPO_ROOT / "logs"))
    p.add_argument("--out", default=None, help="Output dir (default: <experiments>/_report)")
    p.add_argument(
        "--since",
        default=DEFAULT_SINCE,
        help="Only include runs started on/after this ISO date (default: "
             f"{DEFAULT_SINCE}, the active-sampling implementation date). "
             "Pass 'all' to include every run.",
    )
    args = p.parse_args()

    exp_dir = Path(args.experiments)
    logs = _index_logs(Path(args.logs))
    out_dir = Path(args.out) if args.out else (exp_dir / "_report")
    out_dir.mkdir(parents=True, exist_ok=True)
    since = _parse_since(args.since)

    runs = []
    n_excluded = 0
    for trace_path in sorted(exp_dir.glob("*/trace.json")):
        try:
            trace = json.loads(trace_path.read_text())
        except Exception as exc:
            print(f"skip {trace_path}: {type(exc).__name__}: {exc}")
            continue
        run_id = trace_path.parent.name
        started = _run_started(trace, run_id)
        if since is not None and (started is None or started < since):
            n_excluded += 1
            continue
        log = _match_log(started, logs)
        runs.append({"run_id": run_id, "trace": trace, "log_path": log.path if log else None})

    runs.sort(key=lambda r: r["trace"].get("started_utc", ""), reverse=True)

    for run in runs:
        _write_detail(run, out_dir, all_runs=runs)
    _write_index(runs, out_dir, since=since)
    n_pruned = _prune_stale_pages(out_dir, {r["run_id"] for r in runs})

    if since is not None:
        print(f"cutoff --since {since.date()}: excluded {n_excluded} older runs, "
              f"pruned {n_pruned} stale pages")
    print(f"wrote {len(runs)} run pages to {out_dir}")
    print(f"open: {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
