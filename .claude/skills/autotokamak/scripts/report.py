#!/usr/bin/env python3
# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Palantir-styled HTML report for an autotokamak agent run.

Produces a single self-contained HTML file (all CSS inline, images
base64-embedded) that combines:

  - trace.json  → header, timeline, per-round plan + execution
  - workspace/outputs/report.json  → surrogate winner, models tried, RMSEs
  - workspace/outputs/eval_plots/*.png → visualization gallery
  - workspace/outputs/study.db (if optuna present) → Optuna convergence
  - workspace/meta_trace.json (if meta-loop) → per-iteration ActionDecision log
  - score.hard_gates + score.quality → gate checklist and quality bars

Usage:
    python report.py --run-id 20260707T205832Z
    python report.py --workspace examples/surrogate_automl
    python report.py --trace path/to/trace.json --out report.html
    python report.py --latest                 # newest completed run

Design language: black background, near-white text, IBM Plex Mono + Inter,
one amber accent (#f97316) for the winner + top score. Sharp corners, thin
1px borders, dense information layout. No gradients, no rounded corners,
no drop shadows on data.
"""

from __future__ import annotations

import argparse
import base64
import html as _html
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _locate import (  # type: ignore[import-not-found]
    locate_root,
    print_env_header,
    print_json_summary,
    read_only_advisory,
)


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------


@dataclass
class ReportData:
    """Everything the renderer needs. Any field can be None; renderer degrades."""

    trace_path: Path
    trace: dict
    workspace: Path | None
    surrogate_report: dict | None = None
    meta_trace: dict | None = None
    meta_report: dict | None = None
    eval_plots: dict[str, str] = field(default_factory=dict)  # name → base64 png
    optuna_history: dict[str, list[tuple[int, float]]] = field(default_factory=dict)
    log_text: str | None = None
    search_history: list[dict] = field(default_factory=list)
    physics_config: dict | None = None
    physics_config_source: str | None = None


def _read_json(p: Path) -> dict | None:
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(errors="replace"))
    except (json.JSONDecodeError, OSError):
        return None


def _embed_png(p: Path) -> str | None:
    if not p.is_file():
        return None
    try:
        data = p.read_bytes()
    except OSError:
        return None
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


_OPTUNA_EXTRACT_SNIPPET = r"""
import json, sys, optuna
storage = f'sqlite:///{sys.argv[1]}'
out = {}
for s in optuna.study.get_all_study_summaries(storage=storage):
    try:
        st = optuna.load_study(study_name=s.study_name, storage=storage)
    except Exception:
        continue
    best = float('inf'); series = []
    for t in st.trials:
        if t.value is None: continue
        best = min(best, t.value)
        series.append([t.number, best])
    if series:
        out[s.study_name] = series
print(json.dumps(out))
"""


def _load_optuna(study_db: Path) -> dict[str, list[tuple[int, float]]]:
    """Per-study running-best trial values. Falls back to a venv subprocess
    when optuna isn't importable in the current interpreter — the skill's
    report script may be invoked by system python that lacks optuna even
    though the repo's venv has it."""
    if not study_db.is_file():
        return {}
    parsed = None
    try:
        import optuna
        storage = f"sqlite:///{study_db}"
        summaries = optuna.study.get_all_study_summaries(storage=storage)
        parsed = {}
        for s in summaries:
            try:
                study = optuna.load_study(study_name=s.study_name, storage=storage)
            except Exception:
                continue
            best_so_far = float("inf")
            series: list[tuple[int, float]] = []
            for t in study.trials:
                if t.value is None:
                    continue
                best_so_far = min(best_so_far, t.value)
                series.append((t.number, best_so_far))
            if series:
                parsed[s.study_name] = series
        return parsed
    except ImportError:
        pass
    except Exception:
        return {}

    import subprocess, sys as _sys
    root = locate_root()
    venv_py = None
    if root is not None:
        candidate = root / "venv" / "bin" / "python"
        if candidate.is_file():
            venv_py = str(candidate)
    if not venv_py or venv_py == _sys.executable:
        return {}
    try:
        r = subprocess.run(
            [venv_py, "-c", _OPTUNA_EXTRACT_SNIPPET, str(study_db)],
            capture_output=True, text=True, timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError):
        return {}
    if r.returncode != 0 or not r.stdout.strip():
        return {}
    try:
        raw = json.loads(r.stdout)
    except json.JSONDecodeError:
        return {}
    return {k: [(int(a), float(b)) for a, b in v] for k, v in raw.items()}


def _load_search_history(p: Path) -> list[dict]:
    """Parse outputs/search_history.jsonl — one SearchSpec per line."""
    if not p.is_file():
        return []
    out: list[dict] = []
    try:
        text = p.read_text(errors="replace")
    except OSError:
        return []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec.get("spec"), dict):
            merged = dict(rec["spec"])
            if isinstance(rec.get("summary"), dict):
                merged["summary"] = rec["summary"]
            if rec.get("elapsed_seconds") is not None:
                merged["elapsed_seconds"] = rec["elapsed_seconds"]
            rec = merged
        out.append(rec)
    out.sort(key=lambda r: r.get("round", 0))
    return out


_INLINE_KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([^,}]+)")


def _coerce_scalar(s: str):
    s = s.strip().strip('"').strip("'")
    if s in ("true", "True"):
        return True
    if s in ("false", "False"):
        return False
    if s in ("null", "None", "~"):
        return None
    try:
        if "." in s or "e" in s or "E" in s:
            return float(s)
        return int(s)
    except ValueError:
        return s


def _parse_inline_mapping(s: str) -> dict:
    inner = s.strip()
    if inner.startswith("{") and inner.endswith("}"):
        inner = inner[1:-1]
    return {k: _coerce_scalar(v) for k, v in _INLINE_KV_RE.findall(inner)}


def _load_yaml(p: Path) -> dict | None:
    """Load a YAML file. Prefers PyYAML; falls back to a tolerant mini-parser
    that handles the shape of ``dataset_config.yaml`` and ``surrogate_config.yaml``
    (top-level keys, one level of nesting, ``{k: v, ...}`` inline mappings)."""
    if not p.is_file():
        return None
    try:
        import yaml  # type: ignore[import-not-found]
        with p.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else None
    except ImportError:
        pass
    except Exception:
        return None

    try:
        lines = p.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    out: dict = {}
    current_key: str | None = None
    for raw in lines:
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" ") and ":" in line:
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip()
            if not val:
                out[key] = {}
                current_key = key
            elif val.startswith("{") and val.endswith("}"):
                out[key] = _parse_inline_mapping(val)
                current_key = None
            else:
                out[key] = _coerce_scalar(val)
                current_key = None
        elif line.startswith(" ") and current_key is not None and ":" in line:
            sub_key, _, sub_val = line.strip().partition(":")
            sub_key = sub_key.strip()
            sub_val = sub_val.strip()
            if not isinstance(out.get(current_key), dict):
                out[current_key] = {}
            if sub_val.startswith("{") and sub_val.endswith("}"):
                out[current_key][sub_key] = _parse_inline_mapping(sub_val)
            elif sub_val:
                out[current_key][sub_key] = _coerce_scalar(sub_val)
            else:
                out[current_key][sub_key] = {}
    return out or None


def _find_physics_config(workspace: Path) -> tuple[dict | None, str | None]:
    """Locate the dataset_config.yaml that produced the surrogate's training set.

    Search order:
      1. dataset_config.yaml co-located in the surrogate workspace.
      2. surrogate_config.yaml → dataset_h5 → sibling dataset_config.yaml.
      3. ../dataset_generation/dataset_config.yaml (canonical Phase-1 workspace).
    """
    direct = workspace / "dataset_config.yaml"
    if direct.is_file():
        return _load_yaml(direct), str(direct)

    surr_cfg = _load_yaml(workspace / "surrogate_config.yaml")
    if surr_cfg and isinstance(surr_cfg.get("dataset_h5"), str):
        h5_path = (workspace / surr_cfg["dataset_h5"]).resolve()
        sibling = h5_path.parent / "dataset_config.yaml"
        if sibling.is_file():
            return _load_yaml(sibling), str(sibling)
        grandparent = h5_path.parent.parent / "dataset_config.yaml"
        if grandparent.is_file():
            return _load_yaml(grandparent), str(grandparent)

    if surr_cfg and isinstance(surr_cfg.get("dataset_h5"), str):
        h5_dir = (workspace / surr_cfg["dataset_h5"]).resolve().parent
        iter_cfgs = sorted(h5_dir.glob("iter*_config.yaml"))
        if iter_cfgs:
            return _load_yaml(iter_cfgs[-1]), str(iter_cfgs[-1])

    for anc in [workspace, *list(workspace.parents)[:5]]:
        cand = anc / "dataset_generation" / "dataset_config.yaml"
        if cand.is_file():
            return _load_yaml(cand), str(cand)

    return None, None


def _resolve_phase2_outputs_dir(workspace: Path) -> Path:
    """Return the directory that holds the Phase-2 winner + study + history.

    For a bare Phase-2 workspace this is ``<ws>/outputs``. For a meta-loop
    workspace the Phase-2 artifacts live inside a per-iteration sub-workspace
    at ``<ws>/surrogate_runs/iterN/outputs`` — pick the highest iterN that
    actually contains a ``report.json`` (falling back to the highest that
    contains anything at all).
    """
    direct = workspace / "outputs"
    if (direct / "report.json").is_file():
        return direct

    runs = workspace / "surrogate_runs"
    if runs.is_dir():
        candidates = sorted(
            (p for p in runs.glob("iter*") if p.is_dir()),
            key=lambda p: int(p.name[4:]) if p.name[4:].isdigit() else -1,
        )
        with_report = [c for c in candidates if (c / "outputs" / "report.json").is_file()]
        if with_report:
            return with_report[-1] / "outputs"
        with_any = [c for c in candidates if (c / "outputs").is_dir()]
        if with_any:
            return with_any[-1] / "outputs"

    return direct  # non-existent; downstream loaders return None gracefully


def load_report_data(trace_path: Path, log_path: Path | None) -> ReportData:
    trace = _read_json(trace_path) or {}
    workspace = None
    if isinstance(trace.get("prompt"), dict):
        ws_raw = trace["prompt"].get("workspace")
        if ws_raw:
            workspace = Path(ws_raw)

    data = ReportData(trace_path=trace_path, trace=trace, workspace=workspace)

    if workspace and workspace.is_dir():
        phase2_out = _resolve_phase2_outputs_dir(workspace)
        data.surrogate_report = _read_json(phase2_out / "report.json")
        data.meta_trace = _read_json(workspace / "meta_trace.json")
        data.meta_report = _read_json(workspace / "report.json")
        plots_dir = phase2_out / "eval_plots"
        if plots_dir.is_dir():
            for png in sorted(plots_dir.glob("*.png")):
                embedded = _embed_png(png)
                if embedded:
                    data.eval_plots[png.stem] = embedded
        data.optuna_history = _load_optuna(phase2_out / "study.db")
        data.search_history = _load_search_history(phase2_out / "search_history.jsonl")
        # physics config lookup: the surrogate config lives in the phase2
        # sub-workspace (meta-loop) or the workspace root (bare phase-2).
        data.physics_config, data.physics_config_source = _find_physics_config(
            phase2_out.parent if phase2_out.parent != workspace else workspace
        )

    if log_path and log_path.is_file():
        try:
            data.log_text = log_path.read_text(errors="replace")
        except OSError:
            pass
    return data


LOG_TS_RE = re.compile(r"_(\d{8}T\d{6}Z)\.log$")


def _match_log_for_trace(logs_dir: Path, started_utc: str | None) -> Path | None:
    if not logs_dir.is_dir() or not started_utc:
        return None
    try:
        t0 = datetime.fromisoformat(started_utc.replace("Z", "+00:00"))
    except ValueError:
        return None
    best: Path | None = None
    best_delta: float | None = None
    for p in logs_dir.glob("*.log"):
        m = LOG_TS_RE.search(p.name)
        if not m:
            continue
        try:
            ts = datetime.strptime(m.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        delta = (t0 - ts).total_seconds()
        if delta < -5 or delta > 300:
            continue
        if best_delta is None or abs(delta) < abs(best_delta):
            best, best_delta = p, delta
    return best


# ---------------------------------------------------------------------------
# thought-process extraction
# ---------------------------------------------------------------------------


def _summarize_step(step: dict) -> str:
    """Pull the first meaningful sentence out of the step's result_excerpt."""
    exc = step.get("result_excerpt") or ""
    if not exc:
        return ""
    exc = exc.strip()
    for para in exc.split("\n\n"):
        para = para.strip()
        if len(para) < 20 or para.startswith(("```", "---", "===")):
            continue
        first = para.split(". ")[0].strip()
        if len(first) > 40:
            return first[:280] + ("…" if len(first) > 280 else "")
    return (exc.split("\n\n")[0][:280] + "…") if exc else ""


def extract_thought_narrative(data: ReportData) -> list[dict]:
    """Round-by-round summary suitable for the "reasoning" section.

    Returns [{round, n_steps, n_ok, first_summary, last_summary, hygiene}, ...].
    """
    rounds = data.trace.get("rounds") or []
    out = []
    for rnd in rounds:
        execution = rnd.get("execution") or []
        n_ok = sum(1 for st in execution if st.get("ok"))
        first_summary = _summarize_step(execution[0]) if execution else ""
        last_summary = _summarize_step(execution[-1]) if execution else ""
        plan_names = [ps.get("name", "?") for ps in (rnd.get("plan_steps") or [])]
        out.append(
            {
                "round": rnd.get("round"),
                "n_steps": len(execution),
                "n_ok": n_ok,
                "plan_names": plan_names,
                "first_summary": first_summary,
                "last_summary": last_summary,
            }
        )
    return out


def extract_model_comparison(data: ReportData) -> list[dict]:
    """Comparison rows: model | trials | best_value | delta_to_winner."""
    rep = data.surrogate_report
    if not rep:
        return []
    winner = rep.get("winner_model_name")
    models_tried = rep.get("models_tried") or []
    best_by_model = rep.get("best_by_model") or {}
    trials_by_model = rep.get("trials_by_model") or {}

    winner_value = None
    if winner and winner in best_by_model:
        winner_value = float(best_by_model[winner])

    rows = []
    for m in (list(best_by_model.keys()) or models_tried):
        best = best_by_model.get(m)
        try:
            best_f = float(best) if best is not None else None
        except (TypeError, ValueError):
            best_f = None
        delta = None
        if best_f is not None and winner_value is not None:
            delta = best_f - winner_value
        rows.append(
            {
                "model": m,
                "trials": int(trials_by_model.get(m, 0)) if trials_by_model.get(m) else None,
                "best_value": best_f,
                "delta": delta,
                "is_winner": m == winner,
            }
        )
    rows.sort(key=lambda r: (float("inf") if r["best_value"] is None else r["best_value"]))
    return rows


def extract_meta_iterations(data: ReportData) -> list[dict]:
    mt = data.meta_trace or {}
    iters = mt.get("iterations") or []
    out = []
    for r in iters:
        decision = r.get("decision") or {}
        out.append(
            {
                "iteration": r.get("iteration"),
                "action": decision.get("action"),
                "diagnosis": decision.get("diagnosis"),
                "rmse_after": r.get("rmse_after"),
                "diagnostics_keys": list((r.get("diagnostics") or {}).keys())[:8],
            }
        )
    return out


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


PALETTE = {
    "bg":          "#0a0a0a",
    "surface":     "#111113",
    "surface_hi":  "#17171a",
    "border":      "#242428",
    "border_hi":   "#3a3a40",
    "text":        "#e6e6e6",
    "text_dim":    "#9a9aa0",
    "text_faint":  "#6a6a70",
    "accent":      "#f97316",   # amber — one use only: winner + top score
    "ok":          "#8b8b90",   # grayscale ok
    "fail":        "#c94a4a",
}


CSS = """
:root {{
  color-scheme: dark only;
  --bg: {bg};
  --surface: {surface};
  --surface-hi: {surface_hi};
  --border: {border};
  --border-hi: {border_hi};
  --text: {text};
  --text-dim: {text_dim};
  --text-faint: {text_faint};
  --accent: {accent};
  --ok: {ok};
  --fail: {fail};
}}

@font-face {{
  font-family: "Inter-fallback";
  src: local("Inter"), local("SF Pro Text"), local("Segoe UI");
}}
@font-face {{
  font-family: "Mono-fallback";
  src: local("IBM Plex Mono"), local("JetBrains Mono"), local("SF Mono"), local("Menlo");
}}

* {{ box-sizing: border-box; }}

html, body {{
  background: var(--bg);
  color: var(--text);
  margin: 0;
  padding: 0;
  font-family: "Inter-fallback", -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
  font-size: 13px;
  line-height: 1.55;
  letter-spacing: 0.005em;
  -webkit-font-smoothing: antialiased;
}}

.mono {{ font-family: "Mono-fallback", "IBM Plex Mono", "SF Mono", ui-monospace, Menlo, monospace; font-size: 12px; letter-spacing: 0; }}

a {{ color: var(--text); text-decoration: none; border-bottom: 1px dotted var(--border-hi); }}
a:hover {{ color: var(--accent); border-bottom-color: var(--accent); }}

.wrap {{ max-width: 1400px; margin: 0 auto; padding: 24px 32px 64px; }}

/* ---- top strip ---- */
.top {{
  display: flex; align-items: baseline; justify-content: space-between;
  padding-bottom: 16px; margin-bottom: 24px; border-bottom: 1px solid var(--border);
}}
.brand {{ font-family: "Mono-fallback", monospace; font-size: 11px; letter-spacing: 0.14em; color: var(--text-dim); text-transform: uppercase; }}
.brand b {{ color: var(--text); font-weight: 600; }}
.brand .sep {{ color: var(--text-faint); margin: 0 8px; }}

.title {{ font-size: 22px; font-weight: 600; letter-spacing: -0.005em; margin: 0 0 4px; }}
.subtitle {{ color: var(--text-dim); font-size: 12px; margin: 0; }}

/* ---- overview grid ---- */
.grid {{ display: grid; gap: 1px; background: var(--border); border: 1px solid var(--border); margin-bottom: 32px; }}
.grid-4 {{ grid-template-columns: repeat(4, 1fr); }}
.grid-2 {{ grid-template-columns: repeat(2, 1fr); }}
.cell {{ background: var(--surface); padding: 18px 20px; }}
.cell h3 {{
  font-family: "Mono-fallback", monospace; font-size: 10px; letter-spacing: 0.16em;
  color: var(--text-faint); text-transform: uppercase; margin: 0 0 10px; font-weight: 500;
}}
.cell .big {{ font-size: 28px; font-weight: 600; letter-spacing: -0.02em; line-height: 1; margin-bottom: 4px; }}
.cell .big.mono {{ font-size: 24px; }}
.cell .sub {{ color: var(--text-dim); font-size: 12px; }}
.cell .accent {{ color: var(--accent); }}

/* ---- section headers ---- */
.section {{ margin: 40px 0 12px; }}
.section h2 {{
  font-family: "Mono-fallback", monospace; font-size: 11px; letter-spacing: 0.18em;
  color: var(--text-faint); text-transform: uppercase; font-weight: 500;
  border-top: 1px solid var(--border); padding-top: 16px;
  margin: 0 0 16px; display: flex; align-items: center; justify-content: space-between;
}}
.section h2 .num {{ color: var(--text); font-size: 20px; letter-spacing: -0.02em; font-weight: 600; margin-right: 12px; }}
.section h2 .meta {{ color: var(--text-dim); font-family: "Mono-fallback", monospace; font-size: 10px; letter-spacing: 0.14em; }}

/* ---- badges / pills ---- */
.pill {{
  display: inline-block; padding: 2px 8px; border: 1px solid var(--border-hi);
  font-family: "Mono-fallback", monospace; font-size: 10px; letter-spacing: 0.1em;
  text-transform: uppercase; color: var(--text-dim);
}}
.pill.ok {{ border-color: var(--text); color: var(--text); }}
.pill.fail {{ border-color: var(--fail); color: var(--fail); }}
.pill.accent {{ border-color: var(--accent); color: var(--accent); }}

/* ---- gates ---- */
.gates {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(210px, 1fr)); gap: 1px; background: var(--border); border: 1px solid var(--border); }}
.gate {{ background: var(--surface); padding: 10px 14px; display: flex; align-items: center; justify-content: space-between; }}
.gate-name {{ font-family: "Mono-fallback", monospace; font-size: 11px; color: var(--text-dim); }}
.gate-mark {{ font-family: "Mono-fallback", monospace; font-size: 11px; }}
.gate.pass .gate-mark {{ color: var(--text); }}
.gate.pass .gate-name {{ color: var(--text); }}
.gate.fail .gate-mark {{ color: var(--fail); }}
.gate.fail .gate-name {{ color: var(--fail); }}

/* ---- quality bars ---- */
.qbars {{ background: var(--surface); border: 1px solid var(--border); padding: 20px 24px; }}
.qbar-row {{ display: grid; grid-template-columns: 240px 1fr 80px 60px; gap: 16px; align-items: center; padding: 6px 0; border-bottom: 1px solid var(--surface-hi); }}
.qbar-row:last-child {{ border-bottom: none; }}
.qbar-name {{ font-family: "Mono-fallback", monospace; font-size: 11px; color: var(--text-dim); }}
.qbar-track {{ height: 4px; background: var(--surface-hi); position: relative; overflow: hidden; }}
.qbar-fill {{ height: 100%; background: var(--text-dim); }}
.qbar-fill.hi {{ background: var(--text); }}
.qbar-fill.top {{ background: var(--accent); }}
.qbar-val {{ font-family: "Mono-fallback", monospace; font-size: 12px; text-align: right; color: var(--text); }}
.qbar-w {{ font-family: "Mono-fallback", monospace; font-size: 10px; text-align: right; color: var(--text-faint); }}

/* ---- tables ---- */
table.data {{ width: 100%; border-collapse: collapse; background: var(--surface); border: 1px solid var(--border); }}
table.data th, table.data td {{
  padding: 10px 14px; text-align: left; border-bottom: 1px solid var(--surface-hi);
  font-family: "Mono-fallback", monospace; font-size: 12px;
}}
table.data th {{
  color: var(--text-faint); font-size: 10px; letter-spacing: 0.14em; text-transform: uppercase;
  font-weight: 500; border-bottom: 1px solid var(--border-hi); background: var(--surface-hi);
}}
table.data tr:last-child td {{ border-bottom: none; }}
table.data td.num {{ text-align: right; }}
table.data tr.winner td {{ color: var(--accent); font-weight: 500; }}
table.data tr.winner td:first-child::before {{ content: "▸ "; color: var(--accent); }}

/* ---- winner card ---- */
.winner-card {{
  background: var(--surface); border: 1px solid var(--border); padding: 24px 28px; margin-bottom: 16px;
}}
.winner-card .label {{ font-family: "Mono-fallback", monospace; font-size: 10px; letter-spacing: 0.16em; color: var(--text-faint); text-transform: uppercase; }}
.winner-card .name {{ font-family: "Mono-fallback", monospace; font-size: 24px; color: var(--accent); font-weight: 600; margin: 6px 0 4px; letter-spacing: -0.01em; }}
.winner-card .rmse {{ color: var(--text-dim); font-size: 12px; font-family: "Mono-fallback", monospace; }}
.winner-card .rationale {{ color: var(--text); font-size: 13px; margin-top: 16px; padding-top: 16px; border-top: 1px solid var(--surface-hi); max-width: 780px; }}

/* ---- rounds ---- */
.round {{
  background: var(--surface); border: 1px solid var(--border); margin-bottom: 12px; padding: 16px 20px;
}}
.round-head {{
  display: flex; justify-content: space-between; align-items: baseline;
  padding-bottom: 10px; margin-bottom: 12px; border-bottom: 1px solid var(--surface-hi);
}}
.round-head h4 {{ margin: 0; font-family: "Mono-fallback", monospace; font-size: 12px; letter-spacing: 0.14em; text-transform: uppercase; color: var(--text); font-weight: 500; }}
.round-head .stat {{ font-family: "Mono-fallback", monospace; font-size: 11px; color: var(--text-dim); }}
.round-plan {{ font-family: "Mono-fallback", monospace; font-size: 11px; color: var(--text-dim); }}
.round-plan .step {{ display: block; padding: 3px 0; }}
.round-plan .step .n {{ color: var(--text-faint); display: inline-block; width: 24px; }}
.round-plan .step.ok .n {{ color: var(--text); }}
.round-plan .step.err .n {{ color: var(--fail); }}
.round-summary {{ margin-top: 12px; padding-top: 12px; border-top: 1px solid var(--surface-hi); font-size: 12px; color: var(--text-dim); line-height: 1.7; }}
.round-summary b {{ color: var(--text); font-family: "Mono-fallback", monospace; font-size: 10px; letter-spacing: 0.14em; text-transform: uppercase; font-weight: 500; }}

/* ---- gallery ---- */
.gallery {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1px; background: var(--border); border: 1px solid var(--border); }}
.gfig {{ background: var(--surface); padding: 14px 16px; }}
.gfig img {{ width: 100%; height: auto; display: block; background: var(--surface-hi); border: 1px solid var(--border); }}
.gfig figcaption {{ font-family: "Mono-fallback", monospace; font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--text-dim); margin-top: 10px; }}

/* ---- optuna svg ---- */
.optuna-svg {{ display: block; width: 100%; background: var(--surface); border: 1px solid var(--border); padding: 24px 28px; }}
.optuna-svg svg {{ display: block; width: 100%; height: 260px; }}
.optuna-legend {{ padding: 8px 28px 20px; background: var(--surface); border: 1px solid var(--border); border-top: none; font-family: "Mono-fallback", monospace; font-size: 11px; color: var(--text-dim); display: flex; gap: 20px; flex-wrap: wrap; }}
.optuna-legend .sw {{ display: inline-block; width: 12px; height: 2px; vertical-align: middle; margin-right: 6px; }}

/* ---- provenance table ---- */
.kv {{ display: grid; grid-template-columns: 200px 1fr; gap: 0; border: 1px solid var(--border); background: var(--surface); }}
.kv > * {{ padding: 8px 14px; border-bottom: 1px solid var(--surface-hi); }}
.kv > *:nth-last-child(-n+2) {{ border-bottom: none; }}
.kv dt {{ font-family: "Mono-fallback", monospace; font-size: 10px; letter-spacing: 0.14em; text-transform: uppercase; color: var(--text-faint); margin: 0; }}
.kv dd {{ font-family: "Mono-fallback", monospace; font-size: 12px; color: var(--text); margin: 0; word-break: break-all; }}

/* ---- meta iterations ---- */
.meta-iter {{ background: var(--surface); border: 1px solid var(--border); padding: 14px 18px; margin-bottom: 8px; }}
.meta-iter-head {{ display: flex; justify-content: space-between; align-items: baseline; font-family: "Mono-fallback", monospace; font-size: 11px; }}
.meta-iter-head .idx {{ color: var(--text-faint); }}
.meta-iter-head .action {{ color: var(--accent); letter-spacing: 0.12em; text-transform: uppercase; }}
.meta-iter-head .rmse {{ color: var(--text); }}
.meta-iter-diag {{ margin-top: 10px; font-size: 12px; color: var(--text-dim); line-height: 1.6; }}

/* ---- collapsible search-decision rounds ---- */
details.round summary {{ display: flex; }}
details.round summary::-webkit-details-marker {{ display: none; }}
details.round summary::marker {{ content: ""; }}
details.round summary::before {{
  content: "▸"; color: var(--text-faint); margin-right: 12px;
  font-family: "Mono-fallback", monospace; transition: transform .1s;
}}
details.round[open] summary::before {{ content: "▾"; color: var(--text); }}
details.round summary:hover h4 {{ color: var(--accent); }}
.dim {{ color: var(--text-dim); }}
td.dim {{ color: var(--text-dim); }}

/* ---- footer ---- */
footer {{ margin-top: 64px; padding-top: 20px; border-top: 1px solid var(--border); color: var(--text-faint); font-family: "Mono-fallback", monospace; font-size: 10px; letter-spacing: 0.14em; text-transform: uppercase; }}
""".format(**PALETTE)


def _esc(s: Any) -> str:
    return _html.escape(str(s), quote=True)


def _fmt_num(v: Any, decimals: int = 4) -> str:
    if v is None:
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return _esc(v)
    if f != f:  # NaN
        return "NaN"
    if abs(f) >= 1000 or (abs(f) < 0.01 and f != 0):
        return f"{f:.3e}"
    return f"{f:.{decimals}f}"


def _fmt_duration(a: str | None, b: str | None) -> str:
    if not a or not b:
        return "—"
    try:
        ta = datetime.fromisoformat(a.replace("Z", "+00:00"))
        tb = datetime.fromisoformat(b.replace("Z", "+00:00"))
    except ValueError:
        return "—"
    sec = (tb - ta).total_seconds()
    if sec < 60:
        return f"{sec:.1f}s"
    m, s = divmod(int(sec), 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


def render_overview(data: ReportData) -> str:
    t = data.trace
    score = t.get("score") or {}
    prompt = t.get("prompt") or {}
    total = score.get("total")
    gates = score.get("hard_gates") or {}
    all_gates_pass = bool(gates) and all(gates.values())

    if total is None:
        score_html = '<div class="big mono">—</div><div class="sub">no score recorded</div>'
    else:
        cls = "accent" if (total >= 0.75 and all_gates_pass) else ""
        gate_str = f"{sum(1 for v in gates.values() if v)}/{len(gates)}" if gates else "no gates"
        score_html = (
            f'<div class="big mono {cls}">{total:.3f}</div>'
            f'<div class="sub">composite / hard gates {gate_str}</div>'
        )

    rep = data.surrogate_report or {}
    winner = rep.get("winner_model_name") or "—"
    win_rmse = rep.get("test_psi_rmse") or rep.get("val_psi_rmse")
    winner_html = (
        f'<div class="big mono accent">{_esc(winner)}</div>'
        f'<div class="sub">test rmse {_fmt_num(win_rmse)}</div>'
    )

    n_trials = rep.get("n_total_trials")
    models_tried = rep.get("models_tried") or []
    search_html = (
        f'<div class="big mono">{n_trials or "—"}</div>'
        f'<div class="sub">trials across {len(models_tried) or "—"} models</div>'
    )

    rounds = t.get("rounds") or []
    dur = _fmt_duration(t.get("started_utc"), t.get("finished_utc"))
    timing_html = (
        f'<div class="big mono">{len(rounds)}</div>'
        f'<div class="sub">rounds · {dur}</div>'
    )

    return f"""
<div class="grid grid-4">
  <div class="cell"><h3>score</h3>{score_html}</div>
  <div class="cell"><h3>winner</h3>{winner_html}</div>
  <div class="cell"><h3>search</h3>{search_html}</div>
  <div class="cell"><h3>timing</h3>{timing_html}</div>
</div>
"""


def render_gates(data: ReportData) -> str:
    score = data.trace.get("score") or {}
    gates = score.get("hard_gates") or {}
    if not gates:
        return ""
    cells = []
    for k, v in gates.items():
        cls = "pass" if v else "fail"
        mark = "PASS" if v else "FAIL"
        cells.append(
            f'<div class="gate {cls}">'
            f'<span class="gate-name">{_esc(k)}</span>'
            f'<span class="gate-mark">{mark}</span>'
            f'</div>'
        )
    return f"""
<div class="section">
  <h2><span><span class="num">02</span>Hard gates</span><span class="meta">{len(gates)} checks</span></h2>
  <div class="gates">{''.join(cells)}</div>
</div>
"""


def render_quality(data: ReportData) -> str:
    score = data.trace.get("score") or {}
    quality = score.get("quality") or {}
    if not quality:
        return ""
    details = score.get("details") or {}
    weights = (details.get("weights") if isinstance(details, dict) else None) or {}
    sorted_items = sorted(quality.items(), key=lambda kv: -float(kv[1]))
    top_name = sorted_items[0][0] if sorted_items else None
    rows = []
    for name, val in quality.items():
        try:
            frac = max(0.0, min(1.0, float(val)))
        except (TypeError, ValueError):
            frac = 0.0
        pct = int(frac * 100)
        cls = "top" if name == top_name and frac > 0.6 else ("hi" if frac > 0.6 else "")
        w = weights.get(name)
        w_str = f"w {float(w):.2f}" if isinstance(w, (int, float)) else ""
        rows.append(
            f'<div class="qbar-row">'
            f'<span class="qbar-name">{_esc(name)}</span>'
            f'<span class="qbar-track"><span class="qbar-fill {cls}" style="width:{pct}%"></span></span>'
            f'<span class="qbar-val">{float(val):.3f}</span>'
            f'<span class="qbar-w">{w_str}</span>'
            f'</div>'
        )
    return f"""
<div class="section">
  <h2><span><span class="num">03</span>Quality decomposition</span><span class="meta">{len(quality)} weighted terms</span></h2>
  <div class="qbars">{''.join(rows)}</div>
</div>
"""


def _rationale_for_winner(data: ReportData) -> str:
    """Synthesize a short explanation of why the winning model won.

    Prefers explicit fields in the surrogate report; falls back to comparing
    best_by_model values; last resort explains it as "only viable model."
    """
    rep = data.surrogate_report or {}
    winner = rep.get("winner_model_name")
    if not winner:
        return "No winner recorded — this run did not select a surrogate model."

    agent_words: str | None = None
    if data.search_history:
        term = next(
            (r for r in reversed(data.search_history) if r.get("action") == "terminate"),
            None,
        ) or data.search_history[-1]
        rat = term.get("rationale")
        if isinstance(rat, str) and rat.strip():
            agent_words = rat.strip()

    explicit = rep.get("winner_rationale") or rep.get("agent_notes") or rep.get("rationale")
    if explicit and isinstance(explicit, str):
        agent_words = agent_words or explicit.strip()

    best_by_model = rep.get("best_by_model") or {}
    if len(best_by_model) < 2:
        return (
            f"The agent tried only {len(best_by_model)} model family "
            f"({winner}) — no comparison was possible. Downstream conclusions "
            f"should be interpreted as a single-model result, not a search."
        )

    winner_val = float(best_by_model[winner])
    others = {k: float(v) for k, v in best_by_model.items() if k != winner}
    runner_up = min(others, key=lambda k: others[k])
    runner_val = others[runner_up]
    margin = runner_val - winner_val
    rel = (margin / max(winner_val, 1e-9)) if winner_val else 0.0

    parts = [
        f"{winner} achieved the lowest CV RMSE at {winner_val:.4f}.",
        f"Runner-up {runner_up} landed at {runner_val:.4f} — {margin:.4f} higher "
        f"({rel * 100:.1f}% relative).",
    ]
    if rel < 0.05:
        parts.append(
            "The margin is small; on a different split the winner could plausibly flip. "
            "Consider LOO CV at this N to reduce split-luck (see references/surrogates.md)."
        )
    elif rel > 0.30:
        parts.append(
            "The margin is large; the winner is robust to reasonable split variation."
        )
    numeric = " ".join(parts)
    if agent_words:
        return f"Agent's rationale (final round): {agent_words}\n\n{numeric}"
    return numeric


def render_winner(data: ReportData) -> str:
    rep = data.surrogate_report
    if not rep:
        return ""
    winner = rep.get("winner_model_name") or "—"
    test_rmse = rep.get("test_psi_rmse") or rep.get("val_psi_rmse")
    baseline = rep.get("baseline_rmse") or rep.get("recomputed_baseline_rmse")
    hp = rep.get("winner_hyperparams") or {}
    rationale = _rationale_for_winner(data)
    hp_html = ""
    if hp:
        hp_rows = "".join(
            f'<span class="pill">{_esc(k)}={_esc(v)}</span> ' for k, v in list(hp.items())[:12]
        )
        hp_html = f'<div style="margin-top:10px">{hp_rows}</div>'
    return f"""
<div class="section">
  <h2><span><span class="num">04</span>Selected model</span><span class="meta">agent choice</span></h2>
  <div class="winner-card">
    <div class="label">winner</div>
    <div class="name">{_esc(winner)}</div>
    <div class="rmse">
      test rmse {_fmt_num(test_rmse)}
      &nbsp;·&nbsp; baseline {_fmt_num(baseline)}
      &nbsp;·&nbsp; ratio {_fmt_num((test_rmse / baseline) if (test_rmse and baseline) else None, 3)}
    </div>
    {hp_html}
    <div class="rationale">{_esc(rationale)}</div>
  </div>
</div>
"""


def render_model_comparison(data: ReportData) -> str:
    rows = extract_model_comparison(data)
    if not rows:
        return ""
    tbody = []
    max_best = max((r["best_value"] for r in rows if r["best_value"] is not None), default=1.0)
    for r in rows:
        cls = "winner" if r["is_winner"] else ""
        best_str = _fmt_num(r["best_value"])
        delta_str = "0.0000" if r["is_winner"] else (
            _fmt_num(r["delta"], 4) if r["delta"] is not None else "—"
        )
        trials_str = str(r["trials"]) if r["trials"] is not None else "—"
        bar_pct = 0
        if r["best_value"] is not None and max_best:
            bar_pct = int(100 * min(1.0, r["best_value"] / max_best))
        bar_html = (
            f'<span style="display:inline-block;width:80px;height:3px;background:{PALETTE["surface_hi"]};vertical-align:middle;position:relative">'
            f'<span style="display:block;width:{bar_pct}%;height:100%;background:{"var(--accent)" if r["is_winner"] else "var(--text-dim)"}"></span>'
            f'</span>'
        )
        tbody.append(
            f'<tr class="{cls}">'
            f'<td>{_esc(r["model"])}</td>'
            f'<td class="num">{trials_str}</td>'
            f'<td class="num">{best_str}</td>'
            f'<td class="num">{delta_str}</td>'
            f'<td class="num">{bar_html}</td>'
            f'</tr>'
        )
    return f"""
<div class="section">
  <h2><span><span class="num">05</span>Model comparison</span><span class="meta">sorted by best CV value</span></h2>
  <table class="data">
    <thead><tr><th>model</th><th class="num">trials</th><th class="num">best value</th><th class="num">Δ to winner</th><th class="num" style="width:100px">relative</th></tr></thead>
    <tbody>{''.join(tbody)}</tbody>
  </table>
</div>
"""


def render_optuna(data: ReportData) -> str:
    hist = data.optuna_history
    if not hist:
        return ""

    # Build SVG line chart from scratch — no matplotlib dep.
    pad_l, pad_r, pad_t, pad_b = 50, 20, 20, 30
    W, H = 1200, 260
    inner_w = W - pad_l - pad_r
    inner_h = H - pad_t - pad_b

    all_trials = [t for series in hist.values() for t, _ in series]
    all_vals = [v for series in hist.values() for _, v in series]
    if not all_trials or not all_vals:
        return ""
    max_t = max(all_trials)
    min_v, max_v = min(all_vals), max(all_vals)
    v_range = max(max_v - min_v, 1e-9)

    def _x(t):
        return pad_l + (t / max(1, max_t)) * inner_w

    def _y(v):
        return pad_t + (1 - (v - min_v) / v_range) * inner_h

    # simple monochrome ramp for series
    ramp = [PALETTE["text"], PALETTE["text_dim"], PALETTE["accent"], PALETTE["text_faint"], "#888892"]
    lines_svg = []
    legend_html = []
    for i, (name, series) in enumerate(hist.items()):
        color = ramp[i % len(ramp)]
        pts = " ".join(f"{_x(t):.1f},{_y(v):.1f}" for t, v in series)
        lines_svg.append(
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.4" stroke-linecap="square"/>'
        )
        legend_html.append(
            f'<span><span class="sw" style="background:{color}"></span>{_esc(name)}</span>'
        )

    # y-axis ticks
    ticks_svg = []
    for i in range(5):
        y = pad_t + i * (inner_h / 4)
        val = max_v - i * (v_range / 4)
        ticks_svg.append(
            f'<line x1="{pad_l}" x2="{W - pad_r}" y1="{y}" y2="{y}" stroke="{PALETTE["surface_hi"]}" stroke-width="1"/>'
            f'<text x="{pad_l - 8}" y="{y + 4}" text-anchor="end" fill="{PALETTE["text_faint"]}" '
            f'style="font-family:Mono-fallback,monospace;font-size:10px">{val:.4f}</text>'
        )
    # x-axis ticks (5 evenly)
    for i in range(5):
        t = int(i * max_t / 4)
        x = _x(t)
        ticks_svg.append(
            f'<text x="{x}" y="{H - 8}" text-anchor="middle" fill="{PALETTE["text_faint"]}" '
            f'style="font-family:Mono-fallback,monospace;font-size:10px">{t}</text>'
        )

    svg = (
        f'<svg viewBox="0 0 {W} {H}" preserveAspectRatio="none">'
        f'<rect width="{W}" height="{H}" fill="{PALETTE["surface"]}"/>'
        f'{"".join(ticks_svg)}'
        f'{"".join(lines_svg)}'
        f'</svg>'
    )
    return f"""
<div class="section">
  <h2><span><span class="num">07</span>Optuna convergence</span><span class="meta">best value so far per study</span></h2>
  <div class="optuna-svg">{svg}</div>
  <div class="optuna-legend">{''.join(legend_html)}</div>
</div>
"""


def render_gallery(data: ReportData) -> str:
    if not data.eval_plots:
        return ""
    caption_map = {
        "true_pred_residual":   "True vs predicted ψ  ·  per test sample",
        "per_sample_rmse":      "Per-sample test RMSE  ·  vs baseline",
        "pred_vs_true_scatter": "Pixel-wise predicted vs true",
        "residual_histogram":   "Distribution of per-pixel residuals",
        "pca_variance":         "PCA cumulative explained variance",
        "pca_reconstruction":   "PCA-only vs full-pipeline error",
        "optuna_history":       "Optuna best-value convergence",
    }
    figs = []
    for name, data_uri in data.eval_plots.items():
        caption = caption_map.get(name, name.replace("_", " "))
        figs.append(
            f'<figure class="gfig">'
            f'<img src="{data_uri}" alt="{_esc(caption)}"/>'
            f'<figcaption>{_esc(caption)}</figcaption>'
            f'</figure>'
        )
    return f"""
<div class="section">
  <h2><span><span class="num">08</span>Evaluation</span><span class="meta">{len(figs)} diagnostic plots</span></h2>
  <div class="gallery">{''.join(figs)}</div>
</div>
"""


def render_thought_process(data: ReportData) -> str:
    rounds = extract_thought_narrative(data)
    if not rounds:
        return ""
    blocks = []
    for r in rounds:
        step_html = []
        # zip against actual execution to get per-step ok/err
        exec_steps = (data.trace.get("rounds") or [])[r["round"] - 1].get("execution") or []
        exec_by_num = {st.get("step"): st for st in exec_steps}
        for i, name in enumerate(r["plan_names"], 1):
            st = exec_by_num.get(i)
            cls = "ok" if (st and st.get("ok")) else ("err" if st and st.get("ok") is False else "")
            step_html.append(
                f'<span class="step {cls}"><span class="n">{i:02d}</span>{_esc(name)}</span>'
            )
        first = r["first_summary"]
        last = r["last_summary"]
        summary_html = ""
        if first or last:
            parts = []
            if first:
                parts.append(f'<div><b>opened with</b> {_esc(first)}</div>')
            if last and last != first:
                parts.append(f'<div style="margin-top:6px"><b>closed with</b> {_esc(last)}</div>')
            summary_html = f'<div class="round-summary">{"".join(parts)}</div>'
        blocks.append(f"""
<div class="round">
  <div class="round-head">
    <h4>Round {r["round"]:02d}</h4>
    <div class="stat">{r["n_ok"]:02d} / {r["n_steps"]:02d} steps ok</div>
  </div>
  <div class="round-plan">{''.join(step_html)}</div>
  {summary_html}
</div>
""")
    return f"""
<div class="section">
  <h2><span><span class="num">09</span>Agent reasoning</span><span class="meta">round-by-round plan &amp; outcomes</span></h2>
  {''.join(blocks)}
</div>
"""


def render_meta_iterations(data: ReportData) -> str:
    iters = extract_meta_iterations(data)
    if not iters:
        return ""
    blocks = []
    for it in iters:
        rmse = _fmt_num(it["rmse_after"])
        diag = ", ".join(it["diagnostics_keys"]) or "—"
        action = (it["action"] or "—").upper()
        blocks.append(f"""
<div class="meta-iter">
  <div class="meta-iter-head">
    <span class="idx">iter {it["iteration"]:02d}</span>
    <span class="action">{_esc(action)}</span>
    <span class="rmse">rmse {rmse}</span>
  </div>
  <div class="meta-iter-diag"><b>diagnosis.</b> {_esc(it["diagnosis"] or "—")}</div>
  <div class="meta-iter-diag mono" style="color:{PALETTE["text_faint"]};font-size:11px">signals: {_esc(diag)}</div>
</div>
""")
    return f"""
<div class="section">
  <h2><span><span class="num">10</span>Meta-loop iterations</span><span class="meta">autonomous outer loop</span></h2>
  {''.join(blocks)}
</div>
"""


def render_physics(data: ReportData) -> str:
    """Physics inputs — what tokamak parameter space the surrogate was trained on."""
    cfg = data.physics_config
    if not cfg:
        return ""

    sampling = cfg.get("sampling") or {}
    params = cfg.get("parameters") or {}
    fixed = cfg.get("fixed") or {}
    grid = cfg.get("output_grid") or {}

    PARAM_LABEL = {
        "r0":    ("R₀",        "m",  "major radius"),
        "a":     ("a",         "m",  "minor radius"),
        "kappa": ("κ",         "",   "elongation"),
        "delta": ("δ",         "",   "triangularity"),
        "Ip":    ("Iₚ",        "A",  "plasma current"),
        "ip":    ("Iₚ",        "A",  "plasma current"),
        "z0":    ("Z₀",        "m",  "vertical shift"),
        "F0":    ("F₀",        "T·m","toroidal field function"),
    }

    rows = []
    for key, rng in params.items():
        label, unit, meaning = PARAM_LABEL.get(key, (key, "", ""))
        low = rng.get("low") if isinstance(rng, dict) else None
        high = rng.get("high") if isinstance(rng, dict) else None
        rows.append(
            f'<tr>'
            f'<td>{_esc(label)}</td>'
            f'<td class="dim">{_esc(meaning)}</td>'
            f'<td class="num">{_fmt_num(low)}</td>'
            f'<td class="num">{_fmt_num(high)}</td>'
            f'<td class="dim">{_esc(unit)}</td>'
            f'</tr>'
        )
    swept_html = (
        f'<table class="data"><thead><tr>'
        f'<th>symbol</th><th>meaning</th><th class="num">min</th>'
        f'<th class="num">max</th><th>unit</th>'
        f'</tr></thead><tbody>{"".join(rows)}</tbody></table>'
    ) if rows else '<div class="dim mono">No swept parameters declared.</div>'

    fixed_pills = ""
    if fixed:
        pills = []
        for k, v in fixed.items():
            label, unit, _ = PARAM_LABEL.get(k, (k, "", ""))
            unit_txt = f" {unit}" if unit else ""
            pills.append(f'<span class="pill">{_esc(label)}={_esc(v)}{_esc(unit_txt)}</span>')
        fixed_pills = f'<div style="margin-top:12px">{" ".join(pills)}</div>'

    n_samples = sampling.get("n_samples", "—")
    method = sampling.get("method", "—")
    seed = sampling.get("seed", "—")

    R = grid.get("R") or {}
    Z = grid.get("Z") or {}
    if R and Z:
        grid_txt = (
            f'R ∈ [{_fmt_num(R.get("min"))}, {_fmt_num(R.get("max"))}] · nR={R.get("n","—")}'
            f' &nbsp;·&nbsp; '
            f'Z ∈ [{_fmt_num(Z.get("min"))}, {_fmt_num(Z.get("max"))}] · nZ={Z.get("n","—")}'
        )
    else:
        grid_txt = "—"

    source_txt = ""
    if data.physics_config_source:
        source_txt = f'<div class="dim mono" style="margin-top:16px;font-size:10px">source: {_esc(data.physics_config_source)}</div>'

    return f"""
<div class="section">
  <h2><span><span class="num">01</span>Physics &amp; dataset</span><span class="meta">tokamak parameter space</span></h2>
  <div class="grid grid-2" style="margin-bottom:16px">
    <div class="cell">
      <h3>sampling</h3>
      <div class="big mono">{_esc(n_samples)}</div>
      <div class="sub">samples · method={_esc(method)} · seed={_esc(seed)}</div>
    </div>
    <div class="cell">
      <h3>ψ grid</h3>
      <div class="mono" style="font-size:14px;margin-top:4px">{grid_txt}</div>
    </div>
  </div>
  <div style="margin-bottom:10px">
    <span class="pill accent">SWEPT</span>
    <span class="dim mono" style="margin-left:8px;font-size:11px">latin hypercube over the parameter box below</span>
  </div>
  {swept_html}
  <div style="margin-top:20px">
    <span class="pill">FIXED</span>
    <span class="dim mono" style="margin-left:8px;font-size:11px">held constant across all samples</span>
    {fixed_pills}
  </div>
  {source_txt}
</div>
"""


def render_search_decisions(data: ReportData) -> str:
    """Per-round agent decisions: action, rationale, models tried this round."""
    hist = data.search_history
    if not hist:
        return ""

    blocks = []
    for i, r in enumerate(hist):
        rnd = r.get("round", i + 1)
        action = (r.get("action") or "—").upper()
        rationale = (r.get("rationale") or "").strip()
        models = r.get("models") or []
        n_pca = r.get("n_pca_components")

        model_rows = []
        for m in models:
            name = m.get("name", "?")
            n_trials = m.get("n_trials", "—")
            space = m.get("search_space") or {}
            keys = ", ".join(space.keys()) if space else "—"
            model_rows.append(
                f'<tr><td>{_esc(name)}</td>'
                f'<td class="num">{_esc(n_trials)}</td>'
                f'<td class="dim mono">{_esc(keys)}</td></tr>'
            )
        model_tbl = (
            f'<table class="data" style="margin-top:12px"><thead><tr>'
            f'<th>model</th><th class="num">n_trials</th>'
            f'<th>search space</th></tr></thead>'
            f'<tbody>{"".join(model_rows)}</tbody></table>'
        ) if model_rows else ""

        rationale_html = (
            f'<div class="round-summary"><b>rationale.</b> {_esc(rationale)}</div>'
            if rationale else
            f'<div class="round-summary dim mono" style="font-size:11px">no rationale recorded</div>'
        )

        open_attr = " open" if i == 0 or i == len(hist) - 1 else ""
        blocks.append(f"""
<details class="round"{open_attr}>
  <summary class="round-head" style="cursor:pointer;list-style:none">
    <h4>Round {rnd:02d} · {_esc(action)}</h4>
    <div class="stat">{len(models)} model{"s" if len(models)!=1 else ""} · n_pca {_esc(n_pca)}</div>
  </summary>
  {rationale_html}
  {model_tbl}
</details>
""")

    return f"""
<div class="section">
  <h2><span><span class="num">06</span>Search decisions</span><span class="meta">agent's per-round reasoning</span></h2>
  {''.join(blocks)}
</div>
"""


def render_provenance(data: ReportData) -> str:
    t = data.trace
    prompt = t.get("prompt") or {}
    kvs = [
        ("run id",       t.get("run_id", "—")),
        ("status",       t.get("status", "—")),
        ("started",      t.get("started_utc", "—")),
        ("finished",     t.get("finished_utc") or "—"),
        ("model",        prompt.get("model", "—")),
        ("prompt path",  prompt.get("path", "—")),
        ("workspace",    prompt.get("workspace", "—")),
        ("trace file",   str(data.trace_path)),
    ]
    if data.meta_report:
        kvs.extend([
            ("meta.terminated_by", data.meta_report.get("terminated_by", "—")),
            ("meta.n_iterations",  data.meta_report.get("n_iterations", "—")),
            ("meta.final_rmse",    _fmt_num(data.meta_report.get("final_rmse"))),
            ("meta.baseline_rmse", _fmt_num(data.meta_report.get("baseline_rmse"))),
        ])
    rows = "".join(f"<dt>{_esc(k)}</dt><dd>{_esc(v)}</dd>" for k, v in kvs)
    return f"""
<div class="section">
  <h2><span><span class="num">11</span>Provenance</span><span class="meta">reproducibility</span></h2>
  <dl class="kv">{rows}</dl>
</div>
"""


def render_html(data: ReportData) -> str:
    t = data.trace
    prompt = t.get("prompt") or {}
    run_id = t.get("run_id", "—")
    status = t.get("status", "unknown")
    status_cls = "ok" if status == "completed" else ("fail" if status == "errored" else "")

    header = f"""
<div class="top">
  <div>
    <div class="brand"><b>autotokamak</b><span class="sep">/</span>agent report</div>
    <h1 class="title">{_esc(run_id)}</h1>
    <p class="subtitle mono">
      {_esc(prompt.get("model", "—"))}
      &nbsp;·&nbsp; started {_esc(t.get("started_utc", "—"))}
      &nbsp;·&nbsp; {_esc(Path(prompt.get("path", "")).name)}
    </p>
  </div>
  <div><span class="pill {status_cls}">{_esc(status)}</span></div>
</div>
"""

    body = "".join([
        header,
        render_overview(data),
        render_physics(data),
        render_gates(data),
        render_quality(data),
        render_winner(data),
        render_model_comparison(data),
        render_search_decisions(data),
        render_optuna(data),
        render_gallery(data),
        render_thought_process(data),
        render_meta_iterations(data),
        render_provenance(data),
    ])

    footer = f"""
<footer>
  generated by autotokamak-skill · {_esc(datetime.utcnow().isoformat(timespec="seconds"))}Z
</footer>
"""

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{_esc(run_id)} — autotokamak</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
{body}
{footer}
</div>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _find_trace(root: Path, run_id: str | None, workspace: str | None, trace: str | None,
                latest: bool) -> Path | None:
    if trace:
        p = Path(trace).expanduser()
        return p.resolve() if p.is_file() else None

    exp = root / "experiments"

    if run_id:
        p = exp / run_id / "trace.json"
        return p if p.is_file() else None

    if workspace:
        ws = Path(workspace).expanduser().resolve()
        # find the latest trace whose prompt.workspace matches
        candidates = []
        for tp in sorted(exp.glob("*/trace.json"), reverse=True):
            data = _read_json(tp) or {}
            if str((data.get("prompt") or {}).get("workspace", "")) == str(ws):
                candidates.append(tp)
        return candidates[0] if candidates else None

    if latest:
        # newest completed run
        newest = None
        newest_ts = ""
        for tp in exp.glob("*/trace.json"):
            data = _read_json(tp) or {}
            if data.get("status") != "completed":
                continue
            ts = data.get("started_utc") or ""
            if ts > newest_ts:
                newest, newest_ts = tp, ts
        return newest

    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--run-id", help="Experiments run id (e.g. 20260707T205832Z)")
    src.add_argument("--workspace", help="Path to an example workspace; picks latest matching trace")
    src.add_argument("--trace", help="Direct path to a trace.json")
    src.add_argument("--latest", action="store_true", help="Newest completed run")
    p.add_argument("--out", default=None, help="Output HTML path (default: <experiments>/_report/<run-id>.html)")
    p.add_argument("--open", action="store_true", help="Print the file:// URL after writing")
    args = p.parse_args()

    root = locate_root()
    print_env_header(root)
    if root is None:
        read_only_advisory()

    if not any([args.run_id, args.workspace, args.trace, args.latest]):
        args.latest = True

    trace_path = _find_trace(root, args.run_id, args.workspace, args.trace, args.latest)
    if trace_path is None:
        print("ERROR: could not locate a trace.json matching the selectors.", file=sys.stderr)
        print_json_summary({"ok": False, "error": "no_trace_found"})
        sys.exit(2)

    log_path = None
    try:
        started = (_read_json(trace_path) or {}).get("started_utc")
        log_path = _match_log_for_trace(root / "logs", started)
    except Exception:
        log_path = None

    data = load_report_data(trace_path, log_path)
    html = render_html(data)

    if args.out:
        out = Path(args.out).expanduser().resolve()
    else:
        exp_report_dir = root / "experiments" / "_report"
        exp_report_dir.mkdir(parents=True, exist_ok=True)
        run_id = data.trace.get("run_id") or trace_path.parent.name
        out = exp_report_dir / f"{run_id}-styled.html"

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    size_kb = out.stat().st_size / 1024
    print(f"→ wrote {out}  ({size_kb:.1f} KB)")
    if args.open:
        print(f"   open file://{out}")

    print_json_summary({
        "ok": True,
        "trace": str(trace_path),
        "workspace": str(data.workspace) if data.workspace else None,
        "output_html": str(out),
        "size_kb": round(size_kb, 1),
        "sections_rendered": {
            "overview": True,
            "physics": bool(data.physics_config),
            "gates": bool((data.trace.get("score") or {}).get("hard_gates")),
            "quality": bool((data.trace.get("score") or {}).get("quality")),
            "winner": bool(data.surrogate_report),
            "comparison": bool(extract_model_comparison(data)),
            "search_decisions": len(data.search_history),
            "optuna": bool(data.optuna_history),
            "gallery": len(data.eval_plots),
            "thought_process": len(extract_thought_narrative(data)),
            "meta": len(extract_meta_iterations(data)),
        },
        "root": str(root),
    })
    sys.exit(0)


if __name__ == "__main__":
    main()
