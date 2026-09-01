# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Per-run cost/efficiency report across a benchmark tag.

Answers "what did each cell cost" three ways, best available first:

  1. measured  — a dollar figure the substrate itself reported
                 (claude_sdk ResultMessage, pi per-message usage.cost,
                 dspy litellm cost, ursa OpenAI callback),
  2. derived   — token counts × a model price table (cursor reports tokens
                 but no dollars; only possible when the model was pinned),
  3. proxy     — wall-clock seconds + turn/tool-call counts, always present.

Works RETROACTIVELY: when result.json predates the cost-capture patches,
the raw event streams (pi_events.jsonl / cursor_events.jsonl /
claude_events.jsonl) are re-parsed, so old runs get numbers without paying
for a re-run. Also pulls n_solves_attempted from the agent's report.json —
solver calls are the physics-compute axis of cost.

Usage:
    python tools/cost_report.py --tag matrix-v2-20260810
    python tools/cost_report.py --tag <tag> --price-table my_prices.json

Writes experiments/<tag>/cost_report.csv and prints the table.
Price table format: {"<model-substring>": {"in": $, "out": $, "cache_read": $,
"cache_write": $}} per million tokens; first substring match wins.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# $/1M tokens. Extend via --price-table; substring-matched against the
# run's model string, first hit wins.
DEFAULT_PRICES = {
    "claude-opus": {"in": 15.0, "out": 75.0, "cache_read": 1.5, "cache_write": 18.75},
    "claude-sonnet": {"in": 3.0, "out": 15.0, "cache_read": 0.3, "cache_write": 3.75},
    "claude-haiku": {"in": 1.0, "out": 5.0, "cache_read": 0.1, "cache_write": 1.25},
}


def _price_for(model: str, prices: dict) -> dict | None:
    model = (model or "").lower()
    for key, p in prices.items():
        if key.lower() in model:
            return p
    return None


def _pi_cost_from_events(path: Path) -> tuple[float | None, dict | None]:
    cost, tokens, seen = 0.0, {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}, False
    for ln in path.read_text(errors="replace").splitlines():
        try:
            evt = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if evt.get("type") != "message_end":
            continue
        u = (evt.get("message") or {}).get("usage") or {}
        if not u:
            continue
        seen = True
        cost += float((u.get("cost") or {}).get("total") or 0.0)
        for k in tokens:
            tokens[k] += int(u.get(k) or 0)
    return (round(cost, 4) if seen else None), (tokens if seen else None)


def _cursor_usage_from_events(path: Path) -> dict | None:
    usage = None
    for ln in path.read_text(errors="replace").splitlines():
        try:
            evt = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if evt.get("type") == "result" and isinstance(evt.get("usage"), dict):
            usage = evt["usage"]
    return usage


def _claude_cost_from_events(path: Path) -> float | None:
    cost = None
    for ln in path.read_text(errors="replace").splitlines():
        try:
            evt = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if evt.get("kind") == "ResultMessage" and evt.get("total_cost_usd") is not None:
            cost = float(evt["total_cost_usd"])
    return cost


def _tokens_normalized(usage: dict | None) -> dict:
    """Map substrate-specific usage keys onto in/out/cache_read/cache_write."""
    if not usage:
        return {}
    alias = {
        "in": ("input", "inputTokens", "prompt_tokens", "input_tokens"),
        "out": ("output", "outputTokens", "completion_tokens", "output_tokens"),
        "cache_read": ("cacheRead", "cacheReadTokens", "cache_read_input_tokens"),
        "cache_write": ("cacheWrite", "cacheWriteTokens", "cache_creation_input_tokens"),
    }
    out = {}
    for norm, keys in alias.items():
        for k in keys:
            if usage.get(k) is not None:
                out[norm] = int(usage[k])
                break
    return out


def analyze_run(run_dir: Path, prices: dict) -> dict:
    result = json.loads((run_dir / "result.json").read_text())
    harness = result.get("harness", "")
    cost, source = result.get("cost_usd"), ("measured" if result.get("cost_usd") is not None else None)
    usage = (result.get("extra") or {}).get("usage")

    # Retroactive harvest from raw streams when result.json predates capture.
    if cost is None:
        ev = run_dir / f"{'claude' if harness == 'claude_sdk' else harness}_events.jsonl"
        if harness == "pi" and (run_dir / "pi_events.jsonl").is_file():
            cost, u = _pi_cost_from_events(run_dir / "pi_events.jsonl")
            usage = usage or u
            source = "measured" if cost is not None else source
        elif harness == "claude_sdk" and ev.is_file():
            cost = _claude_cost_from_events(ev)
            source = "measured" if cost is not None else source
    if usage is None and harness == "cursor" and (run_dir / "cursor_events.jsonl").is_file():
        usage = _cursor_usage_from_events(run_dir / "cursor_events.jsonl")

    toks = _tokens_normalized(usage)
    if cost is None and toks:
        p = _price_for(result.get("model", ""), prices)
        if p:
            cost = round(
                toks.get("in", 0) / 1e6 * p["in"]
                + toks.get("out", 0) / 1e6 * p["out"]
                + toks.get("cache_read", 0) / 1e6 * p.get("cache_read", 0)
                + toks.get("cache_write", 0) / 1e6 * p.get("cache_write", 0), 4)
            source = "derived"

    n_solves = ""
    rp = run_dir / "workspace" / "report.json"
    if rp.is_file():
        try:
            n_solves = json.loads(rp.read_text()).get("n_solves_attempted", "")
        except Exception:  # noqa: BLE001
            pass

    fs = result.get("frozen_score") or {}
    extra = result.get("extra") or {}
    return {
        "condition": result.get("condition", run_dir.parent.name),
        "run_id": run_dir.name,
        "status": result.get("status", ""),
        "model": result.get("model", ""),
        "wall_min": round((result.get("wall_seconds") or 0) / 60, 1),
        "turns": extra.get("num_turns") or extra.get("n_tool_calls")
                 or (usage or {}).get("n_lm_calls") or "",
        "tok_in": toks.get("in", ""),
        "tok_out": toks.get("out", ""),
        "tok_cache": toks.get("cache_read", ""),
        "cost_usd": cost if cost is not None else "",
        "cost_source": source or "proxy-only",
        "n_solves": n_solves,
        "rel_l2_mean": round(fs["test_rel_l2"]["mean"], 4)
                        if isinstance(fs.get("test_rel_l2"), dict)
                        and fs["test_rel_l2"].get("mean") is not None else "",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--price-table", default=None,
                    help="JSON file extending/overriding the built-in $/1M table")
    ap.add_argument("--experiments-dir", default=None)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    exp_dir = Path(args.experiments_dir) if args.experiments_dir else repo_root / "experiments"
    prices = dict(DEFAULT_PRICES)
    if args.price_table:
        prices.update(json.loads(Path(args.price_table).read_text()))

    rows = [analyze_run(p.parent, prices)
            for p in sorted((exp_dir / args.tag).glob("*/*/result.json"))]
    if not rows:
        print("No runs found.", file=sys.stderr)
        return 1

    cols = list(rows[0].keys())
    csv_path = exp_dir / args.tag / "cost_report.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for r in sorted(rows, key=lambda r: r["condition"]):
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))
    total = sum(r["cost_usd"] for r in rows if isinstance(r["cost_usd"], (int, float)))
    known = sum(1 for r in rows if isinstance(r["cost_usd"], (int, float)))
    print(f"\nTotal (the {known}/{len(rows)} costed runs): ${total:.2f}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
