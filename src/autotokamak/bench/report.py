# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Cross-condition comparison over ``experiments/<tag>/<condition>/<run_id>/``."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def collect_results(tag_dir: Path) -> list[dict[str, Any]]:
    """Every result.json under experiments/<tag>/, newest run per condition last."""
    rows: list[dict[str, Any]] = []
    for result_path in sorted(Path(tag_dir).glob("*/*/result.json")):
        try:
            data = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        data["_path"] = str(result_path.parent)
        rows.append(data)
    return rows


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def render_table(rows: list[dict[str, Any]]) -> str:
    """Plain-text comparison table, one row per run."""
    if not rows:
        return "(no result.json files found)"
    cols = ["condition", "run_id", "status", "gates", "test_rel_l2_mean",
            "wall_seconds", "cost_usd"]
    table: list[list[str]] = [cols]
    for r in rows:
        contract = r.get("contract") or {}
        gates = contract.get("gates") or {}
        gates_str = f"{sum(map(bool, gates.values()))}/{len(gates)}" if gates else "-"
        frozen = (r.get("frozen_score") or {}).get("test_rel_l2") or {}
        table.append([
            _fmt(r.get("condition")),
            _fmt(r.get("run_id")),
            _fmt(r.get("status")),
            gates_str,
            _fmt(frozen.get("mean")),
            _fmt(r.get("wall_seconds")),
            _fmt(r.get("cost_usd")),
        ])
    widths = [max(len(row[i]) for row in table) for i in range(len(cols))]
    lines = []
    for i, row in enumerate(table):
        lines.append("  ".join(cell.ljust(w) for cell, w in zip(row, widths)))
        if i == 0:
            lines.append("  ".join("-" * w for w in widths))
    return "\n".join(lines)
