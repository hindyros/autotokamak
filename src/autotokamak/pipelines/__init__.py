# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Unified pipeline dispatchers for autotokamak (access levels L0/L1).

Each pipeline (phase1, phase2, meta) runs the pre-written library code; the
--level flag selects the decision provider:
  L0 — scripted heuristic decisions (no LLM, reproducible baseline)
  L1 — LLM-typed decisions via the DSPy pickers

Entry point:
    python -m autotokamak.pipelines <phase1|phase2|meta> [--level <L0|L1>] [opts]

Agent-written pipeline code (L2/L3, any harness) is benchmarked separately:
    python -m autotokamak.bench run --task benchmarks/tasks/<task>.yaml --harness <name>
"""
