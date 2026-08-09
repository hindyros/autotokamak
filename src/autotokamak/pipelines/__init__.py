# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Unified pipeline dispatchers for autotokamak.

Each pipeline (phase1, phase2, meta) can run in two modes:
  fast  — platform library code called directly (run_sweep, automl_loop)
  ursa  — URSA PlanningAgent + ExecutionAgent writes code from scratch

Entry point:
    python -m autotokamak.pipelines <phase1|phase2|meta> --mode <fast|ursa> [opts]
"""
