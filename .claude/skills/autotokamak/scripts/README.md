# scripts/ — CLI reference

All scripts follow the same protocol:

1. Resolve `AUTOTOKAMAK_ROOT` (see `references/repo-locate.md`).
2. Print an env header: `autotokamak: root=... cwd=... python=...`.
3. Dispatch to the underlying repo runner via `subprocess.run` (never `import autotokamak.*`).
4. Emit a JSON summary block between `===AUTOTOKAMAK-JSON===` sentinels.

If no repo is detected, every script enters read-only advisory mode and exits 0 without acting.

## CLI

| Script | Signature | Wraps |
|---|---|---|
| `check_env.py` | `[--verbose]` | (self) — probes Python, OFT, autotokamak, ursa, langchain |
| `run_equilibrium.py` | `--config PATH [--out DIR] [--validate-only]` | `examples/config_driven_equilibrium/run_equilibrium_from_config.py` |
| `sweep_parallel.py` | `--config PATH --workers N` | `examples/config_driven_equilibrium/run_discretization_sweep.py`; splits `cases[]` and fans out via `subprocess.Popen` |
| `run_dataset_sweep.py` | `--config PATH` | `examples/dataset_generation/run_dataset_sweep.py` |
| `run_agent.py` | `--prompt NAME [--model M] [--workspace W] [--no-trace] [--max-iterations N]` | dispatches to `plan_execute[_feedback]` or `meta_loop` per name |
| `probe_feasible.py` | `[--out JSON]` | `tools/probe_feasible_box.py` |
| `eval_surrogate.py` | `[--workspace PATH] [--out DIR]` | `tools/eval_surrogate.py` |
| `run_full_pipeline.py` | `[--level {L0,L1}] [--model M] [--regen-dataset] [--n-samples N] [--time-budget S] [--skip-phase2] [--skip-eval] [--skip-report] [--enable-meta]` | `python -m autotokamak.pipelines phase1` → `phase2`/`meta` → `tools/eval_surrogate.py` → `tools/trace_to_html.py` |
| `run_benchmark.py` | `--task PATH --harness {echo,ursa,dspy,claude_sdk,pi,cursor} [--model M] [--tag T] [--dry-run]` | `python -m autotokamak.bench run` (L2/L3 agent-codegen conditions; see `benchmarks/README.md`) |
| `trace_to_html.py` | `[--experiments DIR] [--logs DIR] [--out DIR]` | `tools/trace_to_html.py` (light-theme, run index) |
| `report.py` | `[--run-id ID \| --workspace PATH \| --trace PATH \| --latest] [--out HTML]` | (self) — dark-mode, single-run rich report with data viz, gates, winner rationale, model comparison, eval gallery, agent reasoning |

## `--prompt` values for `run_agent.py`

- `dataset_generation` → `plan_execute_feedback`
- `oft_discretization_example` → `plan_execute_feedback`
- `oft_example_generation` → `plan_execute` (legacy)
- `surrogate_automl` → `plan_execute_feedback`
- `surrogate_meta` → `meta_loop`

Any other value is rejected.

## Parallelism

- `sweep_parallel.py` is the only wrapper that spawns concurrent children. It uses `subprocess.Popen`, one child per shard, because **OFT_env is a Python-process singleton** and cannot be shared across threads or async tasks in a single interpreter.
- All other wrappers are single-child. Even `run_dataset_sweep.py` is sequential — the underlying `run_dataset_sweep.py` iterates cases inside one Python process (reusing the OFT_env singleton via `get_oft_env`).
- Sub-agent parallelism is a separate axis: see `../SKILL.md` §"Sub-agent dispatch" for when to fan out via the `Agent` tool.

## Output contract

The JSON summary block always contains at least:

```
{
  "ok": bool,
  "returncode": int,
  "root": str,
  "elapsed_seconds": float
}
```

Plus script-specific fields. Parse by looking for the `===AUTOTOKAMAK-JSON===` sentinel — the block between it and `===END-AUTOTOKAMAK-JSON===` is one JSON object.

## Shared internals

`_locate.py` is an internal module (underscore-prefixed). It provides `locate_root()`, `print_env_header()`, `print_json_summary()`, `repo_python()`, `agent_env()`, and `read_only_advisory()`. Do not run it directly.
