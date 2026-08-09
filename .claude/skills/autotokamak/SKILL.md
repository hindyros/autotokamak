---
name: autotokamak
description: Run, extend, and debug the autotokamak Grad-Shafranov equilibrium
  platform — config-driven OFT solves, dataset sweeps, surrogate training,
  and the URSA plan-execute-feedback / meta-loop agent runners. Use when the
  user works with the autotokamak repo, asks to build or solve a tokamak
  equilibrium from YAML, run a parameter sweep, train or evaluate a surrogate
  model for ψ(R,Z), invoke plan_execute_feedback or meta_loop, author a new
  example workspace, or diagnose OFT_env / isoflux-fallback / mesh issues.
  Do not use for unrelated MHD codes.
---

# autotokamak

Specialist for the `autotokamak` platform: a two-layer system pairing a Grad–Shafranov / TokaMaker physics core with URSA-driven agent runners for dataset generation, surrogate AutoML, and an autonomous meta-loop.

## On invocation — prompt for inputs, then route

When the user invokes `/autotokamak` **bare** (no accompanying request, or only greeting-level text), open with a single `AskUserQuestion` call carrying the three questions below, then echo the chosen config and route. When the invocation message already carries a concrete request ("run a sweep with N=100", "score my last run", "how does `meta_loop` decide to `regen_dataset`?"), **skip the prompts** and go straight to the routing table below — those users know what they want.

Ask exactly these three questions in one call:

1. **Which task?** — one-of:
   - `Solve one equilibrium` → `scripts/run_equilibrium.py --config PATH` (ask for the config path in a follow-up)
   - `Generate a dataset (Phase-1)` → write `dataset_config.yaml` from the answers below into a fresh `examples/<name>/` workspace, then `scripts/run_dataset_sweep.py --config PATH`
   - `Run surrogate AutoML (Phase-2)` → `scripts/run_agent.py --prompt surrogate_automl`
   - `Run the meta-loop (Phase-1 → Phase-2, self-improving)` → `scripts/run_agent.py --prompt surrogate_meta`
   - `Just generate a report from an existing run` → `scripts/report.py --latest` (or ask for a run-id)

2. **Tokamak parameter ranges** — one-of:
   - `Use shipped defaults` — R₀ ∈ [0.35, 0.55] m · a ∈ [0.10, 0.20] m · κ ∈ [1.0, 1.6] · δ ∈ [0.0, 0.4] · Iₚ ∈ [80 kA, 200 kA]. Recommend this option first for the common case.
   - `Override per parameter` — follow up with a compact form asking each range.

   Skip this question when the task is "generate a report from an existing run" (no physics inputs needed).

3. **Budgets** — multi-select with defaults preselected:
   - `n_samples` (default 500) — Phase-1 dataset size
   - `time_budget_seconds` (default 1200) — Phase-2 AutoML search
   - `feedback_rounds` (default 3) — number of agent replan rounds

   Skip this question for "solve one equilibrium" (no sweep, no search).

After the answers come back:
- Print a two-line summary of the chosen config and destination workspace path.
- If task = dataset or meta-loop, materialize a real `dataset_config.yaml` from the answers using `autotokamak.data.schema.SweepConfig` (do NOT hand-write YAML — round-trip through the pydantic model so validation catches typos).
- Dispatch the corresponding script from `scripts/`. When the run finishes, offer to render the report.

Do not invent knobs the user was not asked about. If the task needs an input not covered by these three questions (e.g. a mesh_dx override, a custom initial guess), ask one focused follow-up rather than guessing.

## Orient — locate the repo before anything else

Before running any script or citing any path, resolve `$AUTOTOKAMAK_ROOT`:

1. If `AUTOTOKAMAK_ROOT` env var is set and points to a dir with `pyproject.toml`, use it.
2. Else walk up from the current working directory. The repo root is the first parent that contains a `pyproject.toml` whose `[project]` block names `autotokamak`.
3. Else, enter **read-only advisory mode**: answer questions from `references/`, decline any action that would run a solve or edit repo files, and tell the user "no autotokamak checkout detected — set `AUTOTOKAMAK_ROOT` to enable actions."

Full protocol: `references/repo-locate.md`. Every wrapper in `scripts/` performs this resolution and prints its result on stdout.

## Hard constraints — never violate

- **OFT_env is a process-wide singleton.** `OpenFUSIONToolkit.OFT_env(...)` may be constructed **once per Python kernel**. Any solve fan-out MUST be process-level (`subprocess.Popen` / `multiprocessing` with `spawn`), never thread-level, never asyncio-in-one-process. `autotokamak.core.solver.get_oft_env()` caches; call it, do not construct.
- **Python 3.11 or 3.12 only.** Some `ursa-ai==0.15.1` deps don't support 3.13+.
- **Never edit generated agent output.** Files under `experiments/**` and agent-produced workspaces (e.g. `examples/dataset_generation/outputs/`, `examples/surrogate_automl/`, `examples/surrogate_meta/`) are agent artifacts. If they contain a bug, fix the *platform* — a prompt YAML under `src/autotokamak/agent/prompts/`, a runner under `src/autotokamak/agent/runners/`, or a scorer/metric under `src/autotokamak/agent/dspy/` — then regenerate. The generated `run_*.py` inside a workspace is disposable.
- **Prompts have `CONSTRAINTS:` blocks.** Do not add `git`, `pip install`, or `input()` calls into prompts or the runners they drive.
- **Writes go through `autotokamak.core.io.atomic_savez` / `atomic_write_text`.** A crashed sweep must not leave a truncated `.npz`.
- **Wrappers never `import autotokamak.*`.** All `scripts/*.py` in this Skill dispatch via `subprocess.run` to keep the OFT_env singleton isolated per child process.

## Route by task

| Task | Load first |
|---|---|
| Author or edit an equilibrium YAML | `references/configs.md` + `assets/templates/equilibrium.yaml` |
| Author or edit a sweep or dataset YAML | `references/configs.md` §sweep / §dataset |
| Understand core API signatures | `references/core-api.md` |
| Run one solve | `scripts/run_equilibrium.py --config PATH` |
| Fan out N ≥ 2 solves | `scripts/sweep_parallel.py --config PATH --workers N` |
| Generate a training dataset | `scripts/run_dataset_sweep.py --config PATH` |
| Invoke `plan_execute_feedback` or `meta_loop` | `references/agent-runners.md` + `scripts/run_agent.py --prompt NAME` |
| Debug isoflux fallback, bad q95, mesh error | `references/debugging.md` |
| Train / evaluate a surrogate | `references/surrogates.md` + `scripts/eval_surrogate.py` |
| Benchmark an agent harness (L2/L3 codegen conditions) | `benchmarks/README.md` + `scripts/run_benchmark.py --task PATH --harness NAME` |
| Generate a rich HTML report for a run | `scripts/report.py --run-id ID` (or `--latest`) |
| Add a new example workspace under `examples/` | `references/examples-guide.md` + `assets/templates/example_workspace/` |
| Physics background | `references/physics.md` |
| Full architecture | `references/architecture.md` |
| Terminology | `references/glossary.md` |
| Environment / install check | `scripts/check_env.py` |

## Canonical solve recipe

The five-step sequence inside every physics runner:

```python
from autotokamak.core.schema import EquilibriumConfig
from autotokamak.core.geometry import build_lcfs, build_mesh
from autotokamak.core.solver import solve_equilibrium, get_last_solve_info
from autotokamak.core.diagnostics import extract_scalars

cfg = EquilibriumConfig.from_yaml("discretization_config.yaml").model_dump()
lcfs = build_lcfs(**cfg["boundary"])
_, mesh_pts, mesh_lc, mesh_reg = build_mesh(lcfs, mesh_dx=cfg["mesh"]["regions"][0]["dx"])
gs = solve_equilibrium(mesh_pts=mesh_pts, mesh_lc=mesh_lc, mesh_reg=mesh_reg, lcfs=lcfs, cfg=cfg)
info = get_last_solve_info()          # {"isoflux_used": bool, "fallback_reason": str|None}
scalars = extract_scalars(gs)         # R_axis, q_0, q_95, p_axis, ...
```

Full signatures with return-type fields: `references/core-api.md`.

## Sub-agent dispatch — parallelize deliberately

Two hard facts drive these rules:

- **OFT_env is a Python-process singleton.** Threading inside one interpreter shares/corrupts it. Solver parallelism must be process-level.
- **Sub-agents (Task tool)** get their own tool sandbox. Use for lookup, synthesis, or independent long-running compute — not for tight I/O loops.

| Trigger | Agent type | Count | Why |
|---|---|---|---|
| User question spans ≥ 3 subsystems (e.g. "how does `meta_loop` decide to `regen_dataset`?") | `Explore` | one per subsystem, ≤ 3 | Parallel read-only lookup, each covers one directory. |
| Design task touches both `core/` and `agent/` layers, or adds a new prompt YAML | `Plan` | 1 | Bounded-context design, single markdown deliverable. |
| Fan-out ≥ 5 independent equilibrium solves | `general-purpose` | one per shard, ≤ workers | Each sub-agent calls `scripts/run_equilibrium.py` **as a subprocess** — never imports the solver. Sidesteps OFT_env singleton. |
| Fan-out < 5 solves | none | — | Use `scripts/sweep_parallel.py` directly; agent overhead exceeds savings. |
| Evaluate ≥ 2 candidate surrogate architectures | `general-purpose` | one per architecture | Each owns train → eval → report end-to-end. No OFT_env concern (surrogates don't touch OFT). |
| Cross-check a physics claim | `Explore` | ≤ 2, against `references/physics.md` + repo `docs/` | No `WebFetch` unless the user explicitly asks. |
| Reading one config, editing one file, running one solve, reading one traceback | none | — | Overhead > benefit. |
| Anything requiring concurrent `import autotokamak.solver` in one Python process | **forbidden** | — | OFT_env singleton will corrupt state. Always subprocess-out. |

When in doubt: single solve → run it directly. Many solves → subprocess. Broad question → parallel `Explore`. Design decision → single `Plan`.

## Physics primer (30-second version)

- **ψ (poloidal flux)** is a 2D scalar field on the (R, Z) plane. Its contours are the magnetic surfaces.
- **Grad–Shafranov equation**: `Δ*ψ = −μ₀R²p'(ψ) − FF'(ψ)`. TokaMaker solves this on a triangular mesh inside the LCFS.
- **LCFS** (Last Closed Flux Surface): the outermost closed contour. Fixed-boundary solves prescribe it; free-boundary solves solve for it.
- **isoflux constraint**: forces the solved ψ to match a prescribed contour. Fails on extreme shapes → `solve_equilibrium` falls back to unconstrained solve. Always check `get_last_solve_info().isoflux_used`.
- **Safety factor q(ψ)**: how many toroidal turns per poloidal turn. `q_95` (q at 95% flux) is the headline stability number — physical range ≈ 2–8.
- **Shipped-example shaping ranges**: `R0 ∈ [0.35, 0.55]`, `a ∈ [0.10, 0.20]`, `κ ∈ [1.0, 1.6]`, `δ ∈ [0.0, 0.4]`, `Ip ∈ [80 kA, 200 kA]`. Outside this box, expect fallback rate to spike.

Full primer: `references/physics.md`.

## Scripts (one-liners; full CLI in `scripts/README.md`)

- `run_equilibrium.py` — solve one config, emit JSON summary
- `sweep_parallel.py` — process-pool fan-out over a sweep YAML
- `run_dataset_sweep.py` — Phase-1 dataset generation
- `run_agent.py --prompt NAME` — dispatch a named prompt to `plan_execute_feedback` (or `meta_loop` for `surrogate_meta`)
- `probe_feasible.py` — LHS feasibility scan for shaping bounds
- `eval_surrogate.py` — 7 diagnostic PNGs + JSON metrics
- `run_full_pipeline.py` — Phase-1 → Phase-2 → eval → HTML report (chains `python -m autotokamak.pipelines`, `--level L0|L1`)
- `run_benchmark.py` — one cell of the harness × access-level benchmark matrix via `python -m autotokamak.bench run` (see `benchmarks/README.md`)
- `trace_to_html.py` — render `experiments/*/trace.json` to browsable HTML (light theme, run index)
- `report.py` — self-contained dark-mode HTML report for one run: physics & dataset (swept/fixed tokamak params, sampling method, ψ grid) + score gates + quality bars + winner rationale (agent's own words from the terminating round) + model comparison + collapsible per-round search decisions (action, agent rationale, models & search space tried) + Optuna SVG + eval gallery + round-by-round agent reasoning + meta iterations. Base64-embedded images, no external assets.
- `check_env.py` — Python/OFT/repo probe, warn-only

## Debugging quick-checks

| Symptom | First check |
|---|---|
| `RuntimeError: Only one instance of OFT_env` | Someone bypassed `get_oft_env()`. `references/debugging.md#oft-env`. |
| `used_fallback=True` on every sample | Shaping outside feasible box → run `probe_feasible.py`. Or `init_psi.method: isoflux` failing → try `tokamaker_default`. |
| `q_95` is `NaN` | Boundary hit inside `get_q`; usually mesh_dx too coarse (>0.02) or κ too high. |
| Mesh build fails at `add_polygon` | LCFS self-intersects — bad `delta` or too-few `npts` (<40). |
| HDF5 dataset opens but `psi` all zero | Dataset generator crashed mid-write and the atomic rename never happened. Check for temp files in `outputs/`. |
| `plan_execute_feedback` loops without progress | Missing scorer or prompt CONSTRAINTS violation. Inspect `experiments/<run_id>/trace.json`. |

Full triage table: `references/debugging.md`.

## Anti-patterns

- Editing `experiments/**/*.py` or `examples/<generated>/run_*.py` when the fix belongs in `src/autotokamak/`.
- Instantiating a second `OFT_env` (e.g. to "reset state") — will raise; the correct move is `make_solver(..., env=get_oft_env())`.
- Threading over `solve_equilibrium` inside one interpreter (`ThreadPoolExecutor`, `asyncio.gather`) — use subprocess pools.
- Adding `pip install` or `git` calls into a prompt YAML or a generated runner — forbidden by prompt CONSTRAINTS blocks.
- Silently ignoring `get_last_solve_info().isoflux_used == False` in dataset generation — the geometry inputs no longer describe the saved ψ.
