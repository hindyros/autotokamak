# Project Agenda — `autotokamak`

Status: living document. Last updated 2026-06-15 (agenda text); see the shipped-status note below.

This file records what we're trying to build, why, and which decisions are still pending. Read this before adding new prompts, agents, or surrogate-model code — it pins down scope so the repo doesn't drift.

For repo structure, setup, and code conventions, see [`CLAUDE.md`](../CLAUDE.md). This file is the *intent*; CLAUDE.md is the *mechanics*.

> **What has shipped since this agenda was written (2026-07):** several items below are
> now implemented and no longer open questions.
> - **Unified pipelines CLI** — `python -m autotokamak.pipelines <phase1|phase2|meta> --mode <fast|ursa>` is the primary entry point; the "planned prompt pipeline" ASCII flow further down is superseded by it (and the prompts `dataset_explore.yaml` / `surrogate_baseline.yaml` / `benchmark_report.yaml` in that flow were never created).
> - **Phase-2** is implemented as structured AutoML in `src/autotokamak/surrogate/` (`automl_loop.py`, `automl.py`, `zoo.py`, `schema.py`) — the "A/B/C variant" question is resolved in code.
> - **Meta-loop** — `agent/runners/meta_loop.py` (+ `pipelines meta`) is the autonomous Phase-1 → Phase-2 outer loop.
> - **Active learning** — `data/acquire.py` + `data/envelope.py` provide residual-driven acquisition and envelope evaluation (the meta-loop's `enrich_active` action).
> - Week-2/Week-4 milestones (sweeps/HDF5, surrogates) below are done; treat the weekly plan as historical.

---

## 1. Goal

Per discussion with the project's advisor (`deepj87`) on 2026-06-15:

> The optimization problem we are solving for is: **Find the best surrogate model to solve the Grad–Shafranov equation.**
>
> Given a set of 4 models (simple models for now, not PINN, etc.), the goal is to find the best combination of hyperparams, choose the best model, etc. Basically run AutoML to find the best surrogate model to solve the Grad–Shafranov equation. **We are making a PoC.**

And the advisor's immediate redirect:

> Yes... but **first task is trying to get the agent to run the mesh and gather data.**

So the project decomposes into two phases:

- **Phase 1 — agent-driven dataset generation.** An LLM agent autonomously drives TokaMaker, sweeps physics parameters, and writes a training dataset to disk.
- **Phase 2 — agent-driven AutoML over surrogates.** Given that dataset, an LLM agent searches over a small zoo of classical surrogate models and picks the best one.

The novelty across both phases is that the *agent* — not a hand-written script — orchestrates the work. "On the fly, not fixed code" is the framing the advisor used.

---

## 2. Phase 1: Agent-driven data generation (current milestone)

**What it produces:** an HDF5 dataset of `(physics-inputs, ψ-on-common-grid)` pairs from fixed-boundary Grad–Shafranov solves.

**Who authors it:** the URSA `PlanningAgent` + `ExecutionAgent` pair (already wired up in `src/autotokamak/agent/runners/plan_execute_feedback.py`), driven by a prompt YAML.

**Concrete artifacts:**

| Artifact | Path |
|---|---|
| Prompt | `src/autotokamak/agent/prompts/dataset_generation.yaml` |
| Generated workspace | `examples/dataset_generation/` (produced by the agent) |
| Raw dataset | `examples/dataset_generation/outputs/dataset.h5` |

**Author's assumed scope (to be confirmed with advisor):**

- Fixed-boundary Grad–Shafranov only (no free-boundary).
- Single profile family — TokaMaker defaults for pressure and current; no per-sample profile coefficients.
- 5 physics inputs vary: `R0, a, κ, δ, Ip`.
- N = 64 PoC samples, with a smoke run at N = 4 first.
- ψ interpolated onto a common (R, Z) grid (96 × 128) so the output tensor is rectangular.
- Failed solves kept as NaN rows so the dataset stays dense.

**Open questions for the advisor (Phase 1):**

1. Sample budget — 64 enough for a PoC, or push to 256 / 1k?
2. Inputs — keep at 5, or also vary profile coefficients (more interesting physics, less interesting for a first surrogate fit)?
3. Fixed-boundary only, or do we also need free-boundary in the PoC dataset?

---

## 3. Phase 2: AutoML over surrogate models

The candidate model zoo (per advisor): Gaussian process regression, kernel ridge regression, polynomial ridge regression, and a small MLP (`sklearn.MLPRegressor`, ≤ 2 hidden layers). **Explicitly out of scope for the PoC:** PINN, DeepONet, FNO. Those come later.

There are three plausible architectures for how the AutoML loop itself works. They look superficially similar but differ in where the "intelligence" lives:

| Variant | What the agent does | Trade-off |
|---|---|---|
| **A. Agent-as-codegen** | Writes a script that calls Optuna + sklearn; reports the winner. | Reliable, but the agent is doing very little. A 100-line script would do the same. |
| **B. Agent-as-optimizer** | The agent **is** the search loop. Each LLM step is one trial. It reasons about which hyperparameters to try next from prior results. | Genuinely novel; matches the advisor's "on the fly, not fixed code" phrasing. But slower convergence than Optuna in practice, token-expensive, and harder to evaluate. |
| **C. Agent-orchestrates-Optuna** (hybrid) | Agent picks the model zoo, hyperparameter ranges, and stopping rules. Optuna runs the inner search. The agent reads each study's results and decides whether to widen the search, add a model, or move on. | Reliable inner loop **and** adaptive outer reasoning. Matches state-of-the-art agentic-ML papers (e.g. AutoML-Agent, MLR-Copilot). |

**Recommendation for the PoC:** start with **C**. Get a working pipeline that picks a surrogate. Later, swap the Optuna call for an LLM-driven trial loop (variant **B**) and benchmark B against C on the same dataset — that comparison is the publishable result.

---

## 4. Prompt pipeline (planned)

Each prompt YAML produces a concrete checkpointed artifact the next one consumes:

```
dataset_generation.yaml   →  outputs/dataset.h5                             [Phase 1]
   ↓
dataset_explore.yaml      →  EDA report + train/val/test split              [optional]
   ↓
surrogate_baseline.yaml   →  one trained model per candidate, defaults
   ↓
surrogate_automl.yaml     →  hyperparameter search + winner selection       [Phase 2 core]
   ↓
benchmark_report.yaml     →  final comparison report                        [optional]
```

**Two design notes to fold into the surrogate prompts when they're written:**

1. ψ is high-dimensional — 96 × 128 ≈ 12 k outputs per sample. Direct GP regression on 12 k targets is not feasible. The surrogate prompts must instruct the agent to do SVD/PCA reduction first, regress in the reduced space, and reconstruct. Equilibrium ψ datasets are typically very low-rank (95 % variance in < 20 components).
2. "Simple MLP" must be capped explicitly in the prompt (sklearn `MLPRegressor`, ≤ 2 hidden layers, ≤ 256 units), or the agent will reach for PyTorch and the line between "classical" and "deep" disappears.

---

## 5. Open questions for the advisor

Listed in the order they need to be answered:

1. **Phase-2 architecture: A, B, or C?** This changes `surrogate_automl.yaml` fundamentally. Recommendation in §3 is **C**, but the advisor's "on the fly, not fixed code" phrasing could push toward **B**.
2. **PoC success metric:**
   - (i) A working pipeline that picks *a* reasonable surrogate (engineering deliverable), or
   - (ii) A benchmark showing agentic search ≥ Optuna on this problem (research deliverable)?
3. **Phase-1 dataset size and inputs.** See §2 open questions.
4. **Same URSA agent in Phase 2?** Fine for variant C (the agent's decisions are coarse, so URSA's per-step overhead is amortized). For variant B, URSA's per-trial LLM-call overhead may dominate, and we may want a thinner LangGraph-only trial loop.

---

## 6. Explicitly out of scope (so we don't drift)

- PINN / DeepONet / FNO surrogates.
- Free-boundary equilibria.
- EQDSK ingestion / experimental shot data.
- Multi-objective AutoML (accuracy vs inference time vs training time).
- Production-grade dataset sizes (we are doing a PoC).

These are all reasonable extensions *after* the PoC is working, not before.

---

## 7. How this slots into the existing weekly plan

`CLAUDE.md` already sketches a Week-0 → Week-4+ plan:

- Week 0: package refactor (done — `autotokamak.core` exists).
- **Week 2: sweep generators, HDF5 loaders** ← Phase 1 of this agenda lives here.
- Week 3+: metrics, benchmarks, comparison plots.
- **Week 4: surrogate models (PINN, DeepONet, FNO, baselines)** ← Phase 2 of this agenda lives here, but restricted to *baselines* (the classical zoo). PINN/DeepONet/FNO stay out of scope until after the PoC.

The agenda doesn't compete with the weekly plan; it pins down the *intent* inside it.
