# DSPy integration plan for `autotokamak`

Status: **implemented** (the meta-action-picker GEPA optimization of §7 and the Phase-2 search-picker are both shipped; see §7 and §10). This document is retained as the design rationale. Last updated 2026-07-18.

This document records why we are (or aren't) pulling DSPy into the agent stack, what we'd actually do with it given the state of the repo today, and the ordering of work.

For project context see [`project_agenda.md`](project_agenda.md). For repo mechanics see [`../CLAUDE.md`](../CLAUDE.md).

---

## 1. Honest assessment

DSPy is well-suited to projects where:
- each evaluation is cheap (≲1s, deterministic),
- there's a labeled corpus of ≥20–50 `(input, expected_output)` pairs to optimize against,
- the bottleneck is prompt wording (not API grounding, retrieval, or tool selection).

Our project today:
- each evaluation is **expensive** — one full URSA loop on `gpt-5.2` is $1–5 and 15–30 minutes (planner + executor + TokaMaker solves),
- we have **~3 historical runs** (one passed, two failed before the gpt-5-mini → gpt-5.2 + API-reference fix),
- the most recent agent failures were **API-grounding** problems (hallucinated `EquilibriumConfig(r0=..., a=...)` flat constructor, invented `geometry.lcfs_from_params`), not prompt wording.

That means DSPy is **plausible but premature** as a drop-in fix. We need to do groundwork before any optimizer can earn its cost.

## 2. The three-option ladder

We see DSPy paying off here in three steps of increasing ambition:

| Option | What it optimizes | Eval cost / trial | When to start |
|---|---|---|---|
| **C. Pre-flight prompt linter** | A small DSPy module predicts `P(success)` from a candidate prompt YAML and suggests edits. Eval = forward pass of the predictor model only. | ~$0.001 | Now |
| **B. Repair / feedback step** | The replanner prompt invoked after an execution step fails. Builds the `(error → fix)` corpus naturally from agent runs. | $0.10–0.50, 1–3 min | After we have ~10 failed-and-fixed traces logged |
| **A. Full-pipeline planner optimization (the destination)** | URSA's `PlanningAgent` instruction string, against "did the executor produce a working dataset". MIPROv2 territory. | $1–5, 15–30 min | After ~20–50 labeled traces exist |

**Recommendation:** start with **C**, instrument every run to feed **B**, and earn the right to do **A** once the corpus exists. Do not skip ahead.

## 3. Prerequisite: run instrumentation

None of the three options is possible without structured per-run traces. Today the runners stream text to stdout (and the codex / Claude session logs it). That's lossy and unparseable.

**What we need to add (small patch, not on this branch yet):**

Each agent invocation writes `experiments/<run_id>/trace.json` containing:

```jsonc
{
  "run_id": "20260616T120000Z",
  "prompt_yaml_hash": "sha256:...",
  "prompt_yaml_path": "src/autotokamak/agent/prompts/dataset_generation.yaml",
  "model": "openai:gpt-5.2",
  "feedback_rounds": 2,
  "started_utc": "...",
  "finished_utc": "...",
  "plan_steps": [{"name": "...", "description": "..."}, ...],
  "execution": [
    {"step": 1, "ok": true, "result_excerpt": "...", "tool_calls": N},
    {"step": 2, "ok": false, "error": "...", "result_excerpt": "..."},
    ...
  ],
  "artifacts": {
    "workspace": "examples/dataset_generation/",
    "files": ["run_dataset_sweep.py", "dataset_config.yaml", ...],
    "dataset_h5": "examples/dataset_generation/outputs/dataset.h5"
  },
  "metric": { ... emitted by score_run() below ... }
}
```

`experiments/` is already in `.gitignore`. The runner imports a tiny helper that opens the file at startup and appends sections as it progresses.

## 4. The metric function (concrete, today)

See [`../src/autotokamak/agent/dspy/metric.py`](../src/autotokamak/agent/dspy/metric.py).

It is pure Python — no DSPy dependency yet — so it can be run retrospectively against any workspace produced by `dataset_generation.yaml`. Later, when DSPy is installed, it plugs in as `metric=score_run` to any DSPy optimizer.

Composite shape:

```
score_run(workspace_dir, *, requested_n_samples) -> ScoreReport
  - hard gates (boolean, all must pass for nonzero score):
      * three deliverables present
      * outputs/dataset.h5 exists and opens
      * at least one /outputs/success == True
  - quality score (weighted sum, in [0, 1] if all gates pass):
      * 0.40 * (n_succeeded / n_requested)
      * 0.30 * inside_lcfs_finite_fraction
        -- measured by masking psi to the LCFS bbox and checking finite values
           are not artifacts of nearest-fill (variance test)
      * 0.20 * shape_fidelity
        -- correlation between requested (r0, a, kappa) and observed plasma
           centroid + extent measured from psi contours
      * 0.10 * runner_cleanliness
        -- did the runner import from autotokamak.core (heuristic)
```

The 0.30 "inside_lcfs_finite_fraction" term is the metric we wish the recent run had been scored against — it would have flagged the `griddata(nearest)` silent-fill bug at score time, not at code-review time.

## 5. Option C in detail — the prompt linter

Once instrumentation exists and there are ≥10 traces:

```python
# src/autotokamak/agent/dspy/linter.py  (NOT WRITTEN YET — sketch)

import dspy

class PromptQualityPredictor(dspy.Signature):
    """Predict whether an agent prompt YAML will produce a working workspace."""
    prompt_yaml: str = dspy.InputField()
    predicted_score: float = dspy.OutputField(desc="0-1, higher = more likely to succeed")
    weak_spots: list[str] = dspy.OutputField(desc="concrete edits that would raise the score")
```

Optimization:
- Bootstrap on the trace corpus: pairs of `(prompt_yaml_text, score_run(workspace).total_score)`.
- Use `dspy.BootstrapFewShot` (no LLM optimizer calls — cheap).
- Hold out ≥30 % of traces for validation.

Deliverables for C:
- `linter.py` — the DSPy module
- a CLI: `python -m autotokamak.agent.dspy.linter src/autotokamak/agent/prompts/new_prompt.yaml`
- output: predicted score + a list of weak-spot edits

This is a tool you can run *before* burning a $3 agent invocation on a bad prompt. ROI is immediate.

## 6. Option B — repair step optimization (later)

Once Option C exists and we have ≥10 traces *with execution failures*, the next target is the planner's replan step in `plan_execute_feedback.py`. Specifically:

- DSPy signature: `(original_problem, failure_history) -> revised_steps`
- Metric: did the revised plan's next round succeed? (binary on next round + score from §4)
- Optimizer: `dspy.BootstrapFewShot` first; `MIPROv2` once we have ≥30 failure traces

## 7. Option A — meta-action-picker optimization (SHIPPED, Phase 4)

**Status: implemented**. See `src/autotokamak/agent/dspy/{signatures,module,trace_loader,metric_adapter,optimize_meta}.py`. We chose GEPA (Genetic-Pareto) over MIPROv2 because (a) our scorer is multi-criterion — 6 quality terms in `metric_meta.WEIGHTS` — and GEPA can Pareto-optimize without collapsing them, and (b) GEPA's reflective mutation reads the agent's natural-language `diagnosis` field in our traces to inform each prompt mutation.

The optimization target is the meta-action-picker — the LLM call inside `meta_loop.pick_action_via_llm` that decides between `regen_dataset` / `extend_search` / `enrich_active` / `terminate` each iteration (four actions; `enrich_active` is the active-learning acquisition action). Its prompt is now `MetaActionPicker.__doc__` in `signatures.py`, and GEPA mutates that docstring.

### Four-command runbook

```bash
# 1. Build (already done — this PR)
pytest tests/ -v

# 2. Collect traces (~$5-15 on gpt-5-mini, ~30-60min)
./tools/collect_traces.sh --n 15 --model openai:gpt-5-mini

# 3. Run GEPA (~$10-50 in optimization compute)
PYTHONPATH=src/autotokamak python -m autotokamak.agent.dspy.optimize_meta \
    --experiments-dir experiments/ \
    --output src/autotokamak/agent/dspy/optimized/meta_picker.json \
    --auto medium

# 4. A/B verify
./tools/collect_traces.sh --n 5 --use-baseline --tag baseline
./tools/collect_traces.sh --n 5 --tag optimized
# Compare scores in the two experiments_* directories
```

The runner picks up the optimized JSON automatically (via `module.load_module()`); pass `--use-baseline` to force the in-code baseline for A/B comparison.

### Phase-4 follow-ups

- **Phase-2 search-picker optimization** — ✅ **shipped**. `SearchRoundPicker` (`signatures.py`) plus `SearchRoundPickerModule`, `load_search_module`, `make_search_decision_fn`, and `DEFAULT_SEARCH_OPTIMIZED_PATH` in `module.py` implement the surrogate_automl agent's per-round `SearchSpec` decisions with the same GEPA pattern.
- **Phase-1 dataset_generation prompt optimization** — still deferred; lowest leverage; the prompt is already tight and the main failure is solver-side (the isoflux fallback).
- **Cross-prompt optimization** — once all three phase prompts are DSPy modules, run GEPA jointly over them.

The original "Option A — full-pipeline optimization (destination)" framing (optimize the whole URSA planner instruction string with MIPROv2 + $500-1500 budget) is now superseded by the GEPA-on-meta-action-picker approach, which is cheaper and more contained.

## 8. Risks and open questions

- **`gpt-5.2` vs `gpt-5-mini` regression risk.** Our optimization may produce prompts that work *only* on `gpt-5.2`. Track stability across models in the trace.
- **Sweep diversity.** We currently have one prompt task (data-gen). Option A needs ≥3 distinct task families (data-gen, surrogate-baseline, surrogate-AutoML) to avoid overfitting to one workflow.
- **Constraint preservation.** The `CONSTRAINTS:` block in every prompt (`don't pip install`, `don't write into ./ursa/`, etc.) is load-bearing. DSPy optimizers must be configured to leave it untouched — only `TITLE`, `GOAL`, `DELIVERABLES`, `API REFERENCE` are optimizable.
- **DSPy lock-in.** DSPy is a real dependency with its own update cadence. Pin the version in `pyproject.toml` once we adopt.

## 9. Open scope decisions to bring to the advisor

1. **Where does DSPy sit in the Phase 1 / Phase 2 split?** It's orthogonal to "data gen → AutoML over surrogates," and probably belongs as a meta-tool that improves both phases over time. Confirm with advisor.
2. **Budget for Options A, B?** Need a token-budget cap before we run MIPROv2.
3. **Is the prompt-linter framing (Option C) interesting on its own?** Or does the advisor want us to push straight at Option A even though it's premature?

## 10. What lives in the repo today

**Shipped:**
- Run instrumentation (`agent/runners/trace.py`) — writes `experiments/<run_id>/trace.json` for every agent invocation.
- Composite scorers (`agent/dspy/metric.py`, `metric_surrogate.py`, `metric_meta.py`) — pure-Python `ScoreReport` producers, designed to plug into DSPy as metrics.
- **Option A (Phase 4)**: GEPA optimization of the meta-action-picker. See §7. New files:
  - `agent/dspy/signatures.py` — `MetaActionPicker` dspy.Signature
  - `agent/dspy/module.py` — `MetaActionPickerModule(dspy.ChainOfThought)` + ActionDecision coercion
  - `agent/dspy/trace_loader.py` — converts `experiments/<id>/trace.json` + `meta_trace.json` → `dspy.Example`
  - `agent/dspy/metric_adapter.py` — GEPAFeedbackMetric implementation
  - `agent/dspy/optimize_meta.py` — CLI script
  - `tools/collect_traces.sh` — drive N varied agent runs for the trainset

- **Phase-2 search-picker (Phase-4 follow-up)**: `SearchRoundPicker` in `signatures.py` + `SearchRoundPickerModule` / `load_search_module` / `make_search_decision_fn` in `module.py`.

**Deferred / not yet implemented:**
- Option C (`linter.py`) — pre-flight prompt linter
- Option B — replan-step optimization
- Phase-1 dataset_generation prompt optimization via the same GEPA pattern
