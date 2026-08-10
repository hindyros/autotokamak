# Agent runners

Three runners live under `src/autotokamak/agent/runners/`. The primary end-user entry point is **`plan_execute_feedback`**; `plan_execute` is the legacy single-pass variant; `meta_loop` is the autonomous Phase-3 orchestrator.

All three write a structured trace to `experiments/<utc_run_id>/trace.json` unless `--no-trace` is given. That trace is the substrate for DSPy optimization and for `scripts/trace_to_html.py`.

## `plan_execute_feedback` (primary)

One or more plan → execute → re-plan cycles. After each round the planner sees the execution history and can propose follow-up steps to fix failures or terminate. The engine is exposed as `run_feedback_loop(...)` (importable, used by `meta_loop` sub-runs and the `ursa` bench harness); `main()` is the YAML CLI over it.

```
python -m autotokamak.agent.runners.plan_execute_feedback \
    --config src/autotokamak/agent/prompts/dataset_generation.yaml \
    [--model openai:gpt-5-mini] \
    [--workspace examples/dataset_generation] \
    [--no-trace] \
    [--experiments-dir experiments]
```

**Config keys** (see `references/configs.md#4-agent-prompt-config`): `problem` (required), `workspace`, `model`, `symlinks`, `feedback_rounds` (default 2), `validate_after` (default false), `scorer`, `scorer_kwargs`, `expected_artifacts`, `allowed_root_files`, `infra_root_files`.

**Where feedback fires:** after each round, the workspace root is scanned; if extras appear beyond `allowed_root_files + infra_root_files`, the next round's planner prompt gets a `WORKSPACE HYGIENE WARNING` block instructing it to `rm` them. This is how prompt CONSTRAINTS get enforced across rounds.

**Outputs:** artifacts land in `workspace/` (per the prompt's expectations). Trace at `experiments/<run_id>/trace.json`. Score (if `scorer` is set) recorded on the trace.

## `plan_execute` (legacy, single-pass)

```
python -m autotokamak.agent.runners.plan_execute --config PATH [--model M] [--workspace W] [--no-trace]
```

No re-plan, no hygiene enforcement. Prefer `plan_execute_feedback` unless the prompt is intentionally single-shot.

## `meta_loop` (Phase-3 autonomous)

Consumes the `surrogate_meta.yaml` prompt shape (no `problem:` field; structured `ActionDecision` output each iteration).

```
python -m autotokamak.agent.runners.meta_loop \
    --config src/autotokamak/agent/prompts/surrogate_meta.yaml \
    [--model openai:gpt-5.2] \
    [--max-iterations 3] \
    [--use-baseline] \
    [--no-trace]
```

**Config keys:**

```yaml
max_iterations: 3
seed: 0
initial_dataset_h5: examples/dataset_generation/outputs/dataset.h5
base_sweep_config: examples/dataset_generation/dataset_config.yaml
phase2_prompt: src/autotokamak/agent/prompts/surrogate_automl.yaml
workspace: examples/surrogate_meta
model: openai:gpt-5.2
```

**Per-iteration flow:**

1. Compute deterministic diagnostics on the current dataset (via `autotokamak.surrogate.diagnostics.run_all`).
2. Call `pick_action_via_llm` → returns a Pydantic `ActionDecision` with `action ∈ {regen_dataset, extend_search, enrich_active, terminate}` and a one-sentence `diagnosis`.
3. Dispatch via `autotokamak.agent.orchestrator.actions.dispatch`:
   - `regen_dataset(overrides)` → apply overrides to `base_sweep_config`, re-run Phase-1, update `state.current_dataset_h5`.
   - `extend_search(focus)` → run a fresh Phase-2 search, load the winner, update `best_rmse` if improved. Phase-2 mode is `structured` (library AutoML) by default and always when driven by `pipelines meta`; the `codegen` hybrid (URSA writes the Phase-2 code) is reachable only via the meta YAML config.
   - `enrich_active(n_new)` → residual-driven active-learning acquisition of new samples.
   - `terminate(reason, confidence)` → stop.
4. Measure test RMSE with the current best winner on the current dataset.

`meta_loop.run()` also accepts a `phase2_decision_fn=` parameter (structured mode only) that replaces the DSPy search-round picker with a caller-supplied decision function — this is how the L0 scripted policy (`autotokamak.policies.scripted`) drives it without any LLM.

**Outputs:**
- `workspace/iterations/<NNN>/{diagnostics,action,result}.json` — per-iteration audit trail.
- `workspace/meta_trace.json` — full log.
- `workspace/report.json` — final `MetaReport` (n_iterations, terminated_by, final_rmse, baseline_rmse, rmse_history, actions_taken).
- `workspace/winner.pkl` — best surrogate copied from the winning `extend_search` child.
- `experiments/<run_id>/trace.json` — outer meta-run trace.

## DSPy integration (in-flight)

`src/autotokamak/agent/dspy/`:

- `signatures.py` — `MetaActionPicker` DSPy signature (inputs: diagnostics + history + state; output: `ActionDecision`).
- `module.py` — `MetaActionPickerModule` wraps that signature in a `dspy.ChainOfThought`; `load_module(path)` loads a GEPA-optimized JSON checkpoint from `agent/dspy/optimized/meta_picker.json`.
- `trace_loader.py` — converts meta-run traces into DSPy `Example` objects for training.
- `optimize_meta.py` — GEPA CLI to optimize the meta prompt from cached traces.
- `metric_adapter.py` — bridges `score_meta_run` to GEPA's metric protocol.

The `meta_loop` runner uses the DSPy module by default; `--use-baseline` forces the in-code baseline picker for A/B comparison.

## Prompt → runner map

| Prompt YAML | Runner | Rationale |
|---|---|---|
| `dataset_generation.yaml` | `plan_execute_feedback` | Feedback loop enforces workspace hygiene during dataset gen. |
| `oft_example_generation.yaml` | `plan_execute` | Legacy single-shot demo generation. |
| `oft_discretization_example.yaml` | `plan_execute_feedback` | Same as dataset_generation. |
| `surrogate_automl.yaml` | `plan_execute_feedback` | Feedback loop for AutoML rounds. |
| `surrogate_meta.yaml` | `meta_loop` | Structured output; no plan_execute. |

`scripts/run_agent.py` encodes this mapping — pass `--prompt NAME` and the wrapper dispatches to the correct runner.

## Trace format

The trace writer is `autotokamak.bench.trace.RunTrace` and the run scorer is `autotokamak.bench.scoring` (both moved from `agent/runners/` into the `bench` package; the same trace shape is emitted by benchmark harness runs under `experiments/<tag>/<condition>/<run_id>/`). Every runner writes an `experiments/<utc_run_id>/trace.json` shaped like:

```json
{
  "run_id": "20260709T151200Z",
  "started_utc": "...",
  "finished_utc": "...",
  "status": "completed" | "errored" | "interrupted",
  "prompt": {"path": "...", "model": "...", "workspace": "...", "feedback_rounds": 2},
  "rounds": [
    {"round": 1, "plan_steps": [...], "execution": [{"step": 1, "name": "...", "ok": true, "result_excerpt": "...", "started_utc": "...", "finished_utc": "..."}, ...]},
    ...
  ],
  "artifacts": {"files_written": [...], "expected": {...}},
  "score": {"total": 0.87, "hard_gates": {...}, "quality": {...}, "details": {...}},
  "parent_run_id": "..."  // set for meta_loop children
}
```

Render to browsable HTML with `scripts/trace_to_html.py`.
