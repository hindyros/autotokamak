# just_dspy — single-prompt capability test, DSPy condition

## Question

Third arm of the experiment started in [just_ursa/](../just_ursa/README.md):
given the **same single task prompt** — build a Grad–Shafranov ψ(R,Z)
surrogate from scratch, with an adaptive-sampling data campaign under a
campaign structure (up to 10 adaptive rounds x 500 solves, 70% error-reduction early stop) and honest held-out evaluation — can a **DSPy-native agent**
generate the full codebase and actually run it to produce the outputs?

| Condition | Substrate | Harness |
|---|---|---|
| `just_ursa` | URSA `PlanningAgent` + `ExecutionAgent` | existing `agent/runners/plan_execute.py` |
| `just_dspy` | DSPy `ChainOfThought` + `ReAct` | [run_just_dspy.py](run_just_dspy.py) (standalone, ~300 lines) |

The harness shape is deliberately symmetric with the URSA runner (plan →
per-step execute, threading a previous-step summary) so the two conditions
differ in **agent substrate**, not in scaffolding intelligence.

## DSPy best practices used

- **Typed `dspy.Signature` classes** whose *docstring is the prompt* (the
  repo convention from `src/autotokamak/agent/dspy/signatures.py`, and what
  GEPA would mutate if you later optimize this program):
  - `PlanCampaign` — task → `steps: list[str]`, via `ChainOfThought`
  - `ExecuteStep` — (task, previous_summary, step) → `summary`, via `ReAct`
  - `ReviewDeliverables` — typed audit gate → `complete: bool`, `missing: list[str]`
- **`dspy.ReAct` with plain-function tools** (`write_file`, `read_file`,
  `list_files`, `run_shell`) — docstrings + type hints become the tool
  schemas; all file tools are jailed to the workspace.
- **One `dspy.Module` (`SurrogateCampaign`)** composing the three stages with
  control flow in `forward()`: plan → execute steps → review → up to
  `fix_rounds` targeted fix-up ReAct rounds driven by the typed `missing` list.
- Declarative LM config via `dspy.configure(dspy.LM(...))` (litellm string,
  `openai/gpt-5.2`).

Because the program is signatures + modules, it is **GEPA-optimizable later**
— a natural follow-up experiment if the zero-shot run is weak.

## Task parity with just_ursa

[task.yaml](task.yaml)'s `problem:` is byte-identical to
`just_ursa/task.yaml` except the URSA-runtime bullet
(`graph_store.sqlite`/`ursa_metrics`) is dropped — this harness writes
nothing into the workspace. Same parameter box, same campaign structure, same
evaluation grid/metric, same `predict.py` + `report.json` contract, same
`autotokamak`-forbidden rule, same optional OFT source symlink.

## Run

```bash
source venv/bin/activate
set -a && source .env && set +a

python just_dspy/run_just_dspy.py --config just_dspy/task.yaml

# Strict parity with the URSA single-pass condition (no review-driven fixes):
python just_dspy/run_just_dspy.py --config just_dspy/task.yaml --fix-rounds 0
```

Agent outputs land in `just_dspy/workspace/`; the harness trace goes to
`experiments/just_dspy_<UTC>/` (`trace.json` with plan/step summaries/review
verdicts, plus `tool_log.jsonl` with every tool call).

## What to look at afterward

Same checklist as [just_ursa](../just_ursa/README.md#what-to-look-at-afterward)
— completion, `grep -rn autotokamak just_dspy/workspace --include='*.py'`,
adaptive-sampling authenticity, the known traps (`get_psi(False)`, budget
burned on infeasible shapes, `OFT_env` singleton) — plus one DSPy-specific
question: **did the typed review gate catch real gaps?** Compare
`reviews` in `trace.json` against what was actually missing.

Head-to-head: both conditions emit the same `report.json` metrics on the same
grid, so `test_rel_l2` is directly comparable between
`just_ursa/workspace`, `just_dspy/workspace`, and the human-engineered
meta-loop (`examples/surrogate_meta/*/manifest.json`).

## Contents

```
just_dspy/
├── README.md          # this file
├── task.yaml          # the single prompt + harness config
├── run_just_dspy.py   # DSPy plan->execute harness
└── workspace/         # created at run time; everything the agent builds
```
