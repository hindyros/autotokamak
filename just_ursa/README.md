# just_ursa — single-prompt URSA capability test

## Question

Everything this repo does — Phase-1 dataset generation, active-learning
acquisition, Phase-2 surrogate AutoML, scoring — was engineered by humans into
pipelines, prompts, and schemas. **Can a vanilla URSA plan→execute agent do the
same job from one prompt**, with no access to any of that scaffolding?

This folder is that experiment. [task.yaml](task.yaml) contains the single
prompt. It specifies *the task* (surrogate for fixed-boundary Grad–Shafranov
ψ(R,Z), adaptive sampling under a 400-solve budget, honest held-out
evaluation) and *the evaluation interface* (fixed grid, relative-L2 metric,
`predict.py` contract) — and deliberately **nothing** about how this repo
solves it: no `autotokamak.core` API, no dataset schema, no sampling method,
no model zoo, no meta-loop structure.

## Fairness controls

| Control | Why |
|---|---|
| `autotokamak` import/read explicitly forbidden in the prompt | It is importable in the venv; using it would trivially replicate our pipeline. After the run, grep the generated code for `autotokamak` to verify compliance. |
| OFT source symlinked as reference (optional) | Humans had the TokaMaker docs/examples; the agent should too. Side-clone `OpenFUSIONToolkit` at the repo root first, or the symlink is warn-skipped and the agent must work from the installed package alone (a harder condition — note which one you ran). |
| Fixed evaluation grid, physical ψ in Wb, defined metric | Makes its numbers directly comparable to our pipeline's; also plants the ψ-vs-ψ_N trap honestly (the target is *defined*, but discovering `get_psi(False)` is the agent's job). |
| Held-out test protocol prescribed (≥60 uniform-random solves, never touched) | Without it, self-reported metrics are not trustworthy. |
| Hard environment facts stated (one `OFT_env` per process, no pip) | Documented library limitations, not design hints; omitting them only tests patience, not capability. |

## Run

```bash
source venv/bin/activate
set -a && source .env && set +a

# Primary condition: single-pass plan -> execute (purest one-shot test)
PYTHONPATH=src/autotokamak python -m agent.runners.plan_execute \
    --config just_ursa/task.yaml

# Optional second condition: with replan/feedback rounds
PYTHONPATH=src/autotokamak python -m agent.runners.plan_execute_feedback \
    --config just_ursa/task.yaml
```

Outputs land in `just_ursa/workspace/`; the run trace goes to
`experiments/<run_id>/trace.json` (browse with `tools/trace_to_html.py`).
Expect roughly 1–2 h wall-clock if the agent gets the solver loop working.

## What to look at afterward

1. **Did it finish?** `workspace/report.json` exists with real metrics;
   `predict.py` honors the CLI contract.
2. **Did it cheat?** `grep -rn autotokamak just_ursa/workspace --include='*.py'`
   should be empty.
3. **Was the sampling actually adaptive?** Read the acquisition log — is there
   a model-informed selection rule, or a random design wearing a costume? Check
   the `adaptive_vs_initial` evidence in `report.json`.
4. **Known traps** (things our platform handles that the agent must
   rediscover): physical ψ via `get_psi(False)` (default is normalized flux —
   trains a model with a dead `Ip` input); mesh failures on high-δ/κ shapes
   eating the solve budget; single `OFT_env` per process when parallelizing.
5. **Head-to-head numbers.** Same grid, same metric: compare its
   `test_rel_l2` against our meta-loop run
   (`examples/surrogate_meta/*/manifest.json`). For a common yardstick,
   generate one shared set of test parameters, compute ground truth with our
   pipeline, and run the agent's `predict.py` on it.

## Contents

```
just_ursa/
├── README.md      # this file
├── task.yaml      # the single prompt + runner config
└── workspace/     # created at run time; everything the agent builds
```
