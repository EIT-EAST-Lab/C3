# Implementation Checklist

Use this checklist when changing protocol, data loading, credit assignment, evaluation, or paper-facing scripts.

## Task and dataset contracts

- `load_task()` exposes `train_datasets` and `eval_suites` in a form directly consumable by `load_task_datasets()`.
- Local dataset paths resolve correctly even when the current working directory is not the repo root.
- Eval suite names propagate unchanged into dataset `datasource` names.
- Fixture tasks under `tests/fixtures/tasks/` still load successfully.

## MAS protocol contracts

- `RoleGraph` still rejects duplicate names, missing dependencies, and cycles.
- Prompt rendering still supports `{question}`, `{context}`, and prior role outputs without raising on missing keys.
- The paper-default `Reasoner -> Actor` path remains intact in `configs/tasks/*.yaml` and `configs/roles/**/*.json`.

## Algorithm and credit contracts

- The paper-facing C3 path remains centered on `openrlhf/trainer/ppo_utils/experience_maker.py` plus `c3/credit/*`.
- `c3/baselines/group_baseline.py` remains documented as a fallback path, not the primary implementation.
- Changes to `marl_algorithm=auto` behavior are intentional and documented.
- `pytest -q tests/test_credit_mechanism.py` passes (LOO and full-mean advantages match the hand-derived values on the paper's fanout).

## Evaluation and aggregation contracts

- `MathEnv` and `CodeEnv` reward entrypoints remain compatible with the rollout metadata contract.
- `main_results.py` still aggregates by the expected benchmark names, which are
  the main-table suites since the evaluation protocol of 2026-09-15:
  - math: `MATH500`, `Minerva-Math`, `AMC23`, `AIME24`, `AIME25`
  - code: `MBPP-test`, `MBPP+`
  - math appendix, reported apart from the main table: `GSM8K-test`, `CMATH-test`
  - the reported `AIME` column is `AIME24` and `AIME25` merged, the same entry
    `scripts/70_rebuild/final_eval.py` writes
- Analysis bucket metadata remains compatible with `c3/analysis/metrics.py` and `c3/reporting/analysis_results.py`.

## Tests and release gate

- `pytest -q tests` passes.
- Fixture-based smoke passes for both math and code tasks.
- `pytest -q tests/test_release_surface.py` passes.
- `bash scripts/90_audit/pre_release.sh` passes.
- `bash scripts/90_audit/release_gate.sh` passes on a CPU machine.

## Documentation sync

- If the implementation path changed, update `docs/10_code_map.md`.
- If the paper-facing mapping changed, update `docs/20_implementation_audit.md`.
- If release behavior changed, update `README.md`, `docs/00_getting_started.md`, and `docs/50_release_policy.md`.
