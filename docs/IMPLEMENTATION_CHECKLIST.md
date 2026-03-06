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

- The paper-facing C3 path remains centered on `openrlhf/trainer/ppo_utils/experience_maker.py` plus `c3/credit/c3/*`.
- `c3/algorithms/c3.py` remains documented as a fallback path, not the primary implementation.
- Changes to `marl_algorithm=auto` behavior are intentional and documented.

## Evaluation and aggregation contracts

- `MathEnv` and `CodeEnv` reward entrypoints remain compatible with the rollout metadata contract.
- `main_results.py` still aggregates by the expected benchmark names:
  - math: `MATH500`, `CMATH-test`, `GSM8K-test`
  - code: `MBPP-test`, `MBPP+`
- Analysis bucket metadata remains compatible with `c3/analysis/metrics.py` and `c3/tools/analysis_results.py`.

## Tests and release gate

- `pytest -q tests` passes.
- Fixture-based smoke passes for both math and code tasks.
- `bash scripts/audit/pre_release.sh` passes.
- `bash scripts/audit/release_gate.sh` passes.

## Documentation sync

- If the implementation path changed, update `docs/CODE_MAP.md`.
- If the paper-facing mapping changed, update `docs/IMPLEMENTATION_AUDIT.md`.
- If release behavior changed, update `README.md`, `docs/GETTING_STARTED.md`, and `docs/RELEASE_POLICY.md`.
