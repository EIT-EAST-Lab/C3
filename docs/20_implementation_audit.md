# Implementation Audit

This document maps the C3 paper to the implementation in this repository and records the main execution contracts that should remain stable across releases.

## Scope

- paper target: C3, credit assignment by counterfactual continuation, for LLM agent teams that communicate through a shared trace: the 2-agent `Reasoner -> Actor` and 3-agent `Reasoner -> Actor -> Verifier` workflows in training, and workflows of 2 to 10 decision points in the measurements,
- repository target: public release of the reference implementation,
- audit goal: make the paper-facing path, compatibility path, and reproducibility path explicit.

## Core execution graph

```mermaid
flowchart TD
PaperSpec["PaperSpec"] --> TaskConfigs["configs/tasks/*.yaml"]
TaskConfigs --> RoleConfigs["configs/roles/**/*.json"]
TaskConfigs --> TaskLoader["c3/task/config.py"]
TaskLoader --> DatasetLoader["c3/task/datasets.py"]
RoleConfigs --> RoleGraph["c3/protocol/role_graph.py"]
RoleGraph --> PromptRender["c3/protocol/prompt_render.py"]
DatasetLoader --> RolloutGen["c3/protocol/rollout_generator.py"]
PromptRender --> RolloutGen
RolloutGen --> ExperienceMaker["openrlhf/trainer/ppo_utils/experience_maker.py"]
ExperienceMaker --> CreditProvider["c3/credit/*"]
ExperienceMaker --> EnvReward["c3/envs/math/* + c3/envs/code/*"]
CreditProvider --> Trainer["openrlhf/trainer/ppo_trainer.py"]
EnvReward --> Trainer
Trainer --> MainResults["c3/reporting/main_results.py"]
Trainer --> Analysis["c3/analysis/* + c3/reporting/analysis_results.py"]
```

## Paper-to-code matrix

| Paper area | Primary implementation | Role in repo | Key contract |
| --- | --- | --- | --- |
| Protocol and roles | `configs/tasks/math.yaml`, `configs/tasks/code.yaml`, `configs/roles/math/roles_duo.json`, `configs/roles/code/roles_duo.json` | Declares the paper-facing tasks and `Reasoner -> Actor` protocol | Task files are the paper defaults; role files define prompt text, dependencies, and `with_answer` |
| Task loading | `c3/task/config.py` | Canonical loader for task and role configs | `TaskSpec` must expose environment, role graph, and dataset specs in a stable shape |
| Dataset loading | `c3/task/datasets.py` | Converts task configs into HF datasets | Local dataset paths must resolve independent of cwd; eval suite names must remain stable |
| Multi-agent execution | `c3/protocol/role_graph.py`, `c3/protocol/prompt_render.py`, `c3/protocol/rollout_generator.py` | Builds topo order, renders prompts, materializes MAS rollouts | Topology must remain acyclic; prompt rendering must be deterministic; rollout metadata feeds downstream credit/reward |
| C3 credit path | `openrlhf/trainer/ppo_utils/experience_maker.py`, `c3/credit/provider.py`, `c3/credit/materialize.py`, `c3/credit/role_dag.py`, `c3/credit/q_prompt.py` | Computes per-node credit and broadcasts token-level advantages | This is the real C3 path used by training |
| MAPPO baseline | `c3/baselines/mappo.py` | Step-level GAE baseline | Consumes step-ordered episode tensors and returns scalar advantages/returns |
| MAGRPO baseline | `c3/baselines/magrpo.py` | Group-based baseline | Computes group-centered advantages, then broadcasts them over action tokens |
| C3 fallback | `c3/baselines/group_baseline.py` | Compatibility and fallback path only | Not the paper’s primary C3 implementation |
| Training entry | `openrlhf/cli/train_ppo_ray_tooling.py`, `openrlhf/cli/train_ppo_ray.py`, `openrlhf/trainer/ppo_trainer.py` | Normalizes args, creates critics, drives PPO/Q-critic training | `marl_algorithm=auto` does not imply `c3`; Q-critic is enabled only for specific C3 variants |
| Math evaluation | `c3/envs/math/reward.py`, `c3/envs/math/backends/marft/*` | Math reward and answer checking | Current paper path scores only the final actor output |
| Code evaluation | `c3/envs/code/reward.py`, `c3/envs/code/executor.py` | Code reward and sandboxed execution | Reward is a pass-rate style score based on hidden tests in metadata |
| Paper training recipes | `c3/utils/paper_train_contract.py`, `scripts/40_train/paper_train.sh` | The three trained tables of the paper: `main` (Table 2), `long` (Table 7), `a3` (Table 8) | Every hyperparameter comes from the recipe, which names its source in Tables 5 and 6 or Appendix B.2 |
| Evaluation of record | `scripts/70_rebuild/final_eval.py`, `c3/analysis/rebuild/aggregate_5seed.py`, `c3/analysis/rebuild/aggregate_e2.py` | Evaluates each trained model after training and aggregates Tables 2, 7, 8 and 10 | One evaluation run has one k; the aggregations write only the keys their `--manifest` defines, and `configs/results_keys_v3.json` lists the paper's |
| Measurements (Sections 3 and 4) | `scripts/70_rebuild/` drivers, `c3/analysis/rebuild/*` | Reliability, noise law, ablation bias and the critic comparison on a frozen policy | `scripts/70_rebuild/README.md` maps each driver to its table and section |
| Results sweep of the earlier paper versions | `scripts/50_eval/paper_main_results.sh` (`sweep`), `c3/reporting/main_results.py`, `configs/main_results_registry.yaml` | Eval-only sweep and table aggregation of the v1 and v2 protocol | Datasource names must match task-suite names and registry expectations |
| Analyses of the earlier paper versions | `scripts/60_analysis/paper_analysis_figs.sh`, `c3/analysis/*`, `c3/reporting/analysis_results.py`, `c3/reporting/plot_paper_figures.py` | Credit, influence, variance, and plotting pipeline of v1 and v2 | Bucket metadata is the contract for downstream metrics |

## Primary paper-facing defaults

### Tasks

- `math`: `configs/tasks/math.yaml`
- `code`: `configs/tasks/code.yaml`

### Roles

- math duo: `configs/roles/math/roles_duo.json`
- code duo: `configs/roles/code/roles_duo.json`

### Main-results entrypoints

- training: `scripts/40_train/paper_train.sh` (`RECIPE=main|long|a3`)
- evaluation of record: `scripts/70_rebuild/final_eval.py`, aggregated by `c3.analysis.rebuild.aggregate_5seed` and `c3.analysis.rebuild.aggregate_e2`
- measurements of Sections 3 and 4: `scripts/70_rebuild/` (see its README)
- results sweep and analyses of the earlier paper versions: `scripts/50_eval/paper_main_results.sh`, `scripts/60_analysis/paper_analysis_figs.sh`
- fast wiring check: `scripts/30_smoke/smoke.sh`

## Core-path vs compatibility-path

### Paper core path

The paper’s C3 path is:

1. `train_ppo_ray_tooling.py` normalizes the run configuration,
2. `train_ppo_ray.py` decides whether a Q-critic is needed,
3. `ppo_trainer.py` prepares Q-critic batches,
4. `experience_maker.py` materializes tree groups and requests node-level C3 credit,
5. `c3/credit/provider.py` computes scalar credit values,
6. the scalar credit is broadcast back to token-level PPO advantages.

### Compatibility or fallback paths

- `c3/baselines/group_baseline.py` is a fallback and compatibility calculator, not the main C3 algorithm.
- `env_smoke.py` contains compatibility bridging for task loading and should not be used as proof that the training path is correct.
- `configs/roles/critic_preamble.json` is not part of the default paper path and should be treated as optional or experimental until its consumer contract is documented more clearly.

## Release-critical invariants

1. `TaskSpec` must expose `train_datasets` and `eval_suites` in a shape directly consumable by `load_task_datasets()`.
2. Eval suite names in task YAML must propagate into dataset `datasource` names unchanged, because `main_results.py` aggregates by benchmark name.
3. Local dataset paths such as `data/MATH/train.jsonl` must resolve from any working directory, not only when the process runs from repo root.
4. The public release must document that the actual C3 credit path lives in `experience_maker.py` plus `c3/credit/*`, not in `c3/baselines/group_baseline.py`.
5. The public release must keep generated directories (`data/`, `artifacts/`, `ckpt/`, `runs/`, `wandb/`, `models/`) out of the shipped repository surface.

## Known interpretation notes

- `marl_algorithm=auto` for MAS tasks resolves to `magrpo` when `K > 1`; users must explicitly request `c3` for the paper method.
- The math paper path scores the final actor output. Multi-answer aggregation settings in `MathEnv` should be interpreted with that default in mind.
- The code path relies on tests embedded in dataset metadata; it is not symmetric with the math path’s `answer` field.
- `smoke.sh` is a wiring smoke test, not a performance regression test.

## Minimum verification targets

- task and role loading,
- dataset loading and benchmark naming,
- cwd-independent local path resolution,
- `RoleGraph` validity checks,
- prompt rendering contract,
- smoke-level math/code evaluator wiring,
- main-results aggregation on synthetic artifacts,
- analysis bucket parsing and metric aggregation.
