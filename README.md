<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="figs/c3-logo-dark.svg">
    <img src="figs/c3-logo-light.svg" alt="C3, credit assignment by counterfactual continuation" width="360">
  </picture>
  <p><strong>When the trace is the state, credit need not be predicted; it can be exact.</strong></p>
  <p>
    <a href="https://eit-east-lab.github.io/C3/"><img src="https://img.shields.io/badge/Project-Page-B24A2F" alt="Project Page"></a>
    <a href="https://arxiv.org/abs/2603.06859"><img src="https://img.shields.io/badge/arXiv-2603.06859-B31B1B" alt="arXiv 2603.06859"></a>
    <a href="https://arxiv.org/pdf/2603.06859"><img src="https://img.shields.io/badge/PDF-arXiv%20Paper-7B2F1A" alt="arXiv PDF"></a>
    <a href="docs/20_implementation_audit.md"><img src="https://img.shields.io/badge/Paper-Implementation%20Audit-8A2BE2" alt="Paper Implementation Audit"></a>
    <a href="docs/50_release_policy.md"><img src="https://img.shields.io/badge/Release-Policy-0A66C2" alt="Release Policy"></a>
    <a href=".github/workflows/ci-cpu.yml"><img src="https://github.com/EIT-EAST-Lab/C3/actions/workflows/ci-cpu.yml/badge.svg" alt="CPU CI"></a>
    <a href=".github/workflows/ci-cpu.yml"><img src="https://img.shields.io/github/check-runs/EIT-EAST-Lab/C3/main?nameFilter=release-audit&label=release%20audit" alt="Release audit"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License Apache 2.0"></a>
    <a href="pyproject.toml"><img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white" alt="Python 3.11"></a>
  </p>
  <p>
    <a href="https://eit-east-lab.github.io/C3/">Project Page</a> |
    <a href="https://arxiv.org/abs/2603.06859">arXiv</a> |
    <a href="https://arxiv.org/pdf/2603.06859">PDF</a> |
    <a href="docs/00_getting_started.md">Getting Started</a> |
    <a href="docs/10_code_map.md">Code Map</a> |
    <a href="docs/21_implementation_checklist.md">Implementation Checklist</a> |
    <a href="docs/51_release_checklist.md">Release Checklist</a> |
    <a href="CHANGELOG.md">Changelog</a>
  </p>
</div>

Reference implementation for the paper **The Trace Is the State: Exact Credit Assignment for LLM Agent Teams**.

Paper status: preprint on arXiv as [2603.06859](https://arxiv.org/abs/2603.06859), under review. The companion project page is at [eit-east-lab.github.io/C3](https://eit-east-lab.github.io/C3/).

Repository version 0.3.0. See [CHANGELOG.md](CHANGELOG.md) for what changed and why.

## TL;DR

Train a team of LLM agents from a single terminal reward and you have to answer, for every message in the trace, how much it was worth. Credit assignment for such teams has mostly been treated as prediction: a learned critic, a trajectory-level score, or an agent ablation stands in for the counterfactual that defines credit, which is rarely run. Teams that communicate through a shared context are different: when everything a downstream agent reads is written into the trace, the trace is the state, and that counterfactual can be executed. **C3**, credit assignment by counterfactual continuation, substitutes one message at a decision point, continues the run to the terminal reward, and compares each alternative with the continuation-weighted mean return of the others. Its credit is unbiased, exact up to Monte Carlo error, and needs no learned parameters.

<p align="center">
  <a href="#core-mechanism">Mechanism</a> |
  <a href="#key-results">Results</a> |
  <a href="#cpu-quickstart">CPU Quickstart</a> |
  <a href="#gpu-path">GPU Path</a> |
  <a href="#main-workflows">Workflows</a> |
  <a href="#audit-and-release-gate">Release Gate</a>
</p>

## Core Mechanism

<div align="center">
  <img src="figs/fig1_overview.png" alt="Two ways to value an alternative message: predicted by a critic, or executed by C3 when the trace is the state" width="90%">
</div>

**Figure 1**: Two ways to value an alternative message. (a) Credit from predicted counterfactuals: the decision point is not revisited, so a critic fitted on past runs predicts the value of each alternative. (b) Credit from executed counterfactuals, when the trace is the state: C3 resets the run to the trace prefix, substitutes an alternative message, runs the downstream roles to the terminal reward, and compares the mean reward of each alternative with the continuation-weighted mean of the other alternatives. Executed, not predicted: against a 16-continuation reference, C3 from 4 continuations reaches 0.69 rank correlation, and the critic reaches 0.29.

## Key Results

When the trace is the state, a credit signal can be judged as any estimator is: by bias, variance, and agreement with an independent reference. C3 is unbiased under that condition; on the other two criteria, and in training, the paper reports:

- **Agreement with a reference.** On the 2-agent chain, from 4 continuations, C3 ranks alternatives at **0.69** rank correlation against a 16-continuation reference, near the **0.73** at which 2 such references agree; a critic trained on the same rollouts reaches **0.29**.
- **Variance.** Given the sampled alternatives, the variance of each advantage estimate follows a derived law with no term for the number of agents; at the opening decision point of 6 workflows of 2 to 10 decision points, the observed variance is **0.90 to 1.07** times the law.
- **Training.** Used as the advantage in training a 2-agent team, C3 beats MAGRPO, the stronger baseline, by **+8.28** points on MATH500 over 5 training seeds (Holm-corrected p < 0.001).
- **Cost.** At a 2048-token generation limit, C3 spends **37%** fewer training tokens than MAPPO, since only the messages downstream of a decision point are regenerated.

The 5-seed run (Table 2 of the paper):

| Method | MATH500 | AIME 2025 | CMATH | GSM8K | Avg.† |
|:-|:-|:-|:-|:-|:-|
| MAPPO | 69.3 ± 0.9 | 3.3 ± 0.0 | 95.3 ± 0.2 | 92.9 ± 0.1 | 85.8 ± 0.3 |
| MAGRPO | 74.5 ± 0.4 | 5.3 ± 1.8 | 96.1 ± 0.2 | **93.4 ± 0.3** | 88.0 ± 0.1 |
| C3 | **82.8 ± 0.6** | **8.0 ± 1.8** | **96.3 ± 0.2** | **93.4 ± 0.2** | **90.9 ± 0.3** |

Qwen3-4B: greedy accuracy (%) under a 512-token generation limit, mean ± std over 5 training seeds on the 2-agent workflow; best value per column in bold, including ties at the printed precision. †Avg. is the mean of MATH500, CMATH and GSM8K; AIME 2025 (30 problems) is reported beside it, not inside it.

### Reproducing the experiments

[c3/utils/paper_train_contract.py](c3/utils/paper_train_contract.py) holds three training recipes, with the settings of Tables 5 and 6 and Appendix B.2 of the paper. [scripts/40_train/paper_train.sh](scripts/40_train/paper_train.sh) trains MAPPO, MAGRPO and C3 under one of them (GPU):

```bash
export PRETRAIN=Qwen/Qwen3-4B-Instruct-2507
RECIPE=main bash scripts/40_train/paper_train.sh   # Table 2: 2-agent workflow, 512 tokens, training seeds 0 to 4
RECIPE=long bash scripts/40_train/paper_train.sh   # Table 7: 2-agent workflow, 2048 tokens, training seed 0
RECIPE=a3   bash scripts/40_train/paper_train.sh   # Table 8: 3-agent workflow, 512 tokens, training seeds 0 to 4
DRY_RUN=1 RECIPE=main bash scripts/40_train/paper_train.sh   # print the command lines, start nothing
```

Each trained model is evaluated by [scripts/70_rebuild/final_eval.py](scripts/70_rebuild/final_eval.py) (GPU). On the CPU, `python -m c3.analysis.rebuild.aggregate_5seed` turns the records of the 5-seed runs into Tables 2, 8 and 10, and `python -m c3.analysis.rebuild.aggregate_e2` turns those of the long-generation run into Table 7 and its training-token account.

The measurements of Sections 3 and 4 run on a frozen policy: a driver under [scripts/70_rebuild/](scripts/70_rebuild/) generates the rollouts (GPU), and a module under [c3/analysis/rebuild/](c3/analysis/rebuild/) computes the numbers (CPU).

| Measurement | Generation (GPU) | Analysis (CPU) |
|---|---|---|
| Split-half reliability of 6 workflows (Table 1, Figures 2 and 3) | `e1_cells.py` | `aggregate_e1` |
| Signal and noise behind reliability (Table 1 Implied column, Table 3 lower block) | `e1_cells.py` | `e1_signal_noise` |
| Noise law on the grid of branching factors and continuations (Table 3 upper block) | `e1b_cells.py` | `aggregate_e1b` |
| Bias of agent ablation (Section 4, Appendix B.6, Table 9) | `e3a_cells.py` | `aggregate_e3a`, `e3a_strata` |
| Learned critic against C3 (Section 4, Appendix B.7) | `e4_reference_cells.py`; `e4_critic.py` `train` and `score` | `e4_critic.py` `split`, `prepare-reference` and `evaluate` |

Every driver takes `--dry-run`. The full command of each experiment, and the table each code name maps to, are in [scripts/70_rebuild/README.md](scripts/70_rebuild/README.md). The repository ships no data, checkpoints or results.

## Repository Purpose

`C3` is a paper-aligned research codebase designed to support:

- **Protocol Reproduction**: Executing the multi-agent task protocols described in the paper.
- **Experiment Execution**: Running full training sweeps, evaluation-only probes, and performance analyses.
- **Implementation Audit**: Providing a transparent mapping from theoretical mechanisms to executable code.
- **Extension and Testing**: Modifying or testing the framework without relying on private paths or bundled artifacts.

## Release Scope

To maintain a clean public-release surface, this repository does **not** distribute:
- Prepared datasets
- Trained model checkpoints
- Cached model weights
- Generated run-time artifacts (e.g., run logs, checkpoints, reports, and experiment outputs)

Corresponding local directories (`data/`, `artifacts/`, `ckpt/`, `runs/`, `wandb/`, `models/`) are treated as local working outputs and remain empty or absent in the public release. For details, see the [Release Policy](docs/50_release_policy.md).

## Repository Structure

- [c3/](c3/): Core C3 implementation, including multi-agent protocol handling, environments, credit assignment logic, and analysis tools.
- [openrlhf/](openrlhf/): Vendored upstream RLHF training stack, augmented with C3-specific integration points.
- [configs/](configs/): Configurations for tasks, roles, analyses, execution registries, and data manifests.
- [requirements/](requirements/): Lock files for the CPU tier, the current GPU tier, and the paper environment.
- [scripts/](scripts/): Entrypoints, numbered in the order a new user runs them.
- [docs/](docs/): Documentation, numbered in reading order.
- [tests/](tests/): [mechanism/](tests/mechanism) for the credit-assignment arithmetic, [contract/](tests/contract) for the release surface and the data contracts, [unit/](tests/unit) for everything else, and [fixtures/](tests/fixtures) for tiny inputs.
- [project-page/](project-page/): Static companion site for the paper, deployed via GitHub Pages.

The `scripts/` numbers are the workflow:

| Directory | Stage |
|---|---|
| [scripts/10_data/](scripts/10_data/) | Download and pin the datasets |
| [scripts/20_models/](scripts/20_models/) | Pre-download the base models |
| [scripts/30_smoke/](scripts/30_smoke/) | Fast wiring checks and the reproduction preflight |
| [scripts/40_train/](scripts/40_train/) | The paper's training recipes (GPU) |
| [scripts/50_eval/](scripts/50_eval/) | The evaluation runner that `final_eval.py` calls, and the results sweep of the paper's earlier versions (GPU) |
| [scripts/60_analysis/](scripts/60_analysis/) | Analysis figures of the paper's earlier versions |
| [scripts/70_rebuild/](scripts/70_rebuild/) | The experiments of the paper: measurement cells, the critic comparison, the evaluation of record, and the start-accuracy probe (GPU) |
| [scripts/90_audit/](scripts/90_audit/) | Release audit and the local release gate |
| [scripts/_lib/](scripts/_lib/) | Shared shell helpers, sourced rather than run |

## Quick Navigation

- **Project Page**: [C3 Paper Page](https://eit-east-lab.github.io/C3/)
- **Paper**: [arXiv Abstract](https://arxiv.org/abs/2603.06859)
- **PDF**: [arXiv PDF](https://arxiv.org/pdf/2603.06859)
- **New User**: [Getting Started Guide](docs/00_getting_started.md)
- **Code Layout**: [Code Map](docs/10_code_map.md)
- **Paper-to-Code Mapping**: [Implementation Audit](docs/20_implementation_audit.md)
- **Development Invariants**: [Implementation Checklist](docs/21_implementation_checklist.md)
- **Data Provenance**: [Data Sources](docs/30_data_sources.md)
- **How Models Are Measured**: [Evaluation Protocol](docs/32_evaluation_protocol.md)
- **Release Verification**: [Release Checklist](docs/51_release_checklist.md)
- **Upstream Lineage**: [Upstream Provenance](docs/40_upstream.md)
- **Release Notes**: [Changelog](CHANGELOG.md)
- **Network Mirrors**: [docs/31_network_mirrors.md](docs/31_network_mirrors.md)

## System Requirements

The repository installs in two tiers. Pick the one that matches what you want to do.

| | CPU tier | GPU tier |
|---|---|---|
| Hardware | Any x86_64 machine, no GPU | NVIDIA GPUs |
| CUDA runtime | not used | CUDA 13 with [requirements/gpu.lock.txt](requirements/gpu.lock.txt) (torch 2.13.0, which pulls the CUDA 13 runtime wheels), CUDA 12.8 with [requirements/gpu-paper.lock.txt](requirements/gpu-paper.lock.txt) (torch 2.9.0+cu128, the pinned paper environment) |
| Python | 3.11 | 3.11 |
| What it enables | Unit tests, dataset preparation and verification, smoke tests, analysis and plotting tooling on stored run artifacts (loading a critic checkpoint needs the GPU tier with flash-attn), the release gate | Everything in the CPU tier, plus training and the GPU stages of the paper's experiments |
| Lock file | [requirements/cpu.lock.txt](requirements/cpu.lock.txt) | [requirements/gpu.lock.txt](requirements/gpu.lock.txt) |

A pinned GPU environment for the paper's experiments is kept as
[requirements/gpu-paper.lock.txt](requirements/gpu-paper.lock.txt).

## Installation

### CPU tier

```bash
python -m pip install -U pip
python -m pip install -r requirements/cpu.lock.txt
python -m pip install -e . --no-deps
python -m pip check
```

The lock file pins the CPU build of PyTorch, which is published on PyTorch's own
index. The required `--extra-index-url` line is inside the lock file, so no extra
flag is needed. To resolve current versions instead of using the lock, run
`python -m pip install -e ".[cpu,test]"`. On Linux that command resolves PyPI's
default PyTorch wheel, which is the CUDA build; add
`--extra-index-url https://download.pytorch.org/whl/cpu` to stay CPU-only.

### GPU tier

```bash
python -m pip install -U pip
python -m pip install -r requirements/gpu.lock.txt
python -m pip install -e . --no-deps
python -m pip check
```

FlashAttention has no wheels: it builds from source and needs a CUDA toolchain
(`nvcc`). It is therefore absent from [requirements/cpu.lock.txt](requirements/cpu.lock.txt)
and [requirements/gpu.lock.txt](requirements/gpu.lock.txt), and recorded in
[requirements/gpu-paper.lock.txt](requirements/gpu-paper.lock.txt)
(`flash_attn==2.8.3`) because the paper environment had it installed.

The plain consequence: importing the training entry point and running the smoke
test do not need flash-attn, but a training run as scripted does, because the
CLI's `--attn_implementation` defaults to `flash_attention_2`
([openrlhf/cli/train_ppo_ray_tooling.py](openrlhf/cli/train_ppo_ray_tooling.py))
and [scripts/40_train/paper_train.sh](scripts/40_train/paper_train.sh) does not
override it. The analysis subcommand that loads a critic checkpoint also requests
`flash_attention_2` and needs a GPU with flash-attn
([c3/analysis/analysis.py](c3/analysis/analysis.py)); the rest of the analysis
and plotting tooling runs on stored run artifacts and stays in the CPU tier.
Install flash-attn on the GPU machine before running `scripts/40_train`. Ring
attention (the sequence-packing path across ranks) needs both flash-attn and
ring-flash-attn, and raises a `RuntimeError` naming the command below at the
point of use.

The command installs both, and is:

```bash
python -m pip install -e ".[flash]" --no-build-isolation
```

Reproduction scripts export the repository root on `PYTHONPATH` themselves, through
[scripts/_lib/common_env.sh](scripts/_lib/common_env.sh).

## CPU Quickstart

This runs end to end on a laptop: no GPU, no checkpoints, no model weights.

```bash
# 1. Install the CPU tier
python -m pip install -r requirements/cpu.lock.txt
python -m pip install -e . --no-deps

# 2. Check the installation
pytest -q tests

# 3. Run the fixture smoke test (tiny bundled data, no downloads)
bash scripts/30_smoke/smoke.sh --task tests/fixtures/tasks/mini_math.yaml --limit 1 --print_example 0

# 4. Prepare the real datasets and verify them against the manifest
bash scripts/10_data/prepare_all.sh --out_dir data --strict 1

# 5. Run the real-data smoke test
bash scripts/30_smoke/smoke.sh --task math --limit 1 --print_example 0
```

On a CPU machine the smoke test import-checks the `c3` entrypoints only.
`--tier gpu` additionally import-checks the training CLI, which needs the GPU tier;
`--tier auto` (the default) picks `gpu` only when torch reports a usable CUDA
device and the training stack (`vllm` and `ray`) is importable, so a GPU machine
with only the CPU tier installed still gets `cpu`. The resolved tier and the
reason are printed.

## GPU Path

With the GPU tier installed, the paper workflows are:

```bash
RECIPE=main bash scripts/40_train/paper_train.sh           # training (RECIPE=main, long or a3)
python scripts/70_rebuild/final_eval.py --policy ...       # evaluation of record
python scripts/70_rebuild/e1_cells.py --model_root ...     # measurement cells (Sections 3 and 4)
```

Each one is spelled out under [Main Workflows](#main-workflows) and in
[scripts/70_rebuild/README.md](scripts/70_rebuild/README.md).

## Network Mirrors

Every download goes to its official source by default, and no mirror is
hard-coded anywhere in this repository. If your network cannot reach PyPI,
Hugging Face or GitHub directly, opt into a mirror through `HF_ENDPOINT`,
`GITHUB_MIRROR_PREFIX`, `PIP_INDEX_URL` or `PIP_EXTRA_INDEX_URL`; see
[Network Mirrors](docs/31_network_mirrors.md). Data integrity does not depend on
trusting a mirror: `--strict 1` verifies every prepared file against the SHA256
pins in `configs/data_manifest.yaml`.

## Data Preparation

Prepared datasets are generated locally from strictly pinned upstream sources. They are not bundled with the codebase.

```bash
bash scripts/10_data/prepare_all.sh --out_dir data
```

Every dataset is pinned by upstream revision and by the SHA256 of each file it produces, both
recorded in [configs/data_manifest.yaml](configs/data_manifest.yaml). Running with `--strict 1`
checks every file against those hashes, so a set of files either matches the ones the paper trained
on byte for byte or the run fails:

```bash
bash scripts/10_data/prepare_all.sh --out_dir data --strict 1
```

Two checks run alongside it. `scripts/10_data/check_overlap.py` fails the preparation if any
evaluation question appears in any training file, and the loaders refuse a split that the pinned
revision does not carry instead of silently falling back to another one. Both exist because the
manifest shipped before 0.2.0 had a mirror of MATH that merged the train and test splits, and a
CMATH revision with no test split at all; the 0.2.0 entry in [CHANGELOG.md](CHANGELOG.md) records
what changed and who it affected.

See [Data Sources](docs/30_data_sources.md) and [Third-Party Notices](THIRD_PARTY_NOTICES.md) for
provenance details.

## Model Preparation

While Transformers/vLLM will automatically download weights on first use, we provide a dedicated utility to pre-download and cache all required HuggingFace base models. This is highly recommended for reproducibility, offline clusters, or avoiding concurrent download races:

```bash
# Log in to Hugging Face (required for gated models like Qwen)
hf auth login

# List the models, without downloading anything
bash scripts/20_models/download_models.sh --dry_run 1

# Pre-download the models of the paper: Qwen3-4B-Instruct-2507, Qwen3-8B and mmBERT-base
bash scripts/20_models/download_models.sh --models paper --out_dir models
```

`--models early` downloads the base models of the results registry
([configs/main_results_registry.yaml](configs/main_results_registry.yaml)),
which belong to the evaluation protocol of the paper's earlier versions; the
default, `all`, downloads both sets.

## Main Workflows

### Smoke Test
Fast end-to-end wiring check to verify environment and protocol integrity:
```bash
bash scripts/30_smoke/smoke.sh
```

### Paper Training Recipes
Train MAPPO, MAGRPO and C3 under one of the three recipes of the paper
(`main` for Table 2, `long` for Table 7, `a3` for Table 8):
```bash
export PRETRAIN='Qwen/Qwen3-4B-Instruct-2507'
RECIPE=main bash scripts/40_train/paper_train.sh
```

`DRY_RUN=1` prints the command line of every run and starts none. `EXTRA_ARGS`
is appended to every run, for the flags of the machine's hardware layout (GPU
and vLLM engine counts, colocation, offloading). Each run writes its
checkpoints and final model under its own directory, `ckpt/_runs/<run_id>/`.

### Evaluation of Record

The numbers of the trained tables come from evaluations run after training by
[scripts/70_rebuild/final_eval.py](scripts/70_rebuild/final_eval.py). The
5-seed runs (Tables 2 and 8) are evaluated once per training seed, greedily
(one sample per problem) under a 512-token generation limit, on MATH500,
AIME 2025, CMATH and GSM8K. The long-generation run (Table 7) is evaluated with
avg@k under a 2048-token limit, five times under evaluation seeds 0 to 4. One
evaluation run has exactly one k, so the driver groups the suites by their k
and runs one evaluation per group. Which suites, with what decoding, where k
comes from and what the evaluation record holds are all in
[Evaluation Protocol](docs/32_evaluation_protocol.md); the commands for each
table are in [scripts/70_rebuild/README.md](scripts/70_rebuild/README.md).

```bash
# Print the two commands and stop. Needs neither a GPU nor a model.
python scripts/70_rebuild/final_eval.py --policy /models/Qwen3-4B-Instruct-2507 --dry-run

# Evaluate one finished training run of the long-generation recipe and write its record.
python scripts/70_rebuild/final_eval.py \
  --policy ckpt/_runs/paper_long_C3_math_seed0/final_hf \
  --run_id paper_long_C3_math_seed0 \
  --out /abs/path/final_eval_record.json
```

The evaluation that runs during training is a different thing: every 5 percent
of the run in the 5-seed recipes and every 25 percent in the long-generation
recipe, and it feeds the monitoring curve only.

### Earlier Evaluation Protocol

The registry sweep and the analysis figures below implement the evaluation
protocol of the paper's earlier versions (v1 and v2): greedy and pass@10 on the
registry's Qwen2.5 base models, and the credit, influence and variance
diagnostics. They are kept for those versions and do not produce the tables of
the current one.

```bash
bash scripts/50_eval/paper_main_results.sh sweep \
  --registry configs/main_results_registry.yaml \
  --only_methods SFT

bash scripts/60_analysis/paper_analysis_figs.sh fig2 \
  --suite math \
  --run_c3 ckpt/_runs/<C3_run_dir> \
  --run_mappo ckpt/_runs/<MAPPO_run_dir> \
  --run_magrpo ckpt/_runs/<MAGRPO_run_dir> \
  --run_sft ckpt/_runs/_sft_main_results/<SFT_dir> \
  --mappo_critic_ckpt <PATH_TO_MAPPO_CRITIC>
```

### Reliability and Ablation Drivers

The drivers under [scripts/70_rebuild/](scripts/70_rebuild/) measure the credit
estimator itself on a frozen policy. They are not on the training path and they
change nothing in it. Each one takes `--dry-run`, which prints the commands it
would run and the cell count without a GPU or a model, so you can read the full
sweep before spending anything on it.

- [scripts/70_rebuild/e1_cells.py](scripts/70_rebuild/e1_cells.py): emit, and optionally run, the bucket-generation cells of the E1 depth study (Section 3: Table 1, Figures 2 and 3).
- [scripts/70_rebuild/e1b_cells.py](scripts/70_rebuild/e1b_cells.py): the cells of the noise-law grid (Table 3, upper block).
- [scripts/70_rebuild/e3a_cells.py](scripts/70_rebuild/e3a_cells.py): emit, and optionally run, the bucket-generation cells of the E3a bias map (Section 4 and Appendix B.6: the bias of agent ablation).
- [scripts/70_rebuild/e4_reference_cells.py](scripts/70_rebuild/e4_reference_cells.py) and [scripts/70_rebuild/e4_critic.py](scripts/70_rebuild/e4_critic.py): the learned-critic comparison (Section 4 and Appendix B.7).
- [scripts/70_rebuild/eval_probe.py](scripts/70_rebuild/eval_probe.py): measure the start accuracy of a frozen policy on the candidate benchmarks.

```bash
# Print the 60 commands of the full E1 sweep and the cell count.
python scripts/70_rebuild/e1_cells.py --model_root /models --dry-run

# Run one E1 cell.
python scripts/70_rebuild/e1_cells.py --model_root /models \
    --results_root /abs/path/20_data/results --only a3/4b/3

# Print the two E3a commands and the cell count.
python scripts/70_rebuild/e3a_cells.py --model_root /models --dry-run

# Run one E3a cell.
python scripts/70_rebuild/e3a_cells.py --model_root /models \
    --results_root /abs/path/20_data/results --only empty

# Print the two start-accuracy probe commands and stop.
python scripts/70_rebuild/eval_probe.py --policy /models/Qwen3-4B-Instruct-2507 --dry-run

# Run both decodings of the probe and write the summary.
python scripts/70_rebuild/eval_probe.py --policy /models/Qwen3-4B-Instruct-2507 \
    --out /abs/path/probe_summary.json
```

The probe reads the suites declared in
[configs/tasks/math_eval_probe.yaml](configs/tasks/math_eval_probe.yaml). They
are evaluation only; the long-generation run (Table 7) evaluates on four of
them; nothing trains on them. The default
preparation run writes them, so no extra flag is needed; see
[Data Sources](docs/30_data_sources.md).

## Implementation Note

The paper's credit assignment lives in [c3/credit/](c3/credit/) and [openrlhf/trainer/ppo_utils/experience_maker.py](openrlhf/trainer/ppo_utils/experience_maker.py); see the [Implementation Audit](docs/20_implementation_audit.md) for the full paper-to-code mapping.

## Audit and Release Gate

Before making any public release, execute the pre-release checks:

```bash
bash scripts/90_audit/pre_release.sh
```

To run the complete local release gate (including syntax checks and unit tests):
```bash
bash scripts/90_audit/release_gate.sh
```

For a single-command preflight reproduction check:
```bash
bash scripts/30_smoke/preflight_repro.sh --task math
```

These gating scripts verify:
- Absence of hard-coded private paths
- Absence of obvious leaked secrets
- Absence of bundled datasets or generated release-surface artifacts
- English prose with no em or en dashes, and no CJK outside the functional allowlist
- Bash scripting syntax sanity
- Python compilation and test suite sanity
- That the gate itself leaves no artifacts in the working tree

Both scripts run on a CPU machine with the CPU tier installed. The same two
stages run in CI on every push and pull request; see
[.github/workflows/ci-cpu.yml](.github/workflows/ci-cpu.yml).

## What is verified, and on what

Every push runs the CPU tier on Python 3.11: install from `requirements/cpu.lock.txt`, a dependency
consistency check, the full unit suite, both fixture smoke tests, the prose scan, and the
pre-release audit. That is what the badges above report.

Two things are worth naming because a reader cannot see them from a green badge.

- **The credit mechanism is checked without a GPU.** `tests/mechanism/` builds the paper's rollout
  tree by hand and compares the leave-one-out advantages against values computed on paper, so the
  arithmetic at the centre of the method can be confirmed on a laptop in seconds.
- **The GPU tier is installed and imported, not trained end to end, in CI.** No hosted runner has
  the hardware. `requirements/gpu.lock.txt` records exactly which checks were run against it and
  which were not, and `requirements/gpu-paper.lock.txt` pins the environment that produced the
  published numbers.

## Governance

This repository includes standard open-source policy files:

- [CONTRIBUTING.md](CONTRIBUTING.md)
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- [SECURITY.md](SECURITY.md)
- [.github/ISSUE_TEMPLATE](.github/ISSUE_TEMPLATE)
- [.github/pull_request_template.md](.github/pull_request_template.md)

## License and Attribution

- Code license: [Apache-2.0](LICENSE)
- Citation metadata: [CITATION.cff](CITATION.cff)
- Third-party notices: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
- Upstream provenance: [docs/40_upstream.md](docs/40_upstream.md)

## Acknowledgements

We deeply appreciate the open-source community for their foundational work. In particular, we would like to acknowledge:

- **[OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)**: Our RL infrastructure is built upon the OpenRLHF framework. We are grateful to the OpenRLHF team for providing an easy-to-use, scalable, and high-performance agentic RL foundation based on Ray and vLLM.

## Citation

If you find this repository or paper useful for your research, please cite:

```bibtex
@misc{chen2026trace,
  title         = {The Trace Is the State: Exact Credit Assignment for {LLM} Agent Teams},
  author        = {Yanjun Chen and Yirong Sun and Hanlin Wang and Jinghan Wang and Xinming Zhang and Xiaoyu Shen and Wenjie Li and Wei Zhang},
  year          = {2026},
  eprint        = {2603.06859},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi           = {10.48550/arXiv.2603.06859},
  url           = {https://arxiv.org/abs/2603.06859}
}
```
