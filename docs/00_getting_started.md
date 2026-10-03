# Getting Started

This guide takes you from a fresh checkout to a verified installation, prepared
datasets, and the paper-facing entrypoints, without first reading the full
implementation audit.

## About this repository

Reference implementation for the paper **The Trace Is the State: Exact Credit Assignment for LLM Agent Teams**.

Paper status: preprint on arXiv as [2603.06859](https://arxiv.org/abs/2603.06859), under review. The companion project page is at [eit-east-lab.github.io/C3](https://eit-east-lab.github.io/C3/).

Repository version 0.3.1. See [CHANGELOG.md](../CHANGELOG.md) for what changed and why.

`C3` is a paper-aligned research codebase designed to support:

- **Protocol Reproduction**: Executing the multi-agent task protocols described in the paper.
- **Experiment Execution**: Running full training sweeps, evaluation-only probes, and performance analyses.
- **Implementation Audit**: Providing a transparent mapping from theoretical mechanisms to executable code.
- **Extension and Testing**: Modifying or testing the framework without relying on private paths or bundled artifacts.

## Quick navigation

- **Project Page**: [C3 Paper Page](https://eit-east-lab.github.io/C3/)
- **Paper**: [arXiv Abstract](https://arxiv.org/abs/2603.06859)
- **PDF**: [arXiv PDF](https://arxiv.org/pdf/2603.06859)
- **Code Layout**: [Code Map](10_code_map.md)
- **Paper-to-Code Mapping**: [Implementation Audit](20_implementation_audit.md)
- **Development Invariants**: [Implementation Checklist](21_implementation_checklist.md)
- **Data Provenance**: [Data Sources](30_data_sources.md)
- **How Models Are Measured**: [Evaluation Protocol](32_evaluation_protocol.md)
- **Release Surface**: [Release Policy](50_release_policy.md)
- **Release Verification**: [Release Checklist](51_release_checklist.md)
- **Upstream Lineage**: [Upstream Provenance](40_upstream.md)
- **Release Notes**: [Changelog](../CHANGELOG.md)
- **Network Mirrors**: [docs/31_network_mirrors.md](31_network_mirrors.md)

### Governance

This repository includes standard open-source policy files:

- [CONTRIBUTING.md](../CONTRIBUTING.md)
- [CODE_OF_CONDUCT.md](../CODE_OF_CONDUCT.md)
- [SECURITY.md](../SECURITY.md)
- [.github/ISSUE_TEMPLATE](../.github/ISSUE_TEMPLATE)
- [.github/pull_request_template.md](../.github/pull_request_template.md)

## 0. Pick a tier

| | CPU tier | GPU tier |
|---|---|---|
| Hardware | any x86_64 machine, no GPU | NVIDIA GPUs |
| CUDA runtime | not used | CUDA 13 with `requirements/gpu.lock.txt` (torch 2.13.0, which pulls the CUDA 13 runtime wheels), CUDA 12.8 with `requirements/gpu-paper.lock.txt` (torch 2.9.0+cu128, the paper environment) |
| Python | 3.11 | 3.11 |
| Enables | unit tests, dataset preparation and verification, smoke tests, analysis and plotting tooling on stored run artifacts (loading a critic checkpoint needs the GPU tier with flash-attn), the release gate | everything above, plus training and full paper reproduction |
| Lock file | `requirements/cpu.lock.txt` | `requirements/gpu.lock.txt` |

Start with the CPU tier. It is enough to check that the repository works, to
prepare the datasets, and to read the code with a running interpreter beside you.

## 1. Install

### CPU tier

```bash
python -m pip install -U pip
python -m pip install -r requirements/cpu.lock.txt
python -m pip install -e . --no-deps
python -m pip check
```

The CPU build of PyTorch lives on PyTorch's own index; the required
`--extra-index-url` line is inside the lock file. To resolve current versions
rather than the lock, use `python -m pip install -e ".[cpu,test]"`. On Linux that
command resolves PyPI's default PyTorch wheel, which is the CUDA build; add
`--extra-index-url https://download.pytorch.org/whl/cpu` to stay CPU-only.

### GPU tier

```bash
python -m pip install -U pip
python -m pip install -r requirements/gpu.lock.txt
python -m pip install -e . --no-deps
python -m pip check
```

To reproduce the published numbers against the exact stack that produced them,
install `requirements/gpu-paper.lock.txt` instead. It is a verbatim snapshot of
the maintainers' training environment and is intentionally not repaired.

FlashAttention has no wheels: it builds from source and needs a CUDA toolchain
(`nvcc`). It is therefore absent from `requirements/cpu.lock.txt` and
`requirements/gpu.lock.txt`, and recorded in `requirements/gpu-paper.lock.txt`
(`flash_attn==2.8.3`) because the paper environment had it installed.

The plain consequence: `import openrlhf.cli.train_ppo_ray` and
`smoke.sh --tier gpu` do not need flash-attn, but a training run as scripted
does, because the CLI's `--attn_implementation` defaults to
`flash_attention_2` (`openrlhf/cli/train_ppo_ray_tooling.py`) and
`scripts/40_train/paper_train.sh` does not override it. The analysis subcommand
that loads a critic checkpoint also requests `flash_attention_2` and needs a GPU
with flash-attn (`c3/analysis/analysis.py`); the rest of the analysis and
plotting tooling runs on stored run artifacts and stays in the CPU tier. Install
flash-attn on the GPU machine with `pip install -e ".[flash]" --no-build-isolation`
before running `scripts/40_train`. That extra installs flash-attn and
ring-flash-attn: ring attention (the sequence-packing path across ranks) needs
both, and raises a `RuntimeError` naming the install command at the point of use.

Reproduction scripts export the repository root on `PYTHONPATH` themselves, through
[scripts/_lib/common_env.sh](../scripts/_lib/common_env.sh).

## 2. Check the installation

```bash
pytest -q tests
bash scripts/30_smoke/smoke.sh --task tests/fixtures/tasks/mini_math.yaml --limit 1 --print_example 0
bash scripts/30_smoke/smoke.sh --task tests/fixtures/tasks/mini_code.yaml --limit 1 --print_example 0
```

The fixture smokes use the tiny bundled data under `tests/fixtures/` and download
nothing.

## 3. Prepare local datasets

The public repository does not ship prepared datasets. Generate them locally:

```bash
bash scripts/10_data/prepare_all.sh --out_dir data
```

Strict verification, which recomputes every SHA256 and compares it against
`configs/data_manifest.yaml`:

```bash
bash scripts/10_data/prepare_all.sh --out_dir data --strict 1
```

See [30_data_sources.md](30_data_sources.md) for provenance and SHA256 pinning,
and [31_network_mirrors.md](31_network_mirrors.md) if your network cannot reach
Hugging Face or GitHub directly.

## 4. Run a wiring smoke test on real data

This checks task loading, prompt rendering, and evaluator wiring. It is not a
model-quality regression test.

```bash
bash scripts/30_smoke/smoke.sh --task math --limit 1 --print_example 0
bash scripts/30_smoke/smoke.sh --task code --limit 1 --print_example 0
```

On a CPU machine the smoke test import-checks the `c3` entrypoints only.
`--tier gpu` additionally import-checks the training CLI, which needs the GPU tier;
`--tier auto` (the default) import-checks the training CLI only when torch reports
a usable CUDA device and the training stack (`vllm` and `ray`) is importable, so a
GPU machine with only the CPU tier installed still resolves to `cpu`. Force it
either way with `--tier cpu` or `--tier gpu`. The resolved tier and the reason are
printed.

## 5. Run the paper's experiments (GPU tier)

### Base models

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
([configs/main_results_registry.yaml](../configs/main_results_registry.yaml)),
which belong to the evaluation protocol of the paper's earlier versions; the
default, `all`, downloads both sets.

### Training recipes

Three recipes, one per trained table of the paper: `main` (Table 2), `long`
(Table 7) and `a3` (Table 8). Each trains MAPPO, MAGRPO and C3 with the
settings of Tables 5 and 6 and Appendix B.2.

```bash
export PRETRAIN='Qwen/Qwen3-4B-Instruct-2507'
DRY_RUN=1 RECIPE=main bash scripts/40_train/paper_train.sh   # print the command lines only
RECIPE=main bash scripts/40_train/paper_train.sh
```

`DRY_RUN=1` prints the command line of every run and starts none. `EXTRA_ARGS`
is appended to every run, for the flags of the machine's hardware layout (GPU
and vLLM engine counts, colocation, offloading). Each run writes its
checkpoints and final model under its own directory, `ckpt/_runs/<run_id>/`.

### Evaluation and measurements

Every trained model is evaluated by `scripts/70_rebuild/final_eval.py`, and the
records are aggregated on the CPU by `c3.analysis.rebuild.aggregate_5seed`
(Tables 2, 8 and 10) and `c3.analysis.rebuild.aggregate_e2` (Table 7). The
measurements of Sections 3 and 4 have their own drivers under
`scripts/70_rebuild/`. The commands, and which driver produces which table, are
in [scripts/70_rebuild/README.md](../scripts/70_rebuild/README.md). Which suites,
with what decoding, where k comes from and what the evaluation record holds are
all in [Evaluation Protocol](32_evaluation_protocol.md).

### Reliability and ablation drivers

The drivers under [scripts/70_rebuild/](../scripts/70_rebuild/) measure the credit
estimator itself on a frozen policy. They are not on the training path and they
change nothing in it. Each one takes `--dry-run`, which prints the commands it
would run and the cell count without a GPU or a model, so you can read the full
sweep before spending anything on it. What each driver is for is listed in
[10_code_map.md](10_code_map.md#drivers).

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
[configs/tasks/math_eval_probe.yaml](../configs/tasks/math_eval_probe.yaml). They
are evaluation only; the long-generation run (Table 7) evaluates on four of
them; nothing trains on them. The default
preparation run writes them, so no extra flag is needed; see
[Data Sources](30_data_sources.md).

### Workflows of the earlier paper versions

The registry sweep and the analysis figures below implement the evaluation
protocol of the paper's earlier versions (v1 and v2): greedy and pass@10 on the
registry's Qwen2.5 base models, and the credit, influence and variance
diagnostics. They are kept for those versions and do not produce the tables of
the current one.

```bash
bash scripts/50_eval/paper_main_results.sh sweep \
  --registry configs/main_results_registry.yaml \
  --only_methods SFT
```

```bash
bash scripts/60_analysis/paper_analysis_figs.sh fig2 \
  --suite math \
  --run_c3 ckpt/_runs/<C3_run_dir> \
  --run_mappo ckpt/_runs/<MAPPO_run_dir> \
  --run_magrpo ckpt/_runs/<MAGRPO_run_dir> \
  --run_sft ckpt/_runs/_sft_main_results/<SFT_dir> \
  --mappo_critic_ckpt <PATH_TO_MAPPO_CRITIC>
```

## 6. Understand the implementation

- quick repository navigation: [10_code_map.md](10_code_map.md)
- paper-to-code mapping and invariants: [20_implementation_audit.md](20_implementation_audit.md)

## 7. Release hygiene

Before publishing the repository, run:

```bash
bash scripts/90_audit/pre_release.sh
```

Single-command preflight:

```bash
bash scripts/30_smoke/preflight_repro.sh --task math
```

The release surface must not include local generated directories such as `data/`,
`artifacts/`, `ckpt/`, `runs/`, `wandb/`, or `models/`. See
[50_release_policy.md](50_release_policy.md).

For the full local release gate, which runs on a CPU machine, use:

```bash
bash scripts/90_audit/release_gate.sh
```
