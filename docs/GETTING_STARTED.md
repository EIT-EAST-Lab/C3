# Getting Started

This guide is for users who want to install the repository, prepare local data, and run the main paper-facing entrypoints without first reading the full implementation audit.

## 1. Install dependencies

Recommended baseline:

```bash
python -m pip install -U pip
python -m pip install -r requirements.txt --no-build-isolation
python -m pip check
```

Notes:

- Python 3.11 is the reference version.
- The repository is Linux-first and expects a CUDA-compatible PyTorch stack for the full training path.
- Reproduce scripts export the repo root on `PYTHONPATH` automatically.

For lightweight local development and CI-style checks, editable install is also supported:

```bash
python -m pip install -e .[test]
```

## 2. Prepare local datasets

The public repository does not ship prepared datasets. Generate them locally:

```bash
bash scripts/data/prepare_all.sh --out_dir data
```

Strict verification:

```bash
bash scripts/data/prepare_all.sh --out_dir data --strict 1
```

See [DATA_SOURCES.md](DATA_SOURCES.md) for provenance and SHA256 pinning.

## 3. Run a wiring smoke test

This checks task loading, prompt rendering, and evaluator wiring. It is not a model-quality regression test.

```bash
bash scripts/reproduce/smoke.sh --task math --limit 1 --print_example 0
```

You can also run:

```bash
bash scripts/reproduce/smoke.sh --task code --limit 1 --print_example 0
```

## 4. Run paper-facing workflows

### SFT-only eval sweep

```bash
bash scripts/reproduce/paper_main_results.sh sweep \
  --registry configs/main_results_registry.yaml \
  --only_methods SFT
```

### Full training matrix

```bash
export PRETRAIN='Qwen/Qwen2.5-3B-Instruct'
bash scripts/reproduce/paper_train.sh
```

### Full main-results sweep

```bash
bash scripts/reproduce/paper_main_results.sh sweep \
  --registry configs/main_results_registry.yaml
```

### Paper analyses

```bash
bash scripts/reproduce/paper_analysis_figs.sh fig2 \
  --suite math \
  --run_c3 ckpt/_runs/<C3_run_dir> \
  --run_mappo ckpt/_runs/<MAPPO_run_dir> \
  --run_magrpo ckpt/_runs/<MAGRPO_run_dir> \
  --run_sft ckpt/_runs/_sft_main_results/<SFT_dir> \
  --mappo_critic_ckpt <PATH_TO_MAPPO_CRITIC>
```

## 5. Understand the implementation

- quick repository navigation: [CODE_MAP.md](CODE_MAP.md)
- paper-to-code mapping and invariants: [IMPLEMENTATION_AUDIT.md](IMPLEMENTATION_AUDIT.md)

## 6. Release hygiene

Before publishing the repository, run:

```bash
bash scripts/audit/pre_release.sh
```

Single-command preflight:

```bash
bash scripts/reproduce/preflight_repro.sh --task math
```

The release surface must not include local generated directories such as `data/`, `artifacts/`, `ckpt/`, `runs/`, `wandb/`, or `models/`. See [RELEASE_POLICY.md](RELEASE_POLICY.md).

For the full local release gate, use:

```bash
bash scripts/audit/release_gate.sh
```
