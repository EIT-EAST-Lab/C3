# Getting Started

This guide takes you from a fresh checkout to a verified installation, prepared
datasets, and the paper-facing entrypoints, without first reading the full
implementation audit.

## 0. Pick a tier

| | CPU tier | GPU tier |
|---|---|---|
| Hardware | any x86_64 machine, no GPU | NVIDIA GPUs |
| CUDA runtime | not used | CUDA 13 with `requirements/gpu.lock.txt` (torch 2.13.0, which pulls the CUDA 13 runtime wheels), CUDA 12.8 with `requirements/gpu-paper.lock.txt` (torch 2.9.0+cu128, the paper environment) |
| Python | 3.11 | 3.11 |
| Enables | unit tests, dataset preparation and verification, smoke tests, analysis and plotting, the release gate | everything above, plus training and full paper reproduction |
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
rather than the lock, use `python -m pip install -e ".[cpu,test]"`.

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

FlashAttention is an optional extra (`pip install -e ".[flash]" --no-build-isolation`).
It is absent from every lock file because it builds from source and needs a CUDA
toolchain.

## 2. Check the installation

```bash
pytest -q tests
bash scripts/30_smoke/smoke.sh --task tests/fixtures/tasks/mini_math.yaml --limit 1 --print_example 0
bash scripts/30_smoke/smoke.sh --task tests/fixtures/tasks/mini_code.yaml --limit 1 --print_example 0 --skip_import_checks 1
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

`--tier auto` (the default) import-checks the training CLI only when torch reports
a usable CUDA device and the training stack (`vllm` and `ray`) is importable, so a
GPU machine with only the CPU tier installed still resolves to `cpu`. Force it
either way with `--tier cpu` or `--tier gpu`.

## 5. Run the paper-facing workflows (GPU tier)

### SFT-only eval sweep

```bash
bash scripts/50_eval/paper_main_results.sh sweep \
  --registry configs/main_results_registry.yaml \
  --only_methods SFT
```

### Full training matrix

```bash
export PRETRAIN='Qwen/Qwen2.5-3B-Instruct'
bash scripts/40_train/paper_train.sh
```

### Full main-results sweep

```bash
bash scripts/50_eval/paper_main_results.sh sweep \
  --registry configs/main_results_registry.yaml
```

### Paper analyses

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
