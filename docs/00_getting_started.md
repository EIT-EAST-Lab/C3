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

`--tier auto` (the default) import-checks the training CLI only when torch reports
a usable CUDA device and the training stack (`vllm` and `ray`) is importable, so a
GPU machine with only the CPU tier installed still resolves to `cpu`. Force it
either way with `--tier cpu` or `--tier gpu`.

## 5. Run the paper's experiments (GPU tier)

### Training recipes

Three recipes, one per trained table of the paper: `main` (Table 2), `long`
(Table 7) and `a3` (Table 8). Each trains MAPPO, MAGRPO and C3 with the
settings of Tables 5 and 6 and Appendix B.2.

```bash
export PRETRAIN='Qwen/Qwen3-4B-Instruct-2507'
DRY_RUN=1 RECIPE=main bash scripts/40_train/paper_train.sh   # print the command lines only
RECIPE=main bash scripts/40_train/paper_train.sh
```

### Evaluation and measurements

Every trained model is evaluated by `scripts/70_rebuild/final_eval.py`, and the
records are aggregated on the CPU by `c3.analysis.rebuild.aggregate_5seed`
(Tables 2, 8 and 10) and `c3.analysis.rebuild.aggregate_e2` (Table 7). The
measurements of Sections 3 and 4 have their own drivers under
`scripts/70_rebuild/`. The commands, and which driver produces which table, are
in [scripts/70_rebuild/README.md](../scripts/70_rebuild/README.md).

### Workflows of the earlier paper versions

The registry sweep and the analysis figures below implement the evaluation
protocol of the paper's earlier versions (v1 and v2). They are kept for those
versions and do not produce the tables of the current one.

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
