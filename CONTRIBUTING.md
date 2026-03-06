# Contributing to C3

Thanks for your interest in improving C3.

## Scope

This repository is a research codebase for reproducing the paper results. We prioritize:

- reproducibility,
- deterministic data preparation,
- clear provenance for models and datasets,
- small and reviewable pull requests.

## Development setup

1. Create Python 3.11 environment.
2. Install pinned dependencies:

```bash
python -m pip install -U pip
python -m pip install -r requirements.txt --no-build-isolation
python -m pip check
```

3. Prepare datasets:

```bash
bash scripts/data/prepare_all.sh --out_dir data
```

4. Run sanity checks:

```bash
bash scripts/reproduce/smoke.sh
bash scripts/audit/pre_release.sh
```

## Pull request expectations

- Keep changes focused; avoid unrelated refactors.
- Update docs when behavior or CLI contracts change.
- Add/adjust tests or checks when changing data, eval, or training logic.
- Preserve backward compatibility of public CLI flags unless the PR explicitly documents a breaking change.
- Do not commit generated local outputs such as `data/`, `artifacts/`, `ckpt/`, `runs/`, `wandb/`, or `models/`.

## Reproducibility requirements

If your change modifies dataset preparation outputs:

1. Recompute hashes:

```bash
bash scripts/data/prepare_all.sh --out_dir data --update_manifest_sha256 1
```

2. Commit `configs/data_manifest.yaml`.
3. Verify strict mode:

```bash
bash scripts/data/prepare_all.sh --out_dir data --strict 1
```

## Security and disclosure

Please do not open public issues for potential security problems. Follow `SECURITY.md` for private disclosure.
