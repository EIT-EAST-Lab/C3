# Release Checklist

Use this checklist before publishing the repository or cutting a release snapshot.

## Repository surface

- `data/`, `artifacts/`, `ckpt/`, `runs/`, `wandb/`, and `models/` are empty or absent.
- No private paths, ad hoc local scripts, or machine-specific configs have been introduced.
- No secrets, tokens, or cached credentials are present.

## Required local gate

Run the unified local gate:

```bash
bash scripts/audit/release_gate.sh
```

This gate currently runs:

1. `pytest -q tests`
2. fixture-based math smoke
3. fixture-based code smoke
4. dummy figure generation for the plotting path
5. `bash scripts/audit/pre_release.sh`

## Extended paper-facing checks

These are recommended before a public paper artifact release:

1. Strict data verification:

```bash
bash scripts/data/prepare_all.sh --out_dir data --strict 1
```

2. SFT-only main-results sweep:

```bash
bash scripts/reproduce/paper_main_results.sh sweep \
  --registry configs/main_results_registry.yaml \
  --only_methods SFT
```

3. Optional release gate with a real HF base model:

```bash
HF_BASE='Qwen/Qwen2.5-3B-Instruct' RUN_SFT_EVAL=1 bash scripts/audit/release_gate.sh
```

## Documentation

- `README.md` matches the current release policy and entrypoints.
- `docs/GETTING_STARTED.md` matches actual installation and smoke commands.
- `docs/CODE_MAP.md` and `docs/IMPLEMENTATION_AUDIT.md` still match the primary implementation path.
- `docs/DATA_SOURCES.md` and `configs/data_manifest.yaml` remain consistent.

## Governance and metadata

- `LICENSE`, `CITATION.cff`, and `THIRD_PARTY_NOTICES.md` are up to date.
- `.github/workflows/pre-release-audit.yml` and `.github/workflows/ci-lite.yml` still match the intended release gate.
- Community files (`CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`) still describe the current workflow.
