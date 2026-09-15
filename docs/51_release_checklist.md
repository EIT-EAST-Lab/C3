# Release Checklist

Use this checklist before publishing the repository or cutting a release snapshot.

## Repository surface

- `data/`, `artifacts/`, `ckpt/`, `runs/`, `wandb/`, and `models/` are empty or absent.
- No private paths, ad hoc local scripts, or machine-specific configs have been introduced.
- No secrets, tokens, or cached credentials are present.

## Required local gate

Run the unified local gate on a CPU machine with the CPU tier installed. It needs
no GPU and no checkpoints:

```bash
bash scripts/90_audit/release_gate.sh
```

This gate currently runs:

1. `python -m pip check`
2. `pytest -q tests`, including `tests/test_release_surface.py`
3. fixture-based math smoke, with the CPU-tier import checks
4. fixture-based code smoke
5. dummy figure generation for the plotting path
6. strict dataset manifest verification, when a prepared `data/` is present
7. model registry resolution (`download_models.sh --dry_run 1`)
8. `bash scripts/90_audit/pre_release.sh`, which includes the prose scan
9. a check that the gate itself left no artifacts in the working tree

Step 6 is skipped when `data/` is absent, which is the state of a clean checkout,
so a gate run that never prepared the datasets is a seven-step gate: prepare the
datasets first if you want the full one.

Run `tests/test_release_surface.py` on its own when you only want the release
surface contracts:

```bash
pytest -q tests/test_release_surface.py
```

## Extended paper-facing checks

These are recommended before a public paper artifact release:

1. Strict data verification:

```bash
bash scripts/10_data/prepare_all.sh --out_dir data --strict 1
```

2. SFT-only main-results sweep:

```bash
bash scripts/50_eval/paper_main_results.sh sweep \
  --registry configs/main_results_registry.yaml \
  --only_methods SFT
```

3. Optional release gate with a real HF base model:

```bash
HF_BASE='Qwen/Qwen2.5-3B-Instruct' RUN_SFT_EVAL=1 bash scripts/90_audit/release_gate.sh
```

## Packaging

- `pyproject.toml` carries the release version, and `CHANGELOG.md` has an entry
  for it.
- `requirements/cpu.lock.txt` was regenerated from a fresh virtual environment,
  and its header states when and how.
- `requirements/gpu.lock.txt` was regenerated and its header lists what still
  needs a GPU to validate.
- `requirements/gpu-paper.lock.txt` is untouched. It is a historical record.

## Documentation

- `README.md` matches the current release policy and entrypoints.
- `docs/00_getting_started.md` matches actual installation and smoke commands.
- `docs/10_code_map.md` and `docs/20_implementation_audit.md` still match the primary implementation path.
- `docs/30_data_sources.md` and `configs/data_manifest.yaml` remain consistent.
- Every relative documentation link resolves. `tests/test_release_surface.py`
  checks this, so a moved file cannot slip through.

## Governance and metadata

- `LICENSE`, `CITATION.cff`, and `THIRD_PARTY_NOTICES.md` are up to date.
- `.github/workflows/ci-cpu.yml` still matches the intended release gate, and its
  two jobs are green on the release commit.
- Community files (`CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`) still describe the current workflow.
