# Release Policy

This repository is released as an engineering-friendly research codebase for the C3 paper.

## What the public repository ships

- source code under `c3/` and the vendored `openrlhf/` subtree,
- configuration files under `configs/`,
- reproducibility and audit scripts under `scripts/`,
- documentation under `docs/`,
- governance and attribution files such as `LICENSE`, `CITATION.cff`, and `THIRD_PARTY_NOTICES.md`.

## What the public repository does not ship

The following directories are treated as generated local outputs and must remain empty or absent in a public release:

- `data/`
- `artifacts/`
- `ckpt/`
- `runs/`
- `wandb/`
- `models/`

These directories may be created locally while preparing datasets, running experiments, generating plots, or caching models, but they are not part of the release surface.

The one intended exception is tiny synthetic fixture data kept under `tests/fixtures/` (and, if needed in the future, `examples/fixtures/`) for lightweight smoke tests and contract-level CI.

## Reproducibility contract

- Prepared datasets are generated locally from pinned upstream sources via `scripts/data/prepare_all.sh`.
- Dataset provenance and SHA256 pins are tracked in `configs/data_manifest.yaml`.
- Release hygiene is enforced by `scripts/audit/pre_release.sh` and `scripts/audit/no_data_check.py`.

## Maintainer checklist

Before publishing or packaging the repository:

1. Remove or regenerate any local outputs under the directories listed above.
2. Run `bash scripts/audit/pre_release.sh`.
3. Verify that documentation still matches the release policy and the expected local generation workflow.
