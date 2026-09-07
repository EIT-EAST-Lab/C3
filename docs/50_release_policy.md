# Release Policy

This repository is released as an engineering-friendly research codebase for the C3 paper.

## What the public repository ships

- source code under `c3/` and the vendored `openrlhf/` subtree,
- configuration files under `configs/`,
- dependency lock files under `requirements/`,
- reproducibility and audit scripts under `scripts/`,
- documentation under `docs/`,
- governance and attribution files such as `LICENSE`, `CITATION.cff`, and `THIRD_PARTY_NOTICES.md`.

## Dependency tiers

The release ships two installable tiers plus one historical record:

- `requirements/cpu.lock.txt`: a CPU-only environment. Everything except training
  runs in it, so a reader without a GPU can still install the package, run the
  tests, prepare and verify the datasets, run the smoke tests, and produce the
  analysis figures.
- `requirements/gpu.lock.txt`: the current training stack. Installation, imports
  and unit tests are verified on CPU; GPU execution is validated separately and
  the file header lists what that covers.
- `requirements/gpu-paper.lock.txt`: the maintainers' training environment, kept
  verbatim so the published numbers remain reproducible. It is a record, not a
  recommendation, and it is never repaired.

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

- Prepared datasets are generated locally from pinned upstream sources via `scripts/10_data/prepare_all.sh`.
- Dataset provenance and SHA256 pins are tracked in `configs/data_manifest.yaml`.
- Downloads default to official sources. Mirrors are opt-in through the
  environment and never hard-coded; see `docs/31_network_mirrors.md`.
- Release hygiene is enforced by `scripts/90_audit/pre_release.sh`,
  `scripts/90_audit/no_data_check.py`, `scripts/90_audit/scan_prose.py`, and
  `tests/test_release_surface.py`.

## Maintainer checklist

Before publishing or packaging the repository:

1. Remove or regenerate any local outputs under the directories listed above.
2. Run `bash scripts/90_audit/pre_release.sh`.
3. Run `bash scripts/90_audit/release_gate.sh` on a CPU machine.
4. Verify that documentation still matches the release policy and the expected local generation workflow.
5. Add the release entry to `CHANGELOG.md`.
