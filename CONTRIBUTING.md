# Contributing to C3

Thanks for your interest in improving C3.

## Scope

This repository is a research codebase for reproducing the paper results. We prioritize:

- reproducibility,
- deterministic data preparation,
- clear provenance for models and datasets,
- small and reviewable pull requests.

## Development setup

The repository installs in two tiers. Contributing to anything except the
training path needs only the CPU tier, which runs on a laptop.

1. Create a Python 3.11 environment.
2. Install a tier:

```bash
# CPU tier: tests, data preparation, smoke tests, analysis, the release gate
python -m pip install -U pip
python -m pip install -r requirements/cpu.lock.txt
python -m pip install -e . --no-deps
python -m pip check
```

```bash
# GPU tier: everything above, plus training
python -m pip install -r requirements/gpu.lock.txt
python -m pip install -e . --no-deps
python -m pip check
```

`requirements/gpu-paper.lock.txt` is the maintainers' training environment kept
verbatim for reproducing the published numbers. Do not repair it in a pull
request.

3. Prepare datasets:

```bash
bash scripts/10_data/prepare_all.sh --out_dir data
```

If your network cannot reach Hugging Face or GitHub directly, see
[docs/31_network_mirrors.md](docs/31_network_mirrors.md).

4. Run sanity checks:

```bash
pytest -q tests
bash scripts/30_smoke/smoke.sh --task tests/fixtures/tasks/mini_math.yaml --limit 1 --print_example 0
bash scripts/90_audit/pre_release.sh
```

The same checks run in CI on every pull request, in
[.github/workflows/ci-cpu.yml](.github/workflows/ci-cpu.yml).

## Pull request expectations

- Keep changes focused; avoid unrelated refactors.
- Update docs when behavior or CLI contracts change.
- Add/adjust tests or checks when changing data, eval, or training logic.
- Preserve backward compatibility of public CLI flags unless the PR explicitly documents a breaking change.
- Do not commit generated local outputs such as `data/`, `artifacts/`, `ckpt/`, `runs/`, `wandb/`, or `models/`.
- Add an entry to [CHANGELOG.md](CHANGELOG.md) for anything a user would notice.

## Prose

Public prose is English. `scripts/90_audit/scan_prose.py` enforces two rules and
runs in CI:

- no em dash and no en dash in `.md`, `.py`, `.sh`, `.yaml`, `.yml`, `.toml` or
  `.cff` files: use a colon, a comma, a semicolon or brackets;
- no CJK characters outside `scripts/90_audit/cjk_allowlist.txt`.

The allowlist holds the functional CMATH answer cues and punctuation sets, which
must stay byte for byte identical. If you touch one of those lines, keep the
characters and update the allowlist entry rather than translating them.

## Reproducibility requirements

If your change modifies dataset preparation outputs:

1. Recompute hashes:

```bash
bash scripts/10_data/prepare_all.sh --out_dir data --update_manifest_sha256 1
```

2. Commit `configs/data_manifest.yaml`.
3. Verify strict mode:

```bash
bash scripts/10_data/prepare_all.sh --out_dir data --strict 1
```

## Security and disclosure

Please do not open public issues for potential security problems. Follow `SECURITY.md` for private disclosure.
