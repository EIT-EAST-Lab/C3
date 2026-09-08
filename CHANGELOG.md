# Changelog

All notable changes to this repository are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0]

A maintenance release focused on one thing: making the repository installable,
runnable and verifiable by someone who is not us, without a GPU. It fixes the
broken quickstart reported in
[issue #1](https://github.com/EIT-EAST-Lab/C3/issues/1). No algorithmic behavior
changed: no numerical logic, defaults, sampling, rewards, credit computation,
config values, prompts or role definitions were touched.

### Added

- Dependency tiers in `pyproject.toml`: a minimal core, plus `cpu`, `train`,
  `flash`, `test` and `dev` extras, so a laptop user is not asked to install the
  GPU training stack.
- Three lock files under `requirements/`:
  - `cpu.lock.txt`, a CPU-only environment (CPU build of PyTorch, no CUDA
    packages) verified with `pip check`, the full test suite and both fixture
    smoke tests;
  - `gpu.lock.txt`, a current and internally consistent training stack, verified
    on CPU for installation, imports and unit tests;
  - `gpu-paper.lock.txt`, the maintainers' training environment preserved
    verbatim so the published numbers stay reproducible.
- A CPU-verifiable surface: `scripts/30_smoke/smoke.sh` gained
  `--tier auto|cpu|gpu`, which decides whether to import-check the training CLI.
  On a CPU machine the default tier skips that import and the smoke test passes.
- Network mirror knobs, all opt-in and never defaulted to a mirror:
  `HF_ENDPOINT`, `GITHUB_MIRROR_PREFIX`, `PIP_INDEX_URL`, `PIP_EXTRA_INDEX_URL`.
  They are validated and reported by `scripts/_lib/mirrors.sh` and documented in
  `docs/31_network_mirrors.md`. With `GITHUB_MIRROR_PREFIX` set,
  `prepare_all.sh` seeds the EvalPlus cache through the mirror, so MBPP+
  preparation works on networks that cannot reach `github.com`.
- `scripts/90_audit/scan_prose.py`, a prose gate that rejects em and en dashes
  and any CJK character outside `scripts/90_audit/cjk_allowlist.txt`. The
  allowlist records the functional CMATH answer cues and punctuation sets that
  must stay byte for byte identical.
- `tests/test_release_surface.py`, contract tests for the mistakes that actually
  reached a release: untracked preparation scripts, dangling documentation
  links, unanchored ignore patterns, and workflow YAML that never parsed. It
  also resolves every `c3.*` and `openrlhf.*` import statement statically
  against the files git tracks, so a package that is never committed cannot
  ship again.
- `CHANGELOG.md`, this file.

### Changed

- `scripts/` and `docs/` are numbered in the order a new user follows them:

  | Before | After |
  |---|---|
  | `scripts/data/` | `scripts/10_data/` |
  | `scripts/models/` | `scripts/20_models/` |
  | `scripts/reproduce/smoke.sh`, `preflight_repro.sh` | `scripts/30_smoke/` |
  | `scripts/reproduce/paper_train.sh` | `scripts/40_train/` |
  | `scripts/reproduce/paper_main_results.sh` | `scripts/50_eval/` |
  | `scripts/reproduce/paper_analysis_figs.sh` | `scripts/60_analysis/` |
  | `scripts/audit/` | `scripts/90_audit/` |
  | `scripts/reproduce/common_env.sh` | `scripts/_lib/common_env.sh` |
  | `docs/GETTING_STARTED.md` | `docs/00_getting_started.md` |
  | `docs/CODE_MAP.md` | `docs/10_code_map.md` |
  | `docs/IMPLEMENTATION_AUDIT.md` | `docs/20_implementation_audit.md` |
  | `docs/IMPLEMENTATION_CHECKLIST.md` | `docs/21_implementation_checklist.md` |
  | `docs/DATA_SOURCES.md` | `docs/30_data_sources.md` |
  | `docs/UPSTREAM.md` | `docs/40_upstream.md` |
  | `docs/CHANGES_FROM_OPENRLHF.md` | `docs/41_changes_from_openrlhf.md` |
  | `docs/RELEASE_POLICY.md` | `docs/50_release_policy.md` |
  | `docs/RELEASE_CHECKLIST.md` | `docs/51_release_checklist.md` |

- Five package-internal renames. The registered algorithm names, CLI flags,
  subcommands and config values are unchanged; only the module paths moved:

  | Before | After | Why |
  |---|---|---|
  | `c3/credit/c3/` | `c3/credit/counterfactual/` | the doubled name `c3.credit.c3` confused readers |
  | `c3/algorithms/c3.py` | `c3/algorithms/group_baseline.py` | the module is the group-baseline fallback, not the paper method |
  | `c3/text_sanitize.py` | `c3/utils/text_sanitize.py` | it was a loose module at the package root |
  | `c3/tools/c3_env_smoke.py` | `c3/tools/env_smoke.py` | inside a package named `c3` the `c3_` prefix is redundant |
  | `c3/analysis/c3_analysis.py` | `c3/analysis/analysis.py` | same reason; `c3.analysis.c3_analysis` read as badly as `c3.credit.c3` |

- Continuous integration is a single workflow, `.github/workflows/ci-cpu.yml`,
  with a `cpu-tier` job that runs the README quickstart end to end and a
  `release-audit` job that runs the pre-release audit.
- The release gate (`scripts/90_audit/release_gate.sh`) now covers the whole CPU
  tier: dependency consistency, tests, both fixture smokes, the plotting path,
  the dataset manifest when prepared data is present, model registry resolution,
  the pre-release audit, and a check that the gate left no artifacts behind.
- `scripts/20_models/download_models.sh` uses the current `huggingface_hub`
  snapshot API. The removed `resume_download` and `local_dir_use_symlinks`
  arguments are gone; resuming and real files are now the library default.
- All remaining Chinese comments, usage text and user-facing messages are in
  English. The functional CJK strings used by the CMATH parsers are unchanged
  and recorded in the audit allowlist.
- The audit scanners scope their generated-directory exclusions to the
  repository root. An unanchored `data` exclusion meant the data preparation
  scripts were never scanned for hard-coded paths.
- The training entry point imports without flash-attn: the two module-level
  imports in `openrlhf/models/ring_attn_utils.py` are guarded. Ring attention,
  the sequence-packing path across ranks, still requires flash-attn and now says
  so at the point of use instead of at import time.

### Fixed

- The vendored `openrlhf/models` package is in the release. Its six modules
  (`__init__`, `actor`, `loss`, `model`, `ring_attn_utils`, `utils`) were never
  committed, for the same reason the preparation scripts were missing: the
  unanchored `models/` ignore pattern matched `openrlhf/models/` too. On a fresh
  clone the training CLI, the evaluation sweep and
  `scripts/30_smoke/smoke.sh --tier gpu` all failed with
  `ModuleNotFoundError: No module named 'openrlhf.models'`, in every
  environment. The package is restored byte for byte from the release packaging
  snapshot, so no behavior changed. Restoring it exposed a second problem the
  missing package used to hide: `openrlhf/models/ring_attn_utils.py` imported
  `flash_attn` at module level, and no lock file can pin flash-attn because it
  builds from source against a CUDA toolchain. That import is now guarded; see
  the entry under Changed.
- The README quickstart works: `scripts/10_data/prepare_all.sh` and
  `scripts/20_models/download_models.sh` are in the release. They were excluded
  from the previous release by unanchored `.gitignore` patterns, which are now
  anchored to the repository root and covered by a test.
- The repository installs and its tests pass without a GPU. The previous single
  `requirements.txt` failed `pip check`, because it pinned `xformers==0.0.32.post1`
  (which requires `torch==2.8.0`) alongside `torch==2.9.0+cu128`.
- `.github/workflows/pre-release-audit.yml` never ran: its `uses:` and `run:`
  keys were indented at the same level as `- name:`, so the YAML was malformed.
  A test now parses every workflow and checks every step.
- `.gitattributes` normalizes line endings for `.toml`, `.yaml`, `.json`,
  `.jsonl` and `.txt`, so a Windows checkout no longer shows spurious diffs.

### Deprecated

- `c3.credit.c3`, `c3.algorithms.c3`, `c3.text_sanitize`,
  `c3.tools.c3_env_smoke` and `c3.analysis.c3_analysis` are import shims that
  re-export the public symbols from their new locations and raise a
  `DeprecationWarning` naming the replacement. The last two also keep
  `python -m <old path>` working, so existing scripts and `SMOKE_MOD` or
  `ANALYSIS_MOD` overrides do not break. They will be removed in a future
  release.
- The root `requirements.txt` is removed. Use `requirements/cpu.lock.txt`,
  `requirements/gpu.lock.txt`, or `requirements/gpu-paper.lock.txt`.

## [0.1.0]

Initial public release accompanying the arXiv submission.
