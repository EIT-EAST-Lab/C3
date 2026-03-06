# Data Sources & Deterministic Preparation (No Dataset Redistribution)

This repository **does not** redistribute third-party raw datasets.  
Instead, scripts under `scripts/data/` download from pinned upstream sources and produce canonical JSONL files under `data/`.

Generated files under `data/` are local reproducibility outputs, not public-release assets. The public repository ships scripts, manifests, and documentation, but not prepared dataset files.

For reproducibility and auditability, the **single source of truth (SSOT)** is:

- `configs/data_manifest.yaml`

The manifest pins:

1. **Upstream dataset revision** (`source.revision`)
   - For Hugging Face sources, this is an immutable **40-char commit SHA**.
   - For EvalPlus-derived MBPP+, this is a provenance tag:
     - `evalplus@<VERSION>` or
     - `fallback_mbpp@<HF_COMMIT_SHA>`.
2. **Prepared artifact hash** (`sha256`)
   - SHA256 of the final canonical JSONL file.

This creates an end-to-end evidence chain without committing raw datasets.

---

## Naming & Path Contract (Release Invariant)

The release treats the manifest as SSOT, and all configs/scripts/docs must align to it.

### Output name conventions

- Train split artifacts: `<DATASET>-train`
- Test split artifacts: `<DATASET>-test`
- Fixed benchmark names remain unchanged (e.g., `MATH500`)
- Special case: MBPP+ output directory is **always** `MBPP_PLUS` (ASCII-safe)

### Output path conventions

Manifest `output_path` uses repo-relative paths (typically under `data/...`).  
If `--out_dir` is provided, scripts emit files under that directory with the same relative structure (stripping the leading `data/` prefix), so you do **not** get `data/data/...`.

---

## Quick Start

### 1) Prepare all datasets

```bash
bash scripts/data/prepare_all.sh --out_dir data
```

(You can omit `--out_dir data`; it defaults to `data/`.)

### 2) Release verification (strict)

```bash
bash scripts/data/prepare_all.sh --out_dir data --strict 1
```

`--strict 1` fails if:

- a required output still has missing `sha256` pins in `configs/data_manifest.yaml`, or
- a Hugging Face dataset uses a mutable revision (e.g., `main`), or
- MBPP+ provenance is not properly pinned / matched.

### 3) Maintainers: when outputs legitimately change

If you intentionally changed preprocessing logic or bumped upstream pins, regenerate SHA256 pins:

```bash
bash scripts/data/prepare_all.sh --out_dir data --update_manifest_sha256 1
```

Then commit:

```bash
git add configs/data_manifest.yaml
git commit -m "Pin prepared artifact checksums (sha256) after data refresh"
```

---

## Runtime behavior (what the scripts enforce)

### A) `prepare_math.py` / `prepare_code.py`

Preparation scripts:

- use manifest as the only provenance source,
- load upstream datasets with manifest-pinned revisions,
- produce canonical JSONL outputs,
- compute SHA256 on outputs,
- verify SHA256 when present,
- optionally write computed pins back to manifest via `--update_manifest_sha256 1`,
- enforce strict checks under `--strict 1`.

### B) Unified entrypoint `prepare_all.sh`

`prepare_all.sh` forwards key reproducibility flags to both math/code preparation:

- `--out_dir` (or legacy alias `--data_dir`)
- `--overwrite 0|1`
- `--strict 0|1`
- `--update_manifest_sha256 0|1`

---

## Prepared Outputs (Authoritative: `configs/data_manifest.yaml`)

The following list is the **current release target set**, and matches the manifest exactly.

### Math (`scripts/data/prepare_math.py`)

- `data/MATH/train.jsonl`  (manifest name: `MATH-train`)
- `data/GSM8K/test.jsonl`  (manifest name: `GSM8K-test`)
- `data/MATH500/test.jsonl` (manifest name: `MATH500`)
- `data/CMATH/train.jsonl` (manifest name: `CMATH-train`)
- `data/CMATH/test.jsonl`  (manifest name: `CMATH-test`)

### Code (`scripts/data/prepare_code.py`)

- `data/HumanEval/test.jsonl` (manifest name: `HumanEval`)
- `data/APPS/test.jsonl`      (manifest name: `APPS`)
- `data/MBPP/train.jsonl`     (manifest name: `MBPP-train`)
- `data/MBPP/test.jsonl`      (manifest name: `MBPP-test`)
- `data/MBPP_PLUS/test.jsonl` (manifest name: `MBPP+`)

> Note: `MBPP_PLUS` (underscore) is intentional and is the only canonical directory name in this release.

---

## Dataset Notes (Pinned via `configs/data_manifest.yaml`)

Below is a human-readable view of what is pinned.  
For the authoritative values (including exact commits and SHA256), always consult `configs/data_manifest.yaml`.

### MATH-train

- Upstream HF ID: `qwedsacf/competition_math`
- Split: `train`
- Revision: pinned commit SHA (see manifest)
- Output: `data/MATH/train.jsonl`

> Note: In this release, the training artifact is pinned to the specific HF dataset ID above.  
> If you swap the upstream source, you must repin the revision and regenerate sha256 pins.

### GSM8K-test

- Upstream HF ID: `openai/gsm8k` (`config: main`)
- Split: `test`
- Revision: pinned commit SHA
- Output: `data/GSM8K/test.jsonl`

### MATH500

- Upstream HF ID: `HuggingFaceH4/MATH-500`
- Split: `test`
- Revision: pinned commit SHA
- Output: `data/MATH500/test.jsonl`

### CMATH-train / CMATH-test

- Upstream HF ID: `weitianwen/cmath`
- Revisions: pinned commit SHA (same pin for both splits in manifest)
- Outputs:
  - `CMATH-train` → `data/CMATH/train.jsonl` (manifest pins upstream split as `validation`)
  - `CMATH-test`  → `data/CMATH/test.jsonl`  (upstream split `test`)

> Why `validation` is used as the “train artifact”:
> - The manifest defines the release artifact set; the training/eval configs in `configs/tasks/math.yaml`
>   consume `data/CMATH/train.jsonl`.
> - We therefore pin the upstream split explicitly in the manifest to remove ambiguity.

### HumanEval

- Upstream HF ID: `openai/openai_humaneval`
- Split: `test`
- Revision: pinned commit SHA
- Output: `data/HumanEval/test.jsonl`

### APPS

- Upstream HF ID: `codeparrot/apps`
- Split: `test`
- Revision: pinned commit SHA
- Output: `data/APPS/test.jsonl`

### MBPP-train / MBPP-test

- Upstream HF ID: `google-research-datasets/mbpp` (`config: full`)
- Splits: `train`, `test`
- Revision: pinned commit SHA
- Outputs:
  - `data/MBPP/train.jsonl`
  - `data/MBPP/test.jsonl`

### MBPP+ (EvalPlus-pinned with strict provenance)

- Manifest entry name: `MBPP+`
- Output: `data/MBPP_PLUS/test.jsonl`
- Manifest provenance tag (pinned): `evalplus@0.3.1`

Strict-mode behavior:

- If manifest says `evalplus@<VERSION>`:
  - EvalPlus must be importable.
  - Installed EvalPlus version must match exactly.
- If (in a future release) manifest says `fallback_mbpp@<HF_COMMIT_SHA>`:
  - EvalPlus path is disabled; deterministic fallback is forced.
  - The fallback pin must be self-consistent with the MBPP HF revision pin.

---

## Common Failure Cases

- **`[FAIL] source.revision must be immutable`**
  - A Hugging Face dataset uses a mutable ref (e.g., `main`).
  - Fix: replace with a commit SHA in the manifest.

- **`[FAIL] sha256 mismatch`**
  - Prepared output differs from pinned artifact.
  - Typical causes: preprocessing change, upstream drift, stale output file.
  - Fix: re-run with overwrite; if intentional update, regenerate pins with `--update_manifest_sha256 1` and commit.

- **`[FAIL] Missing sha256 pins ...` (strict mode)**
  - Required outputs still have missing `sha256`.
  - Fix: run once with `--update_manifest_sha256 1`, commit manifest, rerun strict.

- **`[FAIL] MBPP+: EvalPlus version mismatch / not importable`**
  - Manifest pins `evalplus@...` but environment does not match.
  - Fix: install the pinned EvalPlus version, or intentionally repin to `fallback_mbpp@...` in a clean environment.

---

## Licensing & Attribution

This repository does not distribute third-party raw data.  
You are responsible for complying with upstream dataset licenses and usage terms.

- Dataset IDs, revisions, and prepared-artifact SHA256 pins are tracked in:
  - `configs/data_manifest.yaml`
- Attribution summary:
  - `THIRD_PARTY_NOTICES.md`

For official license text and citation metadata, refer to each upstream dataset card.
