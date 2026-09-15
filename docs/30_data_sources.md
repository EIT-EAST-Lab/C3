# Data Sources & Deterministic Preparation (No Dataset Redistribution)

This repository **does not** redistribute third-party raw datasets.  
Instead, scripts under `scripts/10_data/` download from pinned upstream sources and produce canonical JSONL files under `data/`.

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
3. **How the artifact is built** (`source.configs`, `source.subtract_split`, `source.notes`)
   - `configs` is a list of upstream configs to load and concatenate, in that order.
   - `subtract_split` names another split of the same revision whose problems are
     removed from this artifact.
   - `notes` states the split and the deduplication rule in words, for every entry.
4. **Expected row count** (`expected_rows`, optional)
   - Checked under `--strict 1` while the `sha256` pin is still empty, which is the
     window right after an upstream revision moved.

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
bash scripts/10_data/prepare_all.sh --out_dir data
```

(You can omit `--out_dir data`; it defaults to `data/`.)

### 2) Release verification (strict)

```bash
bash scripts/10_data/prepare_all.sh --out_dir data --strict 1
```

`--strict 1` fails if:

- a required output still has missing `sha256` pins in `configs/data_manifest.yaml`, or
- a Hugging Face dataset uses a mutable revision (e.g., `main`), or
- MBPP+ provenance is not properly pinned / matched, or
- an artifact that declares `expected_rows` has a different number of rows while its
  `sha256` pin is still empty, or
- a problem in an evaluation file also occurs in a training file (the overlap gate
  below).

### 3) Maintainers: when outputs legitimately change

If you intentionally changed preprocessing logic or bumped upstream pins, regenerate SHA256 pins:

```bash
bash scripts/10_data/prepare_all.sh --out_dir data --update_manifest_sha256 1
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

Two rules are worth stating on their own, because breaking either one is silent:

- A requested split that no upstream file holds is a **hard failure**. The loader
  never substitutes another split, and never falls back to "the only file in the
  repository".
- A multi-config entry loads **every** config in `source.configs` and concatenates
  them, then deduplicates by `unique_id` when present and otherwise by the triple
  of input, answer and solution.

### B) Unified entrypoint `prepare_all.sh`

`prepare_all.sh` forwards key reproducibility flags to both math/code preparation:

- `--out_dir` (or legacy alias `--data_dir`)
- `--overwrite 0|1`
- `--strict 0|1`
- `--update_manifest_sha256 0|1`

### C) Overlap gate `check_overlap.py`

```bash
python scripts/10_data/check_overlap.py --out_dir data
```

`prepare_all.sh --strict 1` runs it last, and a failure fails the whole
preparation. The gate reads every prepared file the manifest lists, normalizes
each problem statement (trim, then collapse runs of whitespace), and intersects
every training file with every evaluation file. Training files are the artifacts
whose manifest name ends in `-train`, per the naming contract above; every other
artifact is an evaluation file.

Exit codes: `0` when no evaluation problem occurs in any training file (it prints
the zero for every pair), `1` when one does (it prints the count and a few of the
shared problems per pair), `2` when the check could not be run at all. Artifacts
that were not prepared, such as HumanEval and APPS under the default flags, are
reported as skipped; a run where nothing at all was readable exits `2` rather than
reporting a vacuous pass. The gate uses the standard library only.

---

## Prepared Outputs (Authoritative: `configs/data_manifest.yaml`)

The following list is the **current release target set**, and matches the manifest exactly.

### Math (`scripts/10_data/prepare_math.py`)

- `data/MATH/train.jsonl`  (manifest name: `MATH-train`)
- `data/GSM8K/test.jsonl`  (manifest name: `GSM8K-test`)
- `data/MATH500/test.jsonl` (manifest name: `MATH500`)
- `data/CMATH/train.jsonl` (manifest name: `CMATH-train`)
- `data/CMATH/test.jsonl`  (manifest name: `CMATH-test`)

### Math, candidate benchmarks (`scripts/10_data/prepare_math.py`)

- `data/MINERVA_MATH/test.jsonl` (manifest name: `Minerva-Math`)
- `data/OLYMPIADBENCH/test.jsonl` (manifest name: `OlympiadBench`)
- `data/AMC23/test.jsonl` (manifest name: `AMC23`)
- `data/AIME24/test.jsonl` (manifest name: `AIME24`)
- `data/AIME25/test.jsonl` (manifest name: `AIME25`)

> Note: these five are prepared only when you pass `--prepare_eval_probe_sets 1`, which defaults to `0`, so the default run and the strict release check prepare exactly the same files they did before. They are evaluation only; nothing trains on them.

### Code (`scripts/10_data/prepare_code.py`)

- `data/HumanEval/test.jsonl` (manifest name: `HumanEval`)
- `data/APPS/test.jsonl`      (manifest name: `APPS`)
- `data/MBPP/train.jsonl`     (manifest name: `MBPP-train`)
- `data/MBPP/test.jsonl`      (manifest name: `MBPP-test`)
- `data/MBPP_PLUS/test.jsonl` (manifest name: `MBPP+`)

> Note: HumanEval and APPS are prepared only when you pass `--prepare_humaneval 1` and `--prepare_apps 1` to `scripts/10_data/prepare_all.sh`. Both default to `0`, so the default run prepares MBPP and MBPP+ only.

> Note: `MBPP_PLUS` (underscore) is intentional and is the only canonical directory name in this release.

---

## Dataset Notes (Pinned via `configs/data_manifest.yaml`)

Below is a human-readable view of what is pinned.  
For the authoritative values (including exact commits and SHA256), always consult `configs/data_manifest.yaml`.

### MATH-train

- Upstream HF ID: `EleutherAI/hendrycks_math`
- Configs: `algebra`, `counting_and_probability`, `geometry`, `intermediate_algebra`,
  `number_theory`, `prealgebra`, `precalculus`
- Split: `train` of each config, concatenated in that order
- Revision: pinned commit SHA (see manifest)
- Output: `data/MATH/train.jsonl`
- Rows keep `level`, `subject` (upstream calls that column `type`) and `solution`.

This is the canonical MATH training split: upstream stores one config per subject,
so the train split of the benchmark is the union of the seven subject train splits.

> Correction, and the reason this section changed. Until now the training artifact
> was pinned to the mirror `qwedsacf/competition_math`. That mirror has a single
> split, it is named `train`, and it holds 12,500 rows: the MATH train set and the
> MATH test set together. Every MATH500 problem is a MATH test problem, so the
> training file contained the whole MATH500 benchmark. The manifest was wrong, and
> the overlap gate now fails on exactly this mistake.

> Note: if you swap the upstream source, you must repin the revision and regenerate
> the sha256 pins.

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
- Revision: one pinned commit SHA, the same for both entries, and it must be a
  revision that really has both splits.
- Outputs:
  - `CMATH-train` -> `data/CMATH/train.jsonl`: upstream split `validation` (600
    problems) minus every problem whose statement also occurs in the `test` split
    (3 problems), which leaves `expected_rows: 597`.
  - `CMATH-test` -> `data/CMATH/test.jsonl`: upstream split `test`,
    `expected_rows: 1098`.

Why `validation` is the training artifact: upstream ships no train split, and the
training configs in `configs/tasks/math*.yaml` consume `data/CMATH/train.jsonl`. The
manifest therefore names the upstream split explicitly, and `source.subtract_split`
makes the training artifact disjoint from the evaluation artifact. Statements are
compared after trimming and collapsing whitespace, never by id, because the same
problem can carry a different id in two splits.

> Correction, and the reason this section changed. The revision pinned until now
> (`9b140b4015a1c33b1f91fdb04400cefb9ea6db7e`) has no test split at all: it has only
> `validation`. The manifest asked for `test` anyway, the loader fell back to the
> only data file in the repository, and `data/CMATH/test.jsonl` came out as a
> byte-for-byte copy of `data/CMATH/train.jsonl`. Any CMATH number measured on that
> file was measured on training problems. The loader now fails instead of
> substituting a split, and the gate would catch the copy independently.

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

### Candidate evaluation benchmarks (start-accuracy probe)

Five evaluation-only artifacts exist so that the start accuracy of a frozen policy
can be measured on benchmarks that might replace the ones with no headroom left.
They are opt-in (`--prepare_eval_probe_sets 1`), no training configuration reads
them, and no released number depends on them.

| Manifest name | Upstream HF ID | Split | Rows upstream | Output |
|---|---|---|---|---|
| `Minerva-Math` | `math-ai/minervamath` | `test` | 272 | `data/MINERVA_MATH/test.jsonl` |
| `OlympiadBench` | `math-ai/olympiadbench` | `test` | 674 | `data/OLYMPIADBENCH/test.jsonl` |
| `AMC23` | `math-ai/amc23` | `test` | 40 | `data/AMC23/test.jsonl` |
| `AIME24` | `math-ai/aime24` | `test` | 30 | `data/AIME24/test.jsonl` |
| `AIME25` | `math-ai/aime25` | `test` | 30 | `data/AIME25/test.jsonl` |

Three of the five need more than a column rename:

- `AIME24` has no answer column at all. Its answer is the content of the last
  `\boxed{}` of the `solution` column, and a row whose extraction comes back
  empty fails the preparation instead of becoming an unscorable row.
- `OlympiadBench` keeps its answer in `final_answer`, a list of strings, and
  keeps any unit out of it in a separate `unit` column. The reward path compares
  one string, so `source.multi_answer_policy` (`drop`, `first` or `join_comma`)
  and `source.unit_policy` (`drop`, `ignore` or `append`) say what the single
  answer string of a prepared row is. Both ship as `DECIDE-ME` and the builder
  refuses to run while they hold that value, the same contract as a `PIN-ME`
  revision. Of the 674 rows, 93 carry more than one answer and 9 carry a unit,
  so the choice moves the row count as well as the answers, which is why no
  `expected_rows` is pinned for this entry.
- `math-ai/olympiadbench` is used rather than `Hothan/OlympiadBench`, which is
  the same subset. Upstream files the English text-only open-ended maths subset
  under config `OE_TO_maths_en_COMP` and split `train`; the mirror ships the same
  674 rows under a real `test` split, so the split name in the manifest means
  what it says and the repository-file fallback of `prepare_math.py` can find it.

The probe that consumes these files is `scripts/70_rebuild/eval_probe.py`, driven
by `configs/tasks/math_eval_probe.yaml`.

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

- **`[FAIL] <repo>@<rev>: no data file matches split='<split>'`**
  - The pinned revision does not have the split the manifest asks for.
  - Fix: repin to a revision that has it. Do not expect a fallback: preparing some
    other split under this name is the failure this message prevents.

- **`[FAIL] row count mismatch for prepared artifact`**
  - The artifact declares `expected_rows` and came out a different size while its
    `sha256` pin was empty.
  - Typical cause: the upstream revision moved. Fix the pin, or update
    `expected_rows` deliberately if the change is the intended one.

- **`[FAIL] evaluation problems occur in training files`**
  - The overlap gate found an evaluation problem inside a training file.
  - Fix the manifest or the preparation. Never pin around this one: every number
    measured on that benchmark would be measured on training problems.

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
