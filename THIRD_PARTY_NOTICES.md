# Third-Party Notices

This repository (C3) is built on top of, and interoperates with, third-party open-source software and public datasets.
This document summarizes major attributions. It is **not** a substitute for the full license texts of those projects.

> **Important:** This repository does **not** redistribute third-party datasets.
> Dataset files are downloaded by scripts under `scripts/10_data/` from their official sources and prepared into canonical JSONL artifacts.

For the **single source of truth** on dataset IDs, pinned revisions, prepared artifact paths, and SHA256 checksums, see:

- `configs/data_manifest.yaml`
- `docs/30_data_sources.md`

---

## Upstream base: OpenRLHF

- Project: OpenRLHF (OpenRLHF/OpenRLHF)
- License: Apache License 2.0
- Notes: We preserve `openrlhf/` for traceability and compatibility. See `docs/40_upstream.md`.

---

## Major software dependencies (non-exhaustive)

C3 relies on the standard LLM training ecosystem, including (but not limited to):

- Ray (distributed execution)
- vLLM (fast inference engine)
- DeepSpeed (distributed training)
- PyTorch (core ML framework)
- Hugging Face Transformers / Datasets / Hub tooling

Each dependency is governed by its own license; please refer to their upstream repositories and license files.

---

## Datasets used by the paper configs (downloaded; no redistribution)

Below items are summarized **exactly as pinned in `configs/data_manifest.yaml`** (dataset IDs / splits / outputs).
The manifest also includes pinned upstream revisions and prepared-artifact SHA256 values for reproducibility.

### MATH (train) + MATH500

- Academic origin: Hendrycks et al., “Measuring Mathematical Problem Solving With the MATH Dataset”
- Upstream HF IDs (pinned in manifest):
  - `EleutherAI/hendrycks_math` (used for the **train** and **test** artifacts in this release; upstream stores one config per subject, and each artifact is the union of the seven subject configs, see manifest notes)
  - `HuggingFaceH4/MATH-500` (benchmark)
- Prepared outputs (via `scripts/10_data/prepare_math.py`):
  - `data/MATH/train.jsonl`
  - `data/MATH/test_full.jsonl`
  - `data/MATH/pool_informative.jsonl` (a subset of `data/MATH/test_full.jsonl`, selected by the id list in `configs/data/pool_informative_ids.json`)
  - `data/MATH500/test.jsonl`
- License: see the upstream dataset cards; C3 does not redistribute raw data.

### GSM8K

- Academic origin: Cobbe et al., “Training Verifiers to Solve Math Word Problems”
- Upstream HF ID (pinned in manifest): `openai/gsm8k` (`config: main`)
- Prepared output:
  - `data/GSM8K/test.jsonl`
- License: see the upstream dataset card; C3 does not redistribute raw data.

### CMATH (train/test)

- Academic origin: CMATH benchmark (see upstream dataset card for citation details)
- Upstream HF ID (pinned in manifest): `weitianwen/cmath`
- Prepared outputs (via `scripts/10_data/prepare_math.py`):
  - `data/CMATH/train.jsonl` (**from upstream `validation` split**, per manifest pin/notes)
  - `data/CMATH/test.jsonl`
- License: see the upstream dataset card; C3 does not redistribute raw data.

### HumanEval

- Benchmark: HumanEval (see upstream dataset card for citation details)
- Upstream HF ID (pinned in manifest): `openai/openai_humaneval`
- Prepared output (via `scripts/10_data/prepare_code.py`):
  - `data/HumanEval/test.jsonl`
- License: see the upstream dataset card; C3 does not redistribute raw data.

### APPS

- Benchmark: APPS (see upstream dataset card for citation details)
- Upstream HF ID (pinned in manifest): `codeparrot/apps`
- Prepared output (via `scripts/10_data/prepare_code.py`):
  - `data/APPS/test.jsonl`
- License: see the upstream dataset card; C3 does not redistribute raw data.

### MBPP + MBPP+ (EvalPlus)

- Academic origin: MBPP benchmark (see upstream dataset card for citation details)
- Upstream HF ID (pinned in manifest): `google-research-datasets/mbpp` (`config: full`)
- MBPP prepared outputs (via `scripts/10_data/prepare_code.py`):
  - `data/MBPP/train.jsonl`
  - `data/MBPP/test.jsonl`

- MBPP+ (stronger tests) has a manifest entry and is not prepared in this release:
  - Manifest entry name: `MBPP+`
  - Manifest source kind: `evalplus`
  - Manifest pinned provenance tag: `evalplus@0.3.1`
  - Intended output (via `scripts/10_data/prepare_code.py`):
    - `data/MBPP_PLUS/test.jsonl`

> Note on MBPP+ reproducibility:
> - This release pins MBPP+ to an explicit EvalPlus version tag in the manifest.
> - Strict verification (`--strict 1`) enforces that your environment matches the pinned provenance rule.
> - `scripts/10_data/prepare_code.py` refuses to write the artifact since 0.2.4. An
>   EvalPlus task holds `prompt`, `canonical_solution`, `assertion`, `contract`,
>   `entry_point`, `atol`, `base_input`, `plus_input` and `task_id`, and turning the
>   inputs into tests this repository's evaluator can run is an open question.
> - See `docs/30_data_sources.md` for strict-mode behavior and for that question.

### Minerva-Math

- Benchmark: Minerva Math, the OCWCourses problem set released with the Minerva paper (see upstream dataset card for citation details)
- Upstream HF ID (pinned in manifest): `math-ai/minervamath`
- Prepared output (via `scripts/10_data/prepare_math.py`):
  - `data/MINERVA_MATH/test.jsonl`
- License: see the upstream dataset card; C3 does not redistribute raw data.

### OlympiadBench

- Benchmark: OlympiadBench, the open-ended text-only English maths subset (see upstream dataset card for citation details)
- Upstream HF ID (pinned in manifest): `math-ai/olympiadbench`
- Prepared output (via `scripts/10_data/prepare_math.py`):
  - `data/OLYMPIADBENCH/test.jsonl`
- License: see the upstream dataset card; C3 does not redistribute raw data.

### AMC23

- Benchmark: the short-answer problems of AMC 2023 (see upstream dataset card for citation details)
- Upstream HF ID (pinned in manifest): `math-ai/amc23`
- Prepared output (via `scripts/10_data/prepare_math.py`):
  - `data/AMC23/test.jsonl`
- License: see the upstream dataset card; C3 does not redistribute raw data.

### AIME24

- Benchmark: the problems of AIME 2024 (see upstream dataset card for citation details)
- Upstream HF ID (pinned in manifest): `math-ai/aime24`
- Prepared output (via `scripts/10_data/prepare_math.py`):
  - `data/AIME24/test.jsonl`
- License: see the upstream dataset card; C3 does not redistribute raw data.

### AIME25

- Benchmark: the problems of AIME 2025 (see upstream dataset card for citation details)
- Upstream HF ID (pinned in manifest): `math-ai/aime25`
- Prepared output (via `scripts/10_data/prepare_math.py`):
  - `data/AIME25/test.jsonl`
- License: see the upstream dataset card; C3 does not redistribute raw data.

> Note on the five entries above: they are evaluation only; the main task
> evaluates on four of them since the evaluation protocol of 2026-09-15; nothing
> trains on them. `scripts/10_data/prepare_math.py` prepares them by default
> (`--prepare_candidate_benchmarks`, which defaults to `1`).

---

## Acknowledgements

If you believe an attribution is missing or incorrect, please open an issue or PR with:
- the upstream project/dataset,
- a pointer to its license / dataset card,
- and where it is used in this repository.
