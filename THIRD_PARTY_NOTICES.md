# Third-Party Notices

This repository (C3) is built on top of, and interoperates with, third-party open-source software and public datasets.
This document summarizes major attributions. It is **not** a substitute for the full license texts of those projects.

> **Important:** This repository does **not** redistribute third-party datasets.
> Dataset files are downloaded by scripts under `scripts/data/` from their official sources and prepared into canonical JSONL artifacts.

For the **single source of truth** on dataset IDs, pinned revisions, prepared artifact paths, and SHA256 checksums, see:

- `configs/data_manifest.yaml`
- `docs/DATA_SOURCES.md`

---

## Upstream base: OpenRLHF

- Project: OpenRLHF (OpenRLHF/OpenRLHF)
- License: Apache License 2.0
- Notes: We preserve `openrlhf/` for traceability and compatibility. See `docs/UPSTREAM.md`.

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
  - `qwedsacf/competition_math` (used for the **train** artifact in this release; see manifest notes)
  - `HuggingFaceH4/MATH-500` (benchmark)
- Prepared outputs (via `scripts/data/prepare_math.py`):
  - `data/MATH/train.jsonl`
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
- Prepared outputs (via `scripts/data/prepare_math.py`):
  - `data/CMATH/train.jsonl` (**from upstream `validation` split**, per manifest pin/notes)
  - `data/CMATH/test.jsonl`
- License: see the upstream dataset card; C3 does not redistribute raw data.

### HumanEval

- Benchmark: HumanEval (see upstream dataset card for citation details)
- Upstream HF ID (pinned in manifest): `openai/openai_humaneval`
- Prepared output (via `scripts/data/prepare_code.py`):
  - `data/HumanEval/test.jsonl`
- License: see the upstream dataset card; C3 does not redistribute raw data.

### APPS

- Benchmark: APPS (see upstream dataset card for citation details)
- Upstream HF ID (pinned in manifest): `codeparrot/apps`
- Prepared output (via `scripts/data/prepare_code.py`):
  - `data/APPS/test.jsonl`
- License: see the upstream dataset card; C3 does not redistribute raw data.

### MBPP + MBPP+ (EvalPlus)

- Academic origin: MBPP benchmark (see upstream dataset card for citation details)
- Upstream HF ID (pinned in manifest): `google-research-datasets/mbpp` (`config: full`)
- MBPP prepared outputs (via `scripts/data/prepare_code.py`):
  - `data/MBPP/train.jsonl`
  - `data/MBPP/test.jsonl`

- MBPP+ (stronger tests) is prepared from EvalPlus when pinned and available:
  - Manifest entry name: `MBPP+`
  - Manifest source kind: `evalplus`
  - Manifest pinned provenance tag: `evalplus@0.3.1`
  - Prepared output (via `scripts/data/prepare_code.py`):
    - `data/MBPP_PLUS/test.jsonl`

> Note on MBPP+ reproducibility:
> - This release pins MBPP+ to an explicit EvalPlus version tag in the manifest.
> - Strict verification (`--strict 1`) enforces that your environment matches the pinned provenance rule.
> - See `docs/DATA_SOURCES.md` for strict-mode behavior.

---

## Acknowledgements

If you believe an attribution is missing or incorrect, please open an issue or PR with:
- the upstream project/dataset,
- a pointer to its license / dataset card,
- and where it is used in this repository.
