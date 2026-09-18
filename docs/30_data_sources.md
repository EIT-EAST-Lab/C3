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
- a prepared row carries no problem statement, or an evaluation row carries no tests
  (the write door, which every artifact of both preparation scripts passes through,
  whether it is written now or was written by an earlier run and is being verified), or
- a problem in an evaluation file also occurs in a training file (the overlap gate
  below).

A `sha256` pin says that a file is the one we produced. It cannot say that the file is
usable, and the MBPP+ artifact of 2026-09-07 was 378 rows of nothing that matched its
pin on every run. That is what the write door is for, and why it is not optional.

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
- `data/MATH/test_full.jsonl` (manifest name: `MATH-test`)
- `data/MATH/pool_informative.jsonl` (manifest name: `MATH-pool`)
- `data/GSM8K/test.jsonl`  (manifest name: `GSM8K-test`)
- `data/MATH500/test.jsonl` (manifest name: `MATH500`)
- `data/CMATH/train.jsonl` (manifest name: `CMATH-train`)
- `data/CMATH/test.jsonl`  (manifest name: `CMATH-test`)

> Note: `MATH-pool` is the one entry of this list that a default run usually does
> not write. It is a subset of `MATH-test` named by an id list, and while that
> list is empty the builder prints a skip line and writes no file. See the
> `MATH-pool` section below.

### Math, candidate benchmarks (`scripts/10_data/prepare_math.py`)

- `data/MINERVA_MATH/test.jsonl` (manifest name: `Minerva-Math`)
- `data/OLYMPIADBENCH/test.jsonl` (manifest name: `OlympiadBench`)
- `data/AMC23/test.jsonl` (manifest name: `AMC23`)
- `data/AIME24/test.jsonl` (manifest name: `AIME24`)
- `data/AIME25/test.jsonl` (manifest name: `AIME25`)

> Note: these five are prepared by the default run. `scripts/10_data/prepare_math.py --prepare_candidate_benchmarks` defaults to `1`, and `--prepare_eval_probe_sets` is the former name of that flag, kept as an alias for one release. They are evaluation only; the main task evaluates on four of them (Minerva-Math, AMC23, AIME24, AIME25) since the evaluation protocol of 2026-09-15; nothing trains on them.

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
- Prepared rows: 7,496. The seven subject train splits are the 7,500 problem MATH
  train split, and the prepared file is that union after four reductions:

| Reduction | Rows it drops at the pinned revision | Left |
|---|---|---|
| Deduplication by `unique_id`, or by the input, answer and solution triple | 0 | 7,500 |
| The answer gate: a solution ending in an empty `\boxed{}` has no answer to score against | 2 | 7,498 |
| Repeated problem statement, first occurrence kept | 1 | 7,497 |
| Problem statement that also occurs in the test split | 1 | 7,496 |

This is the canonical MATH training split: upstream stores one config per subject,
so the train split of the benchmark is the union of the seven subject train splits.

The last two reductions are the ones added in 0.2.4, and they key on the normalized
problem statement, which is the key `check_overlap.py` compares, so the preparation
and the gate cannot disagree about what a shared problem is. They are the rule
`CMATH-train` and `MBPP-train` already follow. What they drop here is small and
specific:

- The equilateral triangle problem whose answer is 315 is in both splits, filed as
  Level 5 Geometry in `geometry/train` and as Level 3 Precalculus in
  `precalculus/test`. The statement is identical and every other column differs,
  which is why a check on ids or metadata would not see it.
- `algebra/train` holds the Level 5 problem that ends in `What is $x+y$?` twice, with
  two different solution texts, which is why the deduplication above it does not see
  it either.

Without these two reductions the overlap gate fails on `MATH-train x MATH-test`, which
is how both were found.

The subtraction is not a manifest field and cannot be switched off, because there is
no configuration of this repository in which a MATH training artifact should keep a
test problem. It builds the test side with the builder that writes `MATH-test`, so
what it subtracts is exactly what `MATH-test` is prepared as.

Under the paper's one month older training set the number to compare against is 7,500,
the canonical MATH train split: the public repository has dropped the two rows with no
extractable answer since 0.2.0, and drops the repeat and the shared problem from 0.2.4
on.

The manifest pins this artifact by `sha256`, not by `expected_rows`, so the count
above is documentation and the `sha256` pin is the gate.

> Correction, and the reason this section changed. Until now the training artifact
> was pinned to the mirror `qwedsacf/competition_math`. That mirror has a single
> split, it is named `train`, and it holds 12,500 rows: the MATH train set and the
> MATH test set together. Every MATH500 problem is a MATH test problem, so the
> training file contained the whole MATH500 benchmark. The manifest was wrong, and
> the overlap gate now fails on exactly this mistake.

> Note: if you swap the upstream source, you must repin the revision and regenerate
> the sha256 pins.

### MATH-test

- Upstream HF ID: `EleutherAI/hendrycks_math`
- Configs: the same seven subject configs as `MATH-train`
- Split: `test` of each config, concatenated in that order
- Revision: the same pinned commit SHA as `MATH-train`
- Output: `data/MATH/test_full.jsonl`
- Manifest row count: `expected_rows: 5000`

This is the canonical MATH test split, and MATH500 is drawn from it: the 500
problems of that benchmark are 500 of these rows. One builder writes both MATH
artifacts, so the row format, the deduplication key and the answer gate are
`MATH-train`'s; the two manifest entries differ in `source.split` and in nothing
else.

It is an evaluation artifact under the naming contract above, never a training
one. `check_overlap.py` therefore intersects it with every training file on each
strict run, and the count it prints for `MATH-train x MATH-test` has to be zero.
Two things read it: `configs/tasks/math_screen.yaml`, the task file of the E1
screening pass, and the pool below.

The `sha256` pin is set, so it is the gate and `expected_rows: 5000` is
documentation: the strict pass compares the file against the pin and stops
comparing row counts once a pin is there, exactly as in the `MATH-train` section
above. Repinning after a deliberate change to the artifact is
`--update_manifest_sha256 1`.

### MATH-pool

- Built from: `MATH-test`, not from an upstream download
- Selection: `configs/data/pool_informative_ids.json`, the id list the E1
  screening pass writes
- Output: `data/MATH/pool_informative.jsonl` (manifest entry `MATH-pool`, which
  is `optional: true` and pins no `expected_rows`, because the row count is the
  length of that list)

The pool holds the screened questions the E1 reliability measurement runs on.
The repository ships the id list and not the problem statements, so it
redistributes no dataset row. The builder reads `data/MATH/test_full.jsonl`,
takes the rows the list names in the order the list gives, and fails when one of
the ids is not there rather than writing a shorter file.

Both sides compute an id with `row_id` in `c3/analysis/rebuild/pool_ids.py`: the
`unique_id` of the row, or the first 16 hexadecimal characters of the SHA256 of
its problem statement when it has no `unique_id`. The screening pass that writes
the list calls the same function, so the two sides cannot disagree about what a
row is called.

Until that pass has run, `unique_ids` is empty. The builder then prints

```text
[SKIP] MATHPOOL: pool_informative_ids.json is empty; run the E1 screening pass first
```

and writes nothing. The consumers of the prepared file are the six task files of
the depth study (`math_a2`, `math_a3`, `math_mt4`, `math_branch`, `math_c5`,
`math_c10`), which declare it as the `MATHPOOL` evaluation suite; they are read
by the E1 and E3a cell drivers, which default to `--split MATHPOOL`, and by
nothing on the paper training path.

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
- Revision: pinned commit SHA, the same for both entries
- Outputs:
  - `data/MBPP/train.jsonl`, 317 prepared rows
  - `data/MBPP/test.jsonl`, 500 prepared rows

Both evaluation artifacts are prepared first, and `MBPP-train` is written minus every
problem that occurs in either of them. Upstream ships 374 training rows, and 57 of them
go:

| Dropped because the problem also occurs in | Rows | Left |
|---|---|---|
| `MBPP-test` (this revision of the `full` config ships three problems in both splits) | 3 | 371 |
| `MBPP+` (drawn from the sanitized MBPP, so it spans the train id range too) | 54 | 317 |

The second line is what issue 2 uncovered: 108 of the 378 MBPP+ task ids are MBPP
train-split ids, and the overlap gate could not see it while the MBPP+ artifact had no
problem statement in it. The subtraction keys on the same normalized problem text the
gate compares, so the two cannot disagree, and `check_overlap.py` verifies the result
against both evaluation artifacts on every strict run.

Two things a reader of the counts should know. `MBPP-test` holds 500 rows and 499
distinct problems, and `MBPP+` holds 378 rows and 376 distinct problems: upstream repeats
a problem statement inside each of those evaluation files. Neither repeat affects any
gate, which compares training files against evaluation files, and neither is ours to fix,
so both are left as upstream ships them.

As with `MATH-train`, the manifest pins these artifacts by `sha256` and declares
no `expected_rows` for them, so the counts above are documentation and the `sha256`
pin is the gate.

### MBPP+ (EvalPlus-pinned with strict provenance)

- Manifest entry name: `MBPP+`
- Output: `data/MBPP_PLUS/test.jsonl`, 378 rows, about 4.9 MB
- Manifest provenance tag (pinned): `evalplus@0.3.1`, and there is no other source
- Tests: 41,015, which is 1,174 base inputs plus 39,841 plus inputs, 108.5 per task

#### What an EvalPlus task holds, and what the row holds

The row builder that shipped through v0.2.3 read `text`, `code`, `test_list` and
`test_setup_code` out of an EvalPlus task. An EvalPlus 0.3.1 task has none of those
keys, so all 378 prepared rows came out with a value in `task_id` and `source` and
nothing anywhere else, and the `sha256` pin matched that empty file byte for byte on
every strict run. That is issue 2. The keys a task really has, and where each one goes:

| EvalPlus key | Content | Prepared row |
|---|---|---|
| `task_id` | the sanitized MBPP id with a prefix, `Mbpp/100` | `task_id`, as the integer |
| `prompt` | the statement wrapped in a docstring, with the first MBPP assertion appended inside it | `text`, the statement alone |
| `canonical_solution` | the reference implementation | `code`, and the harness in `test_setup_code` |
| `entry_point` | the name of the function the tests call | `entry_point` |
| `base_input` | the original MBPP inputs, 1,174 over the 378 tasks | `test_setup_code`, one `test_list` entry each |
| `plus_input` | the inputs EvalPlus adds, 39,841 over the 378 tasks | `test_setup_code`, one `test_list` entry each |
| `atol` | the float tolerance, `0` for all but 13 tasks and `1e-4` for those | `atol`, and the harness |
| `assertion` | the three original MBPP assert statements as one string | not carried |
| `contract` | input validation for EvalPlus's own input generator | not carried |

`base_input` and `plus_input` are argument lists, not assertions, and they carry no
expected output. EvalPlus computes what the canonical solution returns for each input
(`evalplus.evaluate.get_groundtruth`, cached under the MD5 of the dataset file) and
compares in memory.

The statement is taken out of the docstring rather than written in as it stands. A
docstring never equals a sentence, so a row carrying the raw prompt would leave
`check_overlap.py` matching nothing at all and reporting a clean file whatever the file
held. With the statement extracted, 152 of the 378 match an MBPP problem and the gate
does its job: 54 of them occur in the training file, which is why `MBPP-train` is now
written minus them as well.

#### How the tests work

Each of the 41,015 inputs becomes one entry of `test_list`, and every entry is the same
shape: `assert _c3_case(0)`, `assert _c3_case(1)`, and so on. Everything those calls need
is in `test_setup_code`: the inputs as Python literals, the reference solution, and the
comparison. The expected value is not stored. It is what the reference returns for that
input, computed while the test runs, which is the only reason the whole benchmark fits in
4.9 MB: writing the expected values out comes to 1.3 GB, because `Mbpp/255` alone returns
a value that serializes to 1.29 GB. The row also carries `n_base_inputs` and
`n_plus_inputs`, so the file can be checked against EvalPlus without loading EvalPlus.

The comparison is EvalPlus's, rewritten for a sandbox that whitelists the standard
library and has no numpy: the set comparison for the 7 entry points EvalPlus names, the
output-is-not-None rule for its 3, the two problems it judges against a second
implementation of its own, and its float tolerance, which starts at the task's `atol`,
becomes `1e-06` the first time an expected value is a float, and stays there for the rest
of the task exactly as the EvalPlus loop leaves it.

What keeps a candidate from writing its own verdict is the order the evaluator runs
things in. `c3/envs/code/executor.py` runs `test_setup_code` before the candidate, and
when that raises `NameError` it runs the same block again after the candidate instead.
The generated setup opens by capturing the entry point and then deleting the name, which
raises `NameError` while no candidate has run, so the whole block lands after the
candidate: a candidate that defines its own `_c3_case` is simply overwritten by the real
one. The `del` is also what makes this work for `Mbpp/126`, whose entry point is `sum`,
where a bare reference would find the builtin and raise nothing. The same block rebinds
whatever modules the reference imports, so a candidate cannot reach the reference by
replacing `re` or `math` either.

Three consequences of judging against a live reference rather than against stored values:

- Every input is run twice, once by the reference and once by the candidate, so a task
  costs about twice what EvalPlus spends on it. The row therefore carries
  `timeout_s: 60`, the per-task ceiling EvalPlus itself uses
  (`EVALPLUS_TIMEOUT_PER_TASK`), and `c3/envs/code/reward.py` takes the larger of that
  and whatever the task file asks for. `Mbpp/599` is the one that needs it: its reference
  alone runs for about 30 seconds.
- `cmath` is on the sandbox import whitelist since 0.2.4. Three MBPP+ problems are about
  complex numbers, and neither their reference nor any correct answer to them can run
  without it.
- `Mbpp/596` needs `sys`, which is not on that whitelist and will not be put there for
  one problem. Its reference cannot run, so the problem scores 0 for every candidate,
  including for the reference solution itself. A number measured on this file is
  therefore measured on 378 problems of which one is unreachable; set
  `C3_CODE_EXTRA_IMPORTS=sys` if you want it scored.

#### Strict-mode behavior

- The manifest pins `evalplus@<VERSION>`. EvalPlus must be importable and its installed
  version must match exactly.
- There is no second source. Until 0.2.4 the manifest also accepted
  `fallback_mbpp@<HF_COMMIT_SHA>`, which wrote the MBPP test problems with the three
  original MBPP assertions under the MBPP+ name whenever EvalPlus was unavailable. That
  is MBPP and not MBPP+, and a file that quietly holds another benchmark is worse than a
  file that is missing, so preparation now fails and says so.

### Candidate evaluation benchmarks (start-accuracy probe)

Five evaluation-only artifacts exist: they are benchmarks on which the start
accuracy of a frozen policy is measured before any of them is adopted. They are
evaluation only; the main task evaluates on four of them (Minerva-Math, AMC23,
AIME24, AIME25) since the evaluation protocol of 2026-09-15; nothing trains on
them. They are prepared by the default run
(`--prepare_candidate_benchmarks`, which defaults to `1`).

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
  answer string of a prepared row is. The manifest ships `drop` for both, so a
  row with more than one answer and a row that carries a unit are not prepared at
  all. `DECIDE-ME` is the unset sentinel for either field and the builder refuses
  to run while one of them holds it, the same contract as a `PIN-ME` revision. Of
  the 674 rows, 93 carry more than one answer and 9 carry a unit, so the choice
  moves the row count as well as the answers, which is why no `expected_rows` is
  pinned for this entry.
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

- **`[FAIL] <name>: row N has an empty problem statement`** or
  **`[FAIL] <name>: row N has no tests`**
  - The write door refused the artifact, and nothing was written.
  - Typical cause: the upstream columns are not the ones the row builder reads, which
    is how the empty MBPP+ file and the answerless CMATH files were produced.
  - Fix the field mapping or repin the revision. Never work around this one by
    removing the check: an evaluation suite whose rows have no statement scores
    nothing and reports it as a result.

- **`[FAIL] MBPP+: EvalPlus could not be loaded, and there is no substitute`**
  - EvalPlus fetches MBPP+ from a GitHub release and could not reach it.
  - Fix: set `GITHUB_MIRROR_PREFIX` (see `docs/31_network_mirrors.md`) and rerun through
    `scripts/10_data/prepare_all.sh`, which seeds the EvalPlus cache through the mirror.
    There is deliberately no fallback that prepares something else under this name.

---

## Licensing & Attribution

This repository does not distribute third-party raw data.  
You are responsible for complying with upstream dataset licenses and usage terms.

- Dataset IDs, revisions, and prepared-artifact SHA256 pins are tracked in:
  - `configs/data_manifest.yaml`
- Attribution summary:
  - `THIRD_PARTY_NOTICES.md`

For official license text and citation metadata, refer to each upstream dataset card.
