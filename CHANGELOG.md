# Changelog

All notable changes to this repository are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.4] - 2026-09-16

**This release removes import paths.** Five aliases kept since 0.2.0 are gone,
and four subpackages are renamed. The full mapping is below; nothing else about
the installed package changed. `tests/contract/test_release_surface.py` now
fails if any withdrawn path still resolves, so the old spellings cannot come
back by accident.

### Changed

- Every `c3` subpackage is now named for what it holds. `c3.algorithms` became
  `c3.baselines`, because it holds the comparison methods and not this paper's
  own; `c3.mas` became `c3.protocol`; `c3.tools` became `c3.reporting`, and the
  environment probe that shared it moved to `c3.utils.env_smoke`;
  `c3.integration` became `c3.task`, whose two modules are now `c3.task.config`
  and `c3.task.datasets`; and `c3.credit.counterfactual` flattened into
  `c3.credit`. One file under it was named for the one thing it does not do, so
  its role graph builder is now `c3.credit.role_dag` and its critic input
  formatter is `c3.credit.q_prompt`.

  | Old import path | New import path |
  |---|---|
  | `c3.algorithms.*` | `c3.baselines.*` |
  | `c3.mas.*` | `c3.protocol.*` |
  | `c3.tools.main_results` | `c3.reporting.main_results` |
  | `c3.tools.analysis_results` | `c3.reporting.analysis_results` |
  | `c3.tools.plot_paper_figures` | `c3.reporting.plot_paper_figures` |
  | `c3.tools.env_smoke` | `c3.utils.env_smoke` |
  | `c3.integration.marl_specs` | `c3.task.config` |
  | `c3.integration.task_datasets` | `c3.task.datasets` |
  | `c3.credit.counterfactual.baselines` | `c3.credit.role_dag` and `c3.credit.q_prompt` |
  | `c3.credit.counterfactual.*` (the rest) | `c3.credit.*` |

- The test suite says what kind of test each file is: `tests/mechanism/` for the
  credit-assignment arithmetic, `tests/contract/` for the release surface and
  the data contracts, `tests/unit/` for everything else. `pytest -q tests` is
  unchanged.

### Fixed

- `data/MBPP_PLUS/test.jsonl` was 378 empty rows, and the manifest pinned the
  `sha256` of exactly that file, so `--strict 1` waved it through and the MBPP+
  evaluation suite scored nothing on every run. The row builder read the MBPP
  column names `text`, `code`, `test_list` and `test_setup_code` out of an
  EvalPlus 0.3.1 task, which carries `prompt`, `canonical_solution`, `assertion`,
  `contract`, `entry_point`, `atol`, `base_input`, `plus_input` and `task_id`.
  Reported as issue 2, with the diagnosis. The builder is removed rather than
  half repaired and the pin is cleared: MBPP+ is not prepared in this release,
  because writing a correct row needs two decisions that change what a number on
  this benchmark means, and the MBPP+ section of `docs/30_data_sources.md` now
  states both with the measurements behind them. Prepare the rest of the code
  data with `--prepare_mbpp_plus 0`. The same commit removes `--mbpp_plus_seed`
  and `--mbpp_plus_max_tests`, which sampled a test list that was never read, and
  with them the `int("Mbpp/100")` that failed and left every row of the file
  sharing one sampling seed; an EvalPlus task id that does not parse is now a
  failure rather than a silent `0`.

- `data/MATH/train.jsonl` still shared one problem with `data/MATH/test_full.jsonl`,
  which the overlap gate reported as a `MATH-train x MATH-test` failure, and held
  one problem twice. `EleutherAI/hendrycks_math` at the pinned revision ships the
  equilateral triangle problem whose answer is 315 as Level 5 Geometry in the train
  split and as Level 3 Precalculus in the test split, and ships one Level 5 algebra
  problem twice with two different solution texts. `MATH-train` is now written minus
  the repeated statements and minus every problem that also occurs in the test split,
  which is the rule `CMATH-train` and `MBPP-train` already follow, keyed on the
  normalized problem statement that the gate itself compares. 7,498 rows become
  7,496 and the `sha256` pin moves with them. Reported as issue 2.

### Removed

- The five import aliases introduced in 0.2.0 (`c3.algorithms.c3`,
  `c3.analysis.c3_analysis`, `c3.credit.c3`, `c3.text_sanitize`,
  `c3.tools.c3_env_smoke`). `docs/50_release_policy.md` now states when a shim
  is created, which release removes it, and which test enforces that.

### Added

- A write door in both data preparation scripts. A prepared row now has to carry a
  problem statement under one of the keys `c3/task/datasets.py` reads, and an
  evaluation row has to carry tests the evaluator can run; the first row that does
  not stops the run, and no file is written. The check also runs against an artifact
  that is being verified rather than written, which is the state the empty MBPP+ file
  was in on every rerun after the first. A `sha256` pin says that a file is the one we
  produced, never that it is usable, and this is the difference that let 378 empty
  rows pass as a benchmark.


- The E1 aggregation reads a noise measurement out of every sweep cell without a
  second run: the even and the odd replays of one group are two independent
  estimates of the same advantage vector, so their disagreement measures the
  noise and the derived law predicts it at the half's own budget. One estimator
  serves both readings, called with two continuation seeds or with two halves.
  The summary carries the ratio per cell, its median, and its rank correlation
  with the branching factor.

## [0.2.3] - 2026-09-15

### Fixed

- `scripts/40_train/paper_train.sh` passed none of the paper's batch
  configuration and several of its optimization settings, so a run launched from
  it silently fell back to the trainer's library defaults: instance batch 128
  where the paper's Table 3 says 256, rollout batch 512 where it says 128, micro
  batches 4 and 8 where it says 64 and 32, one PPO epoch per batch where it says
  five, no KL target where it says 0.1, the k1 KL estimator where it says k3,
  and no bf16. This is the second half of issue #1 and the reporter was right.
  The table's recipe now lives in `c3/utils/paper_train_contract.py`, the
  launcher renders its flags from there rather than spelling them out, and
  `tests/test_release_contracts.py` checks that every flag in the recipe exists
  in the trainer's parser, that the launcher does not write any of them a second
  time, and that the two deliberate departures from the table are declared with
  their reasons and actually passed. Those two are the generation cap of 2048
  and the evaluation every tenth of the run, both argued in
  `docs/32_evaluation_protocol.md`.

### Added

- The E1 aggregation writes the two counts behind every sweep reliability
  number: the groups it is read on, and the share of the collected groups that
  the exclusion rule dropped. The derived rows already carried them; the
  exclusion rate moves with the branching factor, so a reliability read on a
  tenth of the pool is a different claim from one read on most of it.

## [0.2.2] - 2026-09-15

### Fixed

- The `scipy` pin added in 0.2.1 was the then-current release, 1.18.1, which
  requires CPython 3.12 and publishes no wheels for 3.11. This repository
  supports 3.11 (`requires-python = ">=3.11"`), both lock files were resolved on
  3.11, and CI installs on 3.11, so the CPU tier could no longer be installed
  from `requirements/cpu.lock.txt` at all. Both locks now pin `scipy==1.17.1`,
  the last release that ships 3.11 wheels, and the header of each lock records
  why that one pin is held by hand rather than taken from a fresh freeze.
  Verified on CPython 3.11.15 in a virtual environment built from the lock alone:
  the install completes, `pip check` is clean, the 627 unit tests pass, both
  fixture smokes pass, and the pre-release audit passes. The same suite passes
  with this pin on 3.12, so the downgrade costs nothing at the other end of the
  supported range.

## [0.2.1] - 2026-09-15

### Fixed

- `scipy` is a dependency of the `c3` package (the rebuild analysis package
  imports it) and is pinned in both lock files. It was missing from
  `pyproject.toml`, `requirements/cpu.lock.txt` and `requirements/gpu.lock.txt`,
  so the `cpu-tier` job of the 0.2.0 release could not collect four test
  modules. With the pin the suite passes on the locked CPU tier (627 tests, none
  skipped).

## [0.2.0] - 2026-09-15

The first release after the maintenance pass of September 2026. The entries under
"Prepared on 2026-09-08" further down belong to this same release; nothing between
0.1.0 and this tag was published.

### Added

- Six workflow configurations for the depth study, all on the same data as
  `configs/tasks/math.yaml`: `math_a3.yaml` (three agents), `math_mt4.yaml`
  (four turns), `math_branch.yaml` (branching), `math_c5.yaml` (five-agent
  chain), `math_c10.yaml` (ten-agent chain), and `math_screen.yaml` for the E1
  screening pass. The role files are
  `configs/roles/math/roles_{trio,mt4,branch,c5,c10}.json`.
- `scripts/70_rebuild/e1_cells.py` and `scripts/70_rebuild/e3a_cells.py`, which
  emit and optionally run the bucket-generation cells of the E1 depth study and
  of the E3a bias map. `--dry-run` prints the commands and the cell count, and
  needs neither a GPU nor a model.
- `scripts/70_rebuild/eval_probe.py`, which measures the start accuracy of a
  frozen policy on the candidate benchmarks. It drives the existing evaluation
  path in two decodings (greedy, and four samples at temperature 0.7) and writes
  a `probe_summary.json` with Wilson intervals per suite.
  `EVAL_PROBE_EXTRA_ARGS` is appended to the trainer flags, which is where the
  attention backend and the GPU counts of the host go.
- `c3/analysis/replay_batched.py`, cross bucket batched counterfactual replay. It
  keeps the semantics of `ReplayRunner.run_bucket` and only changes when the
  calls are issued: every bucket that needs a generation at the same stage is
  handed to the policy in one `sample_many` call.
  `c3.analysis.analysis build-buckets --batched` selects it.
- `build-buckets --meta_json`, which merges experiment meta into every written
  bucket, and `--inject_literal_candidate` with `--null_form`, which put a null
  arm at `j=0` in both the sequential and the batched path and record it in the
  bucket meta.
- `build-buckets --reuse_candidates_from <buckets.jsonl>`, which replays the
  alternatives an earlier run recorded instead of sampling its own. The two runs
  of a noise cell have to share the alternatives and differ only in the replays,
  otherwise the paired difference of their two advantage vectors also carries
  the difference between two different actions. Decision points are matched on
  `question_id` and `ctx_hash`; one the source does not cover is an error naming
  it, not a resampled bucket. The flag is refused together with
  `--inject_literal_candidate`, and refused against a source built for a
  different number of alternatives per bucket; a source bucket that came up
  short of its own request is reused at the count it has. The sequential path
  and the batched path both support it, and every reused bucket records
  `candidate_reuse`, `candidates_from` and `n_alternatives_effective` in its
  meta.
- `build-buckets --batch_prompts` (default 4096), the prompts per engine call of
  the batched path. It does nothing without `--batched`.
- A `Reliability and Ablation Drivers` section in `README.md` and a `Rebuild
  study: reliability and ablation` section in `docs/10_code_map.md`, covering the
  drivers, the analysis modules, the task configurations and the tests this
  release adds.
- `c3/analysis/rebuild/`, the analysis package of the rebuild study: split-half
  reliability with a group bootstrap (`splithalf.py`), the duplicate rate
  (`duplicates.py`), the estimator noise law (`noise_law.py`), answer-level
  plug-in influence with the Miller-Madow correction, in nats (`influence.py`),
  the ablation bias map (`bias_map.py`), the paired coupling contrast
  (`coupling.py`), one `summary.json` writer (`summary.py`), and the aggregators
  `aggregate_e1.py`, `aggregate_e1b.py`, `aggregate_e1c_temp.py`,
  `aggregate_e3a.py` and `aggregate_e5.py`.
- `scripts/10_data/check_overlap.py`, the contamination gate. It fails when an
  evaluation problem also occurs in a training file, `prepare_all.sh --strict 1`
  runs it last, and it imports the standard library only.
- Five candidate evaluation benchmarks in the manifest: Minerva Math,
  OlympiadBench (the English, text-only, open-ended maths subset), AMC23, AIME24
  and AIME25. They are evaluation only; the main task evaluates on four of them
  since the evaluation protocol of 2026-09-15; nothing trains on them.
  `prepare_math.py --prepare_candidate_benchmarks` defaults to `1`, and
  `--prepare_eval_probe_sets`, the former name of that flag, stays as an alias
  for one release.
- `configs/tasks/math_eval_probe.yaml`, the probe task. Its evaluation suites are
  MATH500, the five candidate benchmarks and the two ceiling controls, eight
  suites, each capped at 100 problems.
- Fifteen test files for the above: the two cell drivers and the probe, the
  analysis package, the workflow configurations, the batched replay path,
  the context scope, the bucket guard, the overlap gate, the data-preparation
  gates and the schema alignment. Four fixtures under `tests/fixtures/data/`
  come with them.
- `MATH-test`, the canonical MATH test split, as a manifest entry and as
  `data/MATH/test_full.jsonl`. It is the union of the seven subject `test`
  configs of `EleutherAI/hendrycks_math` at the revision `MATH-train` already
  reads, written by the same builder, so the row format and the deduplication
  rule are shared and cannot drift apart. It is the set MATH500 is drawn from,
  it is an evaluation artifact under the naming contract, and the overlap gate
  checks it against every training file. `prepare_math.py --prepare_math_test`
  defaults to `1`.
- `MATH-pool`, the informative-question pool of the E1 screening pass, as an
  optional manifest entry and as `data/MATH/pool_informative.jsonl`. The
  repository ships the selection as an id list,
  `configs/data/pool_informative_ids.json`, rather than as problem statements, so
  no dataset row is redistributed; `prepare_math.py --prepare_math_pool`
  (default `1`) takes the rows that list names out of `MATH-test`, in that order,
  and fails when one of the ids is not there. While the list is empty the builder
  prints a skip line and writes nothing.
- `c3/analysis/rebuild/pool_ids.py`, the one definition of the id of a prepared
  math row: its `unique_id`, or the first 16 hexadecimal characters of the
  SHA256 of its problem statement. The pool builder and the screening pass that
  writes the id list both call it, so they cannot disagree about what a row is
  called.
- `tests/test_prepare_math_pool.py`, covering the test-split builder, the id
  rule, the id list and the pool builder.
- A contract test that every role prompt states the size of its own team, so a
  prompt copied from a smaller workflow cannot keep the smaller number.
- Attribution entries in `THIRD_PARTY_NOTICES.md` for the five candidate
  evaluation benchmarks, and the two new MATH artifacts under the existing MATH
  entry.
- `aggregate_e1 --key_prefix` names the leading segment of the keys written, so
  the full-suite appendix tree `E1_math500all` writes `E1app.*` instead of
  colliding with the question-pool tree's `E1.*`. Aggregating the appendix tree
  under the default prefix is refused with the flag to pass. Every E1 summary now
  records the tree it read and the prefix it wrote at the top level
  (`results_root`, `key_prefix`).
- The E3a bias map reports a stratum thinner than ten decision points, on stderr
  and in the note of every key read off it. The median split the analysis plan fixes is
  unchanged and no key is dropped: this only makes a degenerate split visible in
  the summary instead of leaving it to be noticed in the stderr of the run.
- `scripts/70_rebuild/replay_tokens.py` counts what one replay of the deep chain
  regenerates downstream, for `E1.c10.median_prefix_tokens`, which needs a
  tokenizer and the model files and so cannot come from the aggregation. Its
  `--dry-run` reports what the bucket files carry without either.
- `scripts/70_rebuild/final_eval.py` and `configs/tasks/math_final_eval.yaml`,
  the evaluation of record: the one evaluation per trained model the main table
  is read off. The estimator is avg@k with k fixed per suite (4 for MATH500,
  Minerva-Math and the two saturation controls; 16 for AMC23 and the two AIME
  files). One evaluation run has exactly one
  `--eval_n_samples_per_prompt`, so the driver groups the suites by k, writes one
  generated task file per group that is the source task with its suites narrowed,
  and runs one evaluation per group through `paper_main_results.sh one`. It
  writes an evaluation record carrying, per suite, the avg@k accuracy, k, the
  problem count, the boxed rate, a Wilson interval and the k rewards of every
  problem, plus the merged 60-problem `AIME` entry. `--dry-run` prints the
  commands and needs neither a GPU nor a model; `FINAL_EVAL_EXTRA_ARGS` carries
  the machine details.
- `docs/32_evaluation_protocol.md` states the protocol: the benchmarks, the
  decoding, the arithmetic that fixes k per suite, the definition of the boxed
  rate and what the evaluation record holds. The README points at it.
- `tests/test_rebuild_final_eval.py`, covering the grouping by sample count, the
  generated per-group task file, avg@k, the boxed rate and its denominator, the
  AIME merge on parts of unequal size, and the suites of the new task file.

### Changed

- `configs/tasks/math.yaml` and the six workflow task files sample by `concat`,
  so every problem appears once per epoch. `merge_epoch`, the previous setting,
  oversamples the smaller source.
- Generation-time context is the transitive ancestors of a role rather than the
  topological prefix, so the two parallel solvers of the branching workflow no
  longer see each other. The chain workflows are unaffected, because prefix and
  ancestors coincide there. E1 cell meta records `context_scope=ancestors`.
- `openrlhf/trainer/ppo_trainer.py` declares `prefix_scope: "ancestors_only"` in
  its Q-critic view configuration. The declaration has no effect yet: the critic
  actor reads its own module constant, `_Q_PREFIX_SCOPE` in
  `openrlhf/trainer/ray/ppo_critic.py`, which is still `topo_prefix`, so the
  critic's view of a prefix is unchanged and SolverB's critic view still shows
  SolverA. The gap is deliberate and pinned by `tests/test_rebuild_workflows.py`
  so that it stays visible until the two constants move together.
- The forms of the E3a null arm are `empty` and `placeholder`, and `deleted`
  stays an alias of `empty`. In this code base an empty message and a deleted
  paragraph reach the downstream roles as the same prompt, so the second form is
  a fixed placeholder message that keeps the role present.
- Estimator conventions for the rebuild study are fixed in code: the noise law
  uses `mean(d^2)/2` and the per-alternative return variance, a cell with two
  alternatives reports direction agreement, p-values are rendered in three
  tiers, and the summary document has a single writer.
- `swap_space` is no longer passed in the vLLM engine arguments, because vLLM
  0.28 rejects it.
- `MATHPOOL` is an evaluation suite of the six task files the depth study reads
  (`math_a2`, `math_a3`, `math_mt4`, `math_branch`, `math_c5`, `math_c10`) and
  not of `configs/tasks/math.yaml`. The pool is what the depth study measures
  on, and both cell drivers now default to `--split MATHPOOL`; nothing trains on
  it, and the task file a reader of the paper runs should not name a file that
  exists only after the screening pass. `configs/tasks/math_a2.yaml` is new and
  exists for that last reason: the two-agent arm used to read the paper's own
  task file, which has no `MATHPOOL` suite, so the two-agent cells could not be
  measured on the pool at all. It is `math.yaml` with the pool suite appended
  and nothing else changed, including the role file. The full-suite appendix
  comparison is the same sweep at `--split MATH500 --results_root
  <root>/E1_math500all`.
- `configs/tasks/math_screen.yaml` describes its `MATHTEST` suite as the
  canonical MATH test split prepared by the manifest, and no longer as a subset
  of the `qwedsacf/competition_math` mirror, which manifest v2 dropped as the
  contaminated source it was.
- `THIRD_PARTY_NOTICES.md` attributes the MATH artifacts to
  `EleutherAI/hendrycks_math`. It still named `qwedsacf/competition_math`, the
  mirror the manifest no longer uses, in a file that states it summarizes the
  manifest exactly.
- The comments of `configs/tasks/math_eval_probe.yaml` count eight evaluation
  suites and name them, which is what the file has declared since AIME 2025 was
  added.
- `docs/30_data_sources.md` describes the candidate benchmarks as benchmarks the
  start accuracy of a frozen policy is measured on before any of them is
  adopted. The previous wording asserted that the current benchmarks have no
  headroom left, which is one of the things the probe is there to measure.
- The manifest pins the `EleutherAI/hendrycks_math` and `weitianwen/cmath`
  revisions, and OlympiadBench ships `multi_answer_policy: drop` and
  `unit_policy: drop`, so its multi-answer and unit rows are not prepared.
- The probe task declares eight evaluation suites, AIME 2025 among them, and the
  manifest test expects pinned revisions.
- The E3a answer extractor counts an answer only where the downstream output
  states one: a `#### ...` line, a `\boxed{...}`, or a `Final answer: ...` style
  anchor. The repository parser's last resort is the last non-empty line, which
  is a sentence of the working and is all but unique to its replay, so accepting
  it as an answer symbol manufactured influence that no answer carried: it stood
  for about a third of the replays and about six tenths of the mean influence on
  a real run. Such a replay now contributes `<NONE>`, which is a symbol of the
  estimator rather than a dropped sample.
- `E1.c10.median_prefix_tokens` is the median number of tokens one replay
  regenerates downstream, not the length of the prefix the decision point
  conditions on. Every E1 cell is measured at the first role in topological
  order, so that prefix is the question and nothing else, the same text in every
  workflow; what the ten-agent chain costs is the downstream text each replay has
  to regenerate. The key keeps its name until the manifest renames it.
- The evaluation suites of `configs/tasks/math.yaml` and of the five workflow
  task files are the five suites of the main table (MATH500, Minerva-Math,
  AMC23, AIME24, AIME25); the workflow files keep `MATHPOOL` as before. GSM8K-test
  and CMATH-test left these files: they are saturation controls, reported once
  after training and never tested, so paying for 2,417 problems at every
  monitoring evaluation bought a curve nobody reads. They are evaluation suites
  of `configs/tasks/math_final_eval.yaml` instead. Minerva-Math, AMC23, AIME24
  and AIME25 are prepared by the default preparation run, which is what
  `--prepare_candidate_benchmarks` defaulting to `1` is for.
- `scripts/40_train/paper_train.sh` generates up to 2048 tokens rather than 512,
  passes `--eval_temperature 0.7` so the evaluation samples the way the rollouts
  do, and evaluates every ten percent of the run at four samples per problem
  (`--eval_every_ratio 0.10`, `--eval_n_samples_per_prompt 4`). At 512 tokens the
  harder suites were measuring truncation: not one AIME problem reached an
  answer inside the cap. The evaluation reads top-p and top-k off the rollout
  flags, which are unchanged at 0.8 and 20.

### Fixed

- Evaluation suites with different prepared columns are aligned before they are
  concatenated (`c3.integration.task_datasets.concatenate_datasets_aligned`: a
  missing column is filled with None, a column whose type differs across suites
  is cast to string). The trainer used to fall back to the first suite alone when
  `concatenate_datasets` raised, so a task that declared eight suites was
  evaluated on one without an error. There is no fallback any more; a failure
  raises. The training-side mixer goes through the same alignment.
- Hybrid-thinking chat templates (Qwen3-8B) are rendered with
  `enable_thinking=False`; every C3 role speaks in plain action messages. The
  switch is passed only when the template has it, so Qwen3-4B-Instruct-2507
  prompts are unchanged. `C3_ENABLE_THINKING=1` turns thinking back on.
- `prepare_math.py`: the CMATH answer column is `golden` upstream, and the
  prepared CMATH files used to carry an empty answer in every row while passing
  the row-count and sha checks. The write path now refuses a row without an
  input or an answer, the strict pass verifies existing files row by row, a bare
  `\boxed 2` (no braces) is extracted, a row whose solution ends in an empty
  `\boxed{}` is dropped and counted (MATH-train is 7,498 rows), and a re-pinning
  run reports a changed sha instead of failing on it.
- Data manifest v2: `MATH-train` is the canonical train split (the 7,500 problem
  train split, assembled from the seven subject configs of
  `EleutherAI/hendrycks_math`), `CMATH-test` is the real test split, and an
  overlap gate was added. Both artifacts were wrong in 0.2.0 and both errors were
  silent. `MATH-train` was pinned to a mirror whose only split is named `train`
  but holds the MATH train and test sets together, 12,500 rows, so the training
  file contained all 500 MATH500 problems. `CMATH-test` was pinned to a revision
  that has no test split, and the loader fell back to the only data file in the
  repository, so the CMATH evaluation file was a byte-for-byte copy of the CMATH
  training file. The loader now fails instead of substituting a split,
  `CMATH-train` is the validation split minus the problems it shares with the test
  split, and `scripts/10_data/check_overlap.py` fails the preparation when any
  evaluation problem occurs in any training file. See `docs/30_data_sources.md`.
- `MBPP-train` is written minus the problems it shares with `MBPP-test`. Upstream
  MBPP (`full` config, revision `4bb6404`) ships three problems in both splits,
  so the training artifact as shipped failed the overlap gate against
  `MBPP-test` and `MBPP+`. `prepare_code.py` now prepares `MBPP-test` first and
  subtracts its problems from `MBPP-train` by the same normalized problem text
  the gate compares, which leaves 371 rows. The `MBPP+` entry keeps
  `evalplus@0.3.1`.
- Three test files carried a maintainer's local interpreter path in the "run it
  like this" line of their docstring (`tests/test_bucket_guard.py`,
  `tests/test_rebuild_bias_coupling.py`, `tests/test_rebuild_summary.py`). The
  line now reads `python -m pytest tests/<file> -q`. The paths failed
  `scripts/90_audit/scan_paths.py`, so `pre_release.sh` and the release-audit
  job stopped at their first step.
- `c3.analysis.buckets.validate_bucket` has its own context-key collision guard.
  It and `ReplayRunner.build_context_hash` used to observe the same `ctx_hash`
  on one process-global guard with fingerprint strings that differ by
  construction, so the first bucket ever written by `build-buckets` was reported
  as a context-key collision. Both checks still fail fast on a real collision.
- `aggregate_e3a` now looks for the second null-action arm under `placeholder`,
  the directory name the cell driver writes, and falls back to the old name
  `deleted` with one line on stderr. With neither on disk the message names both,
  so the operator is not sent looking for a directory that was renamed.
- The E3a answer extractor is the repository's own math parser
  (`c3.envs.math.parsing.parse_math_answer` followed by
  `c3.envs.math.backends.marft.normalize.normalize_expr`), reachable as
  `--extractor math` and now the default. The previous `\boxed{...}` reader was a
  placeholder that found no answer on about half the downstream outputs of a real
  run; it is kept as `--extractor boxed` for the unit tests, where a
  hand-checkable symbol is what is wanted.
- `aggregate_e1` writes repository-relative `source` paths again. It shapes its
  joined cell paths through `summary.source_path`, the same function the other
  aggregation family uses, so aggregating a tree by absolute path no longer puts
  the operator's own machine path into summary.json.

### Prepared on 2026-09-08 for this same release

A maintenance release focused on one thing: making the repository installable,
runnable and verifiable by someone who is not us, without a GPU. It fixes the
broken quickstart reported in
[issue #1](https://github.com/EIT-EAST-Lab/C3/issues/1). No algorithmic behavior
changed: no numerical logic, defaults, sampling, rewards, credit computation,
config values, prompts or role definitions were touched.

#### Added

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
- `tests/test_credit_mechanism.py`, a CPU-runnable test of the paper's
  mechanism. It builds a rollout tree by hand on the paper's fanout and checks
  the LOO and full-mean advantages against values derived independently, in 13
  cases. It was verified to fail when the LOO baseline is broken.
- `CHANGELOG.md`, this file.

#### Changed

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

#### Fixed

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

#### Deprecated

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
