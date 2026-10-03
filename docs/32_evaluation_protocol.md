# Evaluation Protocol

How a trained model is measured: which benchmarks, with what decoding, how many
samples per problem, and what the evaluation record holds. Every number below is
derived, not chosen by taste, and the derivation is in this document.

Sections 1 to 6 are the protocol of the long-generation run, Table 7 of the
paper. The 5-seed runs of Tables 2 and 8 are evaluated more simply, greedily
under a 512-token generation limit; section 7 says how.

Two evaluations exist and they are not the same thing:

| | During training | After training |
|---|---|---|
| Task file | [configs/tasks/math.yaml](../configs/tasks/math.yaml) | [configs/tasks/math_final_eval.yaml](../configs/tasks/math_final_eval.yaml) |
| When | every 25 percent of the run | after training, on the finished model |
| Suites | the five files of the four suites of Table 7 | those five plus the two saturation controls |
| Samples per problem | four, for every suite | per suite, see below |
| What it is for | the monitoring curve | the numbers of Table 7 |
| Driver | [scripts/40_train/paper_train.sh](../scripts/40_train/paper_train.sh) | [scripts/70_rebuild/final_eval.py](../scripts/70_rebuild/final_eval.py) |

## 1. Benchmarks

Table 7 reports four suites:

| Suite | Problems | File |
|---|---|---|
| MATH500 | 500 | `data/MATH500/test.jsonl` |
| Minerva-Math | 272 | `data/MINERVA_MATH/test.jsonl` |
| AMC23 | 40 | `data/AMC23/test.jsonl` |
| AIME | 60 | `data/AIME24/test.jsonl` and `data/AIME25/test.jsonl` |

AIME is two files and one suite: AIME 2024 and AIME 2025 are 30 problems each,
and 30 problems cannot separate anything. They are merged problem by problem, so
the AIME accuracy is the mean over 60 problem scores, which is not the mean of
the two file accuracies unless the two files have equally many problems. The
merge happens in the evaluation record, not in the data, so both files stay
visible and either can still be read on its own.

No significance test is run on the long-generation run. The paired tests of
the paper (Table 10) are over the 5-seed run; see section 7.

GSM8K-test (1,319 problems) and CMATH-test (1,098) are saturation controls in
this run: Table 7 reports them beside the four suites, read the same way. A
benchmark a model already solves can only show that nothing broke.

OlympiadBench is not evaluated. Under the non-thinking decoding of the 4B policy
its accuracy was 2, 4 and 8 percent at generation caps of 512, 2048 and 4096
tokens, and nine tenths of its answers never reached a boxed expression. That
measures formatting, not mathematics.

## 2. Decoding

Temperature 0.7, top-p 0.8, top-k 20. This is the setting Qwen3 documents for
its own non-thinking mode, and it is the setting the training rollouts already
use, so the evaluation samples from the same distribution training optimized.

Generation cap 2048 tokens, in training and in evaluation alike, so that
Minerva-Math, AMC23 and AIME are read with room for a full solution.

Every suite is read this way, the two saturation controls included: one
estimator, one decoding, across the whole protocol.

## 3. Samples per problem: where k comes from

The estimator is avg@k. A problem is sampled k times, its score is the mean of
those k rewards, and the accuracy of a suite is the mean of its problem scores.

The sampling standard error of one evaluation is about

    sqrt( mean p (1 - p) / (N k) )

with N the problems of the suite and p its accuracy. The requirement is that
this noise stays under about 1.5 percentage points per training seed: the
seed-to-seed standard deviation of MATH500 in the 5-seed run of the paper
(Table 2) is at most 0.9 points, and an evaluation whose own noise is larger than that measures the
sampler instead of the method.

Putting the measured start accuracies into that expression:

| Suite | N | p at the start | k = 1 | k = 4 | k = 16 | Chosen k |
|---|---|---|---|---|---|---|
| AIME | 60 | 0.30 | +/- 5.9 | | +/- 1.5 | 16 |
| AMC23 | 40 | 0.75 | +/- 6.9 | | +/- 1.7 | 16 |
| MATH500 | 500 | 0.89 | | +/- 0.7 | | 4 |
| Minerva-Math | 272 | 0.64 | | +/- 1.5 | | 4 |

So k is 16 for AIME and AMC23, and 4 for MATH500 and Minerva-Math. The two
saturation controls take k = 4 as well: nothing is tested on them, but reading
them with the same estimator as everything else costs one setting rather than
two, and a control read greedily could not be compared with a suite read at
avg@4.

The generation count of one model follows from the table: 4,688 for the four
suites of Table 7 (2,000 plus 1,088 plus 640 plus 960), and 9,668 more for the
two controls at k = 4 (5,276 plus 4,392), so about 14,400 in all.

The long-generation run trains one model per method, and each model is
evaluated five times, under evaluation seeds 0 to 4; Table 7 reports the mean
and standard deviation over those five evaluations. The training seed is the
unit of replication of the 5-seed runs (section 7), not of this one.

Before the training runs start, the frozen policy is evaluated twice under this
protocol with two different sampling seeds. The difference between those two
readings is the noise floor of the instrument. A suite whose floor is larger
than twice the standard error in the table above is investigated before anything
is trained.

## 4. One run has one k, so the suites are grouped

The trainer evaluates the whole concatenated evaluation set with a single
`--eval_n_samples_per_prompt`, and reshapes the rewards to (problems, k) with
that one k. Two suites at different k therefore cannot ride in one run.

[scripts/70_rebuild/final_eval.py](../scripts/70_rebuild/final_eval.py) groups
the suites by k, writes one generated task file per group that carries only that
group's suites and is otherwise
[configs/tasks/math_final_eval.yaml](../configs/tasks/math_final_eval.yaml)
unchanged, and runs one evaluation per group through the same path the
start-accuracy probe uses:
[scripts/50_eval/paper_main_results.sh](../scripts/50_eval/paper_main_results.sh)
`one` in its direct mode. With the shipped sample counts that is two runs:

    k = 4   MATH500, Minerva-Math, GSM8K-test, CMATH-test
    k = 16  AMC23, AIME24, AIME25

A group at k = 1 would be decoded greedily; no suite of this protocol asks for
one, and k = 1 remains a value the driver accepts rather than the reading of
anything reported.

```bash
# Print the two commands and stop. Needs neither a GPU nor a model.
python scripts/70_rebuild/final_eval.py --policy /models/Qwen3-4B-Instruct-2507 --dry-run

# Evaluate one finished training run and write its record.
python scripts/70_rebuild/final_eval.py \
    --policy ckpt/_runs/paper_long_C3_math_seed0/final_hf \
    --run_id paper_long_C3_math_seed0 \
    --out /abs/path/final_eval_record.json
```

The generated task files are written under the run root, in `_task_groups/`, and
are rewritten on every run: they are an artifact of the driver, not
configuration to edit. One thing in them is not a copy: `mas.roles_path` is
pinned to an absolute path, because the loader resolves a relative roles path
against the directory of the task file it is reading and the generated file does
not sit beside the task it came from.

`FINAL_EVAL_EXTRA_ARGS` is appended to the trainer flags of every group, which
is where machine details go (the attention backend where flash-attn is not
installed, the GPU counts of a one GPU host).

## 5. What the evaluation record holds

One JSON file per evaluated model, schema `c3.final_eval.record/1`, with one
entry per reported suite and the merged `AIME` entry beside its two parts. Per
suite:

| Field | What it is |
|---|---|
| `accuracy` | avg@k: the mean over problems of the mean of their k rewards |
| `k` | samples per problem of this suite |
| `n_questions` | problems the run actually covered |
| `boxed_rate` | see below |
| `wilson95` | 95 percent Wilson interval over the draws |
| `per_question` | the k rewards of every problem, in dump order |

`boxed_rate` is the share of scored generations whose answer was extracted from
a boxed expression: the evaluation dump carries the reward detail of the
answering role as `reward_info_json`, and this is the share whose
`per_role[0].pred_method` is `boxed`. The denominator is the generations that
carried a reward detail at all, so a generation whose detail is missing or
unparsable is missing data rather than a generation that failed to box. The rate
is a probe for truncation and formatting: a suite whose accuracy falls while its
boxed rate falls with it is losing to the generation cap, not to the problems.

`wilson95` is binomial over problem times sample draws. The samples of one
problem are correlated, so it is narrower than a cluster-aware interval whenever
k is larger than one, and the record says so in `wilson95_basis`. It is a
per-suite sanity band, not the interval any paper claim is made from: the claims
are paired over the five training seeds.

The record also carries, per suite, the pass1 the trainer itself logged and
whether it agrees with the accuracy computed here, so a change in the dump
format shows up as a disagreement rather than as a quiet number.

## 6. Provenance

The suites, the decoding, the sample counts and the noise arithmetic of section
3 are fixed in the frozen analysis plan of the rebuild experiments, revision 18. The
start accuracies in the table are the measurements of the start-accuracy probe,
[scripts/70_rebuild/eval_probe.py](../scripts/70_rebuild/eval_probe.py), on the
frozen policy. Data provenance for every file named here is in
[30_data_sources.md](30_data_sources.md). Every suite named here is prepared by
the default run: `--prepare_candidate_benchmarks` defaults to `1`, which is what
changed when four of the candidates became evaluation suites of the long-generation run.

## 7. The 5-seed runs (Tables 2, 8 and 10)

The 5-seed runs train each method under training seeds 0 to 4, on the 2-agent
workflow (Table 2) and on the 3-agent workflow (Table 8). Each trained model is
evaluated once, greedily (k = 1), under a 512-token generation limit, the limit
it was trained under, on four suites: MATH500, AIME 2025, CMATH-test and
GSM8K-test. The average the tables print is the mean of MATH500, CMATH and
GSM8K; AIME 2025 (30 problems) is reported beside it.

[c3/analysis/rebuild/aggregate_5seed.py](../c3/analysis/rebuild/aggregate_5seed.py)
reads the records and writes each cell as the mean and standard deviation over
the training seeds. For the 2-agent workflow it also writes Table 10: C3 minus
the baseline with the higher MATH500 mean, paired by training seed, a two-sided
paired t test with its 95 percent interval, and Holm's correction over the four
suites. The training seed is the unit of replication. The commands are in
[scripts/70_rebuild/README.md](../scripts/70_rebuild/README.md).

The evaluation that runs during training is a different thing: every 5 percent
of the run in the 5-seed recipes and every 25 percent in the long-generation
recipe, and it feeds the monitoring curve only.
