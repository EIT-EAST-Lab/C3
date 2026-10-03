# c3/analysis/rebuild/aggregate_e2.py
"""
E2 aggregation: scan the training arms, write 20_data/results/E2/summary.json.

    python -m c3.analysis.rebuild.aggregate_e2 \
        --results 20_data/results/E2 --manifest results/manifest.json --out summary.json

Directory layout (the results layout, section 2, in the shape the platform
writes since 2026-09-19):

    <results>/<arm>/<method>/train_s<k>/eval_s<j>/final_eval_record.json
    <results>/<arm>/<method>/train_s<k>/{run_config.json, compute.json, budget_ledger.jsonl,
                                         train.log, eval_during_train.jsonl.metrics.jsonl}

One record per evaluation run, written by `scripts/70_rebuild/final_eval.py`
(schema `c3.final_eval.record/1`). Its `suites` map carries `accuracy` as a
fraction of one, the avg@k over the problems of that suite. Every manifest key
this script writes is a percentage or a difference of percentages, so accuracy
is multiplied by 100 once, here, on the way in.

A damaged record stops the run
------------------------------
An evaluation seed with no record in it yet is the normal state of a tree the
platform is still filling, and it is passed over. A record file that is there
and cannot be trusted is not: by default this script writes nothing at all and
exits `EXIT_BROKEN_RECORDS`, naming every such file and why. Untrusted means it
does not parse, its schema is not `RECORD_SCHEMA`, it lacks one of
`REQUIRED_RECORD_FIELDS`, it says `ok` is not true, it carries no usable suite
accuracy, or its size is more than `RECORD_SIZE_TOLERANCE` of the median away
from the records beside it.

The rule exists because of what happened on 2026-09-22: a transfer stopped and
left one of MAPPO's five records at a quarter of its size, and the aggregation
reported the mean of the other four, with `n` at 4 and a note that still said
five evaluation runs. One line on stderr said the file had been ignored and the
exit code was still zero, which is a warning that has to be read by a person to
work. `--allow-broken-records` asks for the old behaviour on purpose, and then
every key of the affected arm carries `BROKEN_SKIPPED_NOTE`: how many records
were damaged and which, beside the `n` that says how many were read.

The replication unit
--------------------
The number of training seeds an arm has decides how the arm is read, and the
choice is explicit rather than inferred from the data's shape:

- one training seed (`interim`): the five numbers behind a mean are the five evaluation runs of
  the one model. The standard deviation is therefore evaluation noise and not
  seed noise, `n` is the number of evaluation runs, and every note says so. No
  paired interval and no p value is written, because there is nothing to pair:
  an interval computed over evaluation runs of one model would read as the
  interval over training seeds that the manifest key is defined as.
- more than one training seed (`full`): each training seed is first reduced to
  the mean of its evaluation runs, and the mean, the standard deviation and the
  paired tests are taken over the training seeds. `n` is the number of training
  seeds. The paired reading follows `30_analysis/local/paired_n5_local.py`: the
  two-sided paired t test of `scipy.stats.ttest_rel`, a 95 percent interval
  `mean +/- t(0.975, n-1) * sd(differences) / sqrt(n)`, and Holm over the four
  main benchmarks, with `avg4` tested outside that family.

The mode is decided per arm, because an arm is one table of the paper and never
shares a row with another, and the summary records both the per-arm modes and
the single mode when every arm agrees.

The two blocks beside the keys
------------------------------
`tokens` and `curve` are top-level blocks, one entry per training run, and
neither feeds a manifest key:

- `tokens` is `c3.analysis.rebuild.e2_tokens` on the run directory: the
  training token account read off `train.log` and `budget_ledger.jsonl`, with
  the source of every quantity and the ones no file records. Its `warmup` block
  is the critic warmup stage, read off `budget_ledger_warmup.jsonl` and kept out
  of every training total, because a warmup rollout belongs to no optimizer
  step. The per-step rows are left out here (one arm has a few hundred steps);
  the standalone script keeps them. Two of its numbers do reach a manifest key,
  the two named below; the rest of the block is reported and nothing more.
- `curve` is `c3.analysis.rebuild.e2_curves` on the same directory: the
  evaluation points the trainer wrote while it trained (`--eval_every_ratio
  0.25`, four per run). Preregistration revision 18 (d) keeps them out of every
  reported number, so they are carried for plotting and nothing else. Each
  block also carries `bench_of_suite`, the part of the curve's suite names that
  the manifest's bench segments cover, so a plot can be labelled the way the
  tables are. A run whose metrics file never reached the results tree still has
  its points, read off `train.log` by the same reader; the block's `source`
  says which of the two files they came from.

A method of an arm that produced no evaluation record is named in
`extras.methods_without_records`, with the two states told apart: no directory
under the arm at all, or a training run directory carrying the training
artefacts and no record beside them, which is the state of an arm between the
end of its training and the end of its evaluation. No key is written for it
either way, and neither state is an error.

What this script does not write
-------------------------------
- Verdict keys (`E2.*.paired`, `E2.a2_code.margin`). Filling a verdict is the
  maintainers' action; `summary.key_refusal` refuses the unit mechanically.
- In interim mode, `*.ci_lo`, `*.ci_hi`, `*.p` and `*.p_holm`. See above.
- `E2.depth_cost.*` and `E2.ablation.*`, which need arms this round does not
  have.
- Every compute key but `E2.<arm>.compute.<method>.train_tokens_M` and
  `.wall_clock_h`. Those two used to be left empty because which quantity of the
  `tokens` block they name was a decision about the paper; the decision is now
  on record, so this script writes
  them: `train_tokens_M` is `totals.total_tokens_M`, prompt plus generated,
  which is the reading the paper's compute table reports and therefore
  comparable across rounds, and `wall_clock_h` is the ledger span summed over
  the run's contiguous training segments, so a run stopped and resumed on
  another container does not bill the stop as training
  (`e2_tokens.ledger_segments`). Each is
  written from the run it was read off, under that run's own arm and method, so
  a three-agent key is never filled with a two-agent number. `gen_tokens_M`,
  `cont_tokens_M` and `eval_tokens_M` stay empty for the earlier reason, which
  the decision did not touch: they split the account by position in the rollout
  tree and by stage, and no file of a run records that split
  (`e2_tokens.STRUCTURAL_GAPS`). The `*_saving_pct` keys are a ratio between two
  methods and wait until both of them have an account.
- Any key the manifest does not define. The manifest is the single source of
  truth for which keys exist, and the set differs by benchmark: the four main
  ones carry a paired reading while the two appendix controls carry a mean, a
  standard deviation and a plain difference. Rather than keep a second table of
  that here, every write asks the manifest first; the names it turned down are
  counted in `extras`.

The exit code stays 0 through all of this, so a partly collected tree still
produces the keys it supports.

Only numpy, scipy and the standard library are imported (through this package).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
from collections import OrderedDict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from scipy import stats

from . import e2_curves, e2_tokens
from .aggregate_e1 import SummaryBuilder, build_parser, source_path
from .summary import format_p, load_manifest

__all__ = [
    "ARMS",
    "METHODS",
    "METHOD_NAMES",
    "SUPPLEMENTARY_METHOD_RE",
    "supplementary_method",
    "known_method",
    "collect_supplementary",
    "STRONGEST_BASELINE_CANDIDATES",
    "BENCH_SUITES",
    "BENCH_OF_SUITE",
    "AVG4_PARTS",
    "HOLM_FAMILY",
    "AIME_MERGED_QUESTIONS",
    "INTERIM_NOTE",
    "REQUIRED_RECORD_FIELDS",
    "RECORD_SIZE_TOLERANCE",
    "RECORD_SIZE_MIN_GROUP",
    "EXIT_BROKEN_RECORDS",
    "BrokenRecordError",
    "EvalRun",
    "scan_e2_records",
    "accuracy_table",
    "holm",
    "paired_stats",
    "collect_run_extras",
    "collect_missing_methods",
    "collect_tokens",
    "collect_curves",
    "write_compute_keys",
    "build_e2_summary",
    "main",
]

# ---------------------------------------------------------------------------
# vocabulary
# ---------------------------------------------------------------------------

#: Arm directory names, which are also the arm segment of every key.
ARMS = ("a2_4b", "a3_4b", "a2_m2", "a3_m2", "a2_code")

#: Method directory names, which are also the method segment of every key.
METHODS = ("sft", "ppo1a", "mappo", "magrpo", "ccpo", "c3")

#: How the manifest spells a method when a key's value is a name.
METHOD_NAMES = {
    "sft": "SFT start",
    "ppo1a": "single-agent PPO",
    "mappo": "MAPPO",
    "magrpo": "MAGRPO",
    "ccpo": "CCPO",
    "c3": "C3",
}

#: The baselines `E2.<arm>.strongest_baseline.name` ranges over (its manifest note).
#: The preregistered methods and only those: a supplementary checkpoint row (below)
#: is never a candidate, however it scores.
STRONGEST_BASELINE_CANDIDATES = ("mappo", "magrpo", "ccpo")

#: A results tree may also carry a SUPPLEMENTARY method: not one of the
#: preregistered names above but a particular CHECKPOINT of one of them, exported
#: under its own method name so that it travels through this same aggregation
#: rather than through a special case (scripts/70_rebuild/export_tag_hf.py
#: writes `magrpo_s250`, MAGRPO's step-250 save, exactly that way; preregistration
#: revision 25, 2026-09-22).
#:
#: The name is `<preregistered method>_s<step>` and nothing else, so a mistyped
#: method directory is still reported as one rather than quietly read as a row.
#: What such a row gets and what it does not:
#:   * it gets the accuracy cells its own manifest keys define, whose notes say
#:     they deviate from the preregistration;
#:   * it is in NO comparison. The strongest baseline, `c3_minus_best`,
#:     `c3_minus_mappo` and the two decomposition gains range over the
#:     preregistered methods only, because a test against a checkpoint chosen on
#:     the monitoring evaluation is not the preregistered test;
#:   * it gets no compute key, because the tokens and the wall clock belong to
#:     the training run it is a checkpoint OF and are already keyed there.
SUPPLEMENTARY_METHOD_RE = re.compile(r"^(?P<base>[a-z0-9]+)_s(?P<step>[0-9]+)$")

#: Bench segment of a key -> the suite of the evaluation record it reads.
#: The left column is manifest_skeleton.py, the right column is final_eval.py.
#: Preregistration revision 18 (2026-09-15) made the main table MATH500,
#: Minerva-Math, AMC23 and AIME, the last being the 60 problems of 2024 and 2025
#: merged, which is why `aime` reads the record's merged entry rather than a
#: single year. GSM8K and CMATH stayed as appendix saturation controls.
BENCH_SUITES = {
    "math500": "MATH500",
    "minerva": "Minerva-Math",
    "amc23": "AMC23",
    "aime": "AIME",
    "gsm8k": "GSM8K-test",
    "cmath": "CMATH-test",
    "mbppplus": "MBPP+",
    "mbpptest": "MBPP-test",
}

#: The same table read the other way, for the monitoring curve: the suite name
#: the trainer stamps on a datasource -> the bench segment of a manifest key.
#: The curve's own AIME24 and AIME25 are not in it, because the manifest's
#: `aime` is the merged pair and one year is a different benchmark.
BENCH_OF_SUITE = {suite: bench for bench, suite in BENCH_SUITES.items()}

#: The merged AIME entry is the two years together and nothing else. A record
#: that carries only one of them still calls the merge `AIME`, so the count is
#: checked rather than trusted.
AIME_MERGED_QUESTIONS = 60

#: `avg4` is the unweighted mean of these four, taken inside each evaluation run
#: so that the pairing survives. Unweighted: AIME's 60 problems count as much as
#: MATH500's 500, which is what the manifest key means by the mean of the four.
AVG4_PARTS = ("math500", "minerva", "amc23", "aime")

#: The four benchmarks Holm corrects across; `avg4` is tested outside the family.
HOLM_FAMILY = AVG4_PARTS

#: The note every key of an interim arm carries.
INTERIM_NOTE = "replication: 5 evaluation runs of one model"

#: The note every `avg4` key carries, because the weighting is a choice.
AVG4_NOTE = ("unweighted mean of MATH500, Minerva-Math, AMC23 and AIME, "
             "taken inside each evaluation run")

#: The note every cell of a supplementary checkpoint row carries. The manifest's
#: own skeleton sentence says the same thing in the paper's words; this one says
#: it in the summary, so a reader of summary.json alone can see which row is not
#: the preregistered one.
SUPPLEMENTARY_NOTE = ("supplementary row: %s at step %d, not the preregistered final model, "
                      "and in no comparison")

#: The two compute keys of a method, and which quantity of the `tokens` block
#: each one is. The reading is the
#: paper's compute table's: prompt plus generated tokens, so a number here and a
#: number there mean the same thing.
COMPUTE_TOKENS_NOTE = ("training stage only: prompt plus generated tokens summed over the %d "
                       "training steps from train.log")

#: Why a `Global step` line of the log is not a step of the account. The three
#: reasons are counted apart because they mean different things, and the note
#: used to give one sentence for all of them: `%d further step line(s) of the
#: log are not in the total, because they carry no length or could not be read`.
#: That sentence reported the MAGRPO arm's four repeated steps, which are what a
#: resumed run writes and cost the account nothing, as four damaged lines.
#: `read_train_log` counts the repeats
#: and the unreadable lines apart, so the note can too.
COMPUTE_DUPLICATE_NOTE = ("%d further step line(s) repeat a step the log already carried, which is "
                          "what a run resumed from a checkpoint writes, and the account keeps the "
                          "last line of each")
COMPUTE_UNPARSED_NOTE = ("%d further step line(s) could not be read and are in no step of the "
                         "account")
COMPUTE_INCOMPLETE_NOTE = ("%d step(s) of the log carry no length and are not in the total")

#: The line count and the step count disagree by something the three sentences
#: above do not explain. It cannot happen while the log block carries all four
#: of its counters, and saying so is better than printing a breakdown that does
#: not add up.
COMPUTE_UNACCOUNTED_NOTE = ("%d further step line(s) are not in the total and this script cannot "
                            "say which of the three reasons above they fall under")

#: The same sentence for the wall clock. The quantity is the sum of the ledger's
#: contiguous training segments and not the distance from its first row to its
#: last, so a run stopped and resumed on another container does not bill the
#: stop as training (`e2_tokens.ledger_segments`). The count and the hours are
#: written even when both are zero, so
#: every row of the compute table states the same reading of the same quantity.
COMPUTE_WALL_NOTE = ("ledger span summed over contiguous training segments; "
                     "%d gap(s) of %.2f h excluded")

#: What a MAPPO token account leaves out, on the MAPPO key only. That stage runs
#: full rollouts before the first optimizer step and prints `Critic warmup
#: n/total` instead of a `Global step` line, so its tokens are in no status line
#: and in no training total. The wall clock of the same arm leaves the stage out
#: too, for the same reason on the ledger side, and each manifest entry already
#: says so in its own skeleton note, so the sentence is not repeated there.
COMPUTE_WARMUP_NOTE = "the 64 critic-warmup rollouts of MAPPO are not in its ledger"

#: How the MAGRPO arm of this round was trained, on the MAGRPO key only. The
#: container it started on was reclaimed at step 300, so the run was resumed from
#: that checkpoint on a second one and its `train.log` is the two logs appended:
#: steps 301 to 304 appear twice and the token account keeps the last of each.
#: The sentence describes this one run,
#: so it moves or goes when the arm is retrained in one piece.
COMPUTE_RESUMED_NOTE = ("trained in two segments (steps 1 to 300 and 301 to 312) on two "
                        "containers, resumed from the step-300 checkpoint")

#: The record schema this script knows how to read. A later version may mean the
#: same field names differently, so it is refused rather than read hopefully.
RECORD_SCHEMA = "c3.final_eval.record/1"

#: Top-level fields a usable record carries. `schema` says which reading its
#: field names have, `ok` is the evaluation's own verdict on itself, `method` is
#: what the record says it evaluated, and `suites` holds every accuracy this
#: script reads. A file missing one of them is damaged, not old.
REQUIRED_RECORD_FIELDS = ("schema", "ok", "method", "suites")

#: How far a record may sit from the median size of the records beside it before
#: this script calls it damaged, as a fraction of that median. The five records
#: of one training run are the same seven suites over the same problems, so they
#: come out within a few hundred bytes of each other; on 2026-09-22 a transfer
#: stopped and left one of MAPPO's at 351,232 bytes beside four of about
#: 1,446,000, and it still ended in the middle of a string, so it did not parse.
#: A half is far wider than the spread of healthy records and far narrower than
#: a truncation worth catching, and the test is a ratio rather than a byte count
#: so it does not need to know how big a record of a given arm should be.
RECORD_SIZE_TOLERANCE = 0.5

#: Below this many records beside each other the size test is not run: a median
#: of one is that record itself, and a median of two is the midpoint of the pair,
#: so neither can say which of them is the odd one out. A tree that is half
#: written is the normal state while the platform is still evaluating.
RECORD_SIZE_MIN_GROUP = 3

#: What a permissive run says on every key of an arm whose records were not all
#: readable. It rides on every key of that arm rather than on the cells of the
#: one method, because a strongest baseline, a difference and a gain are all
#: computed from cells that may be short, and a sentence that has to be
#: remembered for each derived key is a sentence that will be forgotten.
BROKEN_SKIPPED_NOTE = ("%d evaluation record(s) of this arm were damaged and are in no number "
                       "here (%s); the n of each key is how many were read")

#: The exit code of a refused run. Distinct from 2, which is a bad command line.
EXIT_BROKEN_RECORDS = 3


class BrokenRecordError(RuntimeError):
    """Raised instead of aggregating over records this script cannot trust.

    Carries the list of damaged files so a caller can report each with its own
    reason. The aggregation never half-writes: the error is raised before any
    key is computed, so nothing reaches the summary or, through it, the manifest.
    """

    def __init__(self, broken: Sequence[Mapping[str, Any]]) -> None:
        self.broken: List[Dict[str, Any]] = [dict(item) for item in broken]
        super().__init__("%d damaged evaluation record(s)" % len(self.broken))

#: Files a training run may leave beside its evaluation seeds, so an operator
#: can see from `extras.runs[].files` alone which of them reached the results
#: tree. The archive step that copies a finished run into the results tree
#: copies every one of them except `compute.json`, which nothing writes yet.
RUN_FILES = ("run_config.json", "compute.json", "budget_ledger.jsonl",
             e2_tokens.WARMUP_LEDGER_NAME, e2_tokens.LOG_NAME, e2_curves.METRICS_NAME)

EXPERIMENT = "E2"
SCRIPT = "c3.analysis.rebuild.aggregate_e2"


# ---------------------------------------------------------------------------
# the method vocabulary
# ---------------------------------------------------------------------------


def supplementary_method(name: str) -> Optional[Dict[str, Any]]:
    """`{"base": ..., "checkpoint_step": ...}` for a checkpoint row, else None.

    A name is a supplementary checkpoint row only when it reads
    `<preregistered method>_s<step>`; `magrpo_s250` is one and `not_a_method`,
    `magrpo_step250` and `magrpos250` are not.
    """
    match = SUPPLEMENTARY_METHOD_RE.match(str(name or ""))
    if match is None:
        return None
    base = match.group("base")
    if base not in METHODS:
        return None
    return {"base": base, "checkpoint_step": int(match.group("step"))}


def known_method(name: str) -> bool:
    """Whether a directory name is a method this aggregation reads at all."""
    return name in METHODS or supplementary_method(name) is not None


# ---------------------------------------------------------------------------
# scanning
# ---------------------------------------------------------------------------


class EvalRun:
    """One `final_eval_record.json`: the accuracies of one evaluation run."""

    def __init__(self, arm: str, method: str, train_seed: int, eval_seed: int,
                 path: str, accuracy_pct: Mapping[str, float],
                 n_questions: Mapping[str, int]) -> None:
        self.arm = arm
        self.method = method
        self.train_seed = int(train_seed)
        self.eval_seed = int(eval_seed)
        self.path = path
        self.accuracy_pct: Dict[str, float] = dict(accuracy_pct)
        self.n_questions: Dict[str, int] = dict(n_questions)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "EvalRun(%s/%s/train_s%d/eval_s%d)" % (
            self.arm, self.method, self.train_seed, self.eval_seed)


def _subdirs(path: str) -> List[str]:
    if not os.path.isdir(path):
        return []
    return sorted(name for name in os.listdir(path)
                  if os.path.isdir(os.path.join(path, name)))


def _seed_of(name: str, prefix: str) -> Optional[int]:
    """The integer of a `train_s3` or `eval_s0` directory name, or None."""
    if not name.startswith(prefix):
        return None
    tail = name[len(prefix):]
    if not tail.isdigit():
        return None
    return int(tail)


def _size_outliers(sizes: Mapping[str, int]) -> Dict[str, str]:
    """Paths whose byte size sits too far from the median of the group, with why.

    The group is the records of one training run, which are the same suites over
    the same problems and therefore the same size to within a fraction of a
    percent. A file that a stopped transfer left short is the one shape of
    damage that can still parse, so size is checked before the file is read.
    The median is used rather than the mean because one badly wrong value pulls
    a mean towards itself and can hide behind it.

    Groups under `RECORD_SIZE_MIN_GROUP` records are not tested; see that
    constant. A median of zero, which takes a group of empty files, is not
    tested either: every ratio against it is infinite and the emptiness is
    caught when the files fail to parse.
    """
    if len(sizes) < RECORD_SIZE_MIN_GROUP:
        return {}
    median = float(statistics.median(sizes.values()))
    if median <= 0:
        return {}
    out: Dict[str, str] = {}
    for path, size in sizes.items():
        if abs(size - median) > RECORD_SIZE_TOLERANCE * median:
            out[path] = ("is %d bytes, %.0f%% away from the %d-byte median of the %d records "
                         "of this training run, which is how a stopped transfer looks"
                         % (size, 100.0 * abs(size - median) / median, int(median), len(sizes)))
    return out


def _read_record(path: str, builder: SummaryBuilder,
                 damage: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
    """The parsed record, or None with one line on stderr.

    A file that is there but cannot be used is damage, and every reason is
    appended to `damage` for the caller to refuse on. A directory with no record
    at all is not damage: an unevaluated seed is the normal state of a tree the
    platform is still filling, and that case never reaches here.
    """
    def refuse(reason: str) -> None:
        builder.note("ignored: %s %s" % (source_path(path), reason))
        if damage is not None:
            damage.append(reason)

    try:
        with open(path, "r", encoding="utf-8") as fh:
            record = json.load(fh)
    except (OSError, ValueError) as exc:
        refuse("does not parse (%s)" % exc)
        return None
    if not isinstance(record, dict):
        refuse("is not a JSON object")
        return None
    schema = str(record.get("schema") or "")
    if schema != RECORD_SCHEMA:
        refuse("has schema %r, expected %r" % (schema, RECORD_SCHEMA))
        return None
    missing = [field for field in REQUIRED_RECORD_FIELDS if field not in record]
    if missing:
        refuse("does not carry %s" % ", ".join(missing))
        return None
    if record.get("ok") is not True:
        refuse("reports ok=%r, so the evaluation did not finish clean"
               % (record.get("ok"),))
        return None
    if not isinstance(record.get("suites"), Mapping) or not record.get("suites"):
        refuse("carries an empty or non-object suites map")
        return None
    return record


def scan_e2_records(results_root: str, builder: SummaryBuilder,
                    *, arm_filter: Optional[str] = None,
                    partial_merges: Optional[List[Dict[str, Any]]] = None,
                    broken: Optional[List[Dict[str, Any]]] = None) -> List[EvalRun]:
    """Every evaluation record under the tree, in arm, method, seed order.

    Two kinds of thing are passed over, and they are not the same kind. What the
    layout does not define is skipped with one line on stderr and nothing more:
    an arm or method directory with an unknown name, a seed directory that does
    not end in a number, an evaluation seed with no record in it yet. A tree
    that is half written is the normal state while the platform is still running.

    A record file that is there but cannot be trusted is DAMAGE, and each one is
    appended to `broken` as {path, reason, arm, method, train_seed, eval_seed}:
    it does not parse, its schema is not the one this script reads, it is
    missing a field of `REQUIRED_RECORD_FIELDS`, it says `ok` is not true, it
    carries no usable suite accuracy, or its size is far from the median of the
    records beside it (`_size_outliers`). This function still returns the runs it
    could read; what to do about the damage is the caller's, and
    `build_e2_summary` refuses by default.

    `partial_merges` collects the records whose AIME entry is not the merged 60.
    That is a different benchmark under the same name rather than a damaged
    file, so it drops the AIME cell of that record and is reported, not refused.
    """
    runs: List[EvalRun] = []
    if partial_merges is None:
        partial_merges = []
    if not os.path.isdir(results_root):
        raise ValueError("results root %s is not a directory" % results_root)

    for arm in _subdirs(results_root):
        if arm not in ARMS:
            builder.note("ignored: %s is not an arm name" % source_path(results_root, arm))
            continue
        if arm_filter and arm != arm_filter:
            continue
        for method in _subdirs(os.path.join(results_root, arm)):
            if not known_method(method):
                builder.note("ignored: %s is not a method name"
                             % source_path(results_root, arm, method))
                continue
            method_dir = os.path.join(results_root, arm, method)
            for train_name in _subdirs(method_dir):
                train_seed = _seed_of(train_name, "train_s")
                if train_seed is None:
                    builder.note("ignored: %s is not a training seed directory"
                                 % source_path(results_root, arm, method, train_name))
                    continue
                train_dir = os.path.join(method_dir, train_name)

                # The records of this training run are collected before any is
                # read, because the size test compares each with the median of
                # the others and there is no median until they are all in hand.
                found: List[Tuple[int, str]] = []
                for eval_name in _subdirs(train_dir):
                    eval_seed = _seed_of(eval_name, "eval_s")
                    if eval_seed is None:
                        builder.note("ignored: %s is not an evaluation seed directory"
                                     % source_path(results_root, arm, method, train_name, eval_name))
                        continue
                    path = os.path.join(train_dir, eval_name, "final_eval_record.json")
                    if not os.path.isfile(path):
                        builder.note("ignored: no final_eval_record.json in %s"
                                     % source_path(results_root, arm, method, train_name, eval_name))
                        continue
                    found.append((eval_seed, path))

                sizes: Dict[str, int] = {}
                for _seed, path in found:
                    try:
                        sizes[path] = os.path.getsize(path)
                    except OSError:
                        pass
                outliers = _size_outliers(sizes)

                for eval_seed, path in found:
                    def hurt(reason: str, _seed: int = eval_seed, _path: str = path) -> None:
                        if broken is not None:
                            broken.append({"path": source_path(_path), "reason": reason,
                                           "arm": arm, "method": method,
                                           "train_seed": train_seed, "eval_seed": _seed})

                    if path in outliers:
                        builder.note("ignored: %s %s" % (source_path(path), outliers[path]))
                        hurt(outliers[path])
                        continue
                    damage: List[str] = []
                    record = _read_record(path, builder, damage)
                    if record is None:
                        for reason in damage:
                            hurt(reason)
                        continue
                    accuracy, counts = _accuracies_of(record, path, builder, partial_merges)
                    if not accuracy:
                        builder.note("ignored: %s carries no suite accuracy" % source_path(path))
                        hurt("carries no suite accuracy")
                        continue
                    runs.append(EvalRun(arm, method, train_seed, eval_seed, path, accuracy, counts))
    return runs


def _accuracies_of(record: Mapping[str, Any], path: str, builder: SummaryBuilder,
                   partial_merges: List[Dict[str, Any]]) -> Tuple[Dict[str, float], Dict[str, int]]:
    """Bench segment -> accuracy in percent, plus the problem count behind each.

    `avg4` is formed here, inside the run, so that a later pairing compares the
    same four benchmarks of the same run, and it is the unweighted mean of them.

    The `aime` bench is the record's merged entry, the 60 problems of 2024 and
    2025 together. `final_eval.py` merges whichever parts of that pair the run
    produced, so a run that lost one year still calls its 30 problems `AIME`.
    That is a different benchmark under the same name, so the count is checked
    and a short merge is dropped with a line on stderr rather than reported.
    """
    suites = record.get("suites")
    if not isinstance(suites, Mapping):
        return {}, {}

    out: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for bench, suite in BENCH_SUITES.items():
        entry = suites.get(suite)
        if not isinstance(entry, Mapping):
            continue
        value = entry.get("accuracy")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        if not math.isfinite(float(value)):
            continue
        n_questions = entry.get("n_questions")
        if bench == "aime" and n_questions != AIME_MERGED_QUESTIONS:
            builder.note("ignored: %s has AIME over %r problems, not the %d of the merged "
                         "2024 and 2025 set" % (source_path(path), n_questions, AIME_MERGED_QUESTIONS))
            partial_merges.append({"path": source_path(path), "n_questions": n_questions})
            continue
        out[bench] = float(value) * 100.0
        if isinstance(n_questions, int) and not isinstance(n_questions, bool):
            counts[bench] = n_questions

    if all(part in out for part in AVG4_PARTS):
        out["avg4"] = sum(out[part] for part in AVG4_PARTS) / len(AVG4_PARTS)
    return out, counts


def accuracy_table(runs: Sequence[EvalRun]) -> Dict[Tuple[str, str, str], Dict[int, Dict[int, float]]]:
    """(arm, method, bench) -> {training seed: {evaluation seed: percent}}."""
    table: Dict[Tuple[str, str, str], Dict[int, Dict[int, float]]] = {}
    for run in runs:
        for bench, value in run.accuracy_pct.items():
            cell = table.setdefault((run.arm, run.method, bench), {})
            cell.setdefault(run.train_seed, {})[run.eval_seed] = value
    return table


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------


def _mean(values: Sequence[float]) -> Optional[float]:
    return (sum(values) / len(values)) if values else None


def _sample_std(values: Sequence[float]) -> Optional[float]:
    """Standard deviation with one degree of freedom taken off, or None under two values."""
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def holm(p_values: Sequence[float]) -> List[float]:
    """Holm-corrected p values, the step-down form `30_analysis/local/paired_n5_local.py` uses.

    Sorted ascending, each multiplied by the number of tests still standing, and
    carried forward as a running maximum so the corrected values keep the order
    of the raw ones.
    """
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    total = len(p_values)
    adjusted = [0.0] * total
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, p_values[index] * (total - rank))
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def paired_stats(left: Sequence[float], right: Sequence[float]) -> Dict[str, Any]:
    """The paired reading of `paired_n5_local.py`: difference, 95 percent interval, p.

    `left` and `right` are the per-unit values in the same order, which for a
    full tree means one value per training seed. The interval is the t interval
    of the mean difference and the p value is the two-sided paired t test. A
    single pair, or differences with no spread at all, leave the interval and
    the p value undefined rather than inventing one.
    """
    n = len(left)
    if n != len(right) or n == 0:
        return {"n": 0, "diff": None, "ci_lo": None, "ci_hi": None, "p": None}

    differences = [a - b for a, b in zip(left, right)]
    diff = sum(differences) / n
    out: Dict[str, Any] = {"n": n, "diff": diff, "ci_lo": None, "ci_hi": None, "p": None}
    if n < 2:
        return out

    spread = _sample_std(differences)
    if spread is None or not math.isfinite(spread) or spread <= 0.0:
        return out

    half = float(stats.t.ppf(0.975, n - 1)) * spread / math.sqrt(n)
    out["ci_lo"] = diff - half
    out["ci_hi"] = diff + half
    p_value = float(stats.ttest_rel(list(left), list(right)).pvalue)
    if math.isfinite(p_value):
        out["p"] = p_value
    return out


# ---------------------------------------------------------------------------
# reading one cell under the mode of its arm
# ---------------------------------------------------------------------------


class Series:
    """The numbers one key rests on, already reduced to the arm's unit."""

    def __init__(self, values: Mapping[int, float], sources: Sequence[str], mode: str) -> None:
        self.by_unit: "OrderedDict[int, float]" = OrderedDict(sorted(values.items()))
        self.sources = list(sources)
        self.mode = mode

    @property
    def values(self) -> List[float]:
        return list(self.by_unit.values())

    @property
    def n(self) -> int:
        return len(self.by_unit)

    @property
    def mean(self) -> Optional[float]:
        return _mean(self.values)

    @property
    def std(self) -> Optional[float]:
        return _sample_std(self.values)


def arm_mode(table: Mapping[Tuple[str, str, str], Mapping[int, Mapping[int, float]]], arm: str) -> str:
    """`interim` while the arm has one training seed, `full` once it has more."""
    seeds = set()
    for (cell_arm, _method, _bench), by_seed in table.items():
        if cell_arm == arm:
            seeds.update(by_seed)
    return "full" if len(seeds) > 1 else "interim"


def series_of(table: Mapping[Tuple[str, str, str], Mapping[int, Mapping[int, float]]],
              runs: Sequence[EvalRun], arm: str, method: str, bench: str, mode: str) -> Optional[Series]:
    """The values behind one (arm, method, bench) cell, reduced to the arm's unit.

    Interim: one value per evaluation run, keyed by the evaluation seed.
    Full: one value per training seed, each the mean of that seed's evaluation runs.
    """
    by_seed = table.get((arm, method, bench))
    if not by_seed:
        return None

    sources = [source_path(run.path) for run in runs
               if run.arm == arm and run.method == method and bench in run.accuracy_pct]

    if mode == "interim":
        train_seed = sorted(by_seed)[0]
        return Series(dict(by_seed[train_seed]), sources, mode)

    per_seed = {}
    for train_seed, by_eval in by_seed.items():
        mean = _mean(list(by_eval.values()))
        if mean is not None:
            per_seed[train_seed] = mean
    if not per_seed:
        return None
    return Series(per_seed, sources, mode)


# ---------------------------------------------------------------------------
# compute extras
# ---------------------------------------------------------------------------


def collect_run_extras(results_root: str, runs: Sequence[EvalRun]) -> Dict[str, Any]:
    """What each training run left beside its evaluation seeds.

    A file list and the ledger's own totals, reported and never written to a
    manifest key. The two compute keys a run does fill are written from the
    `tokens` block (`write_compute_keys`), which reads the same ledger through
    `e2_tokens` and says where each number came from; what is here is the
    inventory of the run directory, which no key carries.
    """
    extras: Dict[str, Any] = {}
    for arm, method, train_seed in _training_runs(runs):
        train_dir = os.path.join(results_root, arm, method, "train_s%d" % train_seed)
        entry: Dict[str, Any] = {"files": [], "path": source_path(train_dir)}
        for name in RUN_FILES:
            path = os.path.join(train_dir, name)
            if os.path.isfile(path):
                entry["files"].append(name)
        if "compute.json" in entry["files"]:
            try:
                with open(os.path.join(train_dir, "compute.json"), "r", encoding="utf-8") as fh:
                    loaded = json.load(fh)
                if isinstance(loaded, dict):
                    entry["compute"] = loaded
            except (OSError, ValueError) as exc:
                entry["compute_error"] = str(exc)
        if "budget_ledger.jsonl" in entry["files"]:
            entry["budget_ledger"] = _ledger_totals(os.path.join(train_dir, "budget_ledger.jsonl"))
        extras[_run_label(arm, method, train_seed)] = entry
    return extras


def collect_missing_methods(results_root: str, runs: Sequence[EvalRun]) -> Dict[str, Any]:
    """Per arm found in the tree, the methods that produced no evaluation record.

    A method is missing in one of two ways, and they are two states of the round
    rather than two errors: it has no directory under the arm at all, or it has
    a training run directory carrying the training artefacts and no readable
    `final_eval_record.json`, which is what an arm looks like between the end of
    its training and the end of its evaluation.

    Neither fills a key and neither is fatal. The block exists so that a key the
    summary does not carry can be told apart from a key nobody asked for: the
    manifest defines a cell for every method of `METHODS`, and without this the
    tree's silence about one of them looks the same as a tree that was never
    scanned.
    """
    have = {(run.arm, run.method) for run in runs}
    out: Dict[str, Any] = {}
    for arm in _subdirs(results_root):
        if arm not in ARMS:
            continue
        arm_dir = os.path.join(results_root, arm)
        in_tree = set(_subdirs(arm_dir))
        missing: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        for method in METHODS:
            if (arm, method) in have:
                continue
            if method not in in_tree:
                missing[method] = {"why": "no directory under the arm", "train_dirs": []}
                continue
            train_dirs: List[Dict[str, Any]] = []
            for train_name in _subdirs(os.path.join(arm_dir, method)):
                if _seed_of(train_name, "train_s") is None:
                    continue
                train_dir = os.path.join(arm_dir, method, train_name)
                train_dirs.append({
                    "path": source_path(train_dir),
                    "files": [name for name in RUN_FILES
                              if os.path.isfile(os.path.join(train_dir, name))],
                    "n_eval_dirs": len([name for name in _subdirs(train_dir)
                                        if _seed_of(name, "eval_s") is not None]),
                })
            missing[method] = {
                "why": ("training run directory with no evaluation record" if train_dirs
                        else "no training seed directory under the method"),
                "train_dirs": train_dirs,
            }
        if missing:
            out[arm] = missing
    return out


def collect_supplementary(results_root: str, runs: Sequence[EvalRun]) -> Dict[str, Any]:
    """Per arm, the supplementary checkpoint rows the tree carries.

    A reported block and not a key, because what it says is that a row is NOT
    the preregistered one, and that has to be visible to whoever reads the
    summary without their having to re-derive the naming rule. Every entry
    carries `supplementary: true` and `in_comparisons: false`, so a consumer
    tests one field.

    The entry also carries whatever the run's own `run_config.json` says about
    where the checkpoint came from, which is where the deviation is recorded on
    the machine that exported it.
    """
    out: Dict[str, Any] = {}
    for arm, method, train_seed in _training_runs(runs):
        info = supplementary_method(method)
        if info is None:
            continue
        train_dir = os.path.join(results_root, arm, method, "train_s%d" % train_seed)
        entry: Dict[str, Any] = {
            "supplementary": True,
            "in_comparisons": False,
            "base_method": info["base"],
            "checkpoint_step": info["checkpoint_step"],
            "path": source_path(train_dir),
            "n_records": len([run for run in runs
                              if run.arm == arm and run.method == method]),
        }
        config = os.path.join(train_dir, "run_config.json")
        if os.path.isfile(config):
            try:
                with open(config, "r", encoding="utf-8") as fh:
                    loaded = json.load(fh)
            except (OSError, ValueError) as exc:
                entry["run_config_error"] = str(exc)
            else:
                if isinstance(loaded, dict):
                    entry["run_config"] = {
                        field: loaded[field] for field in
                        ("method", "alg", "checkpoint_tag", "checkpoint_step",
                         "derived_from", "note")
                        if field in loaded}
        out.setdefault(arm, OrderedDict())[method] = entry
    return out


def _training_runs(runs: Sequence[EvalRun]) -> List[Tuple[str, str, int]]:
    """(arm, method, training seed) of every run the records came from, sorted."""
    return sorted({(run.arm, run.method, run.train_seed) for run in runs})


def _run_label(arm: str, method: str, train_seed: int) -> str:
    return "%s/%s/train_s%d" % (arm, method, train_seed)


def collect_tokens(results_root: str, runs: Sequence[EvalRun]) -> Dict[str, Any]:
    """The training token account of each run, from `e2_tokens`.

    One entry per training run, the per-step rows left out. Two numbers of each
    entry reach a manifest key through `write_compute_keys`; the rest of the
    block is reported so that a reader can see which files a number came from
    and which quantities no file of a run holds.
    """
    out: Dict[str, Any] = {}
    for arm, method, train_seed in _training_runs(runs):
        train_dir = os.path.join(results_root, arm, method, "train_s%d" % train_seed)
        out[_run_label(arm, method, train_seed)] = e2_tokens.read_run(
            train_dir, include_steps=False)
    return out


def collect_curves(results_root: str, runs: Sequence[EvalRun]) -> Dict[str, Any]:
    """The monitoring evaluation points of each run, from `e2_curves`.

    The points are the ones the trainer wrote while it trained, four per run at
    `--eval_every_ratio 0.25`. They feed curves only (preregistration revision
    18 (d)), so no key is written from them; `bench_of_suite` is added so a plot
    can carry the same benchmark labels as the tables.
    """
    out: Dict[str, Any] = {}
    for arm, method, train_seed in _training_runs(runs):
        train_dir = os.path.join(results_root, arm, method, "train_s%d" % train_seed)
        document = e2_curves.read_run(train_dir)
        document["bench_of_suite"] = {suite: BENCH_OF_SUITE[suite]
                                      for suite in document.get("suites", [])
                                      if suite in BENCH_OF_SUITE}
        out[_run_label(arm, method, train_seed)] = document
    return out


def _round2(value: Any) -> Optional[float]:
    """Two decimals of a finite number, or None.

    The manifest prints these keys with one decimal, so the second one is the
    margin a reader can check the rounding against, and everything past it is
    the float noise of a product of means.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(float(value)):
        return None
    return round(float(value), 2)


def _log_line_clauses(log: Mapping[str, Any], totals: Mapping[str, Any]) -> List[str]:
    """What the log's `Global step` lines that are not steps of the total were.

    `read_train_log` counts every line carrying `Global step`, the ones it could
    not parse, and the ones that repeat a step it had already read; the totals
    carry the unique steps and, of those, the ones with every length field. So
    `n_lines_with_a_step` splits exactly into unreadable, repeated, and unique,
    and the unique split again into complete and short. Each is named in its own
    sentence, and nothing is said about a reason with a count of zero.

    Returns the sentences to hang on the token note, possibly empty.
    """
    lines = int(log.get("n_lines_with_a_step") or 0)
    if not lines:
        return []
    complete = int(totals.get("n_complete_steps") or 0)
    steps = int(totals.get("n_steps") or 0)
    duplicates = int(log.get("n_duplicate_steps") or 0)
    unparsed = int(log.get("n_unparsed") or 0)
    short = max(steps - complete, 0)

    clauses: List[str] = []
    for count, sentence in ((duplicates, COMPUTE_DUPLICATE_NOTE),
                            (unparsed, COMPUTE_UNPARSED_NOTE),
                            (short, COMPUTE_INCOMPLETE_NOTE)):
        if count:
            clauses.append(sentence % count)
    rest = lines - complete - duplicates - unparsed - short
    if rest:
        clauses.append(COMPUTE_UNACCOUNTED_NOTE % rest)
    return clauses


def write_compute_keys(gate: "ManifestGate", tokens: Mapping[str, Any]) -> List[str]:
    """`compute.<method>.train_tokens_M` and `.wall_clock_h` of every run in hand.

    One run fills the two keys of its own arm and method, and only those: the
    quantities are read off that run's `train.log` and `budget_ledger.jsonl`,
    which is what each key's `source` names. A run whose log carries no complete
    step, or whose ledger carries no timestamp, fills neither, because the only
    way to produce a number there would be to invent one.

    The `tokens` block has one entry per training run that produced an
    evaluation record, so a method still under evaluation is absent here and
    lands by itself once its evaluation records do, with no change to this code.

    Returns the keys written, for the caller's report.
    """
    written: List[str] = []
    for label in sorted(tokens):
        parts = str(label).split("/")
        if len(parts) != 3:
            continue
        arm, method, _train = parts
        block = tokens[label]
        if not isinstance(block, Mapping):
            continue
        totals = block.get("totals") or {}
        ledger = block.get("ledger") or {}
        log = block.get("log") or {}
        base = "E2.%s.compute.%s" % (arm, method)

        tokens_value = _round2(totals.get("total_tokens_M"))
        if tokens_value is not None:
            complete = int(totals.get("n_complete_steps") or 0)
            tail = COMPUTE_TOKENS_NOTE % complete
            for clause in _log_line_clauses(log, totals):
                tail += "; " + clause
            if method == "mappo":
                tail += "; " + COMPUTE_WARMUP_NOTE
            if method == "magrpo":
                tail += "; " + COMPUTE_RESUMED_NOTE
            if gate.add(base + ".train_tokens_M", tokens_value, n=1,
                        source=[log.get("path")] if log.get("path") else [],
                        note=tail):
                written.append(base + ".train_tokens_M")

        wall_value = _round2(ledger.get("wall_clock_h"))
        if wall_value is not None:
            gaps = int(ledger.get("n_gaps_excluded") or 0)
            gap_hours = float(ledger.get("gap_hours_excluded") or 0.0)
            if gate.add(base + ".wall_clock_h", wall_value, n=1,
                        source=[ledger.get("path")] if ledger.get("path") else [],
                        note=COMPUTE_WALL_NOTE % (gaps, gap_hours)):
                written.append(base + ".wall_clock_h")
    return written


def _ledger_totals(path: str) -> Dict[str, Any]:
    """Steps and evaluation calls of one `budget_ledger.jsonl`, or the reason there are none."""
    rows = 0
    steps = set()
    calls = 0
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(row, dict):
                    continue
                rows += 1
                step = row.get("global_step")
                if isinstance(step, int) and not isinstance(step, bool):
                    steps.add(step)
                total = row.get("total_eval_calls")
                if isinstance(total, (int, float)) and not isinstance(total, bool):
                    calls += int(total)
    except OSError as exc:
        return {"error": str(exc)}
    return {"rows": rows, "global_steps": len(steps), "total_eval_calls": calls}


# ---------------------------------------------------------------------------
# the summary
# ---------------------------------------------------------------------------


class ManifestGate:
    """Writes only the keys the manifest defines, and remembers the ones it refused.

    The manifest is the single source of truth for which keys exist, and the set
    differs by bench: the four main benchmarks carry a paired reading, while the
    two appendix controls carry a mean, a standard deviation and a plain
    difference and nothing else (preregistration revision 18). Asking the
    manifest is therefore better than a second table here that could disagree
    with it, and it keeps the expected absences off stderr, where the lines that
    matter are the ones about the result tree.
    """

    def __init__(self, builder: SummaryBuilder) -> None:
        self.builder = builder
        self.skipped: List[str] = []
        #: Appended to the note of every key written while it is set. The caller
        #: sets it around one arm, so a fact about that arm's records cannot be
        #: left off a key by forgetting to thread it through one more function.
        self.note_suffix: str = ""

    def add(self, key: str, value: Any, *, n: int, source: Any, note: str = "") -> bool:
        if key not in self.builder.manifest:
            self.skipped.append(key)
            return False
        if self.note_suffix:
            note = "; ".join(part for part in (note, self.note_suffix) if part)
        return self.builder.add(key, value, n=n, source=source, note=note)


def _note(mode: str, tail: str = "") -> str:
    """The note of one key: the interim stamp when the arm has one training seed."""
    parts = []
    if mode == "interim":
        parts.append(INTERIM_NOTE)
    if tail:
        parts.append(tail)
    return "; ".join(parts)


def _benches_present(table: Mapping[Tuple[str, str, str], Any], arm: str) -> List[str]:
    """Bench segments the arm has numbers for, in manifest order."""
    order = list(BENCH_SUITES) + ["avg4"]
    present = {bench for (cell_arm, _method, bench) in table if cell_arm == arm}
    return [bench for bench in order if bench in present]


def _methods_present(table: Mapping[Tuple[str, str, str], Any], arm: str) -> List[str]:
    """Methods the arm has numbers for: the preregistered ones in manifest order,
    then any supplementary checkpoint rows, by name."""
    present = {method for (cell_arm, method, _bench) in table if cell_arm == arm}
    preregistered = [method for method in METHODS if method in present]
    return preregistered + sorted(method for method in present if method not in METHODS)


def _broken_suffix(broken: Sequence[Mapping[str, Any]], arm: str) -> str:
    """The sentence every key of `arm` carries when its records were not all read."""
    mine = [item for item in broken if item.get("arm") == arm]
    if not mine:
        return ""
    where = ", ".join("%s/train_s%s/eval_s%s"
                      % (item.get("method"), item.get("train_seed"), item.get("eval_seed"))
                      for item in mine)
    return BROKEN_SKIPPED_NOTE % (len(mine), where)


def build_e2_summary(results_root: str, manifest: Mapping[str, Any], *,
                     stream=None, arm_filter: Optional[str] = None,
                     allow_broken_records: bool = False) -> SummaryBuilder:
    """Compute every E2 key the collected arms support.

    Refuses by default: a record file that is there and cannot be trusted raises
    `BrokenRecordError` before any key is computed, so a round that lost one
    evaluation to a stopped transfer cannot quietly report the mean of the
    others under a note that says five. `allow_broken_records=True` goes back to
    reading what is readable, and then every key of that arm says how many
    records were damaged and which (`BROKEN_SKIPPED_NOTE`), beside the n that
    says how many were read.
    """
    builder = SummaryBuilder(EXPERIMENT, SCRIPT, manifest, stream=stream)
    partial_merges: List[Dict[str, Any]] = []
    broken: List[Dict[str, Any]] = []
    runs = scan_e2_records(results_root, builder, arm_filter=arm_filter,
                           partial_merges=partial_merges, broken=broken)
    if broken and not allow_broken_records:
        raise BrokenRecordError(broken)
    table = accuracy_table(runs)

    arms = [arm for arm in ARMS if any(key[0] == arm for key in table)]
    modes = {arm: arm_mode(table, arm) for arm in arms}

    gate = ManifestGate(builder)
    for arm in arms:
        mode = modes[arm]
        gate.note_suffix = _broken_suffix(broken, arm)
        _write_arm(gate, table, runs, arm, mode)
        gate.note_suffix = ""

    distinct = sorted(set(modes.values()))
    builder.extra["results_root"] = source_path(results_root)
    builder.extra["replication"] = {
        "mode": distinct[0] if len(distinct) == 1 else ("mixed" if distinct else "none"),
        "by_arm": modes,
        "interim_note": INTERIM_NOTE,
        "rule": ("one training seed: mean and std over the evaluation runs, no paired reading; "
                 "more than one: mean over the evaluation runs of each training seed first, "
                 "then mean, std and paired reading over the training seeds"),
    }
    builder.extra["tokens"] = collect_tokens(results_root, runs)
    builder.extra["curve"] = collect_curves(results_root, runs)
    compute_keys = write_compute_keys(gate, builder.extra["tokens"])
    extras: Dict[str, Any] = {
        "runs": collect_run_extras(results_root, runs),
        "n_records": len(runs),
        "compute_keys_written": compute_keys,
    }
    if broken:
        extras["broken_records"] = broken
        builder.note("%d damaged evaluation record(s) were read past because "
                     "--allow-broken-records was given; every key of the affected arm says so "
                     "and see extras.broken_records" % len(broken))
    supplementary = collect_supplementary(results_root, runs)
    if supplementary:
        extras["supplementary"] = supplementary
        for arm in sorted(supplementary):
            builder.note("%s: supplementary checkpoint row(s) %s: reported beside the "
                         "preregistered methods and in no comparison; see extras.supplementary"
                         % (arm, ", ".join(sorted(supplementary[arm]))))
    missing_methods = collect_missing_methods(results_root, runs)
    if missing_methods:
        extras["methods_without_records"] = missing_methods
        for arm in sorted(missing_methods):
            builder.note("%s: no evaluation record for %s; see extras.methods_without_records"
                         % (arm, ", ".join(sorted(missing_methods[arm]))))
    if partial_merges:
        extras["aime_partial_merges"] = partial_merges
    if gate.skipped:
        extras["keys_the_manifest_does_not_define"] = sorted(set(gate.skipped))
        builder.note("%d key(s) the manifest does not define were not written; see extras"
                     % len(set(gate.skipped)))
    builder.extra["extras"] = extras
    return builder


def _write_arm(gate: "ManifestGate", table, runs, arm: str, mode: str) -> None:
    """Every key of one arm: the cells, the strongest baseline, the differences."""
    benches = _benches_present(table, arm)
    methods = _methods_present(table, arm)
    series: Dict[Tuple[str, str], Optional[Series]] = {}
    for bench in benches:
        for method in methods:
            series[(bench, method)] = series_of(table, runs, arm, method, bench, mode)

    over = ("evaluation runs of one training seed" if mode == "interim"
            else "training seeds, each the mean of its evaluation runs")

    for bench in benches:
        for method in methods:
            cell = series[(bench, method)]
            if cell is None or cell.mean is None:
                continue
            base = "E2.%s.%s.%s" % (arm, bench, method)
            formed = (AVG4_NOTE if bench == "avg4" else "")
            info = supplementary_method(method)
            extra = ("" if info is None
                     else SUPPLEMENTARY_NOTE % (METHOD_NAMES.get(info["base"], info["base"]),
                                                info["checkpoint_step"]))
            gate.add(base + ".mean", cell.mean, n=cell.n, source=cell.sources,
                     note=_note(mode, "; ".join(x for x in ("mean over the %s" % over, formed, extra) if x)))
            if cell.std is not None:
                gate.add(base + ".std", cell.std, n=cell.n, source=cell.sources,
                         note=_note(mode, "; ".join(x for x in ("std over the %s" % over, formed, extra) if x)))

    best = _strongest_baseline(series, benches)
    if best is not None:
        _write_strongest(gate, arm, mode, series, best)

    _write_differences(gate, arm, mode, series, benches, best)
    _write_gains(gate, arm, mode, series)


def _strongest_baseline(series: Mapping[Tuple[str, str], Optional[Series]],
                        benches: Sequence[str]) -> Optional[str]:
    """The baseline with the highest MATH500 mean, or None when none ran.

    The manifest names MAPPO, MAGRPO and CCPO as the candidates, so SFT and the
    single-agent PPO reference are not eligible however they score.
    """
    if "math500" not in benches:
        return None
    ranked = []
    for method in STRONGEST_BASELINE_CANDIDATES:
        cell = series.get(("math500", method))
        if cell is not None and cell.mean is not None:
            ranked.append((cell.mean, method))
    if not ranked:
        return None
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return ranked[0][1]


def _write_strongest(gate: "ManifestGate", arm: str, mode: str,
                     series: Mapping[Tuple[str, str], Optional[Series]], best: str) -> None:
    cell = series[("math500", best)]
    if cell is None or cell.mean is None:
        return
    gate.add("E2.%s.strongest_baseline.name" % arm, METHOD_NAMES[best],
             n=cell.n, source=cell.sources,
             note=_note(mode, "highest MATH500 mean among MAPPO, MAGRPO and CCPO"))
    gate.add("E2.%s.math500.best.mean" % arm, cell.mean, n=cell.n, source=cell.sources,
             note=_note(mode, "the MATH500 mean of %s" % METHOD_NAMES[best]))
    if cell.std is not None:
        gate.add("E2.%s.math500.best.std" % arm, cell.std, n=cell.n, source=cell.sources,
                 note=_note(mode, "the MATH500 std of %s" % METHOD_NAMES[best]))


def _pair(left: Optional[Series], right: Optional[Series], mode: str) -> Optional[Dict[str, Any]]:
    """The difference between two cells, paired by training seed when there is more than one."""
    if left is None or right is None or left.mean is None or right.mean is None:
        return None
    if mode == "interim":
        return {"n": 1, "diff": left.mean - right.mean, "ci_lo": None, "ci_hi": None, "p": None}
    units = [unit for unit in left.by_unit if unit in right.by_unit]
    if not units:
        return None
    return paired_stats([left.by_unit[u] for u in units], [right.by_unit[u] for u in units])


def _write_differences(gate: "ManifestGate", arm: str, mode: str,
                       series: Mapping[Tuple[str, str], Optional[Series]],
                       benches: Sequence[str], best: Optional[str]) -> None:
    """`c3_minus_best` and `c3_minus_mappo`, with Holm inside each family."""
    for label, other in (("c3_minus_best", best), ("c3_minus_mappo", "mappo")):
        if other is None:
            continue
        results: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        for bench in benches:
            pair = _pair(series.get((bench, "c3")), series.get((bench, other)), mode)
            if pair is not None:
                results[bench] = pair
        if not results:
            continue

        family = [bench for bench in HOLM_FAMILY
                  if bench in results and results[bench].get("p") is not None]
        corrected = dict(zip(family, holm([results[bench]["p"] for bench in family])))

        for bench, pair in results.items():
            base = "E2.%s.%s.%s" % (arm, bench, label)
            if mode == "interim":
                tail = ("difference of two means, each over the evaluation runs of one "
                        "training seed; no paired interval and no p value are defined at "
                        "one training seed")
            else:
                tail = "paired over the %d training seeds both arms have" % pair["n"]
            gate.add(base + ".diff", pair["diff"], n=max(1, int(pair["n"])),
                     source=_pair_sources(series, bench, other),
                     note=_note(mode, tail))
            if mode == "interim":
                continue
            if pair["ci_lo"] is not None:
                gate.add(base + ".ci_lo", pair["ci_lo"], n=pair["n"],
                         source=_pair_sources(series, bench, other), note=_note(mode, tail))
                gate.add(base + ".ci_hi", pair["ci_hi"], n=pair["n"],
                         source=_pair_sources(series, bench, other), note=_note(mode, tail))
            if pair["p"] is not None:
                formatted = format_p(pair["p"])
                if formatted is not None:
                    gate.add(base + ".p", formatted, n=pair["n"],
                             source=_pair_sources(series, bench, other),
                             note=_note(mode, "two-sided paired t test"))
                adjusted = corrected.get(bench, pair["p"])
                inside = "Holm over %d benchmarks" % len(family) if bench in corrected \
                    else "tested outside the Holm family"
                formatted = format_p(adjusted)
                if formatted is not None:
                    gate.add(base + ".p_holm", formatted, n=pair["n"],
                             source=_pair_sources(series, bench, other),
                             note=_note(mode, inside))


def _pair_sources(series: Mapping[Tuple[str, str], Optional[Series]],
                  bench: str, other: str) -> List[str]:
    sources: List[str] = []
    for method in ("c3", other):
        cell = series.get((bench, method))
        if cell is not None:
            sources.extend(cell.sources)
    return sources


def _write_gains(gate: "ManifestGate", arm: str, mode: str,
                 series: Mapping[Tuple[str, str], Optional[Series]]) -> None:
    """The two decomposition keys of the two-agent Qwen3-4B arm."""
    if arm != "a2_4b":
        return
    for key, left, right in (("arch_gain", "mappo", "ppo1a"), ("credit_gain", "c3", "mappo")):
        pair = _pair(series.get(("math500", left)), series.get(("math500", right)), mode)
        if pair is None:
            continue
        gate.add("E2.a2_4b.math500.%s" % key, pair["diff"], n=max(1, int(pair["n"])),
                 source=_pair_sources_for(series, "math500", left, right),
                 note=_note(mode, "%s minus %s on MATH500"
                            % (METHOD_NAMES[left], METHOD_NAMES[right])))


def _pair_sources_for(series, bench: str, left: str, right: str) -> List[str]:
    sources: List[str] = []
    for method in (left, right):
        cell = series.get((bench, method))
        if cell is not None:
            sources.extend(cell.sources)
    return sources


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = build_parser(EXPERIMENT)
    ap.add_argument("--arm", default="", help="aggregate only this arm (default: every arm found)")
    ap.add_argument("--allow-broken-records", action="store_true",
                    help=("read past evaluation records that do not parse, are missing a field, "
                          "say ok is not true, or are far from the size of the records beside "
                          "them, instead of refusing. Every key of the affected arm then says "
                          "how many were damaged and which"))
    args = ap.parse_args(argv)

    arm_filter = str(args.arm or "").strip()
    if arm_filter and arm_filter not in ARMS:
        print("[aggregate_e2] ERROR: --arm %s is not one of %s"
              % (arm_filter, ", ".join(ARMS)), file=sys.stderr)
        return 2

    manifest = load_manifest(args.manifest)
    try:
        builder = build_e2_summary(args.results, manifest, arm_filter=arm_filter or None,
                                   allow_broken_records=args.allow_broken_records)
    except BrokenRecordError as exc:
        print("[aggregate_e2] REFUSED: %d damaged evaluation record(s); nothing was written"
              % len(exc.broken), file=sys.stderr)
        for item in exc.broken:
            print("  %s: %s" % (item["path"], item["reason"]), file=sys.stderr)
        print("  %s was NOT updated. If a summary is already there it is from an earlier run, "
              "and feeding it to manifest_writer would write numbers this tree no longer holds."
              % args.out, file=sys.stderr)
        print("  Repair the file(s) and run again, or pass --allow-broken-records to report the "
              "records that are readable, with the damage named on every key of that arm.",
              file=sys.stderr)
        return EXIT_BROKEN_RECORDS

    builder.write(args.out)
    print("wrote %d key(s) to %s" % (len(builder.keys), args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
