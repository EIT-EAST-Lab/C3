# c3/analysis/rebuild/e2_tokens.py
"""The training token account of one E2 run directory.

    python -m c3.analysis.rebuild.e2_tokens \
        --run 20_data/results/E2/a2_4b/c3/train_s0 --out tokens.json

What a run directory carries
----------------------------
The platform archives `run_config.json`, `budget_ledger.jsonl`, `c3_task.yaml`
and `train.log` beside the evaluation seeds. None of them has a token field:
the ledger counts terminal evaluator calls (`c3/utils/budget_ledger.py`, the
record is `global_step, n_questions_in_batch, n_samples_per_prompt,
total_eval_calls, ts_unix` and nothing about length), and the only place a
length reaches the disk is the status dictionary the trainer prints once per
optimizer step:

    INFO 09-19 05:12:33 ppo_trainer.py] Global step 12 | epoch 1/2 | iter 13/312: {...}

written by `openrlhf/trainer/ppo_trainer.py` at the end of each iteration of
`fit()`. Two of its entries are lengths in tokens, both means over samples:

- `response_length`: generated tokens per sample. It starts as
  `len(output.outputs[0].token_ids)` per sample
  (`openrlhf/trainer/ppo_utils/experience_maker.py`, and the same field per
  node in `c3/protocol/rollout_generator.py`), is averaged over the micro batch
  in `openrlhf/trainer/ray/ppo_actor.py:training_step`, then over the micro
  steps of the epoch in `fit`, and is what the progress bar calls `gen_len`.
- `total_length`: prompt plus generated tokens per sample, the sum of the
  attention mask of the same sequence. The progress bar calls it `tot_len`.

The sample count of the same step is `reward/count`, the number of rows
`_rewardprovider_rollout_metrics` built from the rollout batch
(`openrlhf/trainer/ppo_trainer_plugins.py`), one row per agent message, and it
is in the same status dictionary.

So the account this script can close is

    generated tokens of a step = response_length * reward/count
    total tokens of a step     = total_length    * reward/count
    prompt tokens of a step    = total - generated

summed over the steps. That product is the reading the earlier compute table
was built from (`overhead_ledger.py`, the row
`cum_tt += token_len * effective_ads`), so a number from here is comparable
with the 418.4 M / 616.1 M / 703.1 M of that table, with two differences
recorded rather than hidden:

- that script fell back to 2048 samples and 512 tokens when a step line did not
  carry the field. Nothing here falls back: a step missing either field is
  counted in `incomplete_steps` and left out of the totals.
- that script multiplied MAPPO by eight, because in the January runs MAPPO drew
  one sample per prompt against the eight of C3 and MAGRPO, and the table
  reports a matched budget. Nothing here multiplies: every arm of this round
  launches with `--n_samples_per_prompt 8`. `config.n_samples_per_prompt` is
  reported so the reader can check that before comparing.
- that script read the dictionary from the first `{` to the last `}` of the
  line. Ray sends the stdout of every actor to one stream, so a handful of
  status lines per run end with another actor's message flushed onto them, and
  reading to the last `}` loses those steps. `parse_status_line` reads to the
  `}` that closes the first `{` instead; see its own docstring.

What is not recorded
--------------------
Every quantity this script cannot reach is named in `not_recorded` with the
reason, rather than estimated. The four that matter:

- any split of the tokens by position in the rollout tree, which is what a
  reference-versus-continuation breakdown would need. Each node carries its own
  length beside its position (`c3_depth`, `c3_sibling_id`, `c3_path`,
  `is_leaf`) in `Experience.info`, and the step line holds the mean of each
  field separately, so the joint distribution is gone by the time it is
  written.
- the tokens of the evaluation passes, both the terminal evaluator calls and
  the monitoring evaluation. The ledger counts calls, not their length. The
  evaluation writes its own sums beside its accuracies, in a file this script
  does not open and the archive step does not copy.
- the prompt tokens as a recorded quantity. They are a difference of two
  recorded means here, not a third measurement.

The critic warmup stage
-----------------------
That stage, which the MAPPO arm runs 64 rollouts of before its first optimizer
step, prints `Critic warmup n/total` rather than a `Global step` line, so no
status line reports its tokens. It writes a ledger of its own instead,
`budget_ledger_warmup.jsonl`, one row per generation call with the token sums of
that call. Those rows are reported in the `warmup` block and enter no training
total: a warmup rollout belongs to no optimizer step, and a comparison of
methods is a comparison of training. A run directory without that file, which is
every run launched before the record existed, reports None there and says so.

Wall clock comes from the ledger's `ts_unix`, and it is the training time of the
run rather than the distance between its first and last row. A run that is
stopped and resumed on another container leaves one long hole in the ledger, and
the distance from the first row to the last would count that hole as training.
So the rows are cut into contiguous segments wherever the interval between two
adjacent rows is longer than ten times that ledger's own median row interval,
and `wall_clock_h` is the sum of the segment spans. Inside a segment the span
still holds whatever the run spent there, the monitoring evaluations included;
the setup before the first row and the save after the last are still outside it.
A run with no hole has one segment, so the quantity is unchanged for it, and the
number of holes and the hours dropped are reported beside the sum either way.

Only the standard library is imported.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import statistics
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .summary import source_path

__all__ = [
    "SCHEMA",
    "LOG_NAME",
    "LEDGER_NAME",
    "WARMUP_LEDGER_NAME",
    "CONFIG_NAME",
    "STEP_RE",
    "DOWNTIME_GAP_FACTOR",
    "PROVENANCE",
    "parse_status_line",
    "read_train_log",
    "ledger_segments",
    "read_ledger",
    "read_warmup_ledger",
    "read_run_config",
    "token_account",
    "read_run",
    "main",
]

#: Schema of the document this script writes.
SCHEMA = "c3.e2_tokens/1"

#: The files of a training run this script reads, as the platform names them.
LOG_NAME = "train.log"
LEDGER_NAME = "budget_ledger.jsonl"
CONFIG_NAME = "run_config.json"

#: The ledger of the critic warmup stage, written since the rollouts of that
#: stage were given a record of their own (c3/utils/budget_ledger.py). It is a
#: separate file, and its rows are kept out of every training total here, for
#: the same reason it is separate on the writing side: a warmup rollout belongs
#: to no optimizer step, and the row count of the training ledger is read as the
#: step count of the run. A run directory from before that change does not have
#: it, which is reported rather than treated as a zero.
WARMUP_LEDGER_NAME = "budget_ledger_warmup.jsonl"

#: The head of a status line. Anything may precede it: Ray prefixes the stdout
#: of an actor with `(PPOTrainerActor pid=123)` and the logger prefixes it with
#: a level, a timestamp and a source location.
STEP_RE = re.compile(r"Global step\s+(\d+)")

#: The rest of the same message: `| epoch 1/2 | iter 13/312`. Both counters are
#: printed one-based and are kept that way, under the names they are printed
#: with, so a row of this document reads like the line it came from.
EPOCH_RE = re.compile(r"epoch\s+(\d+)\s*/\s*(\d+)")
ITER_RE = re.compile(r"iter\s+(\d+)(?:\s*/\s*(\d+))?")

#: How much longer than the usual step an interval between two adjacent ledger
#: rows has to be before it counts as downtime rather than training. The unit is
#: that ledger's own median row interval, so a slow arm and a fast one are held
#: to the same rule without a wall-clock constant in it. Ten times the median is far above the
#: spread of a healthy run, whose rows are one optimizer step apart with a
#: monitoring evaluation every so often, and far below the hours a stopped run
#: leaves behind.
DOWNTIME_GAP_FACTOR = 10.0

#: Status entry -> the quantity it carries. The left column is what the trainer
#: prints; the right column is what this script calls it.
LENGTH_FIELDS = (("response_length", "generated"), ("total_length", "total"))

#: Status entries that count the samples of the step, first one wins. The second
#: spelling is the one `overhead_ledger.py` also accepted.
COUNT_FIELDS = ("reward/count", "reward_count")

#: Where each quantity comes from, carried in the output so a reader never has
#: to guess which file and field a number was read off.
PROVENANCE: "Dict[str, Dict[str, str]]" = {
    "steps[].response_length": {
        "file": LOG_NAME,
        "field": "response_length of the `Global step` status dictionary",
        "how": ("generated tokens per sample, mean over the samples of the step; written by "
                "openrlhf/trainer/ppo_trainer.py at the end of one fit() iteration, formed in "
                "openrlhf/trainer/ray/ppo_actor.py:training_step from Experience.info"),
    },
    "steps[].total_length": {
        "file": LOG_NAME,
        "field": "total_length of the same dictionary",
        "how": "prompt plus generated tokens per sample, mean over the same samples",
    },
    "steps[].samples": {
        "file": LOG_NAME,
        "field": "reward/count of the same dictionary",
        "how": ("rows built from the rollout batch by "
                "openrlhf/trainer/ppo_trainer_plugins.py:_rewardprovider_rollout_metrics, "
                "one row per agent message"),
    },
    "steps[].generated_tokens": {
        "file": LOG_NAME,
        "field": "response_length * reward/count",
        "how": "product of two entries of the same status dictionary, not a recorded quantity",
    },
    "steps[].total_tokens": {
        "file": LOG_NAME,
        "field": "total_length * reward/count",
        "how": "product of two entries of the same status dictionary, not a recorded quantity",
    },
    "steps[].prompt_tokens": {
        "file": LOG_NAME,
        "field": "(total_length - response_length) * reward/count",
        "how": "difference of the two means above, not a recorded quantity",
    },
    "ledger.total_eval_calls": {
        "file": LEDGER_NAME,
        "field": "total_eval_calls",
        "how": ("n_questions_in_batch * n_samples_per_prompt of one rollout-generation call, "
                "summed over the rows; c3/utils/budget_ledger.py:make_budget_record"),
    },
    "warmup.warmup_tokens_M": {
        "file": WARMUP_LEDGER_NAME,
        "field": "total_tokens",
        "how": ("prompt plus generated tokens of one warmup rollout-generation call, summed over "
                "the nodes of the call in c3/protocol/rollout_generator.py and over the rows "
                "here. A sum of recorded lengths, where the training total above is a product of "
                "two recorded means, so the two are close in meaning and not identical in "
                "construction"),
    },
    "warmup.generated_tokens_M": {
        "file": WARMUP_LEDGER_NAME,
        "field": "generated_tokens",
        "how": "the generated part of the same sum",
    },
    "ledger.wall_clock_h": {
        "file": LEDGER_NAME,
        "field": "ts_unix",
        "how": ("the rows are cut into contiguous segments wherever two adjacent rows are more "
                "than %g times the median row interval apart, and the spans of the segments are "
                "summed, in hours. A segment span includes the evaluation passes inside it and "
                "excludes the setup before it and the save after it; a stop and resume between "
                "two segments is outside every span. A run with no such hole has one segment, "
                "so its number is the same as last row minus first row"
                % DOWNTIME_GAP_FACTOR),
    },
    "config.*": {
        "file": CONFIG_NAME,
        "field": "args",
        "how": "the launch arguments openrlhf/utils/run_metadata.py wrote when the run started",
    },
}

#: The quantities no file of a run directory carries, with what it would take to
#: record each. They are reported on every run, present or absent, because their
#: absence is a property of the trainer and not of one run.
STRUCTURAL_GAPS = (
    {
        "quantity": "tokens_split_by_position_in_the_rollout_tree",
        "why": ("no per-node length reaches the disk. Each node of a rollout carries its own "
                "response_length beside its position in the tree (c3_depth, c3_sibling_id, "
                "c3_path, is_leaf) in Experience.info "
                "(c3/protocol/rollout_generator.py), but "
                "openrlhf/trainer/ray/ppo_actor.py:training_step reduces every numeric info "
                "field to one mean over the micro batch, so the step line holds the mean depth "
                "and the mean length and not their joint distribution. Which positions count as "
                "the reference rollout and which as a replayed continuation is a definition to "
                "settle before the split can be reported either way"),
        "would_need": ("sums of response_length and total_length per position class, taken in "
                       "the rollout generator where the nodes are still in hand, written into "
                       "the budget ledger record"),
    },
    {
        "quantity": "evaluation_tokens",
        "why": ("no file this script reads carries them: the budget ledger counts terminal "
                "evaluator calls and not their length. Each evaluation pass does write its own "
                "token sums, as eval_total_tokens and eval_generated_tokens beside its "
                "accuracies (openrlhf/trainer/ppo_trainer_plugins.py:evaluate), but that is the "
                "metrics file of the evaluation, which this script does not open and which the "
                "archive step of the platform does not copy into the results tree"),
        "would_need": ("reading eval_during_train.jsonl.metrics.jsonl of the run, in the backup "
                       "copy of the run directory while the archive step leaves it there"),
    },
    {
        "quantity": "prompt_tokens_as_a_measurement",
        "why": ("prompt length is never recorded on its own; what this script reports is the "
                "difference of the two recorded means"),
        "would_need": "nothing new if the difference is accepted as the definition",
    },
)


# ---------------------------------------------------------------------------
# reading one status line
# ---------------------------------------------------------------------------

#: `nan` and `inf` are not Python literals, so a status dictionary holding one
#: cannot be read by ast.literal_eval. They are replaced by None, which makes
#: the entry missing rather than zero, and the line is counted as repaired.
_NON_LITERAL_RE = re.compile(r"(?<![\w.])(-?)(?:nan|inf|Infinity|NaN)(?![\w.])")


def _balanced_end(line: str, start: int) -> int:
    """Index of the `}` that closes the `{` at `start`, or -1 when none does.

    Braces are counted outside string literals only, so a status value that
    carries one of its own (`'\\boxed{x}'`) does not move the close. The scan
    stops at the first close, which is what separates the status dictionary from
    whatever follows it on the same physical line.
    """
    depth = 0
    quote = ""
    index = start
    length = len(line)
    while index < length:
        char = line[index]
        if quote:
            if char == "\\":
                index += 2
                continue
            if char == quote:
                quote = ""
        elif char in "'\"":
            quote = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return -1


def parse_status_line(line: str) -> Optional[Dict[str, Any]]:
    """One `Global step` line taken apart, or None when it is not one.

    The line is what `fit()` prints per optimizer step. Everything before
    `Global step` is ignored, so a Ray actor prefix and the logger's own prefix
    do not matter.

    The dictionary runs from the first `{` to the `}` that closes it, not to the
    last `}` of the line as `overhead_ledger.py` did in January. Ray writes the
    stdout of every actor to the same stream, so a second actor's message is
    sometimes flushed onto the physical line a status dictionary ends on:

        ...iter 97/156: {'policy_loss': ...}(PPOTrainer pid=118945) We did not
        manage to extract a prediction in the correct format. Gold:
        ['\\boxed{H, E}'], Pred: ['E, H']

    Reading to the last `}` there takes the other actor's text into the literal
    and the whole step is lost; reading to the matching `}` takes the status
    dictionary and leaves the rest of the line alone. A line whose first `{`
    never closes carries no readable dictionary and is still refused, so a truly
    damaged line stays out of the account rather than being half read.

    Returns `{global_step, status, repaired, epoch, train_epochs, iter,
    iters_in_epoch}`; the four position fields are None when the message does
    not carry them.
    """
    match = STEP_RE.search(line)
    if match is None:
        return None
    start = line.find("{")
    if start < 0:
        return None
    end = _balanced_end(line, start)
    if end < start:
        return None
    payload = line[start:end + 1]
    try:
        status = ast.literal_eval(payload)
        repaired = False
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        try:
            status = ast.literal_eval(_NON_LITERAL_RE.sub("None", payload))
            repaired = True
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            return None
    if not isinstance(status, dict):
        return None

    head = line[match.end():start]
    out: Dict[str, Any] = {
        "global_step": int(match.group(1)),
        "status": status,
        "repaired": repaired,
        "epoch": None,
        "train_epochs": None,
        "iter": None,
        "iters_in_epoch": None,
    }
    epoch = EPOCH_RE.search(head)
    if epoch is not None:
        out["epoch"] = int(epoch.group(1))
        out["train_epochs"] = int(epoch.group(2))
    iteration = ITER_RE.search(head)
    if iteration is not None:
        out["iter"] = int(iteration.group(1))
        if iteration.group(2) is not None:
            out["iters_in_epoch"] = int(iteration.group(2))
    return out


def _number(status: Mapping[str, Any], names: Sequence[str]) -> Tuple[Optional[float], str]:
    """The first finite number among `names`, and the name it was found under."""
    for name in names:
        value = status.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if not math.isfinite(float(value)):
            continue
        return float(value), name
    return None, ""


def _role_prefixed(status: Mapping[str, Any], field: str) -> Dict[str, float]:
    """`{role: value}` for the `<role>/<field>` entries of a per-role status.

    `policy_sharing_mode per_role` prefixes every actor statistic with the role
    name (openrlhf/trainer/ppo_trainer_plugins.py:ppo_train), so a run launched
    that way has no plain `response_length` at all. The arms of this round use
    `shared`, but a per-role log must not read as a log with no lengths in it.
    """
    out: Dict[str, float] = {}
    tail = "/" + field
    for name, value in status.items():
        if not isinstance(name, str) or not name.endswith(tail):
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if math.isfinite(float(value)):
            out[name[:-len(tail)]] = float(value)
    return out


# ---------------------------------------------------------------------------
# reading the files of a run
# ---------------------------------------------------------------------------


def read_train_log(path: str) -> Dict[str, Any]:
    """Every `Global step` line of one trainer log, in step order.

    A step that appears more than once (a resumed run writes the step it
    restarts from again) keeps its last line, which is the convention the
    evaluation records already follow. Lines that do not parse are counted, not
    dropped silently.
    """
    out: Dict[str, Any] = {
        "path": source_path(path),
        "present": os.path.isfile(path),
        "n_lines_with_a_step": 0,
        "n_unparsed": 0,
        "n_repaired": 0,
        "n_duplicate_steps": 0,
        "steps": [],
    }
    if not out["present"]:
        out["error"] = "no %s in the run directory" % os.path.basename(path)
        return out

    by_step: "Dict[int, Dict[str, Any]]" = {}
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if "Global step" not in line:
                    continue
                out["n_lines_with_a_step"] += 1
                parsed = parse_status_line(line)
                if parsed is None:
                    out["n_unparsed"] += 1
                    continue
                if parsed["repaired"]:
                    out["n_repaired"] += 1
                step = int(parsed["global_step"])
                if step in by_step:
                    out["n_duplicate_steps"] += 1
                by_step[step] = _step_row(parsed)
    except OSError as exc:
        out["error"] = str(exc)
        return out

    out["steps"] = [by_step[step] for step in sorted(by_step)]
    return out


def _step_row(parsed: Mapping[str, Any]) -> Dict[str, Any]:
    """One step of the account: the two means, the sample count, the products."""
    status = parsed["status"]
    row: Dict[str, Any] = {"global_step": int(parsed["global_step"])}
    for name in ("epoch", "train_epochs", "iter", "iters_in_epoch"):
        if parsed.get(name) is not None:
            row[name] = int(parsed[name])

    samples, found_under = _number(status, COUNT_FIELDS)
    row["samples"] = int(samples) if samples is not None else None
    if found_under and found_under != COUNT_FIELDS[0]:
        row["samples_field"] = found_under

    for field, _label in LENGTH_FIELDS:
        value, _ = _number(status, (field,))
        row[field] = value
        if value is None:
            by_role = _role_prefixed(status, field)
            if by_role:
                row[field + "_by_role"] = by_role

    generated = row.get("response_length")
    total = row.get("total_length")
    count = row.get("samples")

    row["generated_tokens"] = (generated * count) if (generated is not None and count is not None) else None
    row["total_tokens"] = (total * count) if (total is not None and count is not None) else None
    if row["generated_tokens"] is not None and row["total_tokens"] is not None:
        row["prompt_tokens"] = row["total_tokens"] - row["generated_tokens"]
    else:
        row["prompt_tokens"] = None

    missing = [name for name in ("response_length", "total_length") if row.get(name) is None]
    if count is None:
        missing.append("reward/count")
    if missing:
        row["missing"] = missing
    return row


def ledger_segments(stamps: Sequence[Any]) -> Dict[str, Any]:
    """Cut a ledger's timestamps into the contiguous stretches the run trained in.

    The rule is the one the driver ruled on 2026-09-22: the interval between two
    adjacent rows is downtime, and not training, when it is longer than
    `DOWNTIME_GAP_FACTOR` times the median interval of that same ledger. The
    wall clock of the run is then the sum of the segment spans, so a stop and
    resume on another container is outside the account instead of inside it.

    The stamps are sorted before they are read, because a resumed run's ledger is
    two files appended to each other and nothing guarantees the order on disk;
    adjacency here means adjacency in time.

    A ledger whose median interval is zero, which takes a run that wrote every
    row inside one second, gets no threshold at all: ten times zero would make
    every interval a hole and the sum of the segments zero, which is the one
    answer that is certainly wrong. Such a ledger is reported as one segment.

    Returns the segments, the sum, and the quantities a note has to carry: how
    many holes were dropped and how many hours they held. `first_last_span_h` is
    the old reading, kept so the two can be compared without reparsing the file.
    """
    ordered = sorted(int(stamp) for stamp in stamps)
    out: Dict[str, Any] = {
        "n_stamps": len(ordered),
        "median_row_interval_s": None,
        "gap_threshold_s": None,
        "segments": [],
        "n_segments": 0,
        "n_gaps_excluded": 0,
        "gap_hours_excluded": 0.0,
        "wall_clock_h": None,
        "first_last_span_h": None,
    }
    if not ordered:
        return out

    out["first_last_span_h"] = (ordered[-1] - ordered[0]) / 3600.0
    intervals = [later - earlier for earlier, later in zip(ordered, ordered[1:])]
    threshold: Optional[float] = None
    if intervals:
        median = float(statistics.median(intervals))
        out["median_row_interval_s"] = median
        if median > 0:
            threshold = DOWNTIME_GAP_FACTOR * median
            out["gap_threshold_s"] = threshold

    cuts = ([index for index, gap in enumerate(intervals) if gap > threshold]
            if threshold is not None else [])

    bounds: List[Tuple[int, int]] = []
    start = 0
    for index in cuts:
        bounds.append((start, index))
        start = index + 1
    bounds.append((start, len(ordered) - 1))

    total_h = 0.0
    for first, last in bounds:
        span_h = (ordered[last] - ordered[first]) / 3600.0
        total_h += span_h
        out["segments"].append({
            "first_ts_unix": ordered[first],
            "last_ts_unix": ordered[last],
            "n_rows": last - first + 1,
            "span_h": span_h,
        })
    out["n_segments"] = len(bounds)
    out["n_gaps_excluded"] = len(cuts)
    out["gap_hours_excluded"] = sum(intervals[index] for index in cuts) / 3600.0
    out["wall_clock_h"] = total_h
    return out


def read_ledger(path: str) -> Dict[str, Any]:
    """The totals of one `budget_ledger.jsonl`, plus the training time it spans.

    `wall_clock_h` is the sum of the contiguous segments of the ledger and not
    the distance from its first row to its last; `ledger_segments` says why and
    how. `first_last_span_h` is that older distance, kept beside it.
    """
    out: Dict[str, Any] = {
        "path": source_path(path),
        "present": os.path.isfile(path),
        "rows": 0,
        "global_steps": 0,
        "total_eval_calls": 0,
        "n_questions_in_batch": [],
        "n_samples_per_prompt": [],
        "marl_algorithm": [],
        "first_ts_unix": None,
        "last_ts_unix": None,
        "wall_clock_h": None,
        "first_last_span_h": None,
        "median_row_interval_s": None,
        "gap_threshold_s": None,
        "n_segments": 0,
        "n_gaps_excluded": 0,
        "gap_hours_excluded": 0.0,
        "segments": [],
    }
    if not out["present"]:
        out["error"] = "no %s in the run directory" % os.path.basename(path)
        return out

    steps = set()
    questions = set()
    samples = set()
    algorithms = set()
    stamps: List[int] = []
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
                out["rows"] += 1
                step = row.get("global_step")
                if isinstance(step, int) and not isinstance(step, bool):
                    steps.add(step)
                calls = row.get("total_eval_calls")
                if isinstance(calls, (int, float)) and not isinstance(calls, bool):
                    out["total_eval_calls"] += int(calls)
                for key, sink in (("n_questions_in_batch", questions),
                                  ("n_samples_per_prompt", samples)):
                    value = row.get(key)
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        sink.add(int(value))
                algorithm = row.get("marl_algorithm")
                if isinstance(algorithm, str) and algorithm:
                    algorithms.add(algorithm)
                stamp = row.get("ts_unix")
                if isinstance(stamp, (int, float)) and not isinstance(stamp, bool):
                    stamps.append(int(stamp))
    except OSError as exc:
        out["error"] = str(exc)
        return out

    out["global_steps"] = len(steps)
    out["n_questions_in_batch"] = sorted(questions)
    out["n_samples_per_prompt"] = sorted(samples)
    out["marl_algorithm"] = sorted(algorithms)
    if stamps:
        out["first_ts_unix"] = min(stamps)
        out["last_ts_unix"] = max(stamps)
        cut = ledger_segments(stamps)
        for field in ("wall_clock_h", "first_last_span_h", "median_row_interval_s",
                      "gap_threshold_s", "n_segments", "n_gaps_excluded",
                      "gap_hours_excluded", "segments"):
            out[field] = cut[field]
    return out


def read_warmup_ledger(path: str) -> Dict[str, Any]:
    """The totals of one `budget_ledger_warmup.jsonl`.

    The rows are the rollout-generation calls of the critic warmup stage, the
    stage the MAPPO arm runs 64 of before its first optimizer step. Each row
    carries the token sums of its call, so `warmup_tokens_M` is a sum of
    recorded numbers and not a product of means, which is what the training
    account of this script has to be.

    Nothing here reaches a training total. A run whose directory has no such
    file, which is every run from before the record existed, comes back with
    every quantity None and a note saying so, rather than with a zero.
    """
    out: Dict[str, Any] = {
        "path": source_path(path),
        "present": os.path.isfile(path),
        "note": "",
        "rows": 0,
        "rows_without_tokens": 0,
        "total_eval_calls": 0,
        "n_nodes": None,
        "generated_tokens": None,
        "total_tokens": None,
        "generated_tokens_M": None,
        "warmup_tokens_M": None,
        "marl_algorithm": [],
        "first_ts_unix": None,
        "last_ts_unix": None,
        "wall_clock_h": None,
    }
    if not out["present"]:
        out["note"] = ("no warmup ledger: %s is not in the run directory, so the rollouts of the "
                       "critic warmup stage are not recorded for this run. A run of an algorithm "
                       "with no warmup stage has none either way; which it is here is decided by "
                       "config.critic_warmup_steps and not by this file"
                       % os.path.basename(path))
        return out

    algorithms = set()
    stamps: List[int] = []
    nodes = 0
    generated = 0
    total = 0
    with_tokens = 0
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
                out["rows"] += 1

                calls = row.get("total_eval_calls")
                if isinstance(calls, (int, float)) and not isinstance(calls, bool):
                    out["total_eval_calls"] += int(calls)

                algorithm = row.get("marl_algorithm")
                if isinstance(algorithm, str) and algorithm:
                    algorithms.add(algorithm)

                stamp = row.get("ts_unix")
                if isinstance(stamp, (int, float)) and not isinstance(stamp, bool):
                    stamps.append(int(stamp))

                count = row.get("n_nodes")
                if isinstance(count, (int, float)) and not isinstance(count, bool):
                    nodes += int(count)

                gen = row.get("generated_tokens")
                tot = row.get("total_tokens")
                ok = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (gen, tot))
                if ok:
                    generated += int(gen)
                    total += int(tot)
                    with_tokens += 1
                else:
                    out["rows_without_tokens"] += 1
    except OSError as exc:
        out["error"] = str(exc)
        out["note"] = "the warmup ledger could not be read: %s" % exc
        return out

    out["marl_algorithm"] = sorted(algorithms)
    if stamps:
        # Deliberately the plain first-to-last distance, not the segmented sum
        # the training ledger above reports: the warmup stage runs in one piece
        # before the first optimizer step, its rows reach no training total and
        # no manifest key, and cutting a stage that is never resumed into
        # segments would only invite the reader to compare two numbers that are
        # not the same quantity.
        out["first_ts_unix"] = min(stamps)
        out["last_ts_unix"] = max(stamps)
        out["wall_clock_h"] = (max(stamps) - min(stamps)) / 3600.0
    if nodes:
        out["n_nodes"] = int(nodes)

    if with_tokens:
        out["generated_tokens"] = int(generated)
        out["total_tokens"] = int(total)
        out["generated_tokens_M"] = generated / 1e6
        out["warmup_tokens_M"] = total / 1e6
        out["note"] = ("%d warmup rollout-generation call(s), token sums taken from %d of them"
                       % (out["rows"], with_tokens))
        if out["rows_without_tokens"]:
            out["note"] += ("; the other %d carry no token field and are left out of the sums"
                            % out["rows_without_tokens"])
    else:
        out["note"] = ("%d warmup rollout-generation call(s), none of which carries a token field, "
                       "so the warmup tokens stay unrecorded for this run" % out["rows"])
    return out


#: Launch arguments worth carrying beside the account, because a token number is
#: only comparable with another one under the same rollout budget.
CONFIG_FIELDS = ("run_id", "marl_algorithm", "policy_sharing_mode", "n_samples_per_prompt",
                 "rollout_batch_size", "train_batch_size", "max_epochs", "train_epochs",
                 "max_samples", "generate_max_len", "prompt_max_len", "c3_fanout", "seed",
                 "critic_warmup_steps")


def read_run_config(path: str) -> Dict[str, Any]:
    """The launch arguments of one run, narrowed to what a token account needs."""
    out: Dict[str, Any] = {"path": source_path(path), "present": os.path.isfile(path)}
    if not out["present"]:
        out["error"] = "no %s in the run directory" % os.path.basename(path)
        return out
    try:
        with open(path, "r", encoding="utf-8") as fh:
            loaded = json.load(fh)
    except (OSError, ValueError) as exc:
        out["error"] = str(exc)
        return out
    args = loaded.get("args") if isinstance(loaded, dict) else None
    if not isinstance(args, Mapping):
        out["error"] = "no args object in %s" % os.path.basename(path)
        return out
    for name in CONFIG_FIELDS:
        if name in args:
            out[name] = args[name]
    return out


# ---------------------------------------------------------------------------
# the account
# ---------------------------------------------------------------------------


def token_account(steps: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """The cumulative account over the steps that carry every field it needs.

    A step missing a field contributes nothing and is counted instead, so a
    total is always the total of a stated number of complete steps. Nothing is
    filled in for a missing field.
    """
    complete = [row for row in steps if not row.get("missing")]
    out: Dict[str, Any] = {
        "n_steps": len(steps),
        "n_complete_steps": len(complete),
        "incomplete_steps": [int(row["global_step"]) for row in steps if row.get("missing")],
        "first_step": int(steps[0]["global_step"]) if steps else None,
        "last_step": int(steps[-1]["global_step"]) if steps else None,
        "samples": None,
        "generated_tokens": None,
        "prompt_tokens": None,
        "total_tokens": None,
        "generated_tokens_M": None,
        "total_tokens_M": None,
    }
    if not complete:
        return out

    out["samples"] = int(sum(int(row["samples"]) for row in complete))
    generated = float(sum(float(row["generated_tokens"]) for row in complete))
    total = float(sum(float(row["total_tokens"]) for row in complete))
    out["generated_tokens"] = generated
    out["total_tokens"] = total
    out["prompt_tokens"] = total - generated
    out["generated_tokens_M"] = generated / 1e6
    out["total_tokens_M"] = total / 1e6
    return out


def read_run(run_dir: str, *, log_name: str = LOG_NAME,
             include_steps: bool = True) -> Dict[str, Any]:
    """The whole token account of one training run directory.

    `include_steps` keeps the per-step rows in the document. The command line
    wants them; the E2 aggregation does not, because one arm has a few hundred
    steps and the summary carries six runs.
    """
    log = read_train_log(os.path.join(run_dir, log_name))
    ledger = read_ledger(os.path.join(run_dir, LEDGER_NAME))
    warmup = read_warmup_ledger(os.path.join(run_dir, WARMUP_LEDGER_NAME))
    config = read_run_config(os.path.join(run_dir, CONFIG_NAME))

    steps = list(log.get("steps") or [])
    totals = token_account(steps)

    not_recorded: List[Dict[str, str]] = [dict(gap) for gap in STRUCTURAL_GAPS]
    if not log.get("present"):
        not_recorded.append({
            "quantity": "every per-step length",
            "why": "%s says %s" % (log["path"], log.get("error", "it is not there")),
            "would_need": ("the trainer log of the run, copied as train.log beside the "
                           "evaluation seeds"),
        })
    elif not steps:
        not_recorded.append({
            "quantity": "every per-step length",
            "why": "%s carries no `Global step` status line" % log["path"],
            "would_need": "a log of a run that reached its first optimizer step",
        })
    elif totals["n_complete_steps"] < totals["n_steps"]:
        missing_fields = sorted({name for row in steps for name in (row.get("missing") or [])})
        not_recorded.append({
            "quantity": "the token account of %d of %d steps"
                        % (totals["n_steps"] - totals["n_complete_steps"], totals["n_steps"]),
            "why": "those step lines carry no %s" % ", ".join(missing_fields),
            "would_need": ("nothing, if the totals of the complete steps are read as the totals "
                           "of those steps and not of the run"),
        })
    if not ledger.get("present"):
        not_recorded.append({
            "quantity": "wall_clock_h and total_eval_calls",
            "why": "%s says %s" % (ledger["path"], ledger.get("error", "it is not there")),
            "would_need": "the budget ledger of the run",
        })
    if warmup.get("warmup_tokens_M") is None:
        not_recorded.append({
            "quantity": "tokens_of_the_critic_warmup",
            "why": ("the critic warmup stage generates full rollouts "
                    "(openrlhf/trainer/ppo_trainer.py:_run_critic_warmup_stage) and the MAPPO arm "
                    "runs 64 of them, but it prints `Critic warmup n/total` rather than a `Global "
                    "step` line, so its tokens are in no status line of the trainer log. The one "
                    "place they are recorded is the warmup ledger, and here %s"
                    % warmup.get("note", "it carries none")),
            "would_need": ("%s of the run; a run launched before that record existed has none, "
                           "and its warmup tokens cannot be recovered after the fact"
                           % WARMUP_LEDGER_NAME),
        })

    document: Dict[str, Any] = {
        "schema": SCHEMA,
        "run_dir": source_path(run_dir),
        "config": config,
        "totals": totals,
        "warmup": warmup,
        "ledger": ledger,
        "log": {name: log[name] for name in log if name != "steps"},
        "provenance": PROVENANCE,
        "not_recorded": not_recorded,
    }
    if include_steps:
        document["steps"] = steps
    else:
        document["steps_note"] = ("the per-step rows are in the output of "
                                  "c3.analysis.rebuild.e2_tokens; only the totals are carried here")
    return document


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m c3.analysis.rebuild.e2_tokens",
        description="The training token account of one E2 run directory")
    ap.add_argument("--run", required=True, help="a training run directory (…/train_s0)")
    ap.add_argument("--out", default="", help="where to write the JSON (default: stdout)")
    ap.add_argument("--log", default=LOG_NAME,
                    help="the trainer log inside the run directory (default: %s)" % LOG_NAME)
    ap.add_argument("--no_steps", action="store_true",
                    help="leave the per-step rows out of the document")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.run):
        print("[e2_tokens] ERROR: %s is not a directory" % args.run, file=sys.stderr)
        return 2

    document = read_run(args.run, log_name=args.log, include_steps=not args.no_steps)
    text = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=False)
    if args.out:
        directory = os.path.dirname(os.path.abspath(args.out))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
        totals = document["totals"]
        print("[e2_tokens] %s: %d of %d step(s) complete, %s M total tokens -> %s"
              % (document["run_dir"], totals["n_complete_steps"], totals["n_steps"],
                 "none" if totals["total_tokens_M"] is None else "%.1f" % totals["total_tokens_M"],
                 args.out))
    else:
        print(text)
    for gap in document["not_recorded"]:
        print("[e2_tokens] not recorded: %s (%s)" % (gap["quantity"], gap["why"]), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
