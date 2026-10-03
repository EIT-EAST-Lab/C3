# c3/analysis/rebuild/e2_curves.py
"""The monitoring curve of one E2 run directory.

    python -m c3.analysis.rebuild.e2_curves \
        --run 20_data/results/E2/a2_4b/c3/train_s0 --out curve.json

Where the points come from
--------------------------
The arms of this round launch with `--eval_every_ratio 0.25`, so
`openrlhf/cli/train_ppo_ray.py` turns that into `eval_steps = round(max_steps *
0.25)` and the trainer evaluates four times inside the run. Each pass ends in
`openrlhf/trainer/ppo_trainer_plugins.py:evaluate`, which builds

    eval_<datasource>_pass1   the mean over problems of the mean over the k
                              samples of that problem, that is avg@k, the same
                              reading `scripts/70_rebuild/final_eval.py` calls
                              `accuracy`
    eval_<datasource>_pass<k> the mean over problems of the best of the k
                              samples, that is pass@k, written only when k > 1

and hands them to `_maybe_dump_eval_metrics_jsonl`. That writer has no
`--eval_metrics_jsonl_path` option on the command line, so the path is always
the fallback: the evaluation dump with `.metrics.jsonl` appended to the file
name. The dump itself is `<run_dir>/eval_during_train.jsonl`, so the curve of a
run is

    <run_dir>/eval_during_train.jsonl.metrics.jsonl

one JSON object per evaluation point, carrying `global_step`, `metrics`,
`temperature`, `n_samples_per_prompt` and `duration_sec`. `datasource` is the
name of the suite the problem came from (`c3/task/datasets.py` writes the
column from the `eval_suites` entry of the task yaml), so the suite names of
the curve are the task's: MATH500, Minerva-Math, AMC23, AIME24, AIME25. The
final record merges the two AIME years into one suite of 60 problems and the
curve does not, which is why a point of this document keeps the years apart.

The fallback when that file did not reach the results tree
----------------------------------------------------------
The trainer writes the metrics file into its own run directory, and an archive
step that copies only `run_config.json`, `budget_ledger.jsonl`, `c3_task.yaml`
and `train.log` into
the results tree leaves that one behind. On a run whose container is gone the metrics
file is therefore unrecoverable, and the only surviving copy of the same
numbers is the line the same evaluation pass logged:

    ... global_step 78, eval_metrics: {'eval_MATH500_pass4': 0.94,
                                       'eval_MATH500_pass1': 0.8875, ...}

So when the metrics file yields no point, this script reads `train.log` of the
same directory instead, takes the dictionary of that line as a Python literal,
and builds points of exactly the shape the metrics file produces. The document
says which of the two it read in its top-level `source`, and a point from the
log carries no `k`, `temperature` or `duration_sec`, because the logged line
carries none: they are left out rather than guessed. Nothing about the
accuracies changes, the log line and the metrics file hold the same two
readings of the same pass.

Both accuracies are stored in percent, multiplied by 100 once, here, which is
what `c3.analysis.rebuild.aggregate_e2` does with the accuracies of the final
record.

The problem count behind a point
--------------------------------
`evaluate` counts the problems of each suite while it forms the means and then
drops the count: `metrics[datasource]["count"]` never reaches `logs`, so the
metrics file has no `n_questions`. The count is recoverable from the other file
the same pass writes, `<run_dir>/eval_during_train.jsonl`, one row per problem
and sample with its `global_step` and `datasource`; this script counts the
distinct `question_id` per suite there. When that file is absent the count is
reported as `null` and named in `not_recorded`, never guessed.

This document feeds curves only. Preregistration revision 18 (d) keeps the
monitoring evaluation out of every reported number: four samples per problem at
temperature 0.7 is not the evaluation of record, which is
`scripts/70_rebuild/final_eval.py`.

Only the standard library is imported.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .summary import source_path

__all__ = [
    "SCHEMA",
    "METRICS_NAME",
    "QUESTIONS_NAME",
    "LOG_NAME",
    "METRIC_RE",
    "LOG_METRIC_RE",
    "PROVENANCE",
    "split_metric_name",
    "read_metrics_file",
    "read_log_metrics",
    "count_questions",
    "read_run",
    "main",
]

#: Schema of the document this script writes.
SCHEMA = "c3.e2_curves/1"

#: The two files of the monitoring evaluation, as the trainer names them.
METRICS_NAME = "eval_during_train.jsonl.metrics.jsonl"
QUESTIONS_NAME = "eval_during_train.jsonl"

#: The trainer log, the fallback source of the same points. It is the file the
#: archive step of the platform does copy into the results tree, which is why a
#: run whose container is gone still has its curve here and nowhere else. Spelt
#: the same as `c3.analysis.rebuild.e2_tokens.LOG_NAME`, which reads the token
#: account off the other kind of line in the same file.
LOG_NAME = "train.log"

#: `eval_MATH500_pass4` -> suite MATH500, k 4. The suite part is greedy and the
#: `_pass<k>` tail is anchored at the end, so a suite name holding an underscore
#: survives.
METRIC_RE = re.compile(r"^eval_(?P<suite>.+)_pass(?P<k>\d+)$")

#: The tail of the line `openrlhf/trainer/ppo_trainer_plugins.py:evaluate` logs
#: after each evaluation pass. Anything may precede it: Ray prefixes the stdout
#: of an actor with `(PPOTrainer pid=118945)` and the logger prefixes a level, a
#: timestamp and a source location, so the head is searched rather than
#: anchored. The dictionary runs greedily to the last brace of the line, which
#: is where the message ends.
LOG_METRIC_RE = re.compile(
    r"global_step\s+(?P<step>\d+),\s*eval_metrics:\s*(?P<payload>\{.*\})")

#: The row kind `_maybe_dump_eval_metrics_jsonl` stamps on every line it writes.
METRICS_KIND = "eval_metrics"

PROVENANCE: "Dict[str, Dict[str, str]]" = {
    "points[].global_step": {
        "file": METRICS_NAME,
        "field": "global_step",
        "how": "the optimizer step the evaluation ran after",
    },
    "points[].suites[].accuracy_pct": {
        "file": METRICS_NAME,
        "field": "metrics.eval_<suite>_pass1",
        "how": ("avg@k over the problems of the suite, a fraction of one in the file, "
                "multiplied by 100 here"),
    },
    "points[].suites[].pass_at_k_pct": {
        "file": METRICS_NAME,
        "field": "metrics.eval_<suite>_pass<k>",
        "how": ("pass@k over the same problems, written only when k > 1, a fraction of one in "
                "the file, multiplied by 100 here"),
    },
    "points[].suites[].n_questions": {
        "file": QUESTIONS_NAME,
        "field": "distinct question_id per (global_step, datasource)",
        "how": ("counted here, because evaluate() drops its own problem count before writing "
                "the metrics file"),
    },
    "points[].k": {
        "file": METRICS_NAME,
        "field": "n_samples_per_prompt",
        "how": "the samples per problem of the monitoring evaluation (--eval_n_samples_per_prompt)",
    },
    "points[].temperature": {
        "file": METRICS_NAME,
        "field": "temperature",
        "how": "the sampling temperature of the monitoring evaluation (--eval_temperature)",
    },
    "points[].duration_sec": {
        "file": METRICS_NAME,
        "field": "duration_sec",
        "how": "wall clock of that evaluation pass",
    },
    "points[] when source is train.log": {
        "file": LOG_NAME,
        "field": "the `global_step <n>, eval_metrics: {...}` line",
        "how": ("the same two readings of the same pass, logged by "
                "openrlhf/trainer/ppo_trainer_plugins.py:evaluate. Read only when the metrics "
                "file yields no point; the line carries the step and the metrics dictionary and "
                "nothing else, so k, temperature and duration_sec are absent from such a point"),
    },
    "source": {
        "file": "%s or %s" % (METRICS_NAME, LOG_NAME),
        "field": "which of the two supplied the points",
        "how": ("the metrics file when it carries points, the trainer log when it does not and "
                "the log does"),
    },
}

#: What the monitoring evaluation does not record, whatever the run did.
STRUCTURAL_GAPS = (
    {
        "quantity": "per_question_correctness_of_a_curve_point",
        "why": ("the metrics file carries suite means only. The per-problem rows are in "
                "eval_during_train.jsonl, which the platform does not copy into the results "
                "tree"),
        "would_need": "that file beside the metrics file",
    },
    {
        "quantity": "the_merged_AIME_suite",
        "why": ("the task yaml lists AIME24 and AIME25 as two evaluation suites, so the curve "
                "has two points per step where the final record has one merged suite of 60 "
                "problems"),
        "would_need": ("merging the two years at read time, which is a decision about what the "
                       "curve plots and not a missing measurement"),
    },
    {
        "quantity": "tokens_of_the_evaluation_pass",
        "why": ("this script reports what the run scored, not what it spent. The two sums are "
                "in the same metrics record as the accuracies, as eval_total_tokens and "
                "eval_generated_tokens (openrlhf/trainer/ppo_trainer_plugins.py:evaluate), for "
                "runs launched after that record existed; this script does not read them"),
        "would_need": "reading those two keys here, on a run whose metrics record carries them",
    },
)


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------


def split_metric_name(name: str) -> Optional[Dict[str, Any]]:
    """`{suite, k}` of one `eval_<suite>_pass<k>` entry, or None."""
    match = METRIC_RE.match(str(name))
    if match is None:
        return None
    suite = match.group("suite")
    if not suite:
        return None
    return {"suite": suite, "k": int(match.group("k"))}


def _fraction(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return value


def _point(row: Mapping[str, Any]) -> Dict[str, Any]:
    """One evaluation point, with its suites in name order."""
    point: Dict[str, Any] = {"global_step": int(row.get("global_step") or 0)}
    for name, key in (("ts", "ts"), ("duration_sec", "duration_sec"),
                      ("temperature", "temperature"), ("n_samples_per_prompt", "k")):
        value = row.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            point[key] = int(value) if key == "k" else float(value)

    metrics = row.get("metrics")
    suites: "Dict[str, Dict[str, Any]]" = {}
    unreadable: List[str] = []
    if isinstance(metrics, Mapping):
        for name, value in metrics.items():
            split = split_metric_name(name)
            if split is None:
                unreadable.append(str(name))
                continue
            fraction = _fraction(value)
            if fraction is None:
                unreadable.append(str(name))
                continue
            entry = suites.setdefault(split["suite"], {"accuracy_pct": None, "pass_at_k_pct": None,
                                                       "k_of_pass_at_k": None, "n_questions": None})
            if split["k"] == 1:
                entry["accuracy_pct"] = fraction * 100.0
            else:
                entry["pass_at_k_pct"] = fraction * 100.0
                entry["k_of_pass_at_k"] = split["k"]
    point["suites"] = {name: suites[name] for name in sorted(suites)}
    if unreadable:
        point["unreadable_metrics"] = sorted(unreadable)
    return point


def read_metrics_file(path: str) -> Dict[str, Any]:
    """Every evaluation point of one metrics file, in step order.

    A step written more than once keeps its last row, the convention the
    results layout already states for the evaluation records; the writer dedupes
    per process, so a second row for a step means a resumed run.
    """
    out: Dict[str, Any] = {
        "path": source_path(path),
        "present": os.path.isfile(path),
        "rows": 0,
        "n_unparsed": 0,
        "n_other_kind": 0,
        "n_duplicate_steps": 0,
        "points": [],
    }
    if not out["present"]:
        out["error"] = "no %s in the run directory" % os.path.basename(path)
        return out

    by_step: "Dict[int, Dict[str, Any]]" = {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    out["n_unparsed"] += 1
                    continue
                if not isinstance(row, dict):
                    out["n_unparsed"] += 1
                    continue
                if str(row.get("kind") or "") != METRICS_KIND:
                    out["n_other_kind"] += 1
                    continue
                step = row.get("global_step")
                if not isinstance(step, int) or isinstance(step, bool):
                    out["n_unparsed"] += 1
                    continue
                out["rows"] += 1
                if step in by_step:
                    out["n_duplicate_steps"] += 1
                by_step[step] = _point(row)
    except OSError as exc:
        out["error"] = str(exc)
        return out

    out["points"] = [by_step[step] for step in sorted(by_step)]
    return out


def read_log_metrics(path: str) -> Dict[str, Any]:
    """Every evaluation point printed into one trainer log, in step order.

    The same document shape as `read_metrics_file`, read off the line the
    trainer logs after each evaluation pass instead of off the file the metrics
    writer dumps. The dictionary of that line is a Python literal, because the
    logger formats the dict with `%s`, so it is read with `ast.literal_eval`,
    which evaluates no call and no name.

    A point from here carries only what the line carries: the step and the
    accuracies. `n_other_kind` stays zero, because a log line has no row kind to
    be of; the field is kept so that a caller can read either block the same
    way. A step printed more than once keeps its last line, the same rule the
    metrics file follows.
    """
    out: Dict[str, Any] = {
        "path": source_path(path),
        "present": os.path.isfile(path),
        "rows": 0,
        "n_unparsed": 0,
        "n_other_kind": 0,
        "n_duplicate_steps": 0,
        "points": [],
    }
    if not out["present"]:
        out["error"] = "no %s in the run directory" % os.path.basename(path)
        return out

    by_step: "Dict[int, Dict[str, Any]]" = {}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                match = LOG_METRIC_RE.search(line)
                if match is None:
                    continue
                try:
                    metrics = ast.literal_eval(match.group("payload"))
                except (ValueError, SyntaxError, MemoryError, RecursionError):
                    out["n_unparsed"] += 1
                    continue
                if not isinstance(metrics, dict):
                    out["n_unparsed"] += 1
                    continue
                step = int(match.group("step"))
                out["rows"] += 1
                if step in by_step:
                    out["n_duplicate_steps"] += 1
                by_step[step] = _point({"global_step": step, "metrics": metrics})
    except OSError as exc:
        out["error"] = str(exc)
        return out

    out["points"] = [by_step[step] for step in sorted(by_step)]
    return out


def count_questions(path: str) -> Dict[str, Any]:
    """`{global_step: {suite: number of distinct question ids}}` of one dump file."""
    out: Dict[str, Any] = {
        "path": source_path(path),
        "present": os.path.isfile(path),
        "rows": 0,
        "n_unparsed": 0,
        "by_step": {},
    }
    if not out["present"]:
        out["error"] = "no %s in the run directory" % os.path.basename(path)
        return out

    seen: "Dict[int, Dict[str, set]]" = {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    out["n_unparsed"] += 1
                    continue
                if not isinstance(row, dict):
                    out["n_unparsed"] += 1
                    continue
                step = row.get("global_step")
                suite = row.get("datasource")
                question = row.get("question_id")
                if not isinstance(step, int) or isinstance(step, bool):
                    continue
                if not isinstance(suite, str) or not suite:
                    continue
                if isinstance(question, bool) or not isinstance(question, int):
                    continue
                out["rows"] += 1
                seen.setdefault(int(step), {}).setdefault(suite, set()).add(int(question))
    except OSError as exc:
        out["error"] = str(exc)
        return out

    out["by_step"] = {str(step): {suite: len(ids) for suite, ids in sorted(suites.items())}
                      for step, suites in sorted(seen.items())}
    return out


def read_run(run_dir: str, *, with_counts: bool = True) -> Dict[str, Any]:
    """The whole monitoring curve of one training run directory.

    `with_counts` reads the per-problem dump to recover the problem count behind
    each suite of each point. It is one streaming pass over a file that holds
    the generated text of every evaluation sample, so it is the expensive half;
    turning it off leaves every `n_questions` at null and says so.

    The points come from the metrics file when it has any, and from `train.log`
    of the same directory when it has none; `source` names which. The log is
    read only in that second case, so a run whose metrics file is there costs
    nothing extra.
    """
    metrics = read_metrics_file(os.path.join(run_dir, METRICS_NAME))
    points = list(metrics.get("points") or [])
    source = METRICS_NAME

    fallback: Dict[str, Any] = {"path": source_path(os.path.join(run_dir, LOG_NAME)),
                                "present": None, "read": False}
    if not points:
        fallback = read_log_metrics(os.path.join(run_dir, LOG_NAME))
        fallback["read"] = True
        if fallback.get("points"):
            points = list(fallback["points"])
            source = LOG_NAME

    counts: Dict[str, Any] = {"path": source_path(os.path.join(run_dir, QUESTIONS_NAME)),
                              "present": None, "read": False}
    if with_counts:
        counts = count_questions(os.path.join(run_dir, QUESTIONS_NAME))
        counts["read"] = True
        for point in points:
            per_suite = counts.get("by_step", {}).get(str(point["global_step"]), {})
            for suite, entry in point["suites"].items():
                if suite in per_suite:
                    entry["n_questions"] = int(per_suite[suite])

    suites = sorted({suite for point in points for suite in point["suites"]})

    not_recorded: List[Dict[str, str]] = [dict(gap) for gap in STRUCTURAL_GAPS]
    if not points:
        if not metrics.get("present"):
            why = "%s says %s" % (metrics["path"], metrics.get("error", "it is not there"))
            would_need = ("that file beside the run, or the points in the trainer log. The "
                          "trainer writes the metrics file into its own run directory; an "
                          "archive step that copies only "
                          "run_config.json, budget_ledger.jsonl, c3_task.yaml and train.log "
                          "into the results tree leaves this one behind, so on such a tree it "
                          "is wherever the run directory was backed up")
        else:
            why = "%s carries no %s row" % (metrics["path"], METRICS_KIND)
            would_need = "a run that reached its first evaluation point"
        not_recorded.append({
            "quantity": "the whole monitoring curve",
            "why": "%s, and %s" % (why, _fallback_why(fallback)),
            "would_need": would_need,
        })
    if not with_counts:
        not_recorded.append({
            "quantity": "n_questions of every suite",
            "why": "the per-problem dump was not read (--no_counts)",
            "would_need": "the same call without --no_counts",
        })
    elif not counts.get("present"):
        not_recorded.append({
            "quantity": "n_questions of every suite",
            "why": "%s says %s" % (counts["path"], counts.get("error", "it is not there")),
            "would_need": "the per-problem dump of the same evaluation passes",
        })

    return {
        "schema": SCHEMA,
        "run_dir": source_path(run_dir),
        "source": source,
        "n_points": len(points),
        "suites": suites,
        "points": points,
        "metrics_file": {name: metrics[name] for name in metrics if name != "points"},
        "log_fallback": {name: fallback[name] for name in fallback if name != "points"},
        "questions_file": {name: counts[name] for name in counts if name != "by_step"},
        "provenance": PROVENANCE,
        "not_recorded": not_recorded,
    }


def _fallback_why(fallback: Mapping[str, Any]) -> str:
    """Why the trainer log did not supply the points either."""
    if not fallback.get("read"):
        return "the %s fallback was not read" % LOG_NAME
    if not fallback.get("present"):
        return "%s says %s" % (fallback["path"], fallback.get("error", "it is not there"))
    return ("%s carries no `global_step <n>, eval_metrics: {...}` line"
            % fallback["path"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m c3.analysis.rebuild.e2_curves",
        description="The monitoring curve of one E2 run directory")
    ap.add_argument("--run", required=True, help="a training run directory (…/train_s0)")
    ap.add_argument("--out", default="", help="where to write the JSON (default: stdout)")
    ap.add_argument("--no_counts", action="store_true",
                    help="do not read the per-problem dump for the problem counts")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.run):
        print("[e2_curves] ERROR: %s is not a directory" % args.run, file=sys.stderr)
        return 2

    document = read_run(args.run, with_counts=not args.no_counts)
    text = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=False)
    if args.out:
        directory = os.path.dirname(os.path.abspath(args.out))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
        print("[e2_curves] %s: %d point(s) over %d suite(s), read from %s -> %s"
              % (document["run_dir"], document["n_points"], len(document["suites"]),
                 document["source"], args.out))
    else:
        print(text)
    for gap in document["not_recorded"]:
        print("[e2_curves] not recorded: %s (%s)" % (gap["quantity"], gap["why"]), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
