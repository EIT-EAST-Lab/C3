# c3/analysis/rebuild/aggregate_e2.py
"""
E2 aggregation: scan the training arms, write 20_data/results/E2/summary.json.

    python -m c3.analysis.rebuild.aggregate_e2 \
        --results 20_data/results/E2 --manifest results/manifest.json --out summary.json

Directory layout (the results layout, section 2, in the shape the platform
writes since 2026-09-19):

    <results>/<arm>/<method>/train_s<k>/eval_s<j>/final_eval_record.json
    <results>/<arm>/<method>/train_s<k>/{run_config.json, compute.json, budget_ledger.jsonl}

One record per evaluation run, written by `scripts/70_rebuild/final_eval.py`
(schema `c3.final_eval.record/1`). Its `suites` map carries `accuracy` as a
fraction of one, the avg@k over the problems of that suite. Every manifest key
this script writes is a percentage or a difference of percentages, so accuracy
is multiplied by 100 once, here, on the way in.

The replication unit
--------------------
The number of training seeds an arm has decides how the arm is read, and the
choice is explicit rather than inferred from the data's shape:

- one training seed (`interim`, preregistration revision 20, Captain
  2026-09-19): the five numbers behind a mean are the five evaluation runs of
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
  mathematics benchmarks, with `avg3` tested outside that family.

The mode is decided per arm, because an arm is one table of the paper and never
shares a row with another, and the summary records both the per-arm modes and
the single mode when every arm agrees.

What this script does not write
-------------------------------
- Verdict keys (`E2.*.paired`, `E2.a2_code.margin`). Filling a verdict is the
  maintainers' action; `summary.key_refusal` refuses the unit mechanically.
- In interim mode, `*.ci_lo`, `*.ci_hi`, `*.p` and `*.p_holm`. See above.
- `E2.depth_cost.*`, `E2.ablation.*` and `E2.a3_4b.compute.*`. The first two
  need arms this round does not have, and the third is a three-agent key that
  must not be filled with two-agent numbers. What the run directories do carry
  about compute is collected into the summary's `extras` instead, where it
  cannot be mistaken for a manifest value.

Keys the manifest does not define are skipped with a line on stderr and the exit
code stays 0, so a partly collected tree still produces the keys it supports.

Only numpy, scipy and the standard library are imported (through this package).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import OrderedDict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from scipy import stats

from .aggregate_e1 import SummaryBuilder, build_parser, source_path
from .summary import format_p, load_manifest

__all__ = [
    "ARMS",
    "METHODS",
    "METHOD_NAMES",
    "STRONGEST_BASELINE_CANDIDATES",
    "BENCH_SUITES",
    "AVG3_PARTS",
    "HOLM_FAMILY",
    "INTERIM_NOTE",
    "EvalRun",
    "scan_e2_records",
    "accuracy_table",
    "holm",
    "paired_stats",
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
STRONGEST_BASELINE_CANDIDATES = ("mappo", "magrpo", "ccpo")

#: Bench segment of a key -> the suite of the evaluation record it reads.
#: The left column is manifest_skeleton.py, the right column is final_eval.py.
BENCH_SUITES = {
    "math500": "MATH500",
    "aime25": "AIME25",
    "cmath": "CMATH-test",
    "gsm8k": "GSM8K-test",
    "mbppplus": "MBPP+",
    "mbpptest": "MBPP-test",
}

#: `avg3` is the mean of these three, taken inside each evaluation run so that
#: the pairing survives.
AVG3_PARTS = ("math500", "cmath", "gsm8k")

#: The four benchmarks Holm corrects across; `avg3` is tested outside the family.
HOLM_FAMILY = ("math500", "aime25", "cmath", "gsm8k")

#: The note every key of an interim arm carries (preregistration revision 20).
INTERIM_NOTE = "replication: 1 training seed x 5 evaluation runs (interim, Captain 2026-09-19)"

#: The record schema this script knows how to read. A later version may mean the
#: same field names differently, so it is refused rather than read hopefully.
RECORD_SCHEMA = "c3.final_eval.record/1"

#: Files a training run may leave beside its evaluation seeds.
RUN_FILES = ("run_config.json", "compute.json", "budget_ledger.jsonl")

EXPERIMENT = "E2"
SCRIPT = "c3.analysis.rebuild.aggregate_e2"


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


def _read_record(path: str, builder: SummaryBuilder) -> Optional[Dict[str, Any]]:
    """The parsed record, or None with one line on stderr."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            record = json.load(fh)
    except (OSError, ValueError) as exc:
        builder.note("ignored: %s does not parse (%s)" % (source_path(path), exc))
        return None
    if not isinstance(record, dict):
        builder.note("ignored: %s is not a JSON object" % source_path(path))
        return None
    schema = str(record.get("schema") or "")
    if schema != RECORD_SCHEMA:
        builder.note("ignored: %s has schema %r, expected %r"
                     % (source_path(path), schema, RECORD_SCHEMA))
        return None
    return record


def scan_e2_records(results_root: str, builder: SummaryBuilder,
                    *, arm_filter: Optional[str] = None) -> List[EvalRun]:
    """Every evaluation record under the tree, in arm, method, seed order.

    Anything the layout does not define is skipped with one line on stderr: an
    arm or method directory with an unknown name, a seed directory that does not
    end in a number, a missing or unreadable record. A tree that is half written
    is the normal state while the platform is still running, so nothing here is
    fatal.
    """
    runs: List[EvalRun] = []
    if not os.path.isdir(results_root):
        raise ValueError("results root %s is not a directory" % results_root)

    for arm in _subdirs(results_root):
        if arm not in ARMS:
            builder.note("ignored: %s is not an arm name" % source_path(results_root, arm))
            continue
        if arm_filter and arm != arm_filter:
            continue
        for method in _subdirs(os.path.join(results_root, arm)):
            if method not in METHODS:
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
                    record = _read_record(path, builder)
                    if record is None:
                        continue
                    accuracy, counts = _accuracies_of(record)
                    if not accuracy:
                        builder.note("ignored: %s carries no suite accuracy" % source_path(path))
                        continue
                    runs.append(EvalRun(arm, method, train_seed, eval_seed, path, accuracy, counts))
    return runs


def _accuracies_of(record: Mapping[str, Any]) -> Tuple[Dict[str, float], Dict[str, int]]:
    """Bench segment -> accuracy in percent, plus the problem count behind each.

    `avg3` is formed here, inside the run, so that a later pairing compares the
    same three benchmarks of the same run. The merged `AIME` entry of the record
    has no manifest key; it is carried under its own name for the summary's
    extras and is never written to `aime25`, which the manifest defines as
    AIME 2025.
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
        out[bench] = float(value) * 100.0
        n_questions = entry.get("n_questions")
        if isinstance(n_questions, int) and not isinstance(n_questions, bool):
            counts[bench] = n_questions

    if all(part in out for part in AVG3_PARTS):
        out["avg3"] = sum(out[part] for part in AVG3_PARTS) / len(AVG3_PARTS)

    merged = suites.get("AIME")
    if isinstance(merged, Mapping):
        value = merged.get("accuracy")
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
            out["aime_merged"] = float(value) * 100.0
            n_questions = merged.get("n_questions")
            if isinstance(n_questions, int) and not isinstance(n_questions, bool):
                counts["aime_merged"] = n_questions
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

    This is reported and never written to a manifest key. The compute keys the
    manifest defines are three-agent keys (`E2.a3_4b.compute.*`), and a
    two-agent number does not belong in them; what a two-agent run can say about
    its own cost belongs here until the manifest has a key for it.
    """
    extras: Dict[str, Any] = {}
    seen = sorted({(run.arm, run.method, run.train_seed) for run in runs})
    for arm, method, train_seed in seen:
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
        extras["%s/%s/train_s%d" % (arm, method, train_seed)] = entry
    return extras


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
    order = list(BENCH_SUITES) + ["avg3"]
    present = {bench for (cell_arm, _method, bench) in table if cell_arm == arm}
    return [bench for bench in order if bench in present]


def _methods_present(table: Mapping[Tuple[str, str, str], Any], arm: str) -> List[str]:
    present = {method for (cell_arm, method, _bench) in table if cell_arm == arm}
    return [method for method in METHODS if method in present]


def build_e2_summary(results_root: str, manifest: Mapping[str, Any], *,
                     stream=None, arm_filter: Optional[str] = None) -> SummaryBuilder:
    """Compute every E2 key the collected arms support."""
    builder = SummaryBuilder(EXPERIMENT, SCRIPT, manifest, stream=stream)
    runs = scan_e2_records(results_root, builder, arm_filter=arm_filter)
    table = accuracy_table(runs)

    arms = [arm for arm in ARMS if any(key[0] == arm for key in table)]
    modes = {arm: arm_mode(table, arm) for arm in arms}

    for arm in arms:
        mode = modes[arm]
        _write_arm(builder, table, runs, arm, mode)

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
    builder.extra["extras"] = {
        "runs": collect_run_extras(results_root, runs),
        "aime_merged": _aime_merged_extra(table, runs, modes),
        "n_records": len(runs),
    }
    return builder


def _aime_merged_extra(table, runs, modes) -> Dict[str, Any]:
    """The merged AIME reading, which the manifest has no key for.

    `final_eval.py` also reports AIME24 and AIME25 merged into one 60 problem
    entry. The manifest's `aime25` is AIME 2025, so the merged reading is kept
    here rather than written to that key.
    """
    out: Dict[str, Any] = {}
    for arm in sorted({key[0] for key in table}):
        for method in _methods_present(table, arm):
            series = series_of(table, runs, arm, method, "aime_merged", modes[arm])
            if series is None or series.mean is None:
                continue
            out["%s/%s" % (arm, method)] = {
                "mean_pct": series.mean,
                "std_pct": series.std,
                "n": series.n,
                "mode": series.mode,
            }
    return out


def _write_arm(builder: SummaryBuilder, table, runs, arm: str, mode: str) -> None:
    """Every key of one arm: the cells, the strongest baseline, the differences."""
    benches = [bench for bench in _benches_present(table, arm) if bench != "aime_merged"]
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
            builder.add(base + ".mean", cell.mean, n=cell.n, source=cell.sources,
                        note=_note(mode, "mean over the %s" % over))
            if cell.std is not None:
                builder.add(base + ".std", cell.std, n=cell.n, source=cell.sources,
                            note=_note(mode, "std over the %s" % over))

    best = _strongest_baseline(series, benches)
    if best is not None:
        _write_strongest(builder, arm, mode, series, best)

    _write_differences(builder, arm, mode, series, benches, best)
    _write_gains(builder, arm, mode, series)


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


def _write_strongest(builder: SummaryBuilder, arm: str, mode: str,
                     series: Mapping[Tuple[str, str], Optional[Series]], best: str) -> None:
    cell = series[("math500", best)]
    if cell is None or cell.mean is None:
        return
    builder.add("E2.%s.strongest_baseline.name" % arm, METHOD_NAMES[best],
                n=cell.n, source=cell.sources,
                note=_note(mode, "highest MATH500 mean among MAPPO, MAGRPO and CCPO"))
    builder.add("E2.%s.math500.best.mean" % arm, cell.mean, n=cell.n, source=cell.sources,
                note=_note(mode, "the MATH500 mean of %s" % METHOD_NAMES[best]))
    if cell.std is not None:
        builder.add("E2.%s.math500.best.std" % arm, cell.std, n=cell.n, source=cell.sources,
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


def _write_differences(builder: SummaryBuilder, arm: str, mode: str,
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
            builder.add(base + ".diff", pair["diff"], n=max(1, int(pair["n"])),
                        source=_pair_sources(series, bench, other),
                        note=_note(mode, tail))
            if mode == "interim":
                continue
            if pair["ci_lo"] is not None:
                builder.add(base + ".ci_lo", pair["ci_lo"], n=pair["n"],
                            source=_pair_sources(series, bench, other), note=_note(mode, tail))
                builder.add(base + ".ci_hi", pair["ci_hi"], n=pair["n"],
                            source=_pair_sources(series, bench, other), note=_note(mode, tail))
            if pair["p"] is not None:
                formatted = format_p(pair["p"])
                if formatted is not None:
                    builder.add(base + ".p", formatted, n=pair["n"],
                                source=_pair_sources(series, bench, other),
                                note=_note(mode, "two-sided paired t test"))
                adjusted = corrected.get(bench, pair["p"])
                inside = "Holm over %d benchmarks" % len(family) if bench in corrected \
                    else "tested outside the Holm family"
                formatted = format_p(adjusted)
                if formatted is not None:
                    builder.add(base + ".p_holm", formatted, n=pair["n"],
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


def _write_gains(builder: SummaryBuilder, arm: str, mode: str,
                 series: Mapping[Tuple[str, str], Optional[Series]]) -> None:
    """The two decomposition keys of the two-agent Qwen3-4B arm."""
    if arm != "a2_4b":
        return
    for key, left, right in (("arch_gain", "mappo", "ppo1a"), ("credit_gain", "c3", "mappo")):
        pair = _pair(series.get(("math500", left)), series.get(("math500", right)), mode)
        if pair is None:
            continue
        builder.add("E2.a2_4b.math500.%s" % key, pair["diff"], n=max(1, int(pair["n"])),
                    source=_pair_sources_for(series, "math500", left, right),
                    note=_note(mode, "%s minus %s on MATH500" % (METHOD_NAMES[left], METHOD_NAMES[right])))


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
    args = ap.parse_args(argv)

    arm_filter = str(args.arm or "").strip()
    if arm_filter and arm_filter not in ARMS:
        print("[aggregate_e2] ERROR: --arm %s is not one of %s"
              % (arm_filter, ", ".join(ARMS)), file=sys.stderr)
        return 2

    manifest = load_manifest(args.manifest)
    builder = build_e2_summary(args.results, manifest, arm_filter=arm_filter or None)
    builder.write(args.out)
    print("wrote %d key(s) to %s" % (len(builder.keys), args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
