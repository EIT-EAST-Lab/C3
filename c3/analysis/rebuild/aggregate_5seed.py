# c3/analysis/rebuild/aggregate_5seed.py
"""
The 5-seed runs: Table 2 and Table 8 of the paper, and the paired test of Table 10.

    python -m c3.analysis.rebuild.aggregate_5seed --results <records root> \
        --out summary.json [--table tables.md] [--manifest configs/results_keys_v3.json]

The paper is "The Trace Is the State: Exact Credit Assignment for LLM Agent
Teams" (arXiv:2603.06859v3). Its 5-seed runs train each method under 5
training seeds (`scripts/40_train/paper_train.sh`, `RECIPE=main` for the
2-agent workflow and `RECIPE=a3` for the 3-agent one) and evaluate every trained
model once, greedily, under a 512-token generation limit, on MATH500, AIME 2025,
CMATH and GSM8K (`scripts/70_rebuild/final_eval.py` with one sample per problem
on every suite).

Directory layout, the one `aggregate_e2` reads:

    <results>/<arm>/<method>/train_s<k>/eval_s<j>/final_eval_record.json

`arm` is `a2_4b` (the 2-agent workflow, Table 2) or `a3_4b` (the 3-agent
workflow, Table 8); `method` is `mappo`, `magrpo` or `c3`; `k` is the training
seed. The protocol has one evaluation per training seed, `eval_s0`. A training
seed with more than one evaluation record is reduced to the mean of them first,
and the summary says so.

What is computed
----------------
- Per arm, method and benchmark: the mean and the sample standard deviation
  (one degree of freedom taken off) over the training seeds of the accuracy in
  percent. The accuracy of a suite is the `accuracy` of the record's suite
  entry times 100. AIME 2025 is the record's own `AIME25` suite, the 30
  problems of 2025, not the merged AIME entry.
- `Avg.`: per training seed, the mean of MATH500, CMATH and GSM8K, then the mean
  and standard deviation of those over the seeds. AIME 2025 is reported beside
  the average, not inside it, as in the tables.
- Table 10, on the 2-agent workflow only: the strongest baseline is the one of
  MAPPO and MAGRPO with the higher MATH500 mean. C3 minus that baseline is
  paired by training seed over the seeds both carry, tested against zero with
  the two-sided paired t test, with the 95 percent t interval of the mean
  difference, and the 4 raw p values (MATH500, AIME 2025, CMATH, GSM8K) are
  corrected together by the Holm step-down procedure (`aggregate_e2.holm`,
  `aggregate_e2.paired_stats`).

What is written
---------------
`--out` is a summary document in the shape every aggregation of this package
writes: keys named `E2main.<arm>.<bench>.<method>.mean`, and so on, each with
its value, n (the training seeds behind it), unit and sources.
Only the keys the `--manifest` defines are written; the default is
`configs/results_keys_v3.json`, the keys of the paper with no values. Beside the
keys, a `tables` block carries every number unrounded with the per-seed values
it came from, and `problems` lists what does not match the protocol (a record
at more than one sample per problem, or at a generation limit other than 512).

`--table` writes Tables 2, 8 and 10 as Markdown at the printed precision: one
decimal for a mean and a standard deviation, two for a difference and its
interval, and the p value tiers of `summary.format_p`.

The training-token column of Table 8 is not aggregated, and the key table has
no key for it.

A damaged record (it does not parse, its schema is not `c3.final_eval.record/1`,
it lacks a required field, or it says `ok` is not true) stops the run: nothing
is written and the exit code is `aggregate_e2.EXIT_BROKEN_RECORDS`.

Only numpy, scipy and the standard library are imported (through this package).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .aggregate_e1 import SummaryBuilder
from .aggregate_e2 import (
    EXIT_BROKEN_RECORDS,
    _read_record,
    _sample_std,
    _seed_of,
    _subdirs,
    holm,
    paired_stats,
)
from .summary import MANIFEST_HELP, RESULTS_KEYS_V3, format_p, load_manifest, source_path

__all__ = [
    "ARMS",
    "METHODS",
    "METHOD_NAMES",
    "BENCHES",
    "AVG_PARTS",
    "HOLM_FAMILY",
    "BASELINES",
    "TESTED_ARM",
    "PROTOCOL_K",
    "PROTOCOL_GENERATE_MAX_LEN",
    "scan_records",
    "seed_table",
    "build_tables",
    "build_summary_keys",
    "render_markdown",
    "main",
]

REPO_ROOT = Path(__file__).resolve().parents[3]

#: Arm directory -> the table of the paper it is.
ARMS = OrderedDict([("a2_4b", "Table 2"), ("a3_4b", "Table 8")])

#: Method directories, in the row order of the tables.
METHODS = ("mappo", "magrpo", "c3")
METHOD_NAMES = {"mappo": "MAPPO", "magrpo": "MAGRPO", "c3": "C3"}

#: Bench segment of a key -> (suite of the evaluation record, column title), in
#: the column order of Tables 2 and 8.
BENCHES = OrderedDict([
    ("math500", ("MATH500", "MATH500")),
    ("aime25", ("AIME25", "AIME 2025")),
    ("cmath", ("CMATH-test", "CMATH")),
    ("gsm8k", ("GSM8K-test", "GSM8K")),
])

#: The average column: MATH500, CMATH and GSM8K, taken inside each seed.
AVG_BENCH = "avg3"
AVG_PARTS = ("math500", "cmath", "gsm8k")

#: Table 10: the four benchmarks tested as one family, on the 2-agent workflow.
HOLM_FAMILY = ("math500", "aime25", "cmath", "gsm8k")
BASELINES = ("mappo", "magrpo")
TESTED_ARM = "a2_4b"

#: The evaluation of the 5-seed runs: one greedy sample per problem, 512 tokens.
PROTOCOL_K = 1
PROTOCOL_GENERATE_MAX_LEN = 512

EXPERIMENT = "E2main"
SCRIPT = "c3.analysis.rebuild.aggregate_5seed"
RECORD_NAME = "final_eval_record.json"


class _Notes:
    """The `note` sink `aggregate_e2._read_record` writes its refusals to."""

    def __init__(self, stream) -> None:
        self.stream = stream

    def note(self, message: str) -> None:
        print(message, file=self.stream)


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------


def scan_records(results_root: str, *, stream=None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """(runs, broken): every readable record under the tree, and the damaged ones.

    A run is {arm, method, train_seed, eval_seed, path, accuracy_pct, k,
    generate_max_len}; `accuracy_pct` maps a bench segment to percent and holds
    the benches the record carries. Directories this layout does not define are
    passed over with one line on stderr.
    """
    out_stream = stream if stream is not None else sys.stderr
    notes = _Notes(out_stream)
    if not os.path.isdir(results_root):
        raise ValueError("results root %s is not a directory" % results_root)

    runs: List[Dict[str, Any]] = []
    broken: List[Dict[str, Any]] = []
    for arm in _subdirs(results_root):
        if arm not in ARMS:
            notes.note("ignored: %s is not an arm of the 5-seed runs" % source_path(os.path.join(results_root, arm)))
            continue
        for method in _subdirs(os.path.join(results_root, arm)):
            if method not in METHODS:
                notes.note("ignored: %s is not a method of the 5-seed runs"
                           % source_path(os.path.join(results_root, arm, method)))
                continue
            method_dir = os.path.join(results_root, arm, method)
            for train_name in _subdirs(method_dir):
                train_seed = _seed_of(train_name, "train_s")
                if train_seed is None:
                    notes.note("ignored: %s is not a training seed directory"
                               % source_path(os.path.join(method_dir, train_name)))
                    continue
                for eval_name in _subdirs(os.path.join(method_dir, train_name)):
                    eval_seed = _seed_of(eval_name, "eval_s")
                    path = os.path.join(method_dir, train_name, eval_name, RECORD_NAME)
                    if eval_seed is None or not os.path.isfile(path):
                        notes.note("ignored: no %s in %s" % (RECORD_NAME, source_path(os.path.dirname(path))))
                        continue
                    damage: List[str] = []
                    record = _read_record(path, notes, damage)
                    if record is None:
                        for reason in damage:
                            broken.append({"path": source_path(path), "reason": reason})
                        continue
                    accuracy, ks = _accuracies(record)
                    if not accuracy:
                        broken.append({"path": source_path(path), "reason": "carries no suite accuracy"})
                        continue
                    runs.append({
                        "arm": arm,
                        "method": method,
                        "train_seed": int(train_seed),
                        "eval_seed": int(eval_seed),
                        "path": source_path(path),
                        "accuracy_pct": accuracy,
                        "k": ks,
                        "generate_max_len": record.get("generate_max_len"),
                    })
    return runs, broken


def _accuracies(record: Mapping[str, Any]) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Bench segment -> accuracy in percent, and bench segment -> the record's k."""
    suites = record.get("suites")
    accuracy: Dict[str, float] = {}
    ks: Dict[str, Any] = {}
    if not isinstance(suites, Mapping):
        return accuracy, ks
    for bench, (suite, _title) in BENCHES.items():
        entry = suites.get(suite)
        if not isinstance(entry, Mapping):
            continue
        value = entry.get("accuracy")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            continue
        accuracy[bench] = float(value) * 100.0
        ks[bench] = entry.get("k")
    return accuracy, ks


def seed_table(runs: Sequence[Mapping[str, Any]]) -> Dict[Tuple[str, str, str], Dict[int, float]]:
    """(arm, method, bench) -> {training seed: percent}, with `avg3` per seed.

    A training seed evaluated more than once is the mean of its evaluations.
    """
    by_eval: Dict[Tuple[str, str, str], Dict[int, List[float]]] = {}
    for run in runs:
        for bench, value in run["accuracy_pct"].items():
            by_eval.setdefault((run["arm"], run["method"], bench), {}).setdefault(run["train_seed"], []).append(value)
    table: Dict[Tuple[str, str, str], Dict[int, float]] = {}
    for cell, by_seed in by_eval.items():
        table[cell] = {seed: sum(values) / len(values) for seed, values in by_seed.items()}

    for arm in ARMS:
        for method in METHODS:
            parts = [table.get((arm, method, bench), {}) for bench in AVG_PARTS]
            if not all(parts):
                continue
            seeds = sorted(set.intersection(*[set(part) for part in parts]))
            if seeds:
                table[(arm, method, AVG_BENCH)] = {
                    seed: sum(part[seed] for part in parts) / len(parts) for seed in seeds
                }
    return table


# ---------------------------------------------------------------------------
# the tables
# ---------------------------------------------------------------------------


def _cell(by_seed: Mapping[int, float]) -> Dict[str, Any]:
    values = [by_seed[seed] for seed in sorted(by_seed)]
    return {
        "mean": sum(values) / len(values),
        "std": _sample_std(values),
        "n": len(values),
        "per_seed": {str(seed): by_seed[seed] for seed in sorted(by_seed)},
    }


def _significance(table: Mapping[Tuple[str, str, str], Mapping[int, float]]) -> Optional[Dict[str, Any]]:
    """Table 10: C3 minus the strongest baseline on the 2-agent workflow, or None."""
    ranked = []
    for method in BASELINES:
        cell = table.get((TESTED_ARM, method, "math500"))
        if cell:
            ranked.append((-(sum(cell.values()) / len(cell)), method))
    if not ranked or not table.get((TESTED_ARM, "c3", "math500")):
        return None
    ranked.sort()
    best = ranked[0][1]

    rows: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    for bench in HOLM_FAMILY:
        c3 = table.get((TESTED_ARM, "c3", bench), {})
        base = table.get((TESTED_ARM, best, bench), {})
        shared = sorted(seed for seed in c3 if seed in base)
        if not shared:
            continue
        pair = paired_stats([c3[s] for s in shared], [base[s] for s in shared])
        pair["seeds"] = shared
        rows[bench] = pair

    family = [bench for bench, pair in rows.items() if pair.get("p") is not None]
    corrected = dict(zip(family, holm([rows[bench]["p"] for bench in family])))
    for bench, pair in rows.items():
        pair["p_holm"] = corrected.get(bench)
        pair["holm_family_size"] = len(family)
    return {"arm": TESTED_ARM, "strongest_baseline": best, "benches": rows}


def build_tables(runs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Every number of Tables 2, 8 and 10, unrounded, with what it was computed from."""
    table = seed_table(runs)
    out: Dict[str, Any] = OrderedDict()
    for arm, paper_table in ARMS.items():
        rows: "OrderedDict[str, Any]" = OrderedDict()
        for method in METHODS:
            cells: "OrderedDict[str, Any]" = OrderedDict()
            for bench in list(BENCHES) + [AVG_BENCH]:
                by_seed = table.get((arm, method, bench))
                if by_seed:
                    cells[bench] = _cell(by_seed)
            if cells:
                rows[method] = cells
        if rows:
            out[arm] = {"paper": paper_table, "rows": rows}
    significance = _significance(table)
    if significance is not None:
        out["significance"] = dict(significance, paper="Table 10")
    return out


def protocol_problems(runs: Sequence[Mapping[str, Any]]) -> List[str]:
    """Records that were not evaluated the way the 5-seed runs are."""
    problems: List[str] = []
    for run in runs:
        off_k = sorted(bench for bench, k in run["k"].items() if k != PROTOCOL_K)
        if off_k:
            problems.append("%s: %s evaluated at k != %d (greedy is one sample per problem)"
                            % (run["path"], ", ".join(off_k), PROTOCOL_K))
        if run.get("generate_max_len") != PROTOCOL_GENERATE_MAX_LEN:
            problems.append("%s: generation limit %r, the 5-seed runs use %d"
                            % (run["path"], run.get("generate_max_len"), PROTOCOL_GENERATE_MAX_LEN))
    return problems


# ---------------------------------------------------------------------------
# the summary keys
# ---------------------------------------------------------------------------


def _sources(runs: Sequence[Mapping[str, Any]], arm: str, methods: Sequence[str]) -> List[str]:
    return sorted({run["path"] for run in runs if run["arm"] == arm and run["method"] in methods})


def _multi_eval_note(runs: Sequence[Mapping[str, Any]], arm: str, method: str) -> str:
    count: Dict[int, int] = {}
    for run in runs:
        if run["arm"] == arm and run["method"] == method:
            count[run["train_seed"]] = count.get(run["train_seed"], 0) + 1
    many = sorted(seed for seed, c in count.items() if c > 1)
    if not many:
        return ""
    return "; training seed(s) %s carry more than one evaluation and are the mean of them" % (
        ", ".join(str(s) for s in many))


def build_summary_keys(builder: SummaryBuilder, runs: Sequence[Mapping[str, Any]],
                       tables: Mapping[str, Any]) -> None:
    """Add the paper's keys to `builder`, which refuses the ones the manifest lacks."""
    for arm in ARMS:
        block = tables.get(arm)
        if not block:
            continue
        workflow = "2-agent" if arm == "a2_4b" else "3-agent"
        for method, cells in block["rows"].items():
            sources = _sources(runs, arm, [method])
            tail = _multi_eval_note(runs, arm, method)
            for bench, cell in cells.items():
                what = ("mean of MATH500, CMATH and GSM8K taken inside each seed"
                        if bench == AVG_BENCH else "greedy accuracy")
                base = "%s.%s.%s.%s" % (EXPERIMENT, arm, bench, method)
                builder.add(base + ".mean", cell["mean"], n=cell["n"], source=sources,
                            note="%s, %s workflow, mean over %d training seed(s)%s"
                                 % (what, workflow, cell["n"], tail))
                if cell["std"] is not None:
                    builder.add(base + ".std", cell["std"], n=cell["n"], source=sources,
                                note="sample standard deviation over the training seeds%s" % tail)

    significance = tables.get("significance")
    if not significance:
        return
    arm = significance["arm"]
    best = significance["strongest_baseline"]
    sources = _sources(runs, arm, ["c3", best])
    seeds_n = len(tables[arm]["rows"].get(best, {}).get("math500", {}).get("per_seed", {}))
    builder.add("%s.%s.strongest_baseline.name" % (EXPERIMENT, arm), METHOD_NAMES[best], n=max(1, seeds_n),
                source=sources, note="the baseline with the higher MATH500 mean over the training seeds")
    for bench, pair in significance["benches"].items():
        base = "%s.%s.%s.c3_minus_best" % (EXPERIMENT, arm, bench)
        count = int(pair["n"])
        tail = "C3 minus %s, paired over training seeds %s" % (
            METHOD_NAMES[best], ", ".join(str(s) for s in pair["seeds"]))
        builder.add(base + ".diff", pair["diff"], n=count, source=sources, note=tail)
        if pair["ci_lo"] is not None:
            builder.add(base + ".ci_lo", pair["ci_lo"], n=count, source=sources,
                        note="lower 95 percent t bound; " + tail)
            builder.add(base + ".ci_hi", pair["ci_hi"], n=count, source=sources,
                        note="upper 95 percent t bound; " + tail)
        raw = format_p(pair.get("p"))
        if raw is not None:
            builder.add(base + ".p", raw, n=count, source=sources,
                        note="two-sided paired t test, raw p = %.6g; %s" % (pair["p"], tail))
        adjusted = format_p(pair.get("p_holm"))
        if adjusted is not None:
            builder.add(base + ".p_holm", adjusted, n=count, source=sources,
                        note="Holm over the %d benchmarks, corrected p = %.6g; %s"
                             % (pair["holm_family_size"], pair["p_holm"], tail))


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _mean_std(cell: Optional[Mapping[str, Any]]) -> str:
    if not cell:
        return ""
    if cell["std"] is None:
        return "%.1f" % cell["mean"]
    return "%.1f \u00b1 %.1f" % (cell["mean"], cell["std"])


def _signed(value: Optional[float]) -> str:
    return "" if value is None else "%+.2f" % value


def render_markdown(tables: Mapping[str, Any]) -> str:
    """Tables 2, 8 and 10 at the precision the paper prints them."""
    lines: List[str] = []
    for arm, paper_table in ARMS.items():
        block = tables.get(arm)
        if not block:
            continue
        workflow = "2-agent" if arm == "a2_4b" else "3-agent"
        lines.append("%s: the 5-seed run on the %s workflow, greedy accuracy (%%), mean \u00b1 std over "
                     "training seeds. Avg. is the mean of MATH500, CMATH and GSM8K." % (paper_table, workflow))
        lines.append("")
        titles = [title for _suite, title in BENCHES.values()] + ["Avg."]
        lines.append("| Method | " + " | ".join(titles) + " |")
        lines.append("|---" * (len(titles) + 1) + "|")
        for method, cells in block["rows"].items():
            row = [_mean_std(cells.get(bench)) for bench in list(BENCHES) + [AVG_BENCH]]
            lines.append("| %s | %s |" % (METHOD_NAMES[method], " | ".join(row)))
        lines.append("")

    significance = tables.get("significance")
    if significance:
        best = METHOD_NAMES[significance["strongest_baseline"]]
        lines.append("Table 10: the 5-seed run, paired differences, C3 minus %s, in points, with 95%% "
                     "intervals and raw and Holm-corrected p values over the training seeds." % best)
        lines.append("")
        lines.append("| Benchmark | Difference | 95% interval | Raw | Holm |")
        lines.append("|---|---|---|---|---|")
        for bench, pair in significance["benches"].items():
            interval = ("%s to %s" % (_signed(pair["ci_lo"]), _signed(pair["ci_hi"]))
                        if pair["ci_lo"] is not None else "")
            raw = format_p(pair.get("p"))
            adjusted = format_p(pair.get("p_holm"))
            lines.append("| %s | %s | %s | %s | %s |" % (
                BENCHES[bench][1], _signed(pair["diff"]), interval,
                "p %s" % raw if raw else "", "p %s" % adjusted if adjusted else ""))
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m c3.analysis.rebuild.aggregate_5seed",
        description="Aggregate the 5-seed runs into Tables 2, 8 and 10.",
    )
    ap.add_argument("--results", required=True, help="root of the evaluation records, <arm>/<method>/train_s<k>/eval_s<j>")
    ap.add_argument("--out", required=True, help="summary.json to write")
    ap.add_argument("--table", default="", help="optional Markdown file for Tables 2, 8 and 10")
    ap.add_argument("--manifest", default=str(REPO_ROOT / RESULTS_KEYS_V3), help=MANIFEST_HELP)
    args = ap.parse_args(argv)

    try:
        runs, broken = scan_records(args.results)
    except ValueError as exc:
        print("[aggregate_5seed] ERROR: %s" % exc, file=sys.stderr)
        return 2
    if broken:
        print("[aggregate_5seed] REFUSED: %d damaged evaluation record(s); nothing was written"
              % len(broken), file=sys.stderr)
        for item in broken:
            print("  %s: %s" % (item["path"], item["reason"]), file=sys.stderr)
        return EXIT_BROKEN_RECORDS

    manifest = load_manifest(args.manifest)
    tables = build_tables(runs)
    problems = protocol_problems(runs)
    for problem in problems:
        print("[aggregate_5seed] PROBLEM %s" % problem, file=sys.stderr)

    builder = SummaryBuilder(EXPERIMENT, SCRIPT, manifest,
                             extra={"results_root": source_path(args.results)})
    build_summary_keys(builder, runs, tables)
    builder.extra["tables"] = tables
    builder.extra["problems"] = problems
    builder.write(args.out)
    print("wrote %d key(s) to %s" % (len(builder.keys), args.out))

    if args.table:
        parent = os.path.dirname(os.path.abspath(args.table))
        os.makedirs(parent, exist_ok=True)
        with open(args.table, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(render_markdown(tables))
        print("wrote %s" % args.table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
