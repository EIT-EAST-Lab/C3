# c3/analysis/rebuild/aggregate_e1c_temp.py
"""
E1c aggregation for the temperature sweep and the alert-band subsets.

    python -m c3.analysis.rebuild.aggregate_e1c_temp \
        --results 20_data/results/E1c --manifest results/manifest.json --out summary.json

Directory layout (the results layout, section 2):

    <results>/temp/{t070_n4,t085_n4,t100_n4,t100_n8}/buckets.jsonl
    <results>/band/{calib,holdout}/buckets.jsonl

The sweep cells fill E1c.temp.<condition>.dup_rate and .rel; the sweep's
directory names carry the branching as well, and the manifest's condition names
drop it for the three branching-4 cells (t070_n4 is the condition t070), so the
mapping is spelled out in TEMP_CONDITIONS below.

The two band subsets are measured (reliability and duplicate rate, reported on
stderr) but nothing is written for E1c.band.trigger_rel: the trigger is a rule
over those two numbers and defining it is the maintainers' action, not this
script's.

The training-monitor keys of E1c (E1c.dupcheck.*) come from training runs, not
from these cells, and are not this script's business.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence

from . import duplicates as dup
from . import splithalf as sh
from .aggregate_e1 import (
    BUCKET_FILE,
    MIN_CANDS,
    N_BOOT,
    N_RANDOM_SPLITS,
    SEED,
    SummaryBuilder,
    build_parser,
    load_manifest,
    rel_note,
    source_path,
)

__all__ = ["TEMP_CONDITIONS", "BAND_SUBSETS", "build_e1c_summary", "main"]

# Directory name -> manifest condition name.
TEMP_CONDITIONS = {
    "t070_n4": "t070",
    "t085_n4": "t085",
    "t100_n4": "t100",
    "t100_n8": "t100_n8",
}
BAND_SUBSETS = ("calib", "holdout")


def _subdirs(path: str) -> List[str]:
    return sorted(d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)))


def _measure(path: str):
    buckets = sh.load_buckets(path)
    cell = sh.cell_reliability(buckets, n_random_splits=N_RANDOM_SPLITS, n_boot=N_BOOT,
                               seed=SEED, min_cands=MIN_CANDS)
    report = dup.cell_duplicate_report(buckets)
    return cell, report


def build_e1c_summary(results_root: str, manifest: Mapping[str, Any], *, stream=None
                      ) -> SummaryBuilder:
    builder = SummaryBuilder("E1c", "c3.analysis.rebuild.aggregate_e1c_temp", manifest,
                             stream=stream)
    if not os.path.isdir(results_root):
        raise ValueError("results root %s is not a directory" % results_root)

    for group in _subdirs(results_root):
        if group not in ("temp", "band"):
            builder.note("ignored: %s is not a temp or band tree" % source_path(results_root, group))

    # 1. the temperature sweep
    temp_root = os.path.join(results_root, "temp")
    if os.path.isdir(temp_root):
        for name in _subdirs(temp_root):
            where = source_path(results_root, "temp", name)
            cond = TEMP_CONDITIONS.get(name)
            if cond is None:
                builder.note("ignored: %s is not a sweep condition" % where)
                continue
            path = os.path.join(temp_root, name, BUCKET_FILE)
            if not os.path.isfile(path):
                builder.note("ignored: %s has no %s" % (where, BUCKET_FILE))
                continue
            cell, report = _measure(path)
            src = source_path(results_root, "temp", name, BUCKET_FILE)
            if report["n"]:
                builder.add("E1c.temp.%s.dup_rate" % cond, report["dup_rate_mean"],
                            n=report["n"], source=src,
                            note="mean over groups of the share of alternatives that repeat "
                                 "another alternative exactly; near-duplicate rate %s"
                                 % ("%.4f" % report["near_dup_rate_mean"]
                                    if report["near_dup_rate_mean"] is not None else "n/a"))
            else:
                builder.note("skip E1c.temp.%s.dup_rate: no group has two alternatives" % cond)
            if cell.n_groups:
                builder.add("E1c.temp.%s.rel" % cond, cell.rel_evenodd, n=cell.n_groups,
                            source=src, note=rel_note(cell))
            else:
                builder.note("skip E1c.temp.%s.rel: no group survives the exclusion rule "
                             "(%d collected)" % (cond, cell.n_total))
    else:
        builder.note("ignored: %s has no temp tree" % results_root)

    # 2. the two band subsets: measured and reported, not written
    band_root = os.path.join(results_root, "band")
    if os.path.isdir(band_root):
        for name in _subdirs(band_root):
            where = source_path(results_root, "band", name)
            if name not in BAND_SUBSETS:
                builder.note("ignored: %s is not a band subset" % where)
                continue
            path = os.path.join(band_root, name, BUCKET_FILE)
            if not os.path.isfile(path):
                builder.note("ignored: %s has no %s" % (where, BUCKET_FILE))
                continue
            cell, report = _measure(path)
            builder.note(
                "band subset %s: rel=%s (n_groups=%d of %d, 95%% interval [%s, %s]), "
                "dup_rate=%s (n=%d)"
                % (name,
                   "%.4f" % cell.rel_evenodd if cell.rel_evenodd is not None else "n/a",
                   cell.n_groups, cell.n_total,
                   "%.4f" % cell.rel_ci_lo if cell.rel_ci_lo is not None else "n/a",
                   "%.4f" % cell.rel_ci_hi if cell.rel_ci_hi is not None else "n/a",
                   "%.4f" % report["dup_rate_mean"] if report["dup_rate_mean"] is not None else "n/a",
                   report["n"]))
        builder.note(
            "skip E1c.band.trigger_rel: the two subsets above are measured, but the rule that "
            "turns them into a trigger level is a judgment the maintainers make")

    builder.note("E1c: %d key(s) written, %d refused" % (len(builder.keys), len(builder.refused)))
    return builder


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser("E1c").parse_args(argv)
    manifest = load_manifest(args.manifest)
    builder = build_e1c_summary(args.results, manifest)
    builder.write(args.out)
    print("wrote %d key(s) to %s" % (len(builder.keys), args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
