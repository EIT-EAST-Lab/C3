# c3/analysis/rebuild/e3a_strata.py
"""The stratified ablation bias of E3a: the numbers of Table 9.

    python -m c3.analysis.rebuild.e3a_strata \\
        --results 20_data/results/E3a/a3/4b --out e3a_strata_summary.json

E3a measures what removing an agent's message does to the value of a decision
point (paper arXiv:2603.06859v3, Section 4 and Appendix B.6). Table 9 splits
the decision points by reward sensitivity within the high-influence half: the
points whose influence is at or above the median, split by whether the sampled
alternatives differed in value (delta Q > 0) or not. For each of the two strata
it reports the count, the mean bias, the median, the interquartile range and
how many points carry zero bias.

`aggregate_e3a` already computes all of them through `bias_map.bias_map_report`
and emits the per-stratum location and spread only when asked
(`bias_map.e3a_key_values(..., extra_strata_keys=True)`). This module makes
that call on the same buckets with the same extractor, keeps the
`E3a.strata.*` keys, and recomputes the nine location and spread numbers a
second time with plain numpy from the per-point bias vector and the same two
masks, so that the summary is only written when both readings agree.

`--results` is the cell directory holding `empty/buckets.jsonl` (the reported
null form); the layout `scripts/70_rebuild/e3a_cells.py` writes. The summary is
standalone: it needs no results manifest.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from . import bias_map
from .aggregate_e3a import DEFAULT_EXTRACTOR, EXTRACTORS, find_arm
from .influence import read_jsonl
from .summary import git_sha, source_path

__all__ = ["TABLE9_KEYS", "LOCATION_KEYS", "strata_keys", "independent_strata", "main"]

STRATA = ("high_infl_diff", "high_infl_nodiff")

#: The location and spread of the two strata: the nine keys the default
#: aggregation leaves out, and the ones recomputed independently here.
LOCATION_KEYS = tuple(
    ["E3a.strata.%s.%s" % (name, field)
     for name in STRATA
     for field in ("mean_bias", "median", "iqr_lo", "iqr_hi")]
    + ["E3a.strata.high_infl_nodiff.zero_n"]
)

#: Every key Table 9 is read off: the counts, the zero-bias counts and the
#: location and spread of the two strata.
TABLE9_KEYS = tuple(
    ["E3a.strata.%s.n" % name for name in STRATA]
    + ["E3a.strata.high_infl_diff.zero_n"]
    + list(LOCATION_KEYS)
)

SCRIPT = "c3.analysis.rebuild.e3a_strata"


def strata_keys(points: Sequence[Mapping[str, Any]], *, stream=None) -> Dict[str, Dict[str, Any]]:
    """The Table 9 keys from decision points, through the aggregation's own code."""
    report = bias_map.bias_map_report(points, stream=stream)
    keys = bias_map.e3a_key_values(report, extra_strata_keys=True)
    return {k: keys[k] for k in TABLE9_KEYS if k in keys}


def independent_strata(points: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """The location and spread again, from the per-point biases with plain numpy."""
    bias = np.asarray([float(p["bias"]) for p in points], dtype=float)
    infl = np.asarray([float(p["influence"]) for p in points], dtype=float)
    dq = np.asarray([float(p["delta_q"]) for p in points], dtype=float)
    high = infl >= np.median(infl)  # the aggregation's default side, "ge"
    masks = {"high_infl_diff": high & (dq > 0.0), "high_infl_nodiff": high & ~(dq > 0.0)}
    out: Dict[str, Any] = {}
    for name, mask in masks.items():
        b = np.sort(bias[mask])
        if not b.size:
            continue
        out["E3a.strata.%s.n" % name] = int(b.size)
        out["E3a.strata.%s.mean_bias" % name] = float(b.mean())
        out["E3a.strata.%s.median" % name] = float(np.median(b))
        # numpy's default percentile (linear interpolation between order
        # statistics), the rule the aggregation uses.
        out["E3a.strata.%s.iqr_lo" % name] = float(np.percentile(b, 25))
        out["E3a.strata.%s.iqr_hi" % name] = float(np.percentile(b, 75))
        out["E3a.strata.%s.zero_n" % name] = int(np.sum(np.abs(b) < bias_map.ZERO_TOL))
    return out


def _same(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is b
    return abs(float(a) - float(b)) <= 1e-12 * max(1.0, abs(float(b)))


def check(keys: Mapping[str, Mapping[str, Any]], again: Mapping[str, Any]) -> List[str]:
    """The location keys on which the two readings disagree."""
    return [k for k in LOCATION_KEYS
            if k in keys and not _same(keys[k].get("value"), again.get(k))]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m c3.analysis.rebuild.e3a_strata",
        description="The stratified E3a ablation bias (Table 9) into a summary JSON.",
    )
    ap.add_argument("--results", required=True,
                    help="the E3a cell directory (or a root above it) holding empty/buckets.jsonl")
    ap.add_argument("--out", required=True, help="where the summary JSON is written")
    ap.add_argument("--extractor", default=DEFAULT_EXTRACTOR, choices=sorted(EXTRACTORS),
                    help="answer extractor, as in aggregate_e3a")
    args = ap.parse_args(argv)

    try:
        empty = find_arm(args.results, "empty")
    except ValueError as exc:
        print("[e3a_strata] %s" % exc, file=sys.stderr)
        return 2
    if empty is None:
        print("[e3a_strata] no empty arm found under %s" % args.results, file=sys.stderr)
        return 2

    points, skipped = bias_map.decision_points(read_jsonl(empty), EXTRACTORS[args.extractor])
    if not points:
        print("[e3a_strata] no usable decision point in %s" % empty, file=sys.stderr)
        return 2
    if sum(skipped.values()):
        print("[e3a_strata] %s: buckets skipped %s" % (empty, skipped), file=sys.stderr)

    keys = strata_keys(points)
    differs = check(keys, independent_strata(points))
    if differs:
        print("[e3a_strata] the independent recomputation disagrees on: %s" % ", ".join(differs),
              file=sys.stderr)
        return 1

    src = [source_path(empty)]
    doc = {
        "experiment": "E3a",
        "generated_by": {
            "script": SCRIPT,
            "git_sha": git_sha(),
            "when": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "n_points": len(points),
        "skipped_buckets": skipped,
        "keys": {
            k: {"value": keys[k]["value"], "n": keys[k]["n"], "source": list(src),
                **({"note": keys[k]["note"]} if keys[k].get("note") else {})}
            for k in TABLE9_KEYS if k in keys
        },
    }
    parent = os.path.dirname(os.path.abspath(args.out))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    for k in TABLE9_KEYS:
        if k in keys:
            print("  %-44s %r  (n=%s)" % (k, keys[k]["value"], keys[k]["n"]))
    print("[e3a_strata] wrote %d keys to %s" % (len(doc["keys"]), args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
