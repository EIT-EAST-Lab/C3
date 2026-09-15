# c3/analysis/rebuild/aggregate_e3a.py
"""Command line aggregation for E3a, the ablation bias map.

    python -m c3.analysis.rebuild.aggregate_e3a \
        --results 20_data/results/E3a --manifest 10_paper/04_rebuild/results/manifest.json \
        --out 20_data/results/E3a/summary.json [--extractor boxed]

Directory layout, from the results contract section 2:

    E3a/<workflow>/<model>/empty/buckets.jsonl
    E3a/<workflow>/<model>/deleted/buckets.jsonl

The empty arm is the reported one (preregistration, note of 2026-09-13: the
empty message is the main reading and the deleted paragraph is reported only as
a paired difference), so every key but `E3a.null_forms.paired_p` comes from it.
A missing deleted arm drops that one key and says so on stderr; it is not
written as null.

This script does not touch `manifest.json`. It reads the key names from it,
refuses any key the manifest does not have, and never writes a verdict key:
filling a verdict is the driver's action.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Callable, Dict, List, Optional

from c3.analysis.rebuild.bias_map import (
    bias_map_report,
    decision_points,
    e3a_key_values,
    null_forms_paired,
)
from c3.analysis.rebuild.influence import (
    boxed_extractor,
    load_manifest,
    read_jsonl,
    source_path,
    write_summary,
)

EXTRACTORS: Dict[str, Callable[[str], Optional[str]]] = {"boxed": boxed_extractor}

BUCKETS = "buckets.jsonl"


def find_arm(results: str, arm: str) -> Optional[str]:
    """Locate one arm's bucket file under the results root.

    Primary layout is `<results>/<workflow>/<model>/<arm>/buckets.jsonl`. If
    `--results` already points at the model level, `<results>/<arm>/buckets.jsonl`
    is accepted as well. Several matches is an error, not a silent pick.
    """
    hits: List[str] = []
    for workflow in sorted(os.listdir(results)) if os.path.isdir(results) else []:
        wpath = os.path.join(results, workflow)
        if not os.path.isdir(wpath):
            continue
        for model in sorted(os.listdir(wpath)):
            candidate = os.path.join(wpath, model, arm, BUCKETS)
            if os.path.isfile(candidate):
                hits.append(candidate)
    direct = os.path.join(results, arm, BUCKETS)
    if os.path.isfile(direct):
        hits.append(direct)
    if len(hits) > 1:
        raise ValueError("several %s arms under %s: %s" % (arm, results, ", ".join(hits)))
    return hits[0] if hits else None


def run(
    results: str,
    manifest_path: str,
    out_path: str,
    extractor: Callable[[str], Optional[str]],
    *,
    stream=None,
) -> Dict[str, Any]:
    """Aggregate and write summary.json. Returns the written document."""
    stream = sys.stderr if stream is None else stream
    manifest = load_manifest(manifest_path)

    empty_path = find_arm(results, "empty")
    if empty_path is None:
        raise FileNotFoundError("no empty arm found under %s" % results)
    deleted_path = find_arm(results, "deleted")

    points, skipped = decision_points(read_jsonl(empty_path), extractor)
    if not points:
        raise ValueError("no usable decision point in %s" % empty_path)
    dropped = sum(skipped.values())
    if dropped:
        print("[aggregate_e3a] %s: %d buckets skipped %s" % (empty_path, dropped, skipped),
              file=stream)

    report = bias_map_report(points)
    paired = None
    if deleted_path is None:
        print("[aggregate_e3a] no deleted arm under %s: E3a.null_forms.paired_p not written"
              % results, file=stream)
    else:
        other, other_skipped = decision_points(read_jsonl(deleted_path), extractor)
        paired = null_forms_paired(points, other)
        if paired["n_pairs"] == 0:
            print("[aggregate_e3a] empty and deleted arms share no question id: "
                  "E3a.null_forms.paired_p not written", file=stream)
        other_dropped = sum(other_skipped.values())
        if other_dropped:
            print("[aggregate_e3a] %s: %d buckets skipped %s"
                  % (deleted_path, other_dropped, other_skipped), file=stream)

    keys = e3a_key_values(report, null_forms=paired)
    sources = [source_path(empty_path)]
    for key, entry in keys.items():
        if key == "E3a.null_forms.paired_p" and deleted_path is not None:
            entry["source"] = sources + [source_path(deleted_path)]
        else:
            entry["source"] = list(sources)

    return write_summary(out_path, "E3a", "c3/analysis/rebuild/aggregate_e3a.py",
                         keys, manifest)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Aggregate E3a into summary.json")
    ap.add_argument("--results", required=True, help="the E3a results directory")
    ap.add_argument("--manifest", required=True, help="10_paper/04_rebuild/results/manifest.json")
    ap.add_argument("--out", required=True, help="where summary.json is written")
    ap.add_argument("--extractor", default="boxed", choices=sorted(EXTRACTORS),
                    help="answer extractor; the driver injects the repository parser through run()")
    args = ap.parse_args(argv)

    try:
        doc = run(args.results, args.manifest, args.out, EXTRACTORS[args.extractor])
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print("[aggregate_e3a] %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return 2
    print("[aggregate_e3a] wrote %d keys to %s" % (len(doc["keys"]), args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
