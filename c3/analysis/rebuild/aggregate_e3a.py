# c3/analysis/rebuild/aggregate_e3a.py
"""Command line aggregation for E3a, the ablation bias map.

    python -m c3.analysis.rebuild.aggregate_e3a \
        --results 20_data/results/E3a --manifest results/manifest.json \
        --out 20_data/results/E3a/summary.json [--extractor math]

Directory layout, from the results layout section 2 as revised on 2026-09-15
(revision 3):

    E3a/<workflow>/<model>/empty/buckets.jsonl
    E3a/<workflow>/<model>/placeholder/buckets.jsonl

`placeholder` is the directory name the cell driver writes. `deleted` is the
name the first version of the layout used for the same arm; it is accepted as
a legacy alias, with one line on stderr when it is what was found.

The empty arm is the reported one (the frozen analysis plan, note of 2026-09-13:
the empty message is the main reading and the placeholder message is reported only
as a paired difference), so every key but `E3a.null_forms.paired_p` comes from
it. A missing second arm drops that one key and says so on stderr; it is not
written as null.

The answer extractor is `--extractor math`, the repository's own math parser
followed by its expression normaliser (`influence.math_extractor`). `boxed` is
kept as an explicit option, but it is a placeholder implementation, for tests:
it reads only a balanced \boxed{...} and on the real buckets it finds nothing on
about half the downstream outputs.

This script does not touch `manifest.json`. It reads the key names from it,
refuses any key the manifest does not have, and never writes a verdict key:
filling a verdict is the maintainers' action.
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
    math_extractor,
    read_jsonl,
    source_path,
    write_summary,
)

EXTRACTORS: Dict[str, Callable[[str], Optional[str]]] = {
    "math": math_extractor,
    "boxed": boxed_extractor,
}
DEFAULT_EXTRACTOR = "math"

BUCKETS = "buckets.jsonl"

#: The second arm's directory name, and the name the first revision of the
#: results layout used for it. The alias is read, never written.
NULL_ARM_DIR = "placeholder"
NULL_ARM_DIR_LEGACY = "deleted"


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


def find_null_arm(results: str, *, stream=None) -> Optional[str]:
    """Locate the second arm: `placeholder`, else its legacy name `deleted`.

    The results layout revision of 2026-09-15 renamed the arm, and the cell driver has
    written `placeholder` since. Bucket trees produced before that carry the old
    name, so it is still read; finding one says so on stderr, because a summary
    built off a legacy tree is worth noticing.
    """
    stream = sys.stderr if stream is None else stream
    path = find_arm(results, NULL_ARM_DIR)
    if path is not None:
        return path
    legacy = find_arm(results, NULL_ARM_DIR_LEGACY)
    if legacy is not None:
        print("[aggregate_e3a] using the legacy arm name %s: %s"
              % (NULL_ARM_DIR_LEGACY, legacy), file=stream)
    return legacy


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
    null_path = find_null_arm(results, stream=stream)

    points, skipped = decision_points(read_jsonl(empty_path), extractor)
    if not points:
        raise ValueError("no usable decision point in %s" % empty_path)
    dropped = sum(skipped.values())
    if dropped:
        print("[aggregate_e3a] %s: %d buckets skipped %s" % (empty_path, dropped, skipped),
              file=stream)

    report = bias_map_report(points, stream=stream)
    paired = None
    if null_path is None:
        print("[aggregate_e3a] no %s arm (nor its legacy name %s) under %s: "
              "E3a.null_forms.paired_p not written"
              % (NULL_ARM_DIR, NULL_ARM_DIR_LEGACY, results), file=stream)
    else:
        other, other_skipped = decision_points(read_jsonl(null_path), extractor)
        paired = null_forms_paired(points, other)
        if paired["n_pairs"] == 0:
            print("[aggregate_e3a] the empty and %s arms share no question id: "
                  "E3a.null_forms.paired_p not written" % NULL_ARM_DIR, file=stream)
        other_dropped = sum(other_skipped.values())
        if other_dropped:
            print("[aggregate_e3a] %s: %d buckets skipped %s"
                  % (null_path, other_dropped, other_skipped), file=stream)

    keys = e3a_key_values(report, null_forms=paired)
    sources = [source_path(empty_path)]
    for key, entry in keys.items():
        if key == "E3a.null_forms.paired_p" and null_path is not None:
            entry["source"] = sources + [source_path(null_path)]
        else:
            entry["source"] = list(sources)

    return write_summary(out_path, "E3a", "c3/analysis/rebuild/aggregate_e3a.py",
                         keys, manifest)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Aggregate E3a into summary.json")
    ap.add_argument("--results", required=True, help="the E3a results directory")
    ap.add_argument("--manifest", required=True,
                    help="the results manifest of the paper build")
    ap.add_argument("--out", required=True, help="where summary.json is written")
    ap.add_argument("--extractor", default=DEFAULT_EXTRACTOR, choices=sorted(EXTRACTORS),
                    help="answer extractor: math is the repository parser plus its expression "
                         "normaliser; boxed is a placeholder implementation, for tests")
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
