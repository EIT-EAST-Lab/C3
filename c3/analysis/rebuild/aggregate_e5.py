# c3/analysis/rebuild/aggregate_e5.py
"""Command line aggregation for E5, the paired policy contrast.

    python -m c3.analysis.rebuild.aggregate_e5 \
        --results 20_data/results/E5 --manifest results/manifest.json \
        --out 20_data/results/E5/summary.json --gold datasets/math500_gold.jsonl \
        [--extractor boxed]

Directory layout, from the results layout section 2:

    E5/sft/buckets.jsonl        the reference arm, written to the .sft keys
    E5/c3_s0/buckets.jsonl      the trained arm, written to the .c3 keys

Correctness here is string equality of the extracted answer symbol with the gold
answer, whitespace stripped. That is a placeholder: real mathematical
equivalence lives in the repository's own parser, and the caller injects it by
calling `run()` with its own `extractor` and `is_correct` instead of using the
`--extractor` flag.

This script does not touch `manifest.json`. It reads the key names from it,
refuses any key the manifest does not have, and never writes a verdict key.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Callable, Dict, List, Mapping, Optional

from c3.analysis.rebuild.coupling import arm_scan, coupling_report, e5_key_values
from c3.analysis.rebuild.influence import (
    boxed_extractor,
    load_manifest,
    read_jsonl,
    source_path,
    write_summary,
)

EXTRACTORS: Dict[str, Callable[[str], Optional[str]]] = {"boxed": boxed_extractor}

BUCKETS = "buckets.jsonl"
ARMS = ("sft", "c3_s0")

#: Field names accepted for the gold answer, in order of preference.
GOLD_FIELDS = ("gold_answer", "gold", "answer", "label")


def read_gold(path: str) -> Dict[Any, str]:
    """Read the gold file: one JSON object per line, question id to answer.

    Two shapes are accepted, because the results layout does not fix one: an object
    carrying `question_id` plus one of the fields in `GOLD_FIELDS`
    (gold_answer, gold, answer, label), or a single-pair object {id: answer}.
    Anything else raises with the line number instead of being guessed at.
    """
    gold: Dict[Any, str] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("%s line %d: not a JSON object" % (path, lineno))
            if "question_id" in row:
                qid = row["question_id"]
                for field in GOLD_FIELDS:
                    if field in row:
                        gold[qid] = str(row[field])
                        break
                else:
                    raise ValueError("%s line %d: no gold field among %s"
                                     % (path, lineno, ", ".join(GOLD_FIELDS)))
            elif len(row) == 1:
                qid, value = next(iter(row.items()))
                gold[qid] = str(value)
            else:
                raise ValueError("%s line %d: cannot tell which field is the gold answer"
                                 % (path, lineno))
    return gold


def make_is_correct(gold: Mapping[Any, str]) -> Callable[[str, Any], bool]:
    """Placeholder correctness: the answer symbol equals the gold, stripped."""

    def is_correct(symbol: str, question_id: Any) -> bool:
        want = gold.get(question_id)
        if want is None:
            return False
        return str(symbol).strip() == str(want).strip()

    return is_correct


def _arm_path(results: str, arm: str) -> str:
    path = os.path.join(results, arm, BUCKETS)
    if not os.path.isfile(path):
        raise FileNotFoundError("missing %s arm: %s" % (arm, path))
    return path


def run(
    results: str,
    manifest_path: str,
    out_path: str,
    extractor: Callable[[str], Optional[str]],
    is_correct: Callable[[str, Any], bool],
    *,
    known_questions: Optional[set] = None,
    stream=None,
) -> Dict[str, Any]:
    """Aggregate and write summary.json. Returns the written document.

    `known_questions`, when given, keeps only buckets whose question id is in
    it. With the placeholder correctness that is the set of questions the gold
    file covers: a question with no gold would otherwise count every upstream
    alternative as wrong and inflate the correction denominator.
    """
    stream = sys.stderr if stream is None else stream
    manifest = load_manifest(manifest_path)
    paths = {arm: _arm_path(results, arm) for arm in ARMS}

    scans: Dict[str, Any] = {}
    dropped = {}
    for arm, path in paths.items():
        buckets = read_jsonl(path)
        if known_questions is not None:
            kept = [b for b in buckets if b.get("question_id") in known_questions]
            dropped[arm] = len(buckets) - len(kept)
            buckets = kept
        else:
            dropped[arm] = 0
        scans[arm], counters = arm_scan(buckets, extractor, is_correct)
        if dropped[arm]:
            print("[aggregate_e5] %s: %d buckets dropped for having no gold answer"
                  % (path, dropped[arm]), file=stream)
        if counters["duplicate_question_id"] or counters["no_question_id"]:
            print("[aggregate_e5] %s: %d duplicate and %d missing question ids"
                  % (path, counters["duplicate_question_id"], counters["no_question_id"]),
                  file=stream)

    report = coupling_report(scans["sft"], scans["c3_s0"])
    if report["n_questions"] == 0:
        raise ValueError("the two arms under %s share no question id" % results)

    keys = e5_key_values(report)
    sources = [source_path(paths["sft"]), source_path(paths["c3_s0"])]
    for entry in keys.values():
        entry["source"] = list(sources)
    if "E5.n_questions" in keys and any(dropped.values()):
        keys["E5.n_questions"]["note"] += "; %d and %d buckets had no gold answer" % (
            dropped["sft"], dropped["c3_s0"])

    return write_summary(out_path, "E5", "c3/analysis/rebuild/aggregate_e5.py",
                         keys, manifest)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Aggregate E5 into summary.json")
    ap.add_argument("--results", required=True, help="the E5 results directory")
    ap.add_argument("--manifest", required=True,
                    help="the results manifest of the paper build")
    ap.add_argument("--out", required=True, help="where summary.json is written")
    ap.add_argument("--gold", required=True, help="jsonl of question id to gold answer")
    ap.add_argument("--extractor", default="boxed", choices=sorted(EXTRACTORS),
                    help="answer extractor; the caller injects the repository parser through run()")
    args = ap.parse_args(argv)

    try:
        gold = read_gold(args.gold)
        doc = run(args.results, args.manifest, args.out, EXTRACTORS[args.extractor],
                  make_is_correct(gold), known_questions=set(gold))
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print("[aggregate_e5] %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return 2
    print("[aggregate_e5] wrote %d keys to %s" % (len(doc["keys"]), args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
