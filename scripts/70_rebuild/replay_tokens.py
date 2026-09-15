#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Count what one replay of the deep chain regenerates downstream, in tokens.

This produces the one E1 key the aggregation cannot. It is separate from
`c3.analysis.rebuild.aggregate_e1` because it needs a tokenizer and the model
files, which the aggregation deliberately does not: the aggregation reads bucket
files with numpy and scipy and runs anywhere, this runs where the model lives.

    python scripts/70_rebuild/replay_tokens.py --dry-run \\
        --results 20_data/results/E1

    python scripts/70_rebuild/replay_tokens.py \\
        --results /abs/20_data/results/E1 \\
        --tokenizer /models/Qwen3-4B-Instruct-2507 \\
        --manifest /abs/results/manifest.json \\
        --out /abs/20_data/results/E1/replay_tokens_summary.json

Layout read (the results layout, section 2):

    <results>/c10/<model>/sweep_n{2,3,4,6,8}/buckets.jsonl

WHAT IS COUNTED, AND WHY IT IS NOT THE PREFIX (the maintainers' decision of
2026-09-15).
The statistic is the number of tokens the downstream role regenerates in ONE
replay, taken over every replay of every alternative of every bucket of the c10
cells. The key it fills is still called `E1.c10.median_prefix_tokens`, which is
the name the manifest carries; renaming the key is the maintainers' action, not
this script's.

The measured position of every E1 cell is the FIRST role in topological order
(`scripts/70_rebuild/e1_cells.py`, `measured_positions`), so the upstream prefix
at that position is always just the question: it is the same text for every
alternative and every workflow, and its length says nothing about depth. What
does make the ten-agent chain expensive, and what makes its credit estimate
noisy, is how much text has to be regenerated downstream each time an
alternative is replayed. That is the quantity here.

WHAT THE BUCKET FILES CARRY (measured on 2026-09-15 over the twelve
`buckets.jsonl` on disk, 9799 buckets and 28426 alternatives). Every
`candidates[j]` has `next_actions` as a `list[str]` and `returns` as a
`list[float]` of the same length, one entry per replay, which is also what the
bucket schema guard in `c3/analysis/buckets.py` requires. So `next_actions[k]`
is the downstream text of replay k, and `returns[k]` its return.

One limit, worth knowing before the number is read: `next_actions` records the
output of the NEXT role only (`c3/analysis/replay.py` appends `captured_next`,
which is `cfg.next_role`), while a replay of a ten-agent chain regenerates every
role after the target. The count below is therefore the first downstream role's
regenerated text, which is a lower bound on what the replay actually costs.
`record_next_teammate` off would leave `next_actions` empty altogether, and the
dry run says so per cell rather than quietly reporting a median of nothing.

`--dry-run` reports the cells, their buckets, their replays and how many of
those replays carry downstream text, without a tokenizer and without importing
transformers, which is how it is testable on a machine with neither.

TWO CONVENTIONS ARE NOT PINNED and are therefore flags, never defaults taken
silently: which median an even count takes (`--median`, both readings always
reported in the note), and whether the tokenizer's special tokens are counted
(`--add_special_tokens`, off).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from c3.analysis.rebuild.aggregate_e1 import (  # noqa: E402
    APPENDIX_KEY_PREFIX,
    BUCKET_FILE,
    DEFAULT_KEY_PREFIX,
    MODELS,
    SWEEP_N,
    key_prefix_refusal,
    source_path,
)
from c3.analysis.rebuild.summary import load_manifest, write_summary  # noqa: E402

#: The workflow this key is about, and the key itself. The key name is the
#: manifest's; see the module docstring.
WORKFLOW = "c10"
KEY_TAIL = "c10.median_prefix_tokens"

#: How several role outputs are joined when one replay records more than one.
#: No bucket on disk does (`next_actions[k]` is a string), and the bucket schema
#: guard refuses anything else, so this is a convention for a shape the writer
#: cannot currently produce. It is the join `c3.mas.prompt_render` uses.
TEXT_JOIN = "\n\n"


# ---------------------------------------------------------------------------
# reading the cells
# ---------------------------------------------------------------------------


def find_cells(results_root: str, workflow: str = WORKFLOW) -> Dict[Tuple[str, int], str]:
    """Map (model, branching factor) to that cell's bucket file.

    Only the sweep cells of one workflow, and only the model and branching
    names the results layout defines; anything else is not this script's business.
    """
    cells: Dict[Tuple[str, int], str] = {}
    wf_dir = os.path.join(results_root, workflow)
    if not os.path.isdir(wf_dir):
        return cells
    for model in sorted(os.listdir(wf_dir)):
        if model not in MODELS or not os.path.isdir(os.path.join(wf_dir, model)):
            continue
        for rule in sorted(os.listdir(os.path.join(wf_dir, model))):
            if not rule.startswith("sweep_n"):
                continue
            try:
                n = int(rule[len("sweep_n"):])
            except ValueError:
                continue
            if n not in SWEEP_N:
                continue
            path = os.path.join(wf_dir, model, rule, BUCKET_FILE)
            if os.path.isfile(path):
                cells[(model, n)] = path
    return cells


def read_buckets(path: str) -> List[Dict[str, Any]]:
    """One JSON object per line, blank lines skipped."""
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def downstream_text(entry: Any) -> Optional[str]:
    """The regenerated downstream text of one replay, or None when there is none.

    A `next_actions` entry is a string on every bucket written so far. A mapping
    or a list is read as several downstream roles of the same replay and their
    texts are joined with :data:`TEXT_JOIN`, in the order the container gives
    them, so that a future writer recording the whole tail of the chain is
    counted rather than dropped. An entry of any other type, and an entry that
    is empty once stripped, is not a text.
    """
    if isinstance(entry, str):
        return entry if entry.strip() else None
    if isinstance(entry, Mapping):
        parts = [str(v) for v in entry.values() if isinstance(v, str) and v.strip()]
    elif isinstance(entry, (list, tuple)):
        parts = [str(v) for v in entry if isinstance(v, str) and v.strip()]
    else:
        return None
    joined = TEXT_JOIN.join(parts)
    return joined if joined.strip() else None


def bucket_replays(bucket: Mapping[str, Any]) -> Tuple[List[str], int, Dict[str, int]]:
    """(texts, replays recorded, reasons) of one bucket.

    A replay is one entry of `candidates[j].returns`, which is what the runner
    actually ran; `next_actions[k]` is the downstream text of that same replay.
    Every alternative counts: the credit-set convention of `meta.credit_n`
    selects the alternatives the ESTIMATOR uses, and this statistic is about what
    a replay costs, not about the estimate.
    """
    texts: List[str] = []
    reasons: Dict[str, int] = {}
    n_replays = 0

    cands = bucket.get("candidates") or []
    if not isinstance(cands, list):
        reasons["bucket has no candidates list"] = 1
        return texts, 0, reasons

    for cand in cands:
        if not isinstance(cand, Mapping):
            reasons["candidate is not an object"] = reasons.get(
                "candidate is not an object", 0) + 1
            continue
        nexts = cand.get("next_actions")
        nexts = nexts if isinstance(nexts, list) else []
        rets = cand.get("returns")
        rets = rets if isinstance(rets, list) else []
        here = max(len(rets), len(nexts))
        n_replays += here
        for k in range(here):
            text = downstream_text(nexts[k]) if k < len(nexts) else None
            if text is None:
                why = ("candidate.next_actions is empty, so no downstream text was recorded"
                       if not nexts else
                       "replay has no downstream text in candidate.next_actions")
                reasons[why] = reasons.get(why, 0) + 1
                continue
            texts.append(text)

    return texts, n_replays, reasons


# ---------------------------------------------------------------------------
# counting
# ---------------------------------------------------------------------------


def load_tokenizer(path: str) -> Any:
    """`transformers.AutoTokenizer` for a local model directory.

    Imported here rather than at module level, so `--dry-run` needs neither
    transformers nor the model files.
    """
    from transformers import AutoTokenizer  # noqa: WPS433 (deliberately local)

    return AutoTokenizer.from_pretrained(path, trust_remote_code=False)


def count_tokens(tokenizer: Any, text: str, *, add_special_tokens: bool = False) -> int:
    """Length in tokens of one string."""
    return len(tokenizer(text, add_special_tokens=add_special_tokens)["input_ids"])


def medians(counts: Sequence[int]) -> Tuple[Optional[int], Optional[float]]:
    """(low median, linear median) of the counts, (None, None) when there are none.

    The low median is an observed count and therefore an integer, which is the
    format the manifest gives this key; the linear median is the ordinary one.
    Both are reported, because which of the two the paper prints is not settled.
    """
    values = sorted(int(c) for c in counts)
    if not values:
        return None, None
    return int(statistics.median_low(values)), float(statistics.median(values))


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------


def scan(results_root: str) -> Dict[str, Any]:
    """Read every c10 cell and report what a replay length could be read off.

    Returns {cells, n_buckets, n_replays, n_usable, reasons, texts}. `texts` is
    the downstream text of every replay that recorded one, in cell order, which
    is what the tokenizer is then handed.
    """
    cells = find_cells(results_root)
    reasons: Dict[str, int] = {}
    texts: List[str] = []
    rows: List[Dict[str, Any]] = []
    n_buckets = 0
    n_replays = 0

    for (model, n) in sorted(cells):
        path = cells[(model, n)]
        buckets = read_buckets(path)
        cell_replays = 0
        cell_usable = 0
        for bucket in buckets:
            bucket_texts, replays, bucket_reasons = bucket_replays(bucket)
            cell_replays += replays
            cell_usable += len(bucket_texts)
            texts.extend(bucket_texts)
            for why, count in bucket_reasons.items():
                reasons[why] = reasons.get(why, 0) + count
        n_buckets += len(buckets)
        n_replays += cell_replays
        rows.append({
            "model": model,
            "n": n,
            "source": source_path(results_root, WORKFLOW, model, "sweep_n%d" % n, BUCKET_FILE),
            "buckets": len(buckets),
            "replays": cell_replays,
            "with_text": cell_usable,
        })

    return {
        "cells": rows,
        "n_buckets": n_buckets,
        "n_replays": n_replays,
        "n_usable": len(texts),
        "reasons": reasons,
        "texts": texts,
    }


def run(args: argparse.Namespace, *, stream=None) -> int:
    """Scan, then either report the scan or write the summary."""
    out = sys.stderr if stream is None else stream
    scan_result = scan(args.results)

    for row in scan_result["cells"]:
        print("[replay_tokens] %s: %d bucket(s), %d replay(s), %d with downstream text"
              % (row["source"], row["buckets"], row["replays"], row["with_text"]), file=out)
    for why, count in sorted(scan_result["reasons"].items()):
        print("[replay_tokens] %d replay(s) contribute nothing: %s" % (count, why), file=out)

    if not scan_result["cells"]:
        print("[replay_tokens] no %s cell under %s" % (WORKFLOW, args.results), file=out)

    if args.dry_run:
        print("# cells: %d" % len(scan_result["cells"]))
        print("# buckets: %d" % scan_result["n_buckets"])
        print("# replays: %d" % scan_result["n_replays"])
        print("# with downstream text: %d" % scan_result["n_usable"])
        return 0

    if scan_result["n_usable"] == 0:
        print("[replay_tokens] ERROR: no replay carries downstream text, so there is "
              "nothing to count", file=out)
        return 2

    tokenizer = load_tokenizer(args.tokenizer)
    counts = [count_tokens(tokenizer, text, add_special_tokens=args.add_special_tokens)
              for text in scan_result["texts"]]
    median_low, median_linear = medians(counts)
    value = median_low if args.median == "low" else int(round(median_linear))
    mean_tokens = sum(counts) / float(len(counts))
    tokenizer_name = os.path.basename(str(args.tokenizer).rstrip("/\\"))

    key = args.key_prefix + "." + KEY_TAIL
    note = ("tokens regenerated downstream per replay, tokenizer %s, special tokens %s; "
            "low median %d, linear median %.1f, mean %.1f; min %d, max %d; "
            "%d of %d recorded replays carried downstream text"
            % (tokenizer_name,
               "counted" if args.add_special_tokens else "not counted",
               median_low, median_linear, mean_tokens, min(counts), max(counts),
               len(counts), scan_result["n_replays"]))
    keys = {key: {
        "value": int(value),
        "n": len(counts),
        "source": [row["source"] for row in scan_result["cells"]],
        "note": note,
    }}

    manifest = load_manifest(args.manifest)
    doc = write_summary(args.out, "E1", "scripts/70_rebuild/replay_tokens.py", keys, manifest,
                        stream=out,
                        extra={"results_root": source_path(args.results),
                               "key_prefix": args.key_prefix,
                               "replay_tokens": {
                                   "statistic": "tokens regenerated downstream in one replay",
                                   "tokenizer": tokenizer_name,
                                   "add_special_tokens": bool(args.add_special_tokens),
                                   "median_low": int(median_low),
                                   "median_linear": float(median_linear),
                                   "mean_tokens": float(mean_tokens),
                                   "min_tokens": int(min(counts)),
                                   "max_tokens": int(max(counts)),
                                   "n_cells": len(scan_result["cells"]),
                                   "n_buckets": int(scan_result["n_buckets"]),
                                   "n_replays": int(scan_result["n_replays"]),
                                   "n_replays_counted": len(counts),
                               }})
    print("wrote %d key(s) to %s" % (len(doc["keys"]), args.out))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/70_rebuild/replay_tokens.py",
        description="Median length in tokens of what one deep-chain replay regenerates "
                    "downstream.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results", required=True, help="Root of the E1 result tree.")
    parser.add_argument("--tokenizer", default="",
                        help="Local model directory for transformers.AutoTokenizer.")
    parser.add_argument("--manifest", default="",
                        help="The paper's results/manifest.json.")
    parser.add_argument("--out", default="", help="summary.json to write.")
    parser.add_argument("--key_prefix", default=DEFAULT_KEY_PREFIX,
                        help="Leading segment of the key written; %s for the appendix tree."
                             % APPENDIX_KEY_PREFIX)
    parser.add_argument("--median", default="low", choices=("low", "linear"),
                        help="Which median an even number of replays takes. Both are "
                             "reported in the note either way.")
    parser.add_argument("--add_special_tokens", action="store_true",
                        help="Count the tokenizer's special tokens as part of the text.")
    parser.add_argument("--dry-run", "--dry_run", dest="dry_run", action="store_true",
                        help="Report what the buckets carry and write nothing. Needs neither "
                             "transformers nor the model files.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    refusal = key_prefix_refusal(args.results, args.key_prefix)
    if refusal is not None:
        print("[replay_tokens] ERROR: %s" % refusal, file=sys.stderr)
        return 2
    if not args.dry_run:
        missing = [name for name in ("tokenizer", "manifest", "out")
                   if not getattr(args, name)]
        if missing:
            print("[replay_tokens] ERROR: a real run needs %s"
                  % ", ".join("--" + name for name in missing), file=sys.stderr)
            return 2

    try:
        return run(args)
    except (OSError, ValueError, KeyError) as exc:
        print("[replay_tokens] ERROR: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
