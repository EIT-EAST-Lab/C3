#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Count the prefix the deep-chain decision points condition on, in tokens.

This produces the one E1 key the aggregation cannot:
`E1.c10.median_prefix_tokens`, "median length of the prefix the ten-agent
chain's evaluation points condition on". It is separate from
`c3.analysis.rebuild.aggregate_e1` because it needs a tokenizer and the model
files, which the aggregation deliberately does not: the aggregation reads bucket
files with numpy and scipy and runs anywhere, this runs where the model lives.

    python scripts/70_rebuild/prefix_tokens.py --dry-run \\
        --results 20_data/results/E1 --prefix_scope context

    python scripts/70_rebuild/prefix_tokens.py \\
        --results /abs/20_data/results/E1 --prefix_scope context \\
        --tokenizer /models/Qwen3-4B-Instruct-2507 \\
        --manifest 10_paper/04_rebuild/results/manifest.json \\
        --out /abs/20_data/results/E1/prefix_tokens_summary.json

Layout read (results contract section 2):

    <results>/c10/<model>/sweep_n{2,3,4,6,8}/buckets.jsonl

The summary goes through `c3.analysis.rebuild.summary.write_summary`, the one
writer of this package, so its document shape, its refusal rule and its git
stamp are the same as every other summary of the rebuild study.

WHAT THE BUCKET FILES DO NOT CARRY (state of 2026-09-15, read off the trees in
20_data/results). A bucket records `restart.question`,
`restart.role_outputs_prefix` and `restart.roles_topo`, but NOT the prompt the
target role was actually handed. The prompt is assembled at generation time by
`c3.mas.prompt_render.build_render_context` plus the role prompt template plus
the chat template, and none of the three is stored. Worse for this key, the
measured position of every cell is the FIRST role in topological order
(`scripts/70_rebuild/e1_cells.py`, `measured_positions`), so
`role_outputs_prefix` is empty in all 3350 buckets on disk: under the
`context` scope the ten-agent chain would measure a prefix of zero, which is not
what the key means.

So one of two things has to happen before this script produces a real number,
and which one is the driver's call:

  (a) the bucket code stores the rendered target-role prompt on each bucket (one
      string field, plus enough of its identity to trust it), and this script
      reads it with `--prefix_scope rendered --rendered_field <name>`; or
  (b) the measured position of the c10 cells moves down the chain, so
      `role_outputs_prefix` is non-empty, and `--prefix_scope context` (or
      `question_and_context`) measures something.

`--dry-run` reports exactly this, per cell, without a tokenizer and without
importing transformers, which is how it is testable on a machine with neither.

THREE CONVENTIONS ARE NOT YET PINNED and are therefore flags, never defaults
taken silently: which text counts as the prefix (`--prefix_scope`, required),
which median an even count takes (`--median`, both readings always reported in
the note), and whether the tokenizer's special tokens are counted
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

#: The workflow this key is about, and the key itself.
WORKFLOW = "c10"
KEY_TAIL = "c10.median_prefix_tokens"

#: What "the prefix" may mean. None of the three is a default: see the module
#: docstring.
PREFIX_SCOPES = ("context", "question_and_context", "rendered")

#: How `build_render_context` joins the outputs a role may read.
CONTEXT_JOIN = "\n\n"


# ---------------------------------------------------------------------------
# reading the cells
# ---------------------------------------------------------------------------


def find_cells(results_root: str, workflow: str = WORKFLOW) -> Dict[Tuple[str, int], str]:
    """Map (model, branching factor) to that cell's bucket file.

    Only the sweep cells of one workflow, and only the model and branching
    names the contract defines; anything else is not this script's business.
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


def _dig(obj: Any, dotted: str) -> Any:
    """Follow a dotted field name through nested mappings, None when it breaks."""
    cur = obj
    for part in str(dotted).split("."):
        if not isinstance(cur, Mapping):
            return None
        cur = cur.get(part)
    return cur


def prefix_text(bucket: Mapping[str, Any], scope: str,
                rendered_field: str = "") -> Tuple[Optional[str], str]:
    """The text of one bucket's prefix, or (None, why it is not there).

    `context` is the upstream outputs joined the way
    `c3.mas.prompt_render.build_render_context` joins them, in the topological
    order the bucket recorded. `question_and_context` puts the question in front
    of it. `rendered` reads one stored field, whose name the caller gives,
    because no such field exists yet.
    """
    if scope not in PREFIX_SCOPES:
        raise ValueError("unknown prefix scope %r; known scopes: %s"
                         % (scope, ", ".join(PREFIX_SCOPES)))
    restart = bucket.get("restart") or {}
    if not isinstance(restart, Mapping):
        return None, "bucket has no restart block"

    if scope == "rendered":
        if not rendered_field:
            return None, "--prefix_scope rendered needs --rendered_field"
        value = _dig(bucket, rendered_field)
        if not isinstance(value, str) or not value:
            return None, "no %s on this bucket" % rendered_field
        return value, ""

    outputs = restart.get("role_outputs_prefix")
    if not isinstance(outputs, Mapping):
        return None, "bucket has no restart.role_outputs_prefix"
    topo = restart.get("roles_topo")
    order = [str(r) for r in topo] if isinstance(topo, Sequence) and not isinstance(topo, str) \
        else sorted(str(k) for k in outputs)
    parts = [str(outputs[r]) for r in order if outputs.get(r)]
    context = CONTEXT_JOIN.join(parts)

    if scope == "context":
        if not context:
            return None, "restart.role_outputs_prefix is empty at the measured position"
        return context, ""

    question = restart.get("question")
    if not isinstance(question, str) or not question:
        return None, "bucket has no restart.question"
    return (question + CONTEXT_JOIN + context) if context else question, ""


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


def scan(results_root: str, scope: str, rendered_field: str = "") -> Dict[str, Any]:
    """Read every c10 cell and report what a prefix count could be read off.

    Returns {cells: [...], n_buckets, n_usable, reasons, texts}. `texts` is the
    prefix string of every usable bucket, in cell order, which is what the
    tokenizer is then handed.
    """
    cells = find_cells(results_root)
    reasons: Dict[str, int] = {}
    texts: List[str] = []
    rows: List[Dict[str, Any]] = []
    n_buckets = 0

    for (model, n) in sorted(cells):
        path = cells[(model, n)]
        buckets = read_buckets(path)
        usable = 0
        for bucket in buckets:
            text, why = prefix_text(bucket, scope, rendered_field)
            if text is None:
                reasons[why] = reasons.get(why, 0) + 1
                continue
            usable += 1
            texts.append(text)
        n_buckets += len(buckets)
        rows.append({
            "model": model,
            "n": n,
            "source": source_path(results_root, WORKFLOW, model, "sweep_n%d" % n, BUCKET_FILE),
            "buckets": len(buckets),
            "usable": usable,
        })

    return {
        "cells": rows,
        "n_buckets": n_buckets,
        "n_usable": len(texts),
        "reasons": reasons,
        "texts": texts,
    }


def run(args: argparse.Namespace, *, stream=None) -> int:
    """Scan, then either report the scan or write the summary."""
    out = sys.stderr if stream is None else stream
    scan_result = scan(args.results, args.prefix_scope, args.rendered_field)

    for row in scan_result["cells"]:
        print("[prefix_tokens] %s: %d bucket(s), %d with a prefix"
              % (row["source"], row["buckets"], row["usable"]), file=out)
    for why, count in sorted(scan_result["reasons"].items()):
        print("[prefix_tokens] %d bucket(s) contribute nothing: %s" % (count, why), file=out)

    if not scan_result["cells"]:
        print("[prefix_tokens] no %s cell under %s" % (WORKFLOW, args.results), file=out)

    if args.dry_run:
        print("# cells: %d" % len(scan_result["cells"]))
        print("# buckets: %d" % scan_result["n_buckets"])
        print("# with a prefix: %d" % scan_result["n_usable"])
        return 0

    if scan_result["n_usable"] == 0:
        print("[prefix_tokens] ERROR: no bucket carries a prefix under --prefix_scope %s, "
              "so there is nothing to count" % args.prefix_scope, file=out)
        return 2

    tokenizer = load_tokenizer(args.tokenizer)
    counts = [count_tokens(tokenizer, text, add_special_tokens=args.add_special_tokens)
              for text in scan_result["texts"]]
    median_low, median_linear = medians(counts)
    value = median_low if args.median == "low" else int(round(median_linear))

    key = args.key_prefix + "." + KEY_TAIL
    note = ("prefix scope %s, tokenizer %s, special tokens %s; low median %d, linear median %.1f; "
            "min %d, max %d"
            % (args.prefix_scope, os.path.basename(str(args.tokenizer).rstrip("/\\")),
               "counted" if args.add_special_tokens else "not counted",
               median_low, median_linear, min(counts), max(counts)))
    keys = {key: {
        "value": int(value),
        "n": len(counts),
        "source": [row["source"] for row in scan_result["cells"]],
        "note": note,
    }}

    manifest = load_manifest(args.manifest)
    doc = write_summary(args.out, "E1", "scripts/70_rebuild/prefix_tokens.py", keys, manifest,
                        stream=out,
                        extra={"results_root": source_path(args.results),
                               "key_prefix": args.key_prefix})
    print("wrote %d key(s) to %s" % (len(doc["keys"]), args.out))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/70_rebuild/prefix_tokens.py",
        description="Median prefix length in tokens of the deep-chain decision points.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results", required=True, help="Root of the E1 result tree.")
    parser.add_argument("--prefix_scope", required=True, choices=PREFIX_SCOPES,
                        help="Which text is the prefix. Not settled: see the module docstring.")
    parser.add_argument("--rendered_field", default="",
                        help="Dotted bucket field holding the rendered prompt, for "
                             "--prefix_scope rendered. No such field exists yet.")
    parser.add_argument("--tokenizer", default="",
                        help="Local model directory for transformers.AutoTokenizer.")
    parser.add_argument("--manifest", default="",
                        help="The paper's results/manifest.json.")
    parser.add_argument("--out", default="", help="summary.json to write.")
    parser.add_argument("--key_prefix", default=DEFAULT_KEY_PREFIX,
                        help="Leading segment of the key written; %s for the appendix tree."
                             % APPENDIX_KEY_PREFIX)
    parser.add_argument("--median", default="low", choices=("low", "linear"),
                        help="Which median an even number of buckets takes. Both are reported "
                             "in the note either way.")
    parser.add_argument("--add_special_tokens", action="store_true",
                        help="Count the tokenizer's special tokens as part of the prefix.")
    parser.add_argument("--dry-run", "--dry_run", dest="dry_run", action="store_true",
                        help="Report what the buckets carry and write nothing. Needs neither "
                             "transformers nor the model files.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    refusal = key_prefix_refusal(args.results, args.key_prefix)
    if refusal is not None:
        print("[prefix_tokens] ERROR: %s" % refusal, file=sys.stderr)
        return 2
    if not args.dry_run:
        missing = [name for name in ("tokenizer", "manifest", "out")
                   if not getattr(args, name)]
        if missing:
            print("[prefix_tokens] ERROR: a real run needs %s"
                  % ", ".join("--" + name for name in missing), file=sys.stderr)
            return 2

    try:
        return run(args)
    except (OSError, ValueError, KeyError) as exc:
        print("[prefix_tokens] ERROR: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
