# c3/analysis/rebuild/aggregate_e1b.py
"""
E1b aggregation: the noise-law grid, written to 20_data/results/E1b/summary.json.

    python -m c3.analysis.rebuild.aggregate_e1b \
        --results 20_data/results/E1b --manifest results/manifest.json --out summary.json

Directory layout (results contract section 2):

    <results>/<workflow>/<model>/n{2,4,8}_c{2,4,8}/{seedA,seedB}/buckets.jsonl

The manifest names a cell by branching and replays alone (E1b is collected at
one position: the two-agent Reasoner), so a tree that holds two workflows or two
models for the same (n, c) makes the key ambiguous; such a cell is skipped with
a line on stderr rather than resolved by a rule this script would be inventing.

The measured-over-predicted ratio and its two grid summaries come from
noise_law.py; the estimator convention it uses is that module's default.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from . import noise_law as nl
from . import splithalf as sh
from .aggregate_e1 import (
    BUCKET_FILE,
    MODELS,
    N_BOOT,
    SEED,
    SummaryBuilder,
    build_parser,
    load_manifest,
    source_path,
)

__all__ = ["scan_e1b_cells", "build_e1b_summary", "main"]

WORKFLOWS = ("a2", "a3", "mt4", "branch", "c5", "c10")
CELL_RE = re.compile(r"^n(\d+)_c(\d+)$")
SEED_DIRS = ("seedA", "seedB")
# Element-wise variance convention of the measured noise (work order section 1.3).
DDOF = 1


def _subdirs(path: str) -> List[str]:
    return sorted(d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)))


def scan_e1b_cells(results_root: str, builder: SummaryBuilder
                   ) -> Dict[Tuple[int, int], Tuple[str, str, str, str]]:
    """Map (n, c) to (workflow, model, seedA path, seedB path).

    Unknown directories are ignored with one line each; a cell present under two
    workflow or model directories is dropped, since the manifest key cannot tell
    them apart.
    """
    if not os.path.isdir(results_root):
        raise ValueError("results root %s is not a directory" % results_root)
    found: Dict[Tuple[int, int], List[Tuple[str, str, str, str]]] = {}
    for wf in _subdirs(results_root):
        if wf not in WORKFLOWS:
            builder.note("ignored: %s is not a workflow name" % source_path(results_root, wf))
            continue
        for model in _subdirs(os.path.join(results_root, wf)):
            if model not in MODELS:
                builder.note("ignored: %s is not a model name"
                             % source_path(results_root, wf, model))
                continue
            for cell in _subdirs(os.path.join(results_root, wf, model)):
                where = source_path(results_root, wf, model, cell)
                match = CELL_RE.match(cell)
                if not match:
                    builder.note("ignored: %s is not an n<n>_c<c> cell" % where)
                    continue
                paths = [os.path.join(results_root, wf, model, cell, s, BUCKET_FILE)
                         for s in SEED_DIRS]
                missing = [p for p in paths if not os.path.isfile(p)]
                if missing:
                    builder.note("ignored: %s is missing %s"
                                 % (where, ", ".join(os.path.basename(os.path.dirname(p))
                                                     for p in missing)))
                    continue
                key = (int(match.group(1)), int(match.group(2)))
                found.setdefault(key, []).append((wf, model, paths[0], paths[1]))

    cells: Dict[Tuple[int, int], Tuple[str, str, str, str]] = {}
    for key in sorted(found):
        entries = found[key]
        if len(entries) > 1:
            builder.note(
                "skip E1b.noise_law.ratio.n%d.c%d: %d directories carry this cell (%s) and "
                "the manifest key names only branching and replays"
                % (key[0], key[1], len(entries),
                   ", ".join("%s/%s" % (e[0], e[1]) for e in entries)))
            continue
        cells[key] = entries[0]
    return cells


def build_e1b_summary(results_root: str, manifest: Mapping[str, Any], *, stream=None,
                      n_boot: int = N_BOOT, seed: int = SEED, ddof: int = DDOF
                      ) -> SummaryBuilder:
    builder = SummaryBuilder("E1b", "c3.analysis.rebuild.aggregate_e1b", manifest, stream=stream)
    cells = scan_e1b_cells(results_root, builder)

    reports: Dict[Tuple[int, int], Dict[str, Any]] = {}
    sources: Dict[Tuple[int, int], List[str]] = {}
    for key in sorted(cells):
        n, c = key
        wf, model, path_a, path_b = cells[key]
        rep = nl.cell_noise_ratio(sh.load_buckets(path_a), sh.load_buckets(path_b),
                                  n_boot=n_boot, seed=seed, ddof=ddof)
        src = [source_path(results_root, wf, model, "n%d_c%d" % (n, c), s, BUCKET_FILE)
               for s in SEED_DIRS]
        sources[key] = src
        if rep["ratio"] is None:
            builder.note("skip E1b.noise_law.ratio.n%d.c%d: no paired group produced a ratio "
                         "(%d pairs, %d skipped)" % (n, c, rep["n_pairs"], rep["n_skipped"]))
            continue
        reports[key] = rep
        builder.add("E1b.noise_law.ratio.n%d.c%d" % (n, c), rep["ratio"],
                    n=rep["n_groups"], source=src,
                    note="paired by %s; %d of %d groups usable; bootstrap 95%% interval "
                         "[%.4f, %.4f]" % (rep["pair_field"], rep["n_groups"], rep["n_pairs"],
                                           rep["ci_lo"], rep["ci_hi"]))

    if reports:
        worst = max(sorted(reports), key=lambda k: abs(reports[k]["ratio"] - 1.0))
        dev = 100.0 * abs(reports[worst]["ratio"] - 1.0)
        all_src = [p for key in sorted(reports) for p in sources[key]]
        builder.add("E1b.noise_law.max_dev_pct", dev, n=len(reports), source=all_src,
                    note="largest |ratio - 1| over the %d collected cells, from n%d_c%d"
                         % (len(reports), worst[0], worst[1]))
        builder.add("E1b.noise_law.worst_ratio", reports[worst]["ratio"],
                    n=reports[worst]["n_groups"], source=sources[worst],
                    note="the cell farthest from one is n%d_c%d, over %d cells"
                         % (worst[0], worst[1], len(reports)))
    else:
        builder.note("skip E1b.noise_law.{max_dev_pct,worst_ratio}: no cell produced a ratio")

    builder.note("E1b: %d key(s) written, %d refused" % (len(builder.keys), len(builder.refused)))
    return builder


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser("E1b").parse_args(argv)
    manifest = load_manifest(args.manifest)
    builder = build_e1b_summary(args.results, manifest)
    builder.write(args.out)
    print("wrote %d key(s) to %s" % (len(builder.keys), args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
