# c3/analysis/rebuild/aggregate_e1.py
"""
E1 aggregation: scan the reliability cells, write 20_data/results/E1/summary.json.

    python -m c3.analysis.rebuild.aggregate_e1 \
        --results 20_data/results/E1 --manifest results/manifest.json --out summary.json

Directory layout (results contract section 2, as revised on 2026-09-14):

    <results>/<workflow>/<model>/sweep_n{2,3,4,6,8}/buckets.jsonl

The sweep cells are the only measurement. The fixed-budget and per-decision
rows of the manifest are read off the sweep cells, because a fixed instance
budget B=8 split over K decision points gives ceil(8 / K) alternatives at the
measured position (K=2 -> 4, K=3 -> 3, K=4 -> 2), and the constant per-decision
rule is branching 4:

    E1.per_decision_n4.<wf>.<model>.*   <- sweep_n4
    E1.fixed_b8.a2.<model>.*            <- sweep_n4
    E1.fixed_b8.a3.<model>.*            <- sweep_n3
    E1.fixed_b8.mt4.<model>.*           <- sweep_n2

Two alternatives is a special case of the estimator, not a missing measurement
(driver ruling A3, 2026-09-15). A two-alternative group's split-half correlation
can only be +1 or -1, so the sweep row of that cell carries the direction
agreement rate (the share of +1) while the fixed-budget row of mt4, whose unit
is a correlation, carries the mean correlation itself; the two are the same
reading, since mean rho = 2 x agreement - 1. Both are computed on the same pool,
the one the exclusion rule leaves with min_cands=2.

This script never writes the manifest and never writes a verdict key: filling a
value is the driver's action, through 30_analysis/rebuild/manifest_writer.py.
Keys the manifest does not define are skipped with a line on stderr, so the key
set of a summary is always a subset of the manifest's.

Only numpy, scipy and the standard library are imported (through this package).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import OrderedDict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from . import splithalf as sh
from .summary import build_summary, git_sha, key_refusal, load_manifest, write_summary

__all__ = [
    "SummaryBuilder",
    "load_manifest",
    "git_sha",
    "scan_e1_cells",
    "build_e1_summary",
    "main",
]

# Vocabulary of the layout (results contract section 2).
WORKFLOWS = ("a2", "a3", "mt4", "branch", "c5", "c10")
MODELS = ("4b", "m2", "code")
SWEEP_N = (2, 3, 4, 6, 8)
# Directory names the revised contract retired; seeing one is worth a line.
RETIRED_RULES = ("fixed_b8", "pd_n4")

# The six workflows the cross-workflow keys range over, in schema order.
PD_WORKFLOWS = ("a2", "a3", "mt4", "c5", "c10", "branch")
# Fixed instance budget B=8: the measured position gets ceil(8 / K) alternatives.
FIXED_B8_SOURCE = {"a2": (4, 2), "a3": (3, 3), "mt4": (2, 4)}  # workflow -> (n, K)
PD_SOURCE_N = 4

# The printed name of each workflow, for the manifest keys whose value is a name
# rather than a number (driver ruling B7, 2026-09-15). The directory name is an
# internal label; these are the words the paper uses.
ARM_DISPLAY_NAMES = {
    "a2": "two-agent",
    "a3": "three-agent",
    "mt4": "four-turn",
    "branch": "branching",
    "c5": "five-agent chain",
    "c10": "ten-agent chain",
}

# Statistical settings (preregistration and results contract section 4).
MIN_CANDS = 3
# Two-alternative cells: the exclusion rule's alternative count is relaxed to 2,
# which is the only way the manifest's branching-2 rows can be measured at all.
MIN_CANDS_N2 = 2
N_BOOT = 2000
N_RANDOM_SPLITS = 200
SEED = 0

BUCKET_FILE = "buckets.jsonl"

# Said on every row read off a two-alternative cell (ruling A3).
N2_NOTE = "at two alternatives rho is +1 or -1; mean = 2 x agreement - 1"


# -------------------------
# summary plumbing (shared by the other aggregators in this package)
#
# `load_manifest`, `git_sha` and the document writer itself come from
# summary.py, the one summary implementation of this package (ruling B13).
# `source_path` below is a different function from `summary.source_path`: it
# joins a results root with a cell's sub-path and keeps the root exactly as the
# caller spelled it.
# -------------------------


def _as_source_list(source: Any) -> List[str]:
    if isinstance(source, str):
        return [source]
    return [str(s) for s in source]


class SummaryBuilder:
    """Collects the keys of one experiment one at a time, then hands them to
    `summary.write_summary` (ruling B13: one summary implementation for the whole
    package). This class is the collecting half; the document shape, the git sha,
    the timestamp and the empty-note rule all live in summary.py.

    Refusals (each reported on stderr, none fatal): a key the manifest does not
    define, a verdict key, a value that is not a finite number or a non-empty
    string, and a non-positive n. The first two are `summary.key_refusal`, the
    same rule the other aggregation path applies; the last two are this path's
    own, because it is handed raw estimator output. They are refusals rather
    than errors because a partly collected experiment should still produce the
    keys it can support.
    """

    def __init__(self, experiment: str, script: str, manifest: Mapping[str, Any],
                 *, stream=None) -> None:
        self.experiment = experiment
        self.script = script
        self.manifest = manifest
        self.keys: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self.refused: List[str] = []
        self.stream = stream if stream is not None else sys.stderr

    def note(self, message: str) -> None:
        print(message, file=self.stream)

    def refuse(self, key: str, why: str) -> bool:
        self.refused.append(key)
        self.note("skip %s: %s" % (key, why))
        return False

    def add(self, key: str, value: Any, *, n: int, source: Any, note: str = "") -> bool:
        why = key_refusal(key, self.manifest)
        if why is not None:
            return self.refuse(key, why)
        entry = self.manifest[key]
        if isinstance(value, bool) or value is None:
            return self.refuse(key, "no value")
        if isinstance(value, (int, float)):
            if not math.isfinite(float(value)):
                return self.refuse(key, "value is not finite")
        elif isinstance(value, str):
            if not value.strip():
                return self.refuse(key, "empty string value")
        else:
            return self.refuse(key, "value type %s" % type(value).__name__)
        if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
            return self.refuse(key, "n must be a positive integer, got %r" % (n,))
        self.keys[key] = {
            "value": value,
            "n": int(n),
            "unit": entry.get("unit"),
            "source": _as_source_list(source),
            "note": note,
        }
        return True

    def payload(self) -> Dict[str, Any]:
        """The summary document, built by the package's one implementation."""
        return build_summary(self.experiment, self.script, self.keys, self.manifest,
                             stream=self.stream)

    def write(self, path: str) -> str:
        """Write the document and return the path it was written to."""
        write_summary(path, self.experiment, self.script, self.keys, self.manifest,
                      stream=self.stream)
        return path


def source_path(results_root: str, *parts: str) -> str:
    """The path string that goes into a summary's `source`, in the shape the
    caller gave the results root (relative stays relative), forward slashes."""
    root = str(results_root).replace("\\", "/").rstrip("/")
    return "/".join([root] + [str(p) for p in parts])


def excluded_note(cell: sh.CellResult) -> str:
    return "excluded %d/%d groups" % (cell.n_total - cell.n_groups, cell.n_total)


def rel_note(cell: sh.CellResult) -> str:
    """The exclusion count plus the random-split reading of the same groups, which
    no manifest key carries but which says whether the deterministic split is
    representative."""
    note = excluded_note(cell)
    if cell.rel_random is not None:
        note += "; random-split reading %.4f over %d splits per group" % (
            cell.rel_random, N_RANDOM_SPLITS)
    return note


def wgv_note(cell: sh.CellResult) -> str:
    note = ("pooled over the %d groups that pass the alternative and replay counts, "
            "flat halves included" % cell.n_var_groups)
    if cell.wgv_ci_lo is not None:
        note += "; group bootstrap 95%% interval [%.5f, %.5f]" % (cell.wgv_ci_lo, cell.wgv_ci_hi)
    return note


# -------------------------
# scanning
# -------------------------


def _subdirs(path: str) -> List[str]:
    return sorted(d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)))


def scan_e1_cells(results_root: str, builder: SummaryBuilder) -> Dict[Tuple[str, str, int], str]:
    """Map (workflow, model, n) to the cell's buckets.jsonl path.

    Anything the layout does not define is ignored with one line on stderr.
    """
    cells: Dict[Tuple[str, str, int], str] = {}
    if not os.path.isdir(results_root):
        raise ValueError("results root %s is not a directory" % results_root)
    for wf in _subdirs(results_root):
        if wf not in WORKFLOWS:
            builder.note("ignored: %s is not a workflow name" % source_path(results_root, wf))
            continue
        for model in _subdirs(os.path.join(results_root, wf)):
            if model not in MODELS:
                builder.note("ignored: %s is not a model name" % source_path(results_root, wf, model))
                continue
            for rule in _subdirs(os.path.join(results_root, wf, model)):
                where = source_path(results_root, wf, model, rule)
                if rule in RETIRED_RULES:
                    builder.note(
                        "ignored: %s is retired by the contract revision of 2026-09-14; "
                        "these rows are derived from the sweep cells" % where)
                    continue
                n = None
                if rule.startswith("sweep_n"):
                    try:
                        n = int(rule[len("sweep_n"):])
                    except ValueError:
                        n = None
                if n is None or n not in SWEEP_N:
                    builder.note("ignored: %s is not a sweep cell" % where)
                    continue
                path = os.path.join(results_root, wf, model, rule, BUCKET_FILE)
                if not os.path.isfile(path):
                    builder.note("ignored: %s has no %s" % (where, BUCKET_FILE))
                    continue
                cells[(wf, model, n)] = path
    return cells


class _CellStore:
    """Loads each cell once and keeps its buckets and its reliability result.

    A cell is cached per (path, min_cands): the two-alternative cells are read
    with the relaxed alternative count, every other cell with the preregistered
    one, and a tree could in principle be asked for both.
    """

    def __init__(self, min_cands: int = MIN_CANDS, n_boot: int = N_BOOT,
                 n_random_splits: int = N_RANDOM_SPLITS, seed: int = SEED) -> None:
        self.min_cands = min_cands
        self.n_boot = n_boot
        self.n_random_splits = n_random_splits
        self.seed = seed
        self._buckets: Dict[str, List[Dict[str, Any]]] = {}
        self._cells: Dict[Tuple[str, int], sh.CellResult] = {}

    def buckets(self, path: str) -> List[Dict[str, Any]]:
        if path not in self._buckets:
            self._buckets[path] = sh.load_buckets(path)
        return self._buckets[path]

    def cell(self, path: str, min_cands: Optional[int] = None) -> sh.CellResult:
        want = self.min_cands if min_cands is None else int(min_cands)
        if (path, want) not in self._cells:
            self._cells[(path, want)] = sh.cell_reliability(
                self.buckets(path), n_random_splits=self.n_random_splits,
                n_boot=self.n_boot, seed=self.seed, min_cands=want)
        return self._cells[(path, want)]


# -------------------------
# E1 keys
# -------------------------


def _add_cell_block(builder: SummaryBuilder, prefix: str, cell: sh.CellResult,
                    source: str, note_head: str, rel_extra: str = "") -> None:
    """The six keys every reliability row of the manifest carries.

    `rel_extra` is appended to the `.rel` note alone, where a row needs a word
    about the statistic itself (the two-alternative rows do).
    """
    head = (note_head + "; ") if note_head else ""
    tail = ("; " + rel_extra) if rel_extra else ""
    builder.add(prefix + ".rel", cell.rel_evenodd, n=cell.n_groups, source=source,
                note=head + rel_note(cell) + tail)
    builder.add(prefix + ".rel_ci_lo", cell.rel_ci_lo, n=cell.n_groups, source=source,
                note=head + "percentile bootstrap over groups, B=%d" % N_BOOT)
    builder.add(prefix + ".rel_ci_hi", cell.rel_ci_hi, n=cell.n_groups, source=source,
                note=head + "percentile bootstrap over groups, B=%d" % N_BOOT)
    builder.add(prefix + ".wgv", cell.wgv, n=cell.n_var_groups, source=source,
                note=head + wgv_note(cell))
    builder.add(prefix + ".n_groups", cell.n_groups, n=cell.n_groups, source=source,
                note=head + excluded_note(cell))
    builder.add(prefix + ".excluded_pct", cell.excluded_pct, n=cell.n_groups, source=source,
                note=head + "denominator is the %d collected groups" % cell.n_total)


def build_e1_summary(results_root: str, manifest: Mapping[str, Any], *,
                     stream=None, store: Optional[_CellStore] = None) -> SummaryBuilder:
    """Compute every E1 key the collected cells support."""
    builder = SummaryBuilder("E1", "c3.analysis.rebuild.aggregate_e1", manifest, stream=stream)
    store = store or _CellStore()
    cells = scan_e1_cells(results_root, builder)

    def rel_path(wf: str, model: str, n: int) -> str:
        return source_path(results_root, wf, model, "sweep_n%d" % n, BUCKET_FILE)

    # 1. the sweep itself
    for (wf, model, n) in sorted(cells):
        path = cells[(wf, model, n)]
        key = "E1.sweep_n.%s.%s.n%d.rel" % (wf, model, n)
        two = (n == 2)
        cell = store.cell(path, MIN_CANDS_N2 if two else None)
        if cell.n_groups == 0:
            builder.note("skip %s: no group survives the exclusion rule (%d collected)"
                         % (key, cell.n_total))
            continue
        if two:
            # The manifest's unit here is an agreement rate, not a correlation.
            if cell.n2_direction_agreement is None:
                builder.note(
                    "skip %s: the cell holds groups with other than two alternatives, so a "
                    "direction-agreement rate is not the statistic of this pool" % key)
                continue
            builder.add(key, cell.n2_direction_agreement, n=cell.n_groups,
                        source=rel_path(wf, model, n),
                        note="%s; %s" % (rel_note(cell), N2_NOTE))
            continue
        builder.add(key, cell.rel_evenodd, n=cell.n_groups, source=rel_path(wf, model, n),
                    note=rel_note(cell))

    # 2. the constant per-decision rule, read off sweep_n4
    for (wf, model, n) in sorted(cells):
        if n != PD_SOURCE_N:
            continue
        cell = store.cell(cells[(wf, model, n)])
        if cell.n_groups == 0:
            builder.note("skip E1.per_decision_n4.%s.%s.*: no group survives the exclusion rule"
                         % (wf, model))
            continue
        _add_cell_block(builder, "E1.per_decision_n4.%s.%s" % (wf, model), cell,
                        rel_path(wf, model, n),
                        "derived from sweep_n4 (constant per-decision budget, branching 4)")

    # 3. the fixed instance budget, read off the sweep cell of the integer split
    for wf, (n, k) in sorted(FIXED_B8_SOURCE.items()):
        for model in MODELS:
            if (wf, model, n) not in cells:
                if any(c[0] == wf and c[1] == model for c in cells):
                    builder.note("skip E1.fixed_b8.%s.%s.*: its source cell sweep_n%d was "
                                 "not collected" % (wf, model, n))
                continue
            two = (n == 2)
            cell = store.cell(cells[(wf, model, n)], MIN_CANDS_N2 if two else None)
            head = "derived from sweep_n%d (fixed budget B=8, K=%d)" % (n, k)
            if cell.n_groups == 0:
                builder.note(
                    "skip E1.fixed_b8.%s.%s.*: %s, and no group survives the exclusion rule "
                    "(%d collected)" % (wf, model, head, cell.n_total))
                continue
            # These rows print a correlation, so the two-alternative cell reports the
            # mean rho of the same pool its sweep row reports as an agreement rate.
            _add_cell_block(builder, "E1.fixed_b8.%s.%s" % (wf, model), cell,
                            rel_path(wf, model, n), head, N2_NOTE if two else "")

    # 4. the branching-8 minus branching-3 difference, paired by question
    deltas: Dict[Tuple[str, str], sh.DeltaResult] = {}
    for wf in WORKFLOWS:
        for model in MODELS:
            hi, lo = cells.get((wf, model, 8)), cells.get((wf, model, 3))
            if hi is None or lo is None:
                if hi is not None or lo is not None:
                    builder.note("skip E1.sweep_n.%s.%s.delta_n8_n3{,_ci_lo}: only one of the "
                                 "branching-8 and branching-3 cells was collected" % (wf, model))
                continue
            delta = sh.paired_cell_delta(store.buckets(hi), store.buckets(lo),
                                         n_boot=N_BOOT, seed=SEED, min_cands=MIN_CANDS)
            if delta.delta is None:
                builder.note("skip E1.sweep_n.%s.%s.delta_n8_n3: no question is usable on both sides"
                             % (wf, model))
                continue
            deltas[(wf, model)] = delta
            src = [rel_path(wf, model, 8), rel_path(wf, model, 3)]
            note = "paired over %d questions present and usable in both cells" % delta.n_pairs
            builder.add("E1.sweep_n.%s.%s.delta_n8_n3" % (wf, model), delta.delta,
                        n=delta.n_pairs, source=src, note=note)
            builder.add("E1.sweep_n.%s.%s.delta_n8_n3_ci_lo" % (wf, model), delta.ci_lo,
                        n=delta.n_pairs, source=src,
                        note=note + "; percentile bootstrap over questions, B=%d" % N_BOOT)

    # 5. spread of the per-decision rule over the six workflows on Qwen3-4B
    have = [wf for wf in PD_WORKFLOWS
            if (wf, "4b", PD_SOURCE_N) in cells
            and store.cell(cells[(wf, "4b", PD_SOURCE_N)]).n_groups > 0]
    if len(have) == len(PD_WORKFLOWS):
        rels = {wf: store.cell(cells[(wf, "4b", PD_SOURCE_N)]).rel_evenodd for wf in have}
        spread, lo_name, hi_name = sh.cell_range(rels)
        ci_lo, ci_hi = sh.range_bootstrap(
            {wf: store.buckets(cells[(wf, "4b", PD_SOURCE_N)]) for wf in have},
            N_BOOT, SEED, MIN_CANDS)
        src = [rel_path(wf, "4b", PD_SOURCE_N) for wf in have]
        builder.add("E1.per_decision_n4.4b.rel_min", rels[lo_name], n=len(have), source=src,
                    note="smallest of the six workflows at branching 4 (directory %s)" % lo_name)
        builder.add("E1.per_decision_n4.4b.rel_max", rels[hi_name], n=len(have), source=src,
                    note="largest of the six workflows at branching 4 (directory %s)" % hi_name)
        builder.add("E1.per_decision_n4.4b.rel_range", spread, n=len(have), source=src,
                    note="%s minus %s; group bootstrap 95%% interval of the range "
                         "[%.4f, %.4f]" % (hi_name, lo_name,
                                           ci_lo if ci_lo is not None else float("nan"),
                                           ci_hi if ci_hi is not None else float("nan")))
    else:
        builder.note("skip E1.per_decision_n4.4b.rel_{min,max,range}: %d of the six "
                     "workflows have a usable branching-4 cell" % len(have))

    # 6. the binding workflow of the branching criterion on Qwen3-4B
    have_delta = [wf for wf in PD_WORKFLOWS if (wf, "4b") in deltas]
    if len(have_delta) == len(PD_WORKFLOWS):
        worst = min(sorted(have_delta), key=lambda wf: deltas[(wf, "4b")].delta)
        src = [rel_path(wf, "4b", n) for wf in have_delta for n in (8, 3)]
        note = "smallest rise over the six workflows; the binding workflow is the " \
               "directory %s" % worst
        builder.add("E1.sweep_n.4b.delta_n8_n3_min", deltas[(worst, "4b")].delta,
                    n=len(have_delta), source=src, note=note)
        builder.add("E1.sweep_n.4b.delta_n8_n3_min_ci_lo", deltas[(worst, "4b")].ci_lo,
                    n=len(have_delta), source=src,
                    note=note + "; the interval is that workflow's own")
        builder.add("E1.sweep_n.4b.delta_n8_n3_min_arm", ARM_DISPLAY_NAMES[worst],
                    n=len(have_delta), source=src,
                    note=note + "; printed name of the directory %s" % worst)
    else:
        builder.note("skip E1.sweep_n.4b.delta_n8_n3_min{,_ci_lo,_arm}: %d of the six "
                     "workflows have both branching-8 and branching-3 cells" % len(have_delta))

    builder.note("E1: %d key(s) written, %d refused" % (len(builder.keys), len(builder.refused)))
    return builder


def build_parser(experiment: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Aggregate the %s cells into a summary.json" % experiment)
    ap.add_argument("--results", required=True, help="root of the experiment's result tree")
    ap.add_argument("--manifest", required=True, help="the paper's results/manifest.json")
    ap.add_argument("--out", required=True, help="summary.json to write")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser("E1").parse_args(argv)
    manifest = load_manifest(args.manifest)
    builder = build_e1_summary(args.results, manifest)
    builder.write(args.out)
    print("wrote %d key(s) to %s" % (len(builder.keys), args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
