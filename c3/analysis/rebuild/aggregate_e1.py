# c3/analysis/rebuild/aggregate_e1.py
"""
E1 aggregation: scan the reliability cells, write 20_data/results/E1/summary.json.

    python -m c3.analysis.rebuild.aggregate_e1 \
        --results 20_data/results/E1 --manifest results/manifest.json --out summary.json

Directory layout (the results layout, section 2, as revised on 2026-09-14):

    <results>/<workflow>/<model>/sweep_n{2,3,4,6,8}/buckets.jsonl

Two result trees have this layout, and they are different measurements: `E1` is
the question-pool study the paper reports, `E1_math500all` the full-suite
appendix comparison. Their keys must not collide, so the appendix tree is
aggregated with `--key_prefix E1app` and writes `E1app.*`. Running the appendix
tree under the default prefix is refused rather than silently overwritten, and
every summary records the tree it read (`results_root`) and the prefix it wrote
(`key_prefix`) at the top level.

One key of this experiment is not produced here. `E1.c10.median_prefix_tokens`
is the median number of tokens one replay of the ten-agent chain REGENERATES
downstream (the maintainers' decision of 2026-09-15), so it needs a tokenizer
and the model files. It is produced platform side by
`scripts/70_rebuild/replay_tokens.py`, which writes its own summary.json through
this package's writer.

That is not the length of the prefix, although the key is still named for one.
Every E1 cell is measured at the first role in topological order, so the prefix
at the measured position is the question and nothing else: the same text in
every workflow, carrying no information about depth. What the ten-agent chain
actually costs, and what makes its credit estimate noisy, is the downstream text
every replay has to regenerate. The key keeps its manifest name until the
maintainers rename it there.

The sweep cells are the only measurement. The fixed-budget and per-decision
rows of the manifest are read off the sweep cells, because a fixed instance
budget B=8 split over K decision points gives ceil(8 / K) alternatives at the
measured position (K=2 -> 4, K=3 -> 3, K=4 -> 2), and the constant per-decision
rule is branching 4:

    E1.per_decision_n4.<wf>.<model>.*   <- sweep_n4
    E1.fixed_b8.a2.<model>.*            <- sweep_n4
    E1.fixed_b8.a3.<model>.*            <- sweep_n3
    E1.fixed_b8.mt4.<model>.*           <- sweep_n2

Every sweep cell also carries a noise measurement, and it needs no second run:
the even and the odd replays of one group are two independent estimates of the
same advantage vector at the same decision point, so
`Var(A_even - A_odd) / 2` is the noise of one half and the law predicts it at
the half's own budget. That is the reading revision 19 of the preregistration
(2026-09-15) made its secondary criteria a' and b', and it is the same
estimator E1b reads off two continuation seeds: one implementation, in
noise_law.py, called with two different pairs of sides.

    E1.noise_ratio.<wf>.<model>.n<N>{,_ci_lo,_ci_hi}   per cell
    E1.noise_ratio.median                              over the scan cells
    E1.noise_ratio.vs_n_rho{,_ci_lo,_ci_hi}            over the scan cells
    E1.noise_ratio.n4_all_in_band                      the verdict of b'

The last of those is a verdict key, so no script writes it (`summary.key_refusal`
refuses any key whose unit is "verdict"). The aggregation prints the six readings
it would be decided on and leaves the decision to the maintainers.

Two alternatives is a special case of the estimator, not a missing measurement
(the maintainers' decision of 2026-09-15). A two-alternative group's split-half
correlation can only be +1 or -1, so the sweep row of that cell carries the
direction agreement rate (the share of +1) while the fixed-budget row of mt4,
whose unit is a correlation, carries the mean correlation itself; the two are
the same reading, since mean rho = 2 x agreement - 1. Both are computed on the
same pool, the one the exclusion rule leaves with min_cands=2.

This script never writes the manifest and never writes a verdict key: filling a
value is the maintainers' action, taken outside this repository. Keys the
manifest does not define are skipped with a line on stderr, so the key set of a
summary is always a subset of the manifest's.

Only numpy, scipy and the standard library are imported (through this package).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import OrderedDict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from . import noise_law as nl
from . import splithalf as sh
from . import summary as summary_mod
from .summary import build_summary, git_sha, key_refusal, load_manifest, write_summary

__all__ = [
    "SummaryBuilder",
    "load_manifest",
    "git_sha",
    "scan_e1_cells",
    "build_e1_summary",
    "DEFAULT_KEY_PREFIX",
    "APPENDIX_ROOT_NAME",
    "APPENDIX_KEY_PREFIX",
    "key_prefix_refusal",
    "main",
]

# Vocabulary of the layout (the results layout, section 2).
WORKFLOWS = ("a2", "a3", "mt4", "branch", "c5", "c10")
MODELS = ("4b", "m2", "code")
SWEEP_N = (2, 3, 4, 6, 8)
# Directory names the revised layout retired; seeing one is worth a line.
RETIRED_RULES = ("fixed_b8", "pd_n4")

# The six workflows the cross-workflow keys range over, in schema order.
PD_WORKFLOWS = ("a2", "a3", "mt4", "c5", "c10", "branch")
# Fixed instance budget B=8: the measured position gets ceil(8 / K) alternatives.
FIXED_B8_SOURCE = {"a2": (4, 2), "a3": (3, 3), "mt4": (2, 4)}  # workflow -> (n, K)
PD_SOURCE_N = 4

# The printed name of each workflow, for the manifest keys whose value is a name
# rather than a number (the maintainers' decision of 2026-09-15). The directory name is an
# internal label; these are the words the paper uses.
ARM_DISPLAY_NAMES = {
    "a2": "two-agent",
    "a3": "three-agent",
    "mt4": "four-turn",
    "branch": "branching",
    "c5": "five-agent chain",
    "c10": "ten-agent chain",
}

# Statistical settings (the frozen analysis plan and the results layout, section 4).
MIN_CANDS = 3
# Two-alternative cells: the exclusion rule's alternative count is relaxed to 2,
# which is the only way the manifest's branching-2 rows can be measured at all.
MIN_CANDS_N2 = 2
N_BOOT = 2000
N_RANDOM_SPLITS = 200
SEED = 0

BUCKET_FILE = "buckets.jsonl"

# The two result trees with this layout, and the key prefix each one writes.
DEFAULT_KEY_PREFIX = "E1"
APPENDIX_ROOT_NAME = "E1_math500all"
APPENDIX_KEY_PREFIX = "E1app"

# Said on every row read off a two-alternative cell.
N2_NOTE = "at two alternatives rho is +1 or -1; mean = 2 x agreement - 1"


# -------------------------
# summary plumbing (shared by the other aggregators in this package)
#
# `load_manifest`, `git_sha` and the document writer itself come from
# summary.py, the one summary implementation of this package.
# `source_path` below has a different signature from `summary.source_path`: it
# joins a results root with a cell's sub-path. The shaping of the joined path is
# `summary.source_path`, which it calls, so the two families cannot drift.
# -------------------------


def _as_source_list(source: Any) -> List[str]:
    if isinstance(source, str):
        return [source]
    return [str(s) for s in source]


class SummaryBuilder:
    """Collects the keys of one experiment one at a time, then hands them to
    `summary.write_summary` (one summary implementation for the whole
    package). This class is the collecting half; the document shape, the git sha,
    the timestamp and the empty-note rule all live in summary.py.

    Refusals (each reported on stderr, none fatal): a key the manifest does not
    define, a verdict key, a value that is not a finite number or a non-empty
    string, and a non-positive n. The first two are `summary.key_refusal`, the
    same rule the other aggregation path applies; the last two are this path's
    own, because it is handed raw estimator output. They are refusals rather
    than errors because a partly collected experiment should still produce the
    keys it can support.

    `key_prefix` renames the leading segment of every key, which is how one
    result tree writes `E1.*` and its appendix twin writes `E1app.*` off the same
    code. It is empty by default, meaning no rename, and it only ever replaces
    the builder's own experiment segment, so a builder collecting `E1b.*` cannot
    be redirected by it. `extra` is written at the top level of the document,
    beside `experiment`.
    """

    def __init__(self, experiment: str, script: str, manifest: Mapping[str, Any],
                 *, stream=None, key_prefix: str = "",
                 extra: Optional[Mapping[str, Any]] = None) -> None:
        self.experiment = experiment
        self.script = script
        self.manifest = manifest
        self.key_prefix = str(key_prefix or "")
        self.extra: Dict[str, Any] = dict(extra or {})
        self.keys: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self.refused: List[str] = []
        self.stream = stream if stream is not None else sys.stderr

    def key(self, name: str) -> str:
        """The manifest key this builder writes for a canonical key name."""
        if not self.key_prefix:
            return name
        head, sep, tail = name.partition(".")
        if not sep or head != self.experiment:
            return name
        return self.key_prefix + "." + tail

    def note(self, message: str) -> None:
        print(message, file=self.stream)

    def refuse(self, key: str, why: str) -> bool:
        self.refused.append(key)
        self.note("skip %s: %s" % (key, why))
        return False

    def add(self, key: str, value: Any, *, n: int, source: Any, note: str = "") -> bool:
        key = self.key(key)
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
                             stream=self.stream, extra=self.extra)

    def write(self, path: str) -> str:
        """Write the document and return the path it was written to."""
        write_summary(path, self.experiment, self.script, self.keys, self.manifest,
                      stream=self.stream, extra=self.extra)
        return path


def source_path(results_root: str, *parts: str) -> str:
    """The path string that goes into a summary's `source`.

    The root is joined with a cell's sub-path and the whole thing is then put
    through `summary.source_path`, so both aggregation families write the same
    shape: repository-relative from the last `20_data` segment when there is
    one (`20_data/results/E1/a2/4b/sweep_n4/buckets.jsonl`), and the path as
    spelled, with forward slashes, when there is not. Before this the root was
    kept exactly as the caller gave it, which put the operator's own absolute
    path into the summary of every cell aggregated by absolute path.
    """
    root = str(results_root).replace("\\", "/").rstrip("/")
    return summary_mod.source_path("/".join([root] + [str(p) for p in parts]))


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
                        "ignored: %s is retired by the results layout as revised on 2026-09-14; "
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
    with the relaxed alternative count, every other cell with the one fixed in
    the analysis plan, and a tree could in principle be asked for both.
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


def key_prefix_refusal(results_root: str, key_prefix: str) -> Optional[str]:
    """Why this tree must not be aggregated under this key prefix, or None.

    One refusal so far, and it is the collision the two trees make: the appendix
    tree `E1_math500all` under the default prefix would write exactly the key
    names of the question-pool tree, so the second run to finish would look like
    the first. The check is on the last segment of the results root, which is the
    tree's name.
    """
    name = os.path.basename(str(results_root).replace("\\", "/").rstrip("/"))
    if name == APPENDIX_ROOT_NAME and key_prefix == DEFAULT_KEY_PREFIX:
        return ("the results root is the appendix tree %s, whose keys must not collide with the "
                "question-pool tree: pass --key_prefix %s" % (APPENDIX_ROOT_NAME,
                                                              APPENDIX_KEY_PREFIX))
    return None


def build_e1_summary(results_root: str, manifest: Mapping[str, Any], *,
                     stream=None, store: Optional[_CellStore] = None,
                     key_prefix: str = DEFAULT_KEY_PREFIX) -> SummaryBuilder:
    """Compute every E1 key the collected cells support.

    `key_prefix` is the leading segment of the keys written, `E1` for the
    question-pool tree and `E1app` for the appendix tree. It is recorded in the
    summary together with the results root, both in the shape the results layout
    fixes.
    """
    refusal = key_prefix_refusal(results_root, key_prefix)
    if refusal is not None:
        raise ValueError(refusal)
    builder = SummaryBuilder(
        "E1", "c3.analysis.rebuild.aggregate_e1", manifest, stream=stream,
        key_prefix=key_prefix,
        extra={"results_root": summary_mod.source_path(results_root),
               "key_prefix": key_prefix},
    )
    store = store or _CellStore()
    cells = scan_e1_cells(results_root, builder)

    def rel_path(wf: str, model: str, n: int) -> str:
        return source_path(results_root, wf, model, "sweep_n%d" % n, BUCKET_FILE)

    def add_cell_counts(prefix: str, cell: sh.CellResult, source: str) -> None:
        """The two counts behind a reliability number: how many groups it is read on,
        and what share of the collected groups the exclusion rule dropped.

        The derived rows carry these already through `_add_cell_block`. The sweep
        rows need them too, because the exclusion rate moves with n: a reliability
        read on a tenth of the pool is a different claim from one read on most of
        it, and the appendix reports the pair next to every cell.
        """
        builder.add(prefix + ".n_groups", cell.n_groups, n=cell.n_groups, source=source,
                    note=excluded_note(cell))
        builder.add(prefix + ".excluded_pct", cell.excluded_pct, n=cell.n_groups,
                    source=source,
                    note="denominator is the %d collected groups" % cell.n_total)

    # 1. the sweep itself
    for (wf, model, n) in sorted(cells):
        path = cells[(wf, model, n)]
        key = builder.key("E1.sweep_n.%s.%s.n%d.rel" % (wf, model, n))
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
            add_cell_counts("E1.sweep_n.%s.%s.n%d" % (wf, model, n), cell,
                            rel_path(wf, model, n))
            continue
        builder.add(key, cell.rel_evenodd, n=cell.n_groups, source=rel_path(wf, model, n),
                    note=rel_note(cell))
        add_cell_counts("E1.sweep_n.%s.%s.n%d" % (wf, model, n), cell,
                        rel_path(wf, model, n))

    # 2. the constant per-decision rule, read off sweep_n4
    for (wf, model, n) in sorted(cells):
        if n != PD_SOURCE_N:
            continue
        cell = store.cell(cells[(wf, model, n)])
        if cell.n_groups == 0:
            builder.note("skip %s: no group survives the exclusion rule"
                         % builder.key("E1.per_decision_n4.%s.%s.*" % (wf, model)))
            continue
        _add_cell_block(builder, "E1.per_decision_n4.%s.%s" % (wf, model), cell,
                        rel_path(wf, model, n),
                        "derived from sweep_n4 (constant per-decision budget, branching 4)")

    # 3. the fixed instance budget, read off the sweep cell of the integer split
    for wf, (n, k) in sorted(FIXED_B8_SOURCE.items()):
        for model in MODELS:
            if (wf, model, n) not in cells:
                if any(c[0] == wf and c[1] == model for c in cells):
                    builder.note("skip %s: its source cell sweep_n%d was not collected"
                                 % (builder.key("E1.fixed_b8.%s.%s.*" % (wf, model)), n))
                continue
            two = (n == 2)
            cell = store.cell(cells[(wf, model, n)], MIN_CANDS_N2 if two else None)
            head = "derived from sweep_n%d (fixed budget B=8, K=%d)" % (n, k)
            if cell.n_groups == 0:
                builder.note(
                    "skip %s: %s, and no group survives the exclusion rule (%d collected)"
                    % (builder.key("E1.fixed_b8.%s.%s.*" % (wf, model)), head, cell.n_total))
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
                    builder.note("skip %s: only one of the branching-8 and branching-3 cells "
                                 "was collected"
                                 % builder.key("E1.sweep_n.%s.%s.delta_n8_n3{,_ci_lo}"
                                               % (wf, model)))
                continue
            delta = sh.paired_cell_delta(store.buckets(hi), store.buckets(lo),
                                         n_boot=N_BOOT, seed=SEED, min_cands=MIN_CANDS)
            if delta.delta is None:
                builder.note("skip %s: no question is usable on both sides"
                             % builder.key("E1.sweep_n.%s.%s.delta_n8_n3" % (wf, model)))
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
        builder.note("skip %s: %d of the six workflows have a usable branching-4 cell"
                     % (builder.key("E1.per_decision_n4.4b.rel_{min,max,range}"), len(have)))

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
        builder.note("skip %s: %d of the six workflows have both branching-8 and "
                     "branching-3 cells"
                     % (builder.key("E1.sweep_n.4b.delta_n8_n3_min{,_ci_lo,_arm}"),
                        len(have_delta)))

    # 7. the noise ratio of each sweep cell, read off that cell's own halves
    noise: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    for (wf, model, n) in sorted(cells):
        two = (n == 2)
        rep = nl.cell_noise_ratio_halves(
            store.buckets(cells[(wf, model, n)]), n_boot=N_BOOT, seed=SEED,
            min_cands=MIN_CANDS_N2 if two else MIN_CANDS)
        base = "E1.noise_ratio.%s.%s.n%d" % (wf, model, n)
        if rep["ratio"] is None:
            builder.note("skip %s: no group of this cell carries a halves reading "
                         "(%d collected)" % (builder.key(base), rep["n_total"]))
            continue
        noise[(wf, model, n)] = rep
        src = rel_path(wf, model, n)
        note = ("measured over predicted noise of the advantage estimate, read off the even "
                "and odd halves of this cell's own replays (%.1f replays per half per "
                "alternative, so the prediction is at budget %d x %.1f); %d of %d groups "
                "usable, %.0f%% excluded"
                % (rep["replays_per_half"], n, rep["replays_per_half"],
                   rep["n_groups"], rep["n_total"], rep["excluded_pct"]))
        if rep["n_flat_half"]:
            # The one clause this reading does not take from the reliability path
            # (revision 19(h) of the preregistration, 2026-09-15): a flat half is
            # kept here, and the reading the reliability pool would give rides
            # along as a sensitivity rather than as a key of its own.
            note += ("; %d group(s) came back with a flat half and are kept, which the "
                     "correlation cannot do and this reading can" % rep["n_flat_half"])
            if rep["ratio_flat_dropped"] is not None:
                note += ("; dropping them instead would read %.4f on %d group(s)"
                         % (rep["ratio_flat_dropped"], rep["n_groups_flat_dropped"]))
        if rep["n_half_uneven"]:
            note += ("; %d group(s) hold an odd number of replays, so their halves differ by "
                     "one and the budget above is their average"
                     % rep["n_half_uneven"])
        builder.add(base, rep["ratio"], n=rep["n_groups"], source=src, note=note)
        for tail, value in (("_ci_lo", rep["ci_lo"]), ("_ci_hi", rep["ci_hi"])):
            builder.add(base + tail, value, n=rep["n_groups"], source=src,
                        note="percentile bootstrap over groups, B=%d" % N_BOOT)

    # 8. the three cross-cell readings of that ratio (revision 19, criteria a' and b')
    band_lo, band_hi = nl.RATIO_BAND
    if noise:
        scanned = sorted(noise)
        values = [noise[k]["ratio"] for k in scanned]
        src_all = [rel_path(wf, model, n) for (wf, model, n) in scanned]
        models_read = ", ".join(sorted({model for (_, model, _) in scanned}))
        # Pooled across substrates on purpose (the maintainers' decision of
        # 2026-09-15): the law is not a statement about one base model, and the
        # more cells the correlation has the more it can say. The band test of
        # criterion b' is the one reading tied to a substrate, because it compares
        # the six workflows at one branching.
        builder.add("E1.noise_ratio.median", nl.ratio_median(values), n=len(values),
                    source=src_all,
                    note="median over the %d scan cell(s) that carry a ratio, pooled across "
                         "substrates (model(s) %s); the preregistered band is [%.2f, %.2f]"
                         % (len(values), models_read, band_lo, band_hi))
        vs_n = nl.ratio_vs_n([(n, noise[(wf, model, n)]["ratio"])
                              for (wf, model, n) in scanned], n_boot=N_BOOT, seed=SEED)
        if vs_n["rho"] is None:
            builder.note("skip %s{,_ci_lo,_ci_hi}: the %d cell(s) carrying a ratio do not "
                         "vary in branching or in ratio"
                         % (builder.key("E1.noise_ratio.vs_n_rho"), len(values)))
        else:
            head = ("rank correlation of the cell ratio against branching over the %d scan "
                    "cell(s) that carry one, pooled across substrates (model(s) %s)"
                    % (vs_n["n_cells"], models_read))
            builder.add("E1.noise_ratio.vs_n_rho", vs_n["rho"], n=vs_n["n_cells"],
                        source=src_all, note=head)
            boot = ("percentile bootstrap over cells, B=%d (%d draw(s) kept, %d dropped for "
                    "holding one branching or one ratio only)"
                    % (N_BOOT, vs_n["n_draws"], vs_n["n_degenerate"]))
            for tail, value in (("_ci_lo", vs_n["ci_lo"]), ("_ci_hi", vs_n["ci_hi"])):
                builder.add("E1.noise_ratio.vs_n_rho" + tail, value, n=vs_n["n_cells"],
                            source=src_all, note=head + "; " + boot)
    else:
        builder.note("skip %s and %s{,_ci_lo,_ci_hi}: no scan cell carries a ratio"
                     % (builder.key("E1.noise_ratio.median"),
                        builder.key("E1.noise_ratio.vs_n_rho")))

    # The verdict of criterion b', printed and not written: `summary.key_refusal`
    # refuses every key whose unit is "verdict", and filling one is the
    # maintainers' action. Printed on the question-pool tree alone, which is where
    # the preregistration decides E1; the appendix tree carries no verdict key to
    # name here.
    if builder.key_prefix == DEFAULT_KEY_PREFIX:
        n4 = [(wf, noise[(wf, "4b", PD_SOURCE_N)]["ratio"]) for wf in PD_WORKFLOWS
              if (wf, "4b", PD_SOURCE_N) in noise]
        if len(n4) == len(PD_WORKFLOWS):
            outside = [wf for wf, ratio in n4 if not nl.in_band(ratio)]
            builder.note(
                "E1.noise_ratio.n4_all_in_band is a verdict key and no script writes it. "
                "The reading it is decided on, the six workflows at branching 4 on Qwen3-4B: "
                "%s. Inside [%.2f, %.2f]: %d of 6%s."
                % ("; ".join("%s %.3f" % (wf, ratio) for wf, ratio in n4),
                   band_lo, band_hi, len(n4) - len(outside),
                   ("" if not outside else " (outside: %s)" % ", ".join(outside))))
        else:
            builder.note("E1.noise_ratio.n4_all_in_band cannot be read yet: %d of the six "
                         "workflows have a branching-4 ratio on Qwen3-4B" % len(n4))

    builder.note("%s: %d key(s) written, %d refused"
                 % (builder.key_prefix or builder.experiment, len(builder.keys),
                    len(builder.refused)))
    return builder


def build_parser(experiment: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Aggregate the %s cells into a summary.json" % experiment)
    ap.add_argument("--results", required=True, help="root of the experiment's result tree")
    ap.add_argument("--manifest", required=True, help="the paper's results/manifest.json")
    ap.add_argument("--out", required=True, help="summary.json to write")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = build_parser("E1")
    ap.add_argument("--key_prefix", default=DEFAULT_KEY_PREFIX,
                    help="leading segment of the keys written; %s for the question-pool tree, "
                         "%s for the appendix tree %s"
                         % (DEFAULT_KEY_PREFIX, APPENDIX_KEY_PREFIX, APPENDIX_ROOT_NAME))
    args = ap.parse_args(argv)

    refusal = key_prefix_refusal(args.results, args.key_prefix)
    if refusal is not None:
        print("[aggregate_e1] ERROR: %s" % refusal, file=sys.stderr)
        return 2

    manifest = load_manifest(args.manifest)
    builder = build_e1_summary(args.results, manifest, key_prefix=args.key_prefix)
    builder.write(args.out)
    print("wrote %d key(s) to %s" % (len(builder.keys), args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
