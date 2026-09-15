# c3/analysis/rebuild/influence.py
"""Answer-level plug-in mutual information I(J; Y | h), the influence estimator
frozen in the preregistration for the rebuild experiments (E3a, E5).

What it measures: inside one decision point (bucket), J indexes the alternative
upstream messages and Y is the *answer symbol* of the downstream output that
followed. A high value means "which alternative was sent still shows up in the
final answer"; a value near zero means the downstream agent produced the same
answer whatever it was handed.

Frozen conventions (preregistration E5, results contract section 4):
- Y is the ANSWER symbol, not the raw text and not a hash of the whole message.
  The mapping text -> symbol is injected (see `answer_symbol`), so this module
  never parses mathematics itself.
- J is uniform: p_j = 1/n_j, matching `30_analysis/server_tierA/p1rb_analysis.py`
  (`plugin_mi`). This is NOT the sample-mass weighting used by the older
  `c3.analysis.metrics.influence_mi`.
- Entropies use the natural logarithm, so the unit is nats.
- Miller-Madow bias correction: every entropy term gets + (m - 1) / (2 N), with
  m the number of non-empty cells of that distribution and N its sample count.
  MI = H_MM(Y) - sum_j p_j H_MM(Y | j).
- The result is NOT clipped at zero. The correction can push an already small
  plug-in value below zero and the contract says to report it as measured.

Differences from `c3.analysis.metrics.influence_mi` (the old estimator, kept
only for contrast): that one hashes canonicalized full text, adds a Laplace
alpha, pools Y into a top-K vocabulary plus OTHER, weights J by sample mass and
clips at zero. None of that is done here.

The package-level value conventions (p-value strings, manifest loading, the
summary.json writer) now live in `c3.analysis.rebuild.summary`, which both
aggregation families share (driver ruling B13, 2026-09-15). They are imported
back into this module under their old names, so everything that reads them from
here keeps working.

Module-level imports are limited to numpy, scipy and the standard library.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from .summary import (
    build_summary,
    format_p,
    git_sha,
    load_manifest,
    source_path,
    write_summary,
)

__all__ = [
    "NO_ANSWER",
    "boxed_extractor",
    "answer_symbol",
    "plugin_mi_nats",
    "bucket_influence",
    "format_p",
    "read_jsonl",
    "load_manifest",
    "source_path",
    "git_sha",
    "build_summary",
    "write_summary",
]

#: The symbol used when no answer could be extracted from a downstream output.
#: It is a real symbol for the estimator: "produced nothing parseable" is a
#: distinguishable outcome, exactly as in `e11_paired.py` ("<none>").
NO_ANSWER = "<NONE>"

_BOXED = "\\boxed{"


# -----------------------------------------------------------------------------
# text -> answer symbol
# -----------------------------------------------------------------------------


def boxed_extractor(text: Optional[str]) -> str:
    r"""Default extractor: the content of the LAST balanced ``\boxed{...}``.

    Brace matching is done by counting, so ``\boxed{\frac{1}{2}}`` returns
    ``\frac{1}{2}``. Surrounding whitespace is stripped. If there is no
    well-formed occurrence, or the content is empty, the result is
    :data:`NO_ANSWER`.

    Limits, stated because this is a placeholder: no escape handling (a literal
    ``\{`` counts as an opening brace) and no mathematical normalisation.
    The driver replaces this with the repository's own math parser at run time;
    the signature (str -> str) is the contract, not the parsing quality.
    """
    s = "" if text is None else str(text)
    out = NO_ANSWER
    start = s.find(_BOXED)
    while start != -1:
        i = start + len(_BOXED)
        depth = 1
        while i < len(s) and depth > 0:
            ch = s[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            i += 1
        if depth == 0:
            content = s[start + len(_BOXED): i - 1].strip()
            out = content if content else NO_ANSWER
        start = s.find(_BOXED, start + 1)
    return out


def answer_symbol(text: Optional[str], extractor: Callable[[str], Optional[str]]) -> str:
    """Downstream output text -> answer symbol, through the injected extractor.

    `extractor` has signature ``str -> str | None``. A None result, or a result
    that is empty after stripping, becomes :data:`NO_ANSWER`, so the estimator
    always sees a string.
    """
    raw = extractor("" if text is None else str(text))
    if raw is None:
        return NO_ANSWER
    sym = str(raw).strip()
    return sym if sym else NO_ANSWER


# -----------------------------------------------------------------------------
# the estimator
# -----------------------------------------------------------------------------


def plugin_mi_nats(y_by_j: Sequence[Sequence[str]], *, miller_madow: bool = True) -> float:
    """Plug-in I(J; Y) in nats with J uniform, optionally Miller-Madow corrected.

    `y_by_j[j]` is the list of answer symbols observed under alternative j (one
    per replay). Groups with no samples are dropped before anything else, as in
    `p1rb_analysis.py`, and n_j counts the groups that remain. With fewer than
    two groups the mutual information is identically zero (both corrections
    cancel), so 0.0 is returned.

    The value is returned as computed, negative values included.
    """
    groups: List[List[str]] = [list(ys) for ys in y_by_j if len(ys) > 0]
    n_j = len(groups)
    if n_j < 2:
        return 0.0

    p_j = 1.0 / n_j
    n_total = float(sum(len(g) for g in groups))

    p_y: Dict[str, float] = {}
    h_cond = 0.0
    corr_cond = 0.0
    for g in groups:
        counts = Counter(g)
        n_g = float(len(g))
        h_g = 0.0
        for sym, c in counts.items():
            p = c / n_g
            h_g -= p * math.log(p)
            p_y[sym] = p_y.get(sym, 0.0) + p_j * p
        h_cond += p_j * h_g
        corr_cond += p_j * (len(counts) - 1) / (2.0 * n_g)

    h_y = -sum(p * math.log(p) for p in p_y.values() if p > 0.0)
    mi = h_y - h_cond
    if miller_madow:
        m_y = sum(1 for p in p_y.values() if p > 0.0)
        mi += (m_y - 1) / (2.0 * n_total) - corr_cond
    return float(mi)


def _credit_indices(meta: Mapping[str, Any], n_actions: int) -> List[int]:
    """Indices taking part in credit, following `metrics._credit_action_indices`.

    Default: every alternative. With `meta.credit_n` present: the first
    `credit_n` alternatives, clipped into range.
    """
    if n_actions <= 0:
        return []
    raw = meta.get("credit_n", n_actions)
    try:
        credit_n = int(raw)
    except (TypeError, ValueError):
        credit_n = n_actions
    credit_n = max(0, min(n_actions, credit_n))
    return list(range(credit_n))


def _null_arm_index(meta: Mapping[str, Any]) -> Optional[int]:
    raw = meta.get("null_arm", None)
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def bucket_influence(
    bucket: Mapping[str, Any],
    extractor: Callable[[str], Optional[str]],
    *,
    credit_only: bool = True,
) -> float:
    """Influence of one decision point, in nats, Miller-Madow corrected.

    Y comes from `candidates[j].next_actions` mapped through `extractor`.
    With `credit_only` the credit set convention of `meta.credit_n` applies.
    The E3a null arm (`meta.null_arm`) is always excluded: the injected empty
    message is not a sampled alternative, so it must not inflate or deflate the
    spread of downstream answers.
    """
    cands = bucket.get("candidates", []) or []
    meta = bucket.get("meta", {}) or {}
    if not isinstance(meta, Mapping):
        meta = {}

    idx = _credit_indices(meta, len(cands)) if credit_only else list(range(len(cands)))
    j0 = _null_arm_index(meta)
    if j0 is not None:
        idx = [j for j in idx if j != j0]

    y_by_j: List[List[str]] = []
    for j in idx:
        nexts = cands[j].get("next_actions", []) or []
        if not isinstance(nexts, list):
            nexts = [nexts]
        y_by_j.append([answer_symbol(t, extractor) for t in nexts])
    return plugin_mi_nats(y_by_j)


# -----------------------------------------------------------------------------
# package-level conventions: p-value strings, manifest, summary.json
#
# The implementations moved to `summary.py` (ruling B13); they are imported at
# the top of this file so `influence.format_p` and the rest keep their names.
# -----------------------------------------------------------------------------


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read one JSON object per line, skipping blank lines."""
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out
