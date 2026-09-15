# c3/analysis/rebuild/influence.py
"""Answer-level plug-in mutual information I(J; Y | h), the influence estimator
frozen in the analysis plan of the rebuild experiments (E3a, E5).

What it measures: inside one decision point (bucket), J indexes the alternative
upstream messages and Y is the *answer symbol* of the downstream output that
followed. A high value means "which alternative was sent still shows up in the
final answer"; a value near zero means the downstream agent produced the same
answer whatever it was handed.

Frozen conventions (the analysis plan, E5; the results layout, section 4):
- Y is the ANSWER symbol, not the raw text and not a hash of the whole message.
  The mapping text -> symbol is injected (see `answer_symbol`), so this module
  never parses mathematics itself.
- Y exists only where the downstream output STATES an answer: a `#### ...` line,
  a `\boxed{...}`, or a `Final answer: ...` style anchor. An output that trails
  off in prose has no answer and contributes `<NONE>` (the maintainers' decision
  of 2026-09-15; the rule is implemented in `math_extractor` and the reason is
  written out there).
- J is uniform: p_j = 1/n_j, matching the `plugin_mi` of the earlier
  paired-analysis script. This is NOT the sample-mass weighting used by the
  older `c3.analysis.metrics.influence_mi`.
- Entropies use the natural logarithm, so the unit is nats.
- Miller-Madow bias correction: every entropy term gets + (m - 1) / (2 N), with
  m the number of non-empty cells of that distribution and N its sample count.
  MI = H_MM(Y) - sum_j p_j H_MM(Y | j).
- The result is NOT clipped at zero. The correction can push an already small
  plug-in value below zero and the analysis plan says to report it as measured.

Differences from `c3.analysis.metrics.influence_mi` (the old estimator, kept
only for contrast): that one hashes canonicalized full text, adds a Laplace
alpha, pools Y into a top-K vocabulary plus OTHER, weights J by sample mass and
clips at zero. None of that is done here.

The package-level value conventions (p-value strings, manifest loading, the
summary.json writer) now live in `c3.analysis.rebuild.summary`, which both
aggregation families share (the maintainers' decision of 2026-09-15). They are imported
back into this module under their old names, so everything that reads them from
here keeps working.

Module-level imports are limited to numpy, scipy and the standard library. The
repository's own math parser, which `math_extractor` runs, is imported inside
that function instead, so importing this module still costs nothing beyond the
standard library.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

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
    "EXPLICIT_ANSWER_METHODS",
    "boxed_extractor",
    "math_extractor",
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
#: distinguishable outcome, exactly as in the earlier script ("<none>").
NO_ANSWER = "<NONE>"

#: The methods of `c3.envs.math.parsing.parse_math_answer` that mean the output
#: STATED an answer, and therefore the only ones `math_extractor` accepts
#: (the maintainers' decision of 2026-09-15). The parser's remaining methods are `empty`,
#: `empty_lines`, `anchor_inline` and `last_line`; the first two carry no token
#: at all, and the last is the prose fallback this rule exists to reject.
EXPLICIT_ANSWER_METHODS = ("boxed", "hash", "anchor")

_BOXED = "\\boxed{"


# -----------------------------------------------------------------------------
# text -> answer symbol
# -----------------------------------------------------------------------------


def boxed_extractor(text: Optional[str]) -> str:
    r"""Placeholder extractor: the content of the LAST balanced ``\boxed{...}``.

    Brace matching is done by counting, so ``\boxed{\frac{1}{2}}`` returns
    ``\frac{1}{2}``. Surrounding whitespace is stripped. If there is no
    well-formed occurrence, or the content is empty, the result is
    :data:`NO_ANSWER`.

    Limits, stated because this is a placeholder implementation kept for the
    tests: no escape handling (a literal ``\{`` counts as an opening brace) and
    no mathematical normalisation. On the real E3a buckets it fails on about
    half the downstream outputs, because the models end on ``Final answer: ...``
    rather than on a box. :func:`math_extractor` is the extractor the
    aggregation runs; this one stays because its output is hand-checkable, which
    is what the unit tests of the estimator need.
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


#: Cache for the repository's math parser, filled on the first `math_extractor`
#: call so this module's import surface stays numpy / scipy / standard library.
_MATH_PARSER: Optional[Tuple[Callable[[str], Any], Callable[[str], str]]] = None


def _math_parser() -> Tuple[Callable[[str], Any], Callable[[str], str]]:
    """`(parse_math_answer, normalize_expr)` of the repository, imported once.

    `c3.envs.math.parsing.parse_math_answer` is the extraction half and
    `c3.envs.math.backends.marft.normalize.normalize_expr` the normalisation
    half. Both are pure standard library, so this import pulls in no optional
    dependency of the math backends (the SymPy-based graders sit behind the
    guarded stubs of `c3.envs.math.backends.marft.__init__` and are not touched).
    """
    global _MATH_PARSER
    if _MATH_PARSER is None:
        from c3.envs.math.backends.marft.normalize import normalize_expr
        from c3.envs.math.parsing import parse_math_answer
        _MATH_PARSER = (parse_math_answer, normalize_expr)
    return _MATH_PARSER


def math_extractor(text: Optional[str]) -> str:
    r"""The extractor the E3a aggregation runs: EXPLICIT answers only.

    `aggregate_e5` still offers only `boxed_extractor`, so E5 does not reach this
    function yet; putting it there is the maintainers' call, not this module's.

    Three steps, the first two taken from the repository rather than
    reimplemented here:

    1. `parse_math_answer` returns ``(token, method)``. It tries, in order, the
       last ``#### ...`` line (``hash``), the last balanced ``\boxed{...}``
       (``boxed``), a ``Final answer: ...`` style anchor (``anchor``), and
       finally the last non-empty line (``last_line``).
    2. Only :data:`EXPLICIT_ANSWER_METHODS` count as an answer. Any other method,
       ``last_line`` above all, returns :data:`NO_ANSWER` (the maintainers'
       decision of 2026-09-15).
    3. `normalize_expr` rewrites the accepted token into one shape, so
       ``$\frac{1}{2}$`` and ``\frac{1}{2}`` become the same symbol.

    WHY step 2. Influence measures whether the downstream ANSWER moves with the
    alternative that was sent, so the question step 1 has to answer is "what
    answer did this replay state", and the parser's last resort does not answer
    it: it hands back the last line of the working, which is a sentence. Such a
    sentence is all but unique to its replay, so counting it as an answer symbol
    manufactures information no answer carries. Measured on the real E3a credit
    set, the prose fallback stood for about a third of the replays and about six
    tenths of the mean influence (measured 2026-09-15). The analysis plan calls
    this estimator answer-level; "this output states no
    answer" is the honest reading of a replay that ends in prose, and
    :data:`NO_ANSWER` is a real symbol of the estimator rather than a dropped
    sample, so nothing is thrown away by saying so.

    This is the implementation of the estimator, not a criterion: it fixes what
    the word "answer" means in "answer-level", and it changes no threshold, no
    stratum rule and no test.

    Step 3 is best effort and is the repository's, not this module's. It folds
    ``\frac`` and ``\sqrt``, the math delimiters and the LaTeX spacing commands,
    and it does NOT fold ``\dfrac`` or ``\tfrac``, so those keep symbols of their
    own. That is a property of the extractor which shows up in the influence
    numbers, which is why it is written down here.

    Step 3 emptying a non-empty step 1 token is not "no answer": the answer was
    stated and only the normaliser had nothing left to keep (it drops characters
    outside its allowed set), so the stripped step 1 string is returned instead
    of :data:`NO_ANSWER`.
    """
    parse_math_answer, normalize_expr = _math_parser()
    s = "" if text is None else str(text)
    if not s.strip():
        return NO_ANSWER
    token, method = parse_math_answer(s)
    if method not in EXPLICIT_ANSWER_METHODS:
        return NO_ANSWER
    if token is None:
        return NO_ANSWER
    raw = str(token).strip()
    if not raw:
        return NO_ANSWER
    normalised = str(normalize_expr(raw)).strip()
    return normalised if normalised else raw


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
# The implementations moved to `summary.py`; they are imported at
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
