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
summary.json writer) live at the bottom of this file because the work package's
file list is fixed and this is its base module: everything else in
`c3.analysis.rebuild` that WP-R2b owns imports them from here.

Module-level imports are limited to numpy, scipy and the standard library.
"""

from __future__ import annotations

import datetime
import json
import math
import os
import subprocess
from collections import Counter
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

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
# -----------------------------------------------------------------------------


def format_p(p: Optional[float]) -> Optional[str]:
    """The manifest's p-value string, or None when the p value is undefined.

    Convention from `10_paper/04_rebuild/_tools/manifest_skeleton.py`: a LaTeX
    math fragment without the letter p, ``$<\\!0.001$`` below 0.001 and
    ``$= 0.30$`` (two decimals) otherwise. None means "not computable"; the
    caller then leaves the key out of summary.json rather than writing null.
    """
    if p is None:
        return None
    try:
        value = float(p)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    if value < 0.001:
        return "$<\\!0.001$"
    return "$= %.2f$" % value


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


def load_manifest(path: str) -> Dict[str, Dict[str, Any]]:
    """Load `results/manifest.json`: a flat mapping key -> entry."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("manifest is not a JSON object: %s" % path)
    return data


def source_path(path: str) -> str:
    """Path as written into summary.json's `source`.

    The contract prints repository-relative paths such as
    ``20_data/results/E3a/a3/4b/empty/buckets.jsonl``. The aggregate scripts are
    handed an arbitrary directory, so the rule is mechanical: cut at the last
    ``20_data`` segment when there is one, otherwise keep the path as given.
    Separators are normalised to forward slashes either way.
    """
    parts = os.path.abspath(path).replace("\\", "/").split("/")
    if "20_data" in parts:
        cut = len(parts) - 1 - parts[::-1].index("20_data")
        return "/".join(parts[cut:])
    return path.replace("\\", "/")


def git_sha(cwd: Optional[str] = None) -> Optional[str]:
    """HEAD of the repository holding this file, or None if git is unavailable.

    Read-only (`git rev-parse HEAD`). Any failure returns None instead of
    raising: a missing sha must not stop an aggregation.
    """
    where = cwd or os.path.dirname(os.path.abspath(__file__))
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=where,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    sha = out.stdout.decode("utf-8", "replace").strip()
    return sha or None


def build_summary(
    experiment: str,
    script: str,
    keys: Mapping[str, Mapping[str, Any]],
    manifest: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Assemble the summary document of results contract section 3.4.

    `keys` maps a manifest key to {value, n, source, note}; the unit is taken
    from the manifest so the two cannot drift. Two mechanical refusals:
    a key absent from the manifest raises KeyError (the contract's "unknown key,
    exit"), and a key whose unit is "verdict" raises ValueError, because filling
    a verdict is the driver's action and never an aggregation's.
    """
    unknown = [k for k in keys if k not in manifest]
    if unknown:
        raise KeyError("keys absent from the manifest: %s" % ", ".join(sorted(unknown)))
    verdicts = [k for k in keys if str(manifest[k].get("unit")) == "verdict"]
    if verdicts:
        raise ValueError("refusing to write verdict keys: %s" % ", ".join(sorted(verdicts)))

    out_keys: Dict[str, Any] = {}
    for key in sorted(keys):
        entry = keys[key]
        item: Dict[str, Any] = {
            "value": entry.get("value"),
            "n": entry.get("n"),
            "unit": manifest[key].get("unit"),
            "source": list(entry.get("source") or []),
        }
        note = entry.get("note")
        if note:
            item["note"] = note
        out_keys[key] = item

    return {
        "experiment": experiment,
        "generated_by": {
            "script": script,
            "git_sha": git_sha(),
            "when": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "keys": out_keys,
    }


def write_summary(
    out_path: str,
    experiment: str,
    script: str,
    keys: Mapping[str, Mapping[str, Any]],
    manifest: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Build the summary document and write it as UTF-8 JSON."""
    doc = build_summary(experiment, script, keys, manifest)
    parent = os.path.dirname(os.path.abspath(out_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=1, sort_keys=False)
        fh.write("\n")
    return doc
