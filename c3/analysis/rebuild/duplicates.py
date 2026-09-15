# c3/analysis/rebuild/duplicates.py
"""
Duplicate alternatives inside a group (E1c: the duplicate-rate monitor).

Two alternatives count as duplicates when their action text is identical after
normalization (leading and trailing whitespace removed, every run of whitespace
folded to one space). The rate a group reports is the share of its alternatives
that have at least one duplicate, so four alternatives with three identical
texts report 0.75, not the pair-level share.

The near-duplicate rate is the secondary reading: alternatives whose whitespace
split token sets overlap above a Jaccard threshold. It is the same idea the
Tier-A script 30_analysis/server_tierA/entropy_contrast.py used, moved from a
pair rate to the alternative rate so that both numbers have the same denominator.
Exact duplicates are always near duplicates, since identical text gives identical
token sets.

Only numpy and the standard library are imported here.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

import numpy as np

Bucket = Mapping[str, Any]

__all__ = [
    "normalize_text",
    "token_set",
    "jaccard",
    "bucket_duplicate_rate",
    "bucket_near_duplicate_rate",
    "cell_duplicate_report",
]

DEFAULT_JACCARD = 0.9


def normalize_text(s: Any) -> str:
    """Strip the ends and fold every run of whitespace to a single space.

    Case is left alone: two alternatives that differ only in capitalization are
    different actions, and the exact-duplicate rule is meant to catch the
    sampler returning the same string twice.
    """
    if s is None:
        return ""
    return " ".join(str(s).split())


def token_set(s: Any) -> Set[str]:
    """Whitespace split tokens of the normalized text."""
    return set(normalize_text(s).split())


def _jaccard_sets(ta: Set[str], tb: Set[str]) -> float:
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / max(1, len(ta | tb))


def jaccard(a: Any, b: Any) -> float:
    """Token Jaccard overlap of two texts; two empty texts count as identical."""
    return _jaccard_sets(token_set(a), token_set(b))


def _texts(bucket: Bucket) -> List[str]:
    cands = bucket.get("candidates") or []
    return [str(c.get("action_text", "") or "") for c in cands]


def bucket_duplicate_rate(bucket: Bucket) -> Optional[float]:
    """Share of the group's alternatives that repeat another alternative exactly.

    None when the group carries fewer than two alternatives, where the question
    does not arise.
    """
    texts = [normalize_text(t) for t in _texts(bucket)]
    if len(texts) < 2:
        return None
    counts = Counter(texts)
    return sum(1 for t in texts if counts[t] > 1) / len(texts)


def bucket_near_duplicate_rate(bucket: Bucket, *, jaccard: float = DEFAULT_JACCARD) -> Optional[float]:
    """Share of the group's alternatives whose token overlap with some other
    alternative is above the threshold (strictly above, as in the Tier-A script).
    """
    texts = _texts(bucket)
    if len(texts) < 2:
        return None
    sets = [token_set(t) for t in texts]
    flagged = [False] * len(texts)
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if _jaccard_sets(sets[i], sets[j]) > jaccard:
                flagged[i] = True
                flagged[j] = True
    return sum(flagged) / len(texts)


def cell_duplicate_report(
    buckets: Sequence[Bucket], *, jaccard: float = DEFAULT_JACCARD
) -> Dict[str, Any]:
    """Mean duplicate and near-duplicate rate over the groups of one cell.

    `n` counts the groups entering the means: groups with fewer than two
    alternatives are left out of both.
    """
    dups: List[float] = []
    nears: List[float] = []
    for b in buckets:
        rate = bucket_duplicate_rate(b)
        if rate is None:
            continue
        dups.append(rate)
        near = bucket_near_duplicate_rate(b, jaccard=jaccard)
        nears.append(float(near) if near is not None else 0.0)
    return {
        "dup_rate_mean": float(np.mean(dups)) if dups else None,
        "near_dup_rate_mean": float(np.mean(nears)) if nears else None,
        "n": len(dups),
    }
