# c3/analysis/rebuild/noise_law.py
"""
E1b: measured against predicted estimator noise of the leave-one-out advantage.

Each cell of the grid holds the same groups replayed twice under two independent
continuation seeds (seedA and seedB). For one group:

  A^1, A^2   the two leave-one-out advantage vectors, each from all of that
             seed's replays
  measured   mean((A^1 - A^2)^2) / 2, the element-wise second moment over the
             group's n alternatives (ddof=0, the default)
  predicted  sigma^2 n^2 / (B_u (n - 1)), with B_u = n c and sigma^2 the mean
             over alternatives of the within-alternative sample variance
             (ddof=1) of that alternative's replay returns
  ratio      measured / predicted

The cell value is the mean ratio over its groups, with a percentile bootstrap
interval over groups. The grid value `max_dev_pct` is the largest |ratio - 1| in
percent and `worst_ratio` is the cell ratio farthest from one.

Both conventions are the driver's ruling of 2026-09-15 (preregistration
revision 10), and both follow from the derivation:

- the difference of the two advantage vectors has mean exactly zero, because a
  leave-one-out vector sums to zero, so the mean of squares is the estimator of
  its variance and its expectation is sigma^2 n^2 / (B_u (n - 1)) exactly. An
  unbiased sample variance (ddof=1) would instead multiply every group by
  n / (n - 1), which at two alternatives doubles the ratio whatever the data
  says. `ddof` stays a keyword so that scaling can still be shown, but the
  default is the convention the two sides share.
- sigma^2 in the law is the return variance of a single alternative. Pooling
  every return of the group instead would add the spread between alternatives,
  which drives the ratio down whenever the alternatives differ in quality and
  has nothing to do with the estimator being right.

Only numpy and the standard library plus this package's splithalf helpers are
imported here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .splithalf import boot_ci, candidate_returns, loo_adv

Bucket = Mapping[str, Any]

__all__ = [
    "DDOF",
    "GroupNoise",
    "pair_buckets",
    "group_noise",
    "cell_noise_ratio",
    "grid_report",
]

PAIR_FIELDS = ("bucket_id", "question_id")

# Element-wise convention of the measured noise: mean of squares, not an
# unbiased sample variance (driver ruling A1, preregistration revision 10).
DDOF = 0


@dataclass
class GroupNoise:
    """One group's measured and predicted advantage noise."""

    key: str
    n: int
    c: float
    measured: float
    predicted: float
    ratio: float


def _index_by(buckets: Sequence[Bucket], field: str) -> Optional[Dict[str, Bucket]]:
    """Map field value to bucket, or None when the field is missing or repeats."""
    out: Dict[str, Bucket] = {}
    for b in buckets:
        val = b.get(field)
        if val is None:
            return None
        key = str(val)
        if key in out:
            return None
        out[key] = b
    return out or None


def pair_buckets(
    buckets_a: Sequence[Bucket], buckets_b: Sequence[Bucket], *, field: str = "auto"
) -> Tuple[str, List[Tuple[str, Bucket, Bucket]]]:
    """Pair the two seeds' groups, by bucket_id when that identifies them and by
    question_id otherwise. Returns the field used and the pairs, ordered by key.
    """
    fields = PAIR_FIELDS if field == "auto" else (field,)
    for name in fields:
        ia, ib = _index_by(buckets_a, name), _index_by(buckets_b, name)
        if ia is None or ib is None:
            continue
        common = sorted(set(ia) & set(ib))
        if common:
            return name, [(k, ia[k], ib[k]) for k in common]
    raise ValueError(
        "cannot pair the two seeds: no field of %s is present, unique and shared"
        % (fields,)
    )


def group_noise(
    bucket_a: Bucket, bucket_b: Bucket, *, key: str = "", ddof: int = DDOF
) -> Optional[GroupNoise]:
    """Measured and predicted advantage noise of one group, or None when the
    group cannot carry the comparison (different alternative counts on the two
    seeds, fewer than two alternatives, no replays, no alternative with two
    replays to estimate its variance from, or a group whose returns are all
    identical so the prediction is zero).
    """
    rets_a, rets_b = candidate_returns(bucket_a), candidate_returns(bucket_b)
    n = len(rets_a)
    if n < 2 or len(rets_b) != n:
        return None
    if any(len(r) == 0 for r in rets_a) or any(len(r) == 0 for r in rets_b):
        return None

    adv_a = loo_adv([float(np.mean(r)) for r in rets_a])
    adv_b = loo_adv([float(np.mean(r)) for r in rets_b])
    measured = float(np.var(adv_a - adv_b, ddof=ddof)) / 2.0

    # sigma^2: the within-alternative sample variance, averaged over alternatives.
    # The two seeds are independent replays of the same alternative, so they pool
    # (the group in E1b is the pair, as the cell is collected).
    per_alt: List[float] = []
    for ra, rb in zip(rets_a, rets_b):
        vals = [float(v) for v in ra] + [float(v) for v in rb]
        if len(vals) >= 2:
            per_alt.append(float(np.var(vals, ddof=1)))
    if not per_alt:
        return None
    sigma2 = float(np.mean(per_alt))
    # Replays per alternative per seed: the mean over alternatives and seeds, which
    # is the planned constant c whenever the cell was collected as designed.
    counts = [len(r) for r in rets_a] + [len(r) for r in rets_b]
    c = float(np.mean(counts))
    budget = n * c
    predicted = sigma2 * n * n / (budget * (n - 1))
    if not (predicted > 0):
        return None
    return GroupNoise(key=key, n=n, c=c, measured=measured, predicted=predicted,
                      ratio=measured / predicted)


def cell_noise_ratio(
    buckets_a: Sequence[Bucket],
    buckets_b: Sequence[Bucket],
    *,
    n_boot: int = 2000,
    seed: int = 0,
    ddof: int = DDOF,
) -> Dict[str, Any]:
    """Mean measured-over-predicted ratio for one grid cell, with a bootstrap
    interval over groups.

    `n_skipped` counts paired groups that produced no ratio (see group_noise).
    """
    field, pairs = pair_buckets(buckets_a, buckets_b)
    ratios: List[float] = []
    skipped = 0
    for key, ba, bb in pairs:
        gn = group_noise(ba, bb, key=key, ddof=ddof)
        if gn is None:
            skipped += 1
            continue
        ratios.append(gn.ratio)
    rng = np.random.default_rng([seed, 7])
    lo, hi = boot_ci(ratios, rng, n_boot) if ratios else (None, None)
    return {
        "ratio": float(np.mean(ratios)) if ratios else None,
        "ci_lo": lo,
        "ci_hi": hi,
        "n_groups": len(ratios),
        "n_pairs": len(pairs),
        "n_skipped": skipped,
        "pair_field": field,
        "ratios": ratios,
    }


def grid_report(
    cells: Mapping[Tuple[int, int], Tuple[Sequence[Bucket], Sequence[Bucket]]],
    *,
    n_boot: int = 2000,
    seed: int = 0,
    ddof: int = DDOF,
) -> Dict[str, Any]:
    """Every cell's ratio plus the two grid-level numbers the manifest prints.

    `cells` maps (n, c) to the cell's (seedA buckets, seedB buckets).
    """
    ratio: Dict[Tuple[int, int], Optional[float]] = {}
    ratio_ci: Dict[Tuple[int, int], Tuple[Optional[float], Optional[float]]] = {}
    n_groups: Dict[Tuple[int, int], int] = {}
    for cell in sorted(cells):
        buckets_a, buckets_b = cells[cell]
        rep = cell_noise_ratio(buckets_a, buckets_b, n_boot=n_boot, seed=seed, ddof=ddof)
        ratio[cell] = rep["ratio"]
        ratio_ci[cell] = (rep["ci_lo"], rep["ci_hi"])
        n_groups[cell] = rep["n_groups"]

    filled = {k: v for k, v in ratio.items() if v is not None}
    worst_cell = max(sorted(filled), key=lambda k: abs(filled[k] - 1.0)) if filled else None
    return {
        "ratio": ratio,
        "ratio_ci": ratio_ci,
        "n_groups": n_groups,
        "max_dev_pct": (100.0 * abs(filled[worst_cell] - 1.0)) if worst_cell else None,
        "worst_ratio": filled[worst_cell] if worst_cell else None,
        "worst_cell": worst_cell,
        "n_cells": len(filled),
    }
