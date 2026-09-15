# c3/analysis/rebuild/noise_law.py
"""
E1b: measured against predicted estimator noise of the leave-one-out advantage.

Each cell of the grid holds the same groups replayed twice under two independent
continuation seeds (seedA and seedB). For one group:

  A^1, A^2   the two leave-one-out advantage vectors, each from all of that
             seed's replays
  measured   Var(A^1 - A^2) / 2, the element-wise variance over the group's n
             alternatives (ddof as given, 1 by default per the work order)
  predicted  sigma^2 n^2 / (B_u (n - 1)), with B_u = n c and sigma^2 the sample
             variance (ddof=1) of every replay return in the group
  ratio      measured / predicted

The cell value is the mean ratio over its groups, with a percentile bootstrap
interval over groups. The grid value `max_dev_pct` is the largest |ratio - 1| in
percent and `worst_ratio` is the cell ratio farthest from one.

Read the ratio with the derivation in mind: under independent replays with a
common return variance, the expected value of Var(A^1 - A^2)/2 with ddof=0
equals the predicted quantity exactly, while ddof=1 multiplies it by
n / (n - 1); and pooling sigma^2 over all alternatives makes it the total
return variance, which exceeds the within-alternative variance whenever the
alternatives differ in quality. Both effects move the ratio away from one
without anything being wrong in the data, which is why the estimator keeps a
`ddof` keyword instead of hard-coding one convention.

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
    "GroupNoise",
    "pair_buckets",
    "group_noise",
    "cell_noise_ratio",
    "grid_report",
]

PAIR_FIELDS = ("bucket_id", "question_id")


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
    bucket_a: Bucket, bucket_b: Bucket, *, key: str = "", ddof: int = 1
) -> Optional[GroupNoise]:
    """Measured and predicted advantage noise of one group, or None when the
    group cannot carry the comparison (different alternative counts on the two
    seeds, fewer than two alternatives, no replays, or a group whose returns are
    all identical so the prediction is zero).
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

    flat = [float(v) for r in rets_a for v in r] + [float(v) for r in rets_b for v in r]
    if len(flat) < 2:
        return None
    sigma2 = float(np.var(flat, ddof=1))
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
    ddof: int = 1,
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
    ddof: int = 1,
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
