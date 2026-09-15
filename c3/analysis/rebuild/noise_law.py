# c3/analysis/rebuild/noise_law.py
"""
Measured against predicted estimator noise of the leave-one-out advantage.

Two experiments read this one estimator, and the arithmetic below does not know
which of them is calling:

- E1b, the grid. Each cell holds the same groups replayed twice under two
  independent continuation seeds (seedA and seedB), which share the alternatives
  and redraw only the replays.
- E1, the sweep. Each cell carries one replay set per group, and its even and
  odd replays are two independent estimates of the same advantage vector at the
  same decision point. Every scan cell therefore already carries a noise
  measurement, which is what revision 19 of the preregistration (2026-09-15)
  reads as its secondary criteria a' and b'. Nothing is rerun for it.

For one group, given two sides A and B (two seeds, or the two halves):

  A^1, A^2   the two leave-one-out advantage vectors, each from that side's
             replays
  measured   mean((A^1 - A^2)^2) / 2, the element-wise second moment over the
             group's n alternatives (ddof=0, the default)
  predicted  sigma^2 n^2 / (B (n - 1)), with B = n c, c the mean replays per
             alternative per SIDE, and sigma^2 the mean over alternatives of the
             within-alternative sample variance (ddof=1) of that alternative's
             replay returns, both sides pooled
  ratio      measured / predicted

Reading c per side is what makes one formula serve both callers: a seed of E1b
replays c times, so B_u = n c, while a half of E1 holds c / 2 replays, so
B = n c / 2. There is no second expression to keep in step, and no budget is
written down anywhere: it is counted off the buckets.

The cell value is the mean ratio over its groups, with a percentile bootstrap
interval over groups. The grid value `max_dev_pct` is the largest |ratio - 1| in
percent and `worst_ratio` is the cell ratio farthest from one. The three
cross-cell readings of the sweep (median, rank correlation against branching,
and the band test at branching 4) are `ratio_median`, `ratio_vs_n` and
`all_in_band`.

Both conventions are the maintainers' decision of 2026-09-15 (analysis plan
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

The halves reading adds no arithmetic of its own: it takes the split and the
count clauses of the exclusion rule from splithalf.py (`even_odd_halves`), and
the arithmetic from `returns_noise`, the same function E1b calls. It parts from
the reliability estimator on one clause only, and deliberately (revision 19(h)
of the preregistration, 2026-09-15): a group whose half came back flat stays in
the noise pool, because the noise of a zero estimate is a real reading, while a
constant vector has no ranking for a correlation to take. `cell_noise_ratio_halves`
counts those groups and also reports what the cell would read without them.

Only numpy and the standard library plus this package's splithalf helpers are
imported here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .splithalf import (
    CI_HI_PCT,
    CI_LO_PCT,
    DEFAULT_MIN_CANDS,
    advantages_are_flat,
    boot_ci,
    candidate_returns,
    even_odd_halves,
    loo_adv,
    spearman_rows,
)

Bucket = Mapping[str, Any]

__all__ = [
    "DDOF",
    "RATIO_BAND",
    "GroupNoise",
    "pair_buckets",
    "returns_noise",
    "group_noise",
    "group_noise_halves",
    "cell_noise_ratio",
    "cell_noise_ratio_halves",
    "grid_report",
    "in_band",
    "all_in_band",
    "ratio_median",
    "ratio_vs_n",
]

PAIR_FIELDS = ("bucket_id", "question_id")

# Element-wise convention of the measured noise: mean of squares, not an
# unbiased sample variance (the maintainers' decision, analysis plan revision 10).
DDOF = 0

# The band of the preregistered primary criterion of E1b ("all nine ratios inside
# [0.7, 1.4]", the frozen analysis plan, restated as the primary criterion in
# revision 19 of 2026-09-15). The same band decides the two secondary criteria
# that revision reads off the sweep cells: a' wants the median of the cell ratios
# inside it, b' wants every one of the six workflows at branching 4 inside it.
# A band is a judgement and belongs to the preregistration; this constant only
# spells it once so no caller retypes it.
RATIO_BAND = (0.7, 1.4)


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


def _rows(matrix: np.ndarray) -> List[List[float]]:
    """An (n, k) return matrix as the per-alternative lists `returns_noise` takes."""
    return [[float(v) for v in row] for row in matrix]


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


def returns_noise(
    rets_a: Sequence[Sequence[float]],
    rets_b: Sequence[Sequence[float]],
    *,
    key: str = "",
    ddof: int = DDOF,
) -> Optional[GroupNoise]:
    """Measured and predicted advantage noise of one group, from two independent
    estimates of the same advantage vector.

    `rets_a` and `rets_b` are the per-alternative replay returns of the two
    sides, in the same alternative order. What a side is belongs to the caller:
    E1b hands over two continuation seeds of one group, the E1 sweep hands over
    the even and the odd replays of one group. The arithmetic does not know the
    difference, which is the point of it being one function.

    None when the group cannot carry the comparison (different alternative counts
    on the two sides, fewer than two alternatives, a side with no replays for some
    alternative, no alternative with two replays to estimate its variance from, or
    a group whose returns are all identical so the prediction is zero).
    """
    n = len(rets_a)
    if n < 2 or len(rets_b) != n:
        return None
    if any(len(r) == 0 for r in rets_a) or any(len(r) == 0 for r in rets_b):
        return None

    adv_a = loo_adv([float(np.mean(r)) for r in rets_a])
    adv_b = loo_adv([float(np.mean(r)) for r in rets_b])
    measured = float(np.var(adv_a - adv_b, ddof=ddof)) / 2.0

    # sigma^2: the within-alternative sample variance, averaged over alternatives.
    # The two sides are independent replays of the same alternative, so they pool
    # (in E1b that is the pair of seeds, as the cell is collected; in the sweep it
    # is the group's own replays, the two halves put back together).
    per_alt: List[float] = []
    for ra, rb in zip(rets_a, rets_b):
        vals = [float(v) for v in ra] + [float(v) for v in rb]
        if len(vals) >= 2:
            per_alt.append(float(np.var(vals, ddof=1)))
    if not per_alt:
        return None
    sigma2 = float(np.mean(per_alt))
    # Replays per alternative per SIDE: the mean over alternatives and sides, which
    # is the planned constant c whenever an E1b cell was collected as designed, and
    # half of it when the sides are the two halves of one sweep cell. The budget
    # follows the data rather than a written-down constant.
    counts = [len(r) for r in rets_a] + [len(r) for r in rets_b]
    c = float(np.mean(counts))
    budget = n * c
    predicted = sigma2 * n * n / (budget * (n - 1))
    if not (predicted > 0):
        return None
    return GroupNoise(key=key, n=n, c=c, measured=measured, predicted=predicted,
                      ratio=measured / predicted)


def group_noise(
    bucket_a: Bucket, bucket_b: Bucket, *, key: str = "", ddof: int = DDOF
) -> Optional[GroupNoise]:
    """E1b's reading: the two sides are the two continuation seeds of one group.

    Every condition and every convention is `returns_noise`; this call only says
    which returns are the two sides.
    """
    return returns_noise(candidate_returns(bucket_a), candidate_returns(bucket_b),
                         key=key, ddof=ddof)


def _halves_of(
    bucket: Bucket, min_cands: int
) -> Optional[Tuple[np.ndarray, np.ndarray, bool]]:
    """The group's two halves and whether either of them is flat, or None when the
    alternative-count or replay-count clause already drops the group."""
    halves = even_odd_halves(bucket, min_cands=min_cands)
    if halves is None:
        return None
    even, odd = halves
    flat = (advantages_are_flat(loo_adv(even.mean(axis=1)))
            or advantages_are_flat(loo_adv(odd.mean(axis=1))))
    return even, odd, flat


def group_noise_halves(
    bucket: Bucket,
    *,
    key: str = "",
    min_cands: int = DEFAULT_MIN_CANDS,
    ddof: int = DDOF,
    drop_flat_halves: bool = False,
) -> Optional[GroupNoise]:
    """The sweep's reading: the two sides are the even and the odd replays of one
    group, so a cell measured once still carries a noise measurement.

    None when the group fails a count clause of the exclusion rule (fewer than
    `min_cands` alternatives, fewer than two replays) or when `returns_noise`
    cannot read it. The split and the count clauses come from splithalf.py, so
    this reading and the cell's reliability are taken on the same replays.

    `c` on the result is the mean of the two half sizes, so `n * c` is the budget
    behind ONE half. At the planned four replays that is exactly two per half.

    `drop_flat_halves` is the third clause of that rule, and it is a keyword,
    defaulting to OFF, because the two statistics do not need it equally
    (revision 19(h) of the preregistration, 2026-09-15, which fixes this as the
    implementation and leaves the criteria as they stand). The correlation needs
    the clause: a constant advantage vector has no ranking, so the correlation is
    undefined. The noise reading does not: Var(A_even - A_odd) / 2 is defined when
    one half came out flat, and dropping those groups is a selection on the very
    quantity being measured. A flat half is the case where one side's estimate is
    exactly zero, which is the low tail of the difference, so dropping them lifts
    the measured noise, and it lifts it hardest where flat halves are common, that
    is at few alternatives and few replays. That lift falls with the number of
    alternatives, so it would plant a downward trend in the ratio against
    branching, which is exactly what secondary criterion a' of the preregistration
    tests. Keeping the groups costs comparability with the reliability pool;
    dropping them costs the criterion. `cell_noise_ratio_halves` reports the other
    pool's reading beside this one, so the choice stays visible in the data
    (`test_the_flat_half_clause_lifts_the_halves_ratio` measures what it is worth).
    """
    split = _halves_of(bucket, min_cands)
    if split is None:
        return None
    even, odd, flat = split
    if flat and drop_flat_halves:
        return None
    return returns_noise(_rows(even), _rows(odd), key=key, ddof=ddof)


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


def cell_noise_ratio_halves(
    buckets: Sequence[Bucket],
    *,
    n_boot: int = 2000,
    seed: int = 0,
    min_cands: int = DEFAULT_MIN_CANDS,
    ddof: int = DDOF,
    drop_flat_halves: bool = False,
) -> Dict[str, Any]:
    """Mean measured-over-predicted ratio for one sweep cell, read off the even and
    odd halves of the replays the cell already holds, with a percentile bootstrap
    interval over groups (the unit of these experiments).

    The keys of the report:

      ratio, ci_lo, ci_hi   the cell reading and its interval
      n_groups              groups that produced a ratio
      n_total               groups collected in the cell
      excluded_pct          share of the collected groups that produced none,
                            the same denominator the reliability rows print
      replays_per_half      mean over usable groups of the replays one half holds
                            per alternative, so the budget behind the prediction
                            can be read without reopening the buckets
      n_half_uneven         usable groups whose replay count is odd, so their two
                            halves differ by one replay and `replays_per_half` is
                            their average rather than either half's size; zero in
                            a cell collected at the planned four replays
      n_flat_half           groups that clear the alternative and replay counts
                            but come back with a flat half, counted whichever way
                            `drop_flat_halves` is set (see `group_noise_halves`)
      ratio_flat_dropped    the same cell read on the pool the flat-half clause
                            leaves, and `n_groups_flat_dropped` its size. Always
                            computed, so the sensitivity to that clause is in the
                            data instead of in an argument: the reading the
                            reliability pool would give is right there beside the
                            one this cell reports. With drop_flat_halves=True the
                            two are the same number by construction
      ratios                the per-group values, for a caller that pools cells
    """
    ratios: List[float] = []
    ratios_no_flat: List[float] = []
    half_sizes: List[float] = []
    n_flat_half = 0
    for bucket in buckets:
        split = _halves_of(bucket, min_cands)
        if split is None:
            continue
        even, odd, flat = split
        if flat:
            n_flat_half += 1
            if drop_flat_halves:
                continue
        gn = returns_noise(_rows(even), _rows(odd),
                           key=str(bucket.get("bucket_id") or ""), ddof=ddof)
        if gn is None:
            continue
        ratios.append(gn.ratio)
        if not flat:
            ratios_no_flat.append(gn.ratio)
        half_sizes.append(gn.c)
    rng = np.random.default_rng([seed, 8])
    lo, hi = boot_ci(ratios, rng, n_boot) if ratios else (None, None)
    n_total = len(buckets)
    return {
        "ratio": float(np.mean(ratios)) if ratios else None,
        "ci_lo": lo,
        "ci_hi": hi,
        "n_groups": len(ratios),
        "n_total": n_total,
        "excluded_pct": (100.0 * (n_total - len(ratios)) / n_total) if n_total else None,
        "replays_per_half": float(np.mean(half_sizes)) if half_sizes else None,
        "n_half_uneven": sum(1 for c in half_sizes if not float(c).is_integer()),
        "n_flat_half": n_flat_half,
        "ratio_flat_dropped": float(np.mean(ratios_no_flat)) if ratios_no_flat else None,
        "n_groups_flat_dropped": len(ratios_no_flat),
        "ratios": ratios,
    }


# -------------------------
# the cross-cell readings of the sweep (revision 19, criteria a' and b')
# -------------------------


def in_band(value: Optional[float], band: Tuple[float, float] = RATIO_BAND) -> bool:
    """Is one ratio inside the preregistered band, endpoints included."""
    if value is None or not np.isfinite(value):
        return False
    return band[0] <= float(value) <= band[1]


def all_in_band(values: Sequence[Optional[float]],
                band: Tuple[float, float] = RATIO_BAND) -> Optional[bool]:
    """Are all of these ratios inside the band? None on an empty list, because
    "nothing was measured" is not the same answer as "everything passed"."""
    if not values:
        return None
    return all(in_band(v, band) for v in values)


def ratio_median(values: Sequence[float]) -> Optional[float]:
    """Median of the cell ratios, the statistic criterion a' reads."""
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=float)))


def ratio_vs_n(
    pairs: Sequence[Tuple[float, float]],
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> Dict[str, Any]:
    """Rank correlation of the cell ratio against branching, with a percentile
    bootstrap interval over cells.

    `pairs` is one (n, ratio) per cell. The resampling unit is the cell, because
    the cell is the unit the correlation is computed over; the same choice
    `boot_ci` makes for a mean. Ties are everywhere here (six workflows share each
    branching), so the tie-corrected `spearman_rows` of splithalf.py does the
    ranking, and a resample that draws one branching only, or one ratio only, has
    no correlation at all and is dropped: `n_draws` counts the draws that were
    kept and `n_degenerate` those that were not.

    rho is None when fewer than two cells are given or when the cells do not vary
    in branching or in ratio.
    """
    ns = np.asarray([float(p[0]) for p in pairs], dtype=float)
    rs = np.asarray([float(p[1]) for p in pairs], dtype=float)
    m = ns.size
    out: Dict[str, Any] = {"rho": None, "ci_lo": None, "ci_hi": None, "n_cells": int(m),
                           "n_draws": 0, "n_degenerate": 0}
    if m < 2:
        return out
    rho = float(spearman_rows(ns[None, :], rs[None, :])[0])
    if not np.isfinite(rho):
        return out
    out["rho"] = rho
    if n_boot <= 0:
        return out
    rng = np.random.default_rng([seed, 9])
    idx = rng.integers(0, m, size=(n_boot, m))
    draws = spearman_rows(ns[idx], rs[idx])
    good = np.isfinite(draws)
    out["n_draws"] = int(good.sum())
    out["n_degenerate"] = int(draws.size - good.sum())
    if out["n_draws"]:
        kept = draws[good]
        out["ci_lo"] = float(np.percentile(kept, CI_LO_PCT))
        out["ci_hi"] = float(np.percentile(kept, CI_HI_PCT))
    return out


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
