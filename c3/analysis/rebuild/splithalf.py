# c3/analysis/rebuild/splithalf.py
"""
Split-half reliability of within-group credit, per group and per cell.

This is the rebuild-experiment (E1) reimplementation of the estimator that
30_analysis/local/splithalf_bootstrap_local.py defines; the definitions below
are the same arithmetic, written against the bucket schema of
`c3.analysis.buckets` and packaged as library calls instead of a report script.

Per group (one bucket = one decision point):
  c_min      = smallest number of replays over the group's alternatives
  q_A[j]     = mean return of alternative j on the even replay indices (i < c_min)
  q_B[j]     = mean return of alternative j on the odd replay indices  (i < c_min)
  adv(q)[j]  = q[j] - (sum(q) - q[j]) / (n - 1)          (leave one out)
  rho        = spearman(adv(q_A), adv(q_B))

Exclusion rule (preregistration, restated in the results contract section 4):
fewer than `min_cands` alternatives, fewer than two replays, or a half whose
advantages are constant. An excluded group returns None and never enters a mean.

Note (mechanical): adv() is an increasing affine map of q inside a group
(adv = q * n / (n - 1) - sum(q) / (n - 1)), so it never reorders the
alternatives; the split-half Spearman is the same on the advantages and on the
raw half means. The tests check that claim rather than assuming it.

Only numpy, scipy and the standard library are imported here: the module runs in
an analysis environment with no torch and no serving stack.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import stats

Bucket = Mapping[str, Any]

__all__ = [
    "DEFAULT_MIN_CANDS",
    "CellResult",
    "DeltaResult",
    "load_buckets",
    "candidate_returns",
    "loo_adv",
    "loo_adv_rows",
    "spearman_rows",
    "boot_ci",
    "bucket_splithalf",
    "bucket_splithalf_random",
    "cell_reliability",
    "paired_cell_delta",
    "cell_range",
    "range_bootstrap",
]

# The preregistered exclusion rule: a group needs at least three alternatives.
DEFAULT_MIN_CANDS = 3

# Bootstrap percentiles for a 95% interval.
CI_LO_PCT = 2.5
CI_HI_PCT = 97.5

# A Spearman value this close to one counts as +1 (with two alternatives the
# statistic can only be +1 or -1, so the comparison is exact in practice).
_PLUS_ONE = 0.999


# -------------------------
# IO and small helpers
# -------------------------


def load_buckets(path: str) -> List[Dict[str, Any]]:
    """Read one buckets.jsonl file as a list of plain dicts.

    UTF-8 is explicit (a Windows box with a cp1252 locale otherwise raises on
    these files). Blank lines are skipped; a line that is not JSON raises
    ValueError naming the file and the line number, because silently dropping a
    group would move every number computed from the file.
    """
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError("%s line %d is not JSON: %s" % (path, lineno, exc)) from exc
    return out


def candidate_returns(bucket: Bucket) -> List[List[float]]:
    """The per-alternative return lists of one bucket, in candidate order."""
    cands = bucket.get("candidates") or []
    return [list(c.get("returns") or []) for c in cands]


def n_candidates(bucket: Bucket) -> int:
    """How many alternatives the group carries.

    All of them are resampled from the frozen policy: the real action does not
    take a slot (contract revision 2, `include_real_as_j0 = false`). A group can
    still come back with fewer than the requested number, since the runner drops
    duplicate samples.
    """
    return len(bucket.get("candidates") or [])


def loo_adv(qhat: Sequence[float]) -> Optional[np.ndarray]:
    """Leave-one-out advantage of a vector of per-alternative mean returns."""
    q = np.asarray(qhat, dtype=float)
    n = q.size
    if n < 2:
        return None
    return q - (q.sum() - q) / (n - 1)


def loo_adv_rows(Q: np.ndarray) -> np.ndarray:
    """loo_adv applied along the last axis of a stack of rows."""
    Q = np.asarray(Q, dtype=float)
    total = Q.sum(axis=-1, keepdims=True)
    n = Q.shape[-1]
    return Q - (total - Q) / (n - 1)


def _rank_avg_rows(X: np.ndarray) -> np.ndarray:
    """Tie-corrected average ranks along the last axis of a 2-D array.

    rank_i = #{j : x_j < x_i} + (#{j : x_j == x_i} + 1) / 2, the mean of the
    positions a tie group occupies; the same convention scipy.stats.rankdata uses.
    """
    X = np.asarray(X, dtype=float)
    less = (X[:, :, None] > X[:, None, :]).sum(axis=2)
    eq = (X[:, :, None] == X[:, None, :]).sum(axis=2)
    return less + (eq + 1.0) / 2.0


def spearman_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Row-wise Spearman correlation of two (R, n) arrays.

    A row whose ranks are constant gives nan, which is what scipy returns in
    that case. Used for the random-split path, where one scipy call per split
    would dominate the cost; the tests check it against scipy with and without
    ties.
    """
    ra, rb = _rank_avg_rows(A), _rank_avg_rows(B)
    ca = ra - ra.mean(axis=1, keepdims=True)
    cb = rb - rb.mean(axis=1, keepdims=True)
    num = (ca * cb).sum(axis=1)
    den = np.sqrt((ca * ca).sum(axis=1) * (cb * cb).sum(axis=1))
    out = np.full(num.shape, np.nan, dtype=float)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


def boot_ci(
    values: Sequence[float],
    rng: np.random.Generator,
    n_boot: int,
    lo: float = CI_LO_PCT,
    hi: float = CI_HI_PCT,
) -> Tuple[Optional[float], Optional[float]]:
    """Percentile bootstrap interval for the mean of `values` (units resampled)."""
    v = np.asarray(values, dtype=float)
    m = v.size
    if m == 0 or n_boot <= 0:
        return None, None
    if m == 1:
        return float(v[0]), float(v[0])
    idx = rng.integers(0, m, size=(n_boot, m))
    means = v[idx].mean(axis=1)
    return float(np.percentile(means, lo)), float(np.percentile(means, hi))


# -------------------------
# per-group estimator
# -------------------------


def _group_matrix(bucket: Bucket, min_cands: int) -> Optional[np.ndarray]:
    """The (n, c_min) return matrix of a group, or None when the group is excluded
    by the alternative-count or replay-count clause of the exclusion rule."""
    rets = candidate_returns(bucket)
    if len(rets) < min_cands or len(rets) < 2 or not rets:
        return None
    c_min = min(len(r) for r in rets)
    if c_min < 2:
        return None
    return np.asarray([r[:c_min] for r in rets], dtype=float)


def _spearman_of_halves(q_a: Sequence[float], q_b: Sequence[float]) -> Optional[float]:
    """Spearman of the two half advantage vectors, or None when a half is flat."""
    adv_a, adv_b = loo_adv(q_a), loo_adv(q_b)
    if adv_a is None or adv_b is None:
        return None
    if not (np.std(adv_a) > 0 and np.std(adv_b) > 0):
        return None
    rho = stats.spearmanr(adv_a, adv_b).statistic
    if np.isnan(rho):
        return None
    return float(rho)


def _even_odd_means(vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Half means on the even and odd replay indices below c_min."""
    return vals[:, 0::2].mean(axis=1), vals[:, 1::2].mean(axis=1)


def _random_half_means(vals: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """One random half split: each alternative's replay indices are permuted
    independently and cut into two halves of floor(c_min / 2) replays each (with
    an odd c_min the last drawn replay is dropped, so the two halves are equal in
    size)."""
    n, c = vals.shape
    k = c // 2
    perm = np.argsort(rng.random((n, c)), axis=1)
    g = np.take_along_axis(vals, perm, axis=1)
    return g[:, :k].mean(axis=1), g[:, k:2 * k].mean(axis=1)


def bucket_splithalf(
    bucket: Bucket,
    *,
    min_cands: int = DEFAULT_MIN_CANDS,
    split: Union[str, np.random.Generator] = "evenodd",
) -> Optional[float]:
    """Split-half Spearman of one group, or None when the group is excluded.

    `split` is either "evenodd" (the deterministic split the reference script
    defines, and the value that enters the manifest) or a numpy Generator, in
    which case one random half split is drawn.
    """
    vals = _group_matrix(bucket, min_cands)
    if vals is None:
        return None
    if isinstance(split, str):
        if split != "evenodd":
            raise ValueError('split must be "evenodd" or a numpy Generator, got %r' % (split,))
        q_a, q_b = _even_odd_means(vals)
    elif isinstance(split, np.random.Generator):
        q_a, q_b = _random_half_means(vals, split)
    else:
        raise ValueError('split must be "evenodd" or a numpy Generator, got %r' % (type(split),))
    return _spearman_of_halves(q_a, q_b)


def bucket_splithalf_random(
    bucket: Bucket,
    rng: np.random.Generator,
    *,
    min_cands: int = DEFAULT_MIN_CANDS,
    n_splits: int = 200,
) -> Optional[float]:
    """Mean split-half Spearman of one group over `n_splits` random half splits.

    Returns None when the group is excluded or when no split is computable (every
    drawn split leaves one half with constant advantages).
    """
    vals = _group_matrix(bucket, min_cands)
    if vals is None or n_splits <= 0:
        return None
    n, c = vals.shape
    k = c // 2
    perm = np.argsort(rng.random((n_splits, n, c)), axis=2)
    g = np.take_along_axis(np.broadcast_to(vals, (n_splits, n, c)), perm, axis=2)
    q_a = g[:, :, :k].mean(axis=2)
    q_b = g[:, :, k:2 * k].mean(axis=2)
    rho = spearman_rows(loo_adv_rows(q_a), loo_adv_rows(q_b))
    good = ~np.isnan(rho)
    if not good.any():
        return None
    return float(rho[good].mean())


# -------------------------
# cell level
# -------------------------


@dataclass
class CellResult:
    """Every reliability number one cell (one directory of buckets) produces.

    rel_evenodd is the value that goes into the manifest; rel_random is the
    random-split reading of the same groups, kept as a robustness check.
    n_groups counts the groups entering rel_evenodd, n_var_groups the (larger)
    pool behind wgv: the reference pools the within-group variance over every
    group that passes the alternative-count and replay-count clauses, including
    groups whose halves are flat.

    n2_direction_agreement is the share of +1 among those same rho values, and
    it is defined only when every group entering the mean carries exactly two
    alternatives, since that is the case in which rho can only be +1 or -1. It
    is the value the manifest's branching-2 rows print (driver ruling A3), and
    rel_evenodd of the same pool is 2 x agreement - 1.
    """

    rel_evenodd: Optional[float] = None
    rel_random: Optional[float] = None
    rel_ci_lo: Optional[float] = None
    rel_ci_hi: Optional[float] = None
    wgv: Optional[float] = None
    wgv_ci_lo: Optional[float] = None
    wgv_ci_hi: Optional[float] = None
    n_groups: int = 0
    n_total: int = 0
    excluded_pct: Optional[float] = None
    n2_direction_agreement: Optional[float] = None
    n_var_groups: int = 0
    rhos: List[float] = field(default_factory=list)


def cell_reliability(
    buckets: Sequence[Bucket],
    *,
    n_random_splits: int = 200,
    n_boot: int = 2000,
    seed: int = 0,
    min_cands: int = DEFAULT_MIN_CANDS,
) -> CellResult:
    """Reliability, within-group variance and their bootstrap intervals for one cell.

    The bootstrap resamples groups (the unit of the measurement experiments) and
    reports the 2.5 and 97.5 percentiles of the resampled mean, which is the
    interval the contract asks for.
    """
    rng_split = np.random.default_rng([seed, 1])
    rng_boot_rho = np.random.default_rng([seed, 2])
    rng_boot_var = np.random.default_rng([seed, 3])

    rhos: List[float] = []
    variances: List[float] = []
    rand_means: List[float] = []
    # Groups of exactly two alternatives among those entering the mean. The
    # direction-agreement rate is a statement about that pool, so a collected
    # group the exclusion rule already dropped (a branching-2 group whose two
    # samples came back identical, for instance) must not decide whether the
    # rate is defined.
    two_in_pool = 0

    for b in buckets:
        vals = _group_matrix(b, min_cands)
        if vals is None:
            continue
        adv_full = loo_adv(vals.mean(axis=1))
        variances.append(float(np.var(adv_full, ddof=1)))
        q_a, q_b = _even_odd_means(vals)
        rho = _spearman_of_halves(q_a, q_b)
        if rho is None:
            continue
        rhos.append(rho)
        if vals.shape[0] == 2:
            two_in_pool += 1
        rand = bucket_splithalf_random(
            b, rng_split, min_cands=min_cands, n_splits=n_random_splits
        )
        if rand is not None:
            rand_means.append(rand)

    res = CellResult()
    res.n_total = len(buckets)
    res.n_groups = len(rhos)
    res.n_var_groups = len(variances)
    res.rhos = rhos
    if res.n_total:
        res.excluded_pct = 100.0 * (res.n_total - res.n_groups) / res.n_total
    if rhos:
        res.rel_evenodd = float(np.mean(rhos))
        res.rel_ci_lo, res.rel_ci_hi = boot_ci(rhos, rng_boot_rho, n_boot)
        if two_in_pool == len(rhos):
            res.n2_direction_agreement = float(np.mean(np.asarray(rhos) >= _PLUS_ONE))
    if rand_means:
        res.rel_random = float(np.mean(rand_means))
    if variances:
        res.wgv = float(np.mean(variances))
        res.wgv_ci_lo, res.wgv_ci_hi = boot_ci(variances, rng_boot_var, n_boot)
    return res


# -------------------------
# paired difference between two cells
# -------------------------


@dataclass
class DeltaResult:
    """rel(hi) - rel(lo) over the questions both cells measured."""

    delta: Optional[float] = None
    ci_lo: Optional[float] = None
    ci_hi: Optional[float] = None
    n_pairs: int = 0


def _rhos_by_question(
    buckets: Sequence[Bucket], min_cands: int
) -> Dict[str, List[float]]:
    """Per-question lists of the groups' split-half values (excluded groups dropped)."""
    out: Dict[str, List[float]] = {}
    for b in buckets:
        rho = bucket_splithalf(b, min_cands=min_cands)
        if rho is None:
            continue
        qid = b.get("question_id")
        if qid is None:
            raise ValueError(
                "a usable group has no question_id; the paired difference needs it "
                "(bucket_id=%r)" % (b.get("bucket_id"),)
            )
        out.setdefault(str(qid), []).append(rho)
    return out


def paired_cell_delta(
    buckets_hi: Sequence[Bucket],
    buckets_lo: Sequence[Bucket],
    *,
    n_boot: int = 2000,
    seed: int = 0,
    min_cands: int = DEFAULT_MIN_CANDS,
) -> DeltaResult:
    """Difference of two cells' reliabilities, paired by question.

    Only questions that keep at least one usable group on both sides take part.
    Each side's reliability is its mean over groups inside that question set (the
    same per-group mean the single-cell number uses); the interval resamples
    questions with replacement, so a question's groups move together.
    """
    hi = _rhos_by_question(buckets_hi, min_cands)
    lo = _rhos_by_question(buckets_lo, min_cands)
    qids = sorted(set(hi) & set(lo))
    res = DeltaResult(n_pairs=len(qids))
    if not qids:
        return res

    hi_lists = [np.asarray(hi[q], dtype=float) for q in qids]
    lo_lists = [np.asarray(lo[q], dtype=float) for q in qids]
    hi_sum = np.array([a.sum() for a in hi_lists])
    hi_cnt = np.array([a.size for a in hi_lists], dtype=float)
    lo_sum = np.array([a.sum() for a in lo_lists])
    lo_cnt = np.array([a.size for a in lo_lists], dtype=float)

    res.delta = float(hi_sum.sum() / hi_cnt.sum() - lo_sum.sum() / lo_cnt.sum())

    if n_boot > 0 and len(qids) > 1:
        rng = np.random.default_rng([seed, 5])
        idx = rng.integers(0, len(qids), size=(n_boot, len(qids)))
        d = (hi_sum[idx].sum(axis=1) / hi_cnt[idx].sum(axis=1)
             - lo_sum[idx].sum(axis=1) / lo_cnt[idx].sum(axis=1))
        res.ci_lo = float(np.percentile(d, CI_LO_PCT))
        res.ci_hi = float(np.percentile(d, CI_HI_PCT))
    elif len(qids) == 1:
        res.ci_lo = res.ci_hi = res.delta
    return res


# -------------------------
# spread across cells
# -------------------------


def cell_range(rels: Mapping[str, float]) -> Tuple[float, str, str]:
    """(max - min, name of the smallest cell, name of the largest cell).

    Ties go to the alphabetically first name, so the result does not depend on
    dictionary order.
    """
    if not rels:
        raise ValueError("cell_range needs at least one cell")
    items = sorted(rels.items())
    lo_name, lo_val = min(items, key=lambda kv: kv[1])
    hi_name, hi_val = max(items, key=lambda kv: kv[1])
    return float(hi_val - lo_val), lo_name, hi_name


def range_bootstrap(
    cells: Mapping[str, Sequence[Bucket]],
    n_boot: int = 2000,
    seed: int = 0,
    min_cands: int = DEFAULT_MIN_CANDS,
) -> Tuple[Optional[float], Optional[float]]:
    """Bootstrap interval of the range of cell reliabilities.

    Groups are resampled inside each cell independently (cells are different
    measurements, so there is nothing to pair), the range of the resampled cell
    means is recomputed, and the 2.5 and 97.5 percentiles are returned.
    """
    if not cells:
        raise ValueError("range_bootstrap needs at least one cell")
    rho_lists = {}
    for name, buckets in cells.items():
        rhos = [r for r in (bucket_splithalf(b, min_cands=min_cands) for b in buckets)
                if r is not None]
        if not rhos:
            raise ValueError("cell %r has no usable group" % (name,))
        rho_lists[name] = np.asarray(rhos, dtype=float)
    if n_boot <= 0:
        return None, None

    rng = np.random.default_rng([seed, 6])
    means = np.empty((len(rho_lists), n_boot), dtype=float)
    for row, name in enumerate(sorted(rho_lists)):
        v = rho_lists[name]
        idx = rng.integers(0, v.size, size=(n_boot, v.size))
        means[row] = v[idx].mean(axis=1)
    spread = means.max(axis=0) - means.min(axis=0)
    return float(np.percentile(spread, CI_LO_PCT)), float(np.percentile(spread, CI_HI_PCT))
