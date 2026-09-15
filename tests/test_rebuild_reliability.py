# tests/test_rebuild_reliability.py
"""
Unit tests for the rebuild-experiment reliability family:
c3.analysis.rebuild.{splithalf, duplicates, noise_law, aggregate_e1,
aggregate_e1b, aggregate_e1c_temp}.

Everything here is synthetic: no bucket file from a real run is read, and no
number in these tests is a claim about the experiments. Several cases are the
offline-reproducible half of the self-tests in
30_analysis/local/splithalf_bootstrap_local.py, which is the authoritative
definition of the split-half estimator.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import warnings

import numpy as np
import pytest
from scipy import stats

from c3.analysis.rebuild import aggregate_e1, aggregate_e1b, aggregate_e1c_temp
from c3.analysis.rebuild import duplicates as dup
from c3.analysis.rebuild import noise_law as nl
from c3.analysis.rebuild import splithalf as sh

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# -------------------------
# synthetic buckets
# -------------------------


def make_bucket(returns, *, bucket_id="b0", question_id="q0", texts=None, meta=None):
    """A bucket in the schema of c3.analysis.buckets, with the given returns."""
    cands = []
    for j, r in enumerate(returns):
        cands.append({
            "j": j,
            "action_text": (texts[j] if texts is not None else "alternative %d" % j),
            "returns": [float(v) for v in r],
            "next_actions": ["" for _ in r],
        })
    return {
        "bucket_id": bucket_id,
        "ctx_hash": 0,
        "target_role": "Reasoner",
        "question_id": question_id,
        "restart": {"roles_topo": ["Reasoner", "Actor"], "role_outputs_prefix": {}},
        "candidates": cands,
        "meta": dict(meta or {}),
    }


def bernoulli_cell(ps, c, rng, n_groups, *, prefix="g", question_prefix="q"):
    """n_groups independent groups, alternative j drawing Bernoulli(ps[j])."""
    out = []
    for g in range(n_groups):
        returns = [rng.binomial(1, p, size=c).astype(float) for p in ps]
        out.append(make_bucket(returns, bucket_id="%s%d" % (prefix, g),
                               question_id="%s%d" % (question_prefix, g)))
    return out


def write_jsonl(path, buckets):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for b in buckets:
            fh.write(json.dumps(b, ensure_ascii=False) + "\n")
    return path


# -------------------------
# 1. the estimator on hand-built groups
# -------------------------


def test_perfect_agreement_gives_one():
    # Half means (1, 0.5, 0) on both halves: strictly ordered the same way.
    b = make_bucket([[1, 1, 1, 1], [1, 0, 0, 1], [0, 0, 0, 0]])
    assert sh.bucket_splithalf(b) == pytest.approx(1.0)


def test_reversed_agreement_gives_minus_one():
    # Half means (1, 0.5, 0) against (0, 0.5, 1).
    b = make_bucket([[1, 0, 1, 0], [1, 1, 0, 0], [0, 1, 0, 1]])
    assert sh.bucket_splithalf(b) == pytest.approx(-1.0)


def test_flat_half_is_excluded():
    # Every alternative has even-index mean 1, so the first half's advantages are
    # constant and the group is dropped by the exclusion rule.
    b = make_bucket([[1, 0, 1, 0], [1, 1, 1, 1], [1, 0, 1, 1]])
    assert sh.bucket_splithalf(b) is None


def test_too_few_alternatives_or_replays_excluded():
    two = make_bucket([[1, 0, 1, 0], [0, 1, 0, 1]])
    assert sh.bucket_splithalf(two) is None
    assert sh.bucket_splithalf(two, min_cands=2) == pytest.approx(-1.0)
    one_replay = make_bucket([[1], [0], [1]])
    assert sh.bucket_splithalf(one_replay) is None


def test_even_odd_hand_computed_rho():
    # The reference script's self-test 6: ranks (2.5, 2.5, 1) against (3, 1.5, 1.5).
    b = make_bucket([[1, 1, 1, 1], [1, 0, 1, 0], [0, 0, 0, 0]])
    assert sh.bucket_splithalf(b) == pytest.approx(0.5)


# -------------------------
# 2. the leave-one-out step does not move the statistic
# -------------------------


def test_loo_adv_known_values():
    got = sh.loo_adv([1.0, 0.0, 0.0, 0.0])
    assert np.allclose(got, [1.0, -1 / 3, -1 / 3, -1 / 3])
    assert sh.loo_adv([0.5]) is None
    assert np.allclose(sh.loo_adv([1.0, 0.0]), [1.0, -1.0])


def test_loo_adv_preserves_ranks():
    rng = np.random.default_rng(7)
    for _ in range(200):
        n = int(rng.integers(2, 7))
        q = rng.choice([0.0, 0.25, 0.5, 0.75, 1.0], size=n)
        assert np.array_equal(stats.rankdata(q), stats.rankdata(sh.loo_adv(q)))


def test_splithalf_equals_spearman_of_raw_half_means():
    """The advantage step is an increasing affine map inside a group, so the
    split-half Spearman is the same on the raw half means."""
    rng = np.random.default_rng(101)
    checked = 0
    for _ in range(200):
        n = int(rng.integers(3, 9))
        c = int(rng.choice([2, 4, 6, 8]))
        b = make_bucket([rng.integers(0, 2, size=c).astype(float) for _ in range(n)])
        got = sh.bucket_splithalf(b)
        if got is None:
            continue
        vals = np.asarray([cand["returns"] for cand in b["candidates"]], dtype=float)
        q_a, q_b = vals[:, 0::2].mean(axis=1), vals[:, 1::2].mean(axis=1)
        assert got == pytest.approx(float(stats.spearmanr(q_a, q_b).statistic), abs=1e-12)
        checked += 1
    assert checked > 100


# -------------------------
# 3. pieces ported from the reference script's self-tests
# -------------------------


def test_spearman_rows_matches_scipy_without_ties():
    rng = np.random.default_rng(11)
    A, B = rng.normal(size=(50, 4)), rng.normal(size=(50, 4))
    mine = sh.spearman_rows(A, B)
    for k in range(A.shape[0]):
        assert mine[k] == pytest.approx(float(stats.spearmanr(A[k], B[k]).statistic), abs=1e-12)


def test_spearman_rows_matches_scipy_with_ties():
    rng = np.random.default_rng(13)
    A = rng.choice([0.0, 0.5, 1.0], size=(300, 4))
    B = rng.choice([0.0, 0.5, 1.0], size=(300, 4))
    mine = sh.spearman_rows(A, B)
    with warnings.catch_warnings():  # constant rows are fed on purpose
        warnings.simplefilter("ignore")
        for k in range(A.shape[0]):
            ref = stats.spearmanr(A[k], B[k]).statistic
            if np.isnan(ref):
                assert np.isnan(mine[k])
            else:
                assert mine[k] == pytest.approx(float(ref), abs=1e-12)


def test_spearman_rows_is_nan_on_constant_input():
    out = sh.spearman_rows(np.array([[1.0, 1.0, 1.0, 1.0]]), np.array([[0.0, 1.0, 0.0, 1.0]]))
    assert np.isnan(out[0])


def test_boot_ci_constant_and_deterministic():
    lo, hi = sh.boot_ci([0.7] * 25, np.random.default_rng(3), 200)
    assert lo == pytest.approx(0.7) and hi == pytest.approx(0.7)
    assert sh.boot_ci([0.42], np.random.default_rng(3), 200) == (0.42, 0.42)
    assert sh.boot_ci([], np.random.default_rng(3), 200) == (None, None)
    vals = list(np.random.default_rng(5).normal(0.5, 0.3, size=60))
    a = sh.boot_ci(vals, np.random.default_rng(99), 2000)
    b = sh.boot_ci(vals, np.random.default_rng(99), 2000)
    assert a == b
    assert a[0] < float(np.mean(vals)) < a[1]


def test_min_cands_filter_changes_the_pool():
    """The reference script's self-test 12: the same file, two filters."""
    buckets = [
        make_bucket([[1, 0, 1, 0], [0, 0, 0, 1]], bucket_id="b0", question_id="q0"),
        make_bucket([[1, 0, 1, 0], [0, 0, 0, 1], [1, 1, 1, 1]], bucket_id="b1", question_id="q1"),
    ]
    at3 = sh.cell_reliability(buckets, n_random_splits=10, n_boot=50)
    at2 = sh.cell_reliability(buckets, n_random_splits=10, n_boot=50, min_cands=2)
    assert at3.n_var_groups == 1
    assert at2.n_var_groups == 2
    assert at3.n_total == at2.n_total == 2


def test_random_split_halves_and_degenerate_cases():
    rng = np.random.default_rng(1)
    # Each alternative constant across its replays: every random split rebuilds the
    # same two half vectors, so every drawn rho is exactly +1.
    flat = make_bucket([[1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1], [0.5, 0.5, 0.5, 0.5]])
    assert sh.bucket_splithalf_random(flat, rng, n_splits=40) == pytest.approx(1.0)
    # No variation at all: no split is computable.
    dead = make_bucket([[1, 1], [1, 1], [1, 1]])
    assert sh.bucket_splithalf_random(dead, rng, n_splits=20) is None
    # With an odd replay count both halves take floor(c_min / 2) replays.
    vals = np.arange(15, dtype=float).reshape(3, 5)
    q_a, q_b = sh._random_half_means(vals, np.random.default_rng(2))
    assert q_a.shape == q_b.shape == (3,)
    for row in range(3):
        assert 2 * (q_a[row] + q_b[row]) <= vals[row].sum() + 1e-9


def test_n2_direction_agreement_matches_the_mean():
    rng = np.random.default_rng(21)
    buckets = bernoulli_cell([0.2, 0.8], 4, rng, 60)
    cell = sh.cell_reliability(buckets, n_random_splits=20, n_boot=200, min_cands=2)
    assert cell.n_groups > 20
    assert set(np.round(np.abs(cell.rhos), 6)) == {1.0}
    assert cell.n2_direction_agreement is not None
    assert cell.rel_evenodd == pytest.approx(2 * cell.n2_direction_agreement - 1)
    # With three alternatives in the cell the field is not defined.
    three = sh.cell_reliability(bernoulli_cell([0.2, 0.5, 0.8], 4, rng, 10),
                               n_random_splits=5, n_boot=50)
    assert three.n2_direction_agreement is None


def test_n2_direction_agreement_survives_a_collapsed_group():
    """A branching-2 group whose two samples came back identical carries one
    alternative and is excluded; the cell still reports its agreement rate.

    This is the shape real data takes: the runner drops duplicate samples, so a
    sweep_n2 cell can hold groups with a single alternative. Gating the rate on
    every collected group having two alternatives would throw the whole key away
    because of groups that never entered the mean.
    """
    rng = np.random.default_rng(22)
    buckets = bernoulli_cell([0.2, 0.8], 4, rng, 30)
    collapsed = make_bucket([[1, 0, 1, 0]], bucket_id="collapsed", question_id="qx")
    cell = sh.cell_reliability(buckets + [collapsed], n_random_splits=10, n_boot=100,
                               min_cands=2)
    clean = sh.cell_reliability(buckets, n_random_splits=10, n_boot=100, min_cands=2)
    assert cell.n_total == 31 and clean.n_total == 30
    assert cell.n_groups == clean.n_groups
    assert cell.n2_direction_agreement == pytest.approx(clean.n2_direction_agreement)
    assert cell.rel_evenodd == pytest.approx(2 * cell.n2_direction_agreement - 1)

    # A cell that really does mix widths keeps the rate undefined.
    mixed = sh.cell_reliability(buckets + bernoulli_cell([0.2, 0.5, 0.8], 4, rng, 5,
                                                        prefix="wide", question_prefix="qw"),
                                n_random_splits=10, n_boot=100, min_cands=2)
    assert mixed.n2_direction_agreement is None


def test_pooled_within_group_variance_and_counts():
    """wgv pools every group that passes the alternative and replay counts, which
    is a larger set than the groups entering the reliability mean."""
    usable = make_bucket([[1, 1, 1, 1], [1, 0, 0, 1], [0, 0, 0, 0]],
                         bucket_id="b0", question_id="q0")
    flat_half = make_bucket([[1, 0, 1, 0], [1, 1, 1, 1], [1, 0, 1, 1]],
                            bucket_id="b1", question_id="q1")
    too_few = make_bucket([[1, 0, 1, 0], [0, 1, 0, 1]], bucket_id="b2", question_id="q2")
    cell = sh.cell_reliability([usable, flat_half, too_few], n_random_splits=10, n_boot=200)
    hand = []
    for b in (usable, flat_half):
        vals = np.asarray([c["returns"] for c in b["candidates"]], dtype=float)
        hand.append(float(np.var(sh.loo_adv(vals.mean(axis=1)), ddof=1)))
    assert cell.wgv == pytest.approx(float(np.mean(hand)))
    assert cell.n_var_groups == 2
    assert cell.n_groups == 1
    assert cell.n_total == 3
    assert cell.excluded_pct == pytest.approx(100.0 * 2 / 3)
    assert cell.rel_ci_lo == cell.rel_ci_hi == pytest.approx(cell.rel_evenodd)


def test_cell_reliability_is_deterministic_given_the_seed():
    rng = np.random.default_rng(31)
    buckets = bernoulli_cell([0.1, 0.4, 0.6, 0.9], 4, rng, 40)
    a = sh.cell_reliability(buckets, n_random_splits=50, n_boot=500, seed=3)
    b = sh.cell_reliability(buckets, n_random_splits=50, n_boot=500, seed=3)
    c = sh.cell_reliability(buckets, n_random_splits=50, n_boot=500, seed=4)
    assert (a.rel_evenodd, a.rel_ci_lo, a.rel_ci_hi, a.rel_random) == \
           (b.rel_evenodd, b.rel_ci_lo, b.rel_ci_hi, b.rel_random)
    assert a.rel_evenodd == c.rel_evenodd  # the point estimate does not use the rng
    assert a.rel_ci_lo != c.rel_ci_lo or a.rel_ci_hi != c.rel_ci_hi


# -------------------------
# 4. the paired difference between two cells
# -------------------------


def _usable_question_ids(buckets):
    return {b["question_id"] for b in buckets if sh.bucket_splithalf(b) is not None}


def test_paired_delta_separates_a_reliable_cell_from_a_noisy_one():
    rng = np.random.default_rng(41)
    hi = bernoulli_cell([0.0, 0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0], 16, rng, 60)
    lo = bernoulli_cell([0.2, 0.5, 0.8], 4, rng, 60)
    rel_hi = sh.cell_reliability(hi, n_random_splits=0, n_boot=200).rel_evenodd
    rel_lo = sh.cell_reliability(lo, n_random_splits=0, n_boot=200).rel_evenodd
    assert rel_hi > 0.85 and 0.3 < rel_lo < 0.65
    delta = sh.paired_cell_delta(hi, lo, n_boot=2000, seed=0)
    shared = _usable_question_ids(hi) & _usable_question_ids(lo)
    assert delta.n_pairs == len(shared) < 60
    assert delta.delta == pytest.approx(rel_hi - rel_lo, abs=0.1)
    assert delta.delta > 0.25
    assert delta.ci_lo > 0.0


def test_paired_delta_covers_zero_when_the_two_cells_match():
    rng = np.random.default_rng(42)
    ps = [0.2, 0.45, 0.7, 0.9]
    a = bernoulli_cell(ps, 4, rng, 80)
    b = bernoulli_cell(ps, 4, rng, 80)
    delta = sh.paired_cell_delta(a, b, n_boot=2000, seed=0)
    assert delta.n_pairs > 60
    assert delta.ci_lo < 0.0 < delta.ci_hi


def test_paired_delta_uses_only_shared_questions():
    rng = np.random.default_rng(43)
    hi = bernoulli_cell([0.05, 0.5, 0.95], 4, rng, 20, question_prefix="q")
    lo = bernoulli_cell([0.05, 0.5, 0.95], 4, rng, 20, question_prefix="other")
    assert sh.paired_cell_delta(hi, lo, n_boot=100).n_pairs == 0
    shared = sh.paired_cell_delta(hi, hi, n_boot=100)
    assert shared.delta == pytest.approx(0.0)
    assert shared.n_pairs <= 20


def test_paired_delta_needs_a_question_id():
    b = make_bucket([[1, 1, 1, 1], [1, 0, 0, 1], [0, 0, 0, 0]])
    b.pop("question_id")
    with pytest.raises(ValueError):
        sh.paired_cell_delta([b], [b], n_boot=10)


# -------------------------
# 5. the spread over cells
# -------------------------


def test_cell_range_picks_the_extremes_deterministically():
    spread, lo_name, hi_name = sh.cell_range({"a2": 0.8, "a3": 0.6, "mt4": 0.9})
    assert (spread, lo_name, hi_name) == (pytest.approx(0.3), "a3", "mt4")
    tie_spread, tie_lo, tie_hi = sh.cell_range({"b": 0.5, "a": 0.5})
    assert tie_spread == pytest.approx(0.0) and tie_lo == "a" and tie_hi == "a"


def test_range_bootstrap_brackets_the_observed_range():
    rng = np.random.default_rng(51)
    cells = {}
    for name, ps in (("a2", [0.05, 0.5, 0.95]), ("a3", [0.3, 0.5, 0.7]),
                     ("mt4", [0.1, 0.5, 0.9]), ("c5", [0.35, 0.5, 0.65]),
                     ("c10", [0.4, 0.5, 0.6]), ("branch", [0.2, 0.5, 0.8])):
        cells[name] = bernoulli_cell(ps, 4, rng, 40, prefix=name, question_prefix=name)
    rels = {name: sh.cell_reliability(b, n_random_splits=0, n_boot=100).rel_evenodd
            for name, b in cells.items()}
    spread, lo_name, hi_name = sh.cell_range(rels)
    ci_lo, ci_hi = sh.range_bootstrap(cells, 1000, 0)
    assert 0.0 <= ci_lo <= ci_hi
    assert ci_lo <= spread <= ci_hi
    assert rels[lo_name] <= rels[hi_name]
    with pytest.raises(ValueError):
        sh.range_bootstrap({"empty": [make_bucket([[1, 1], [1, 1], [1, 1]])]}, 10, 0)


# -------------------------
# 6. duplicates
# -------------------------


def test_duplicate_rate_counts_alternatives_not_pairs():
    texts = ["2 + 2 = 4", "2 + 2 = 4", "2  +  2  =  4 ", "the answer is five"]
    b = make_bucket([[1, 0], [1, 0], [1, 0], [0, 1]], texts=texts)
    assert dup.bucket_duplicate_rate(b) == pytest.approx(0.75)
    assert dup.normalize_text("2  +  2  =  4 ") == "2 + 2 = 4"
    assert dup.bucket_duplicate_rate(make_bucket([[1, 0]], texts=["only one"])) is None


def test_duplicate_rate_is_case_sensitive_and_whitespace_insensitive():
    b = make_bucket([[1], [1], [1]], texts=["Answer: 4", "answer: 4", "Answer:\n4"])
    assert dup.bucket_duplicate_rate(b) == pytest.approx(2 / 3)


def test_near_duplicate_threshold():
    ten = ["w%d" % i for i in range(10)]
    same = " ".join(ten)
    one_off = " ".join(ten[:9] + ["different"])
    far = "nothing at all in common here"
    b = make_bucket([[1], [1], [1]], texts=[same, one_off, far])
    # Nine shared tokens out of eleven in the union: 0.818.
    assert dup.jaccard(same, one_off) == pytest.approx(9 / 11)
    assert dup.bucket_near_duplicate_rate(b, jaccard=0.9) == pytest.approx(0.0)
    assert dup.bucket_near_duplicate_rate(b, jaccard=0.8) == pytest.approx(2 / 3)
    # An exact duplicate is always a near duplicate.
    exact = make_bucket([[1], [1], [1]], texts=[same, same, far])
    assert dup.bucket_duplicate_rate(exact) == pytest.approx(2 / 3)
    assert dup.bucket_near_duplicate_rate(exact) == pytest.approx(2 / 3)


def test_cell_duplicate_report_means_over_groups():
    a = make_bucket([[1], [1], [1], [1]], texts=["x", "x", "x", "y"])      # 0.75
    b = make_bucket([[1], [1], [1], [1]], texts=["p", "q", "r", "s"])      # 0.0
    single = make_bucket([[1]], texts=["lonely"])                          # not counted
    report = dup.cell_duplicate_report([a, b, single])
    assert report["n"] == 2
    assert report["dup_rate_mean"] == pytest.approx(0.375)
    assert report["near_dup_rate_mean"] == pytest.approx(0.375)
    empty = dup.cell_duplicate_report([single])
    assert empty == {"dup_rate_mean": None, "near_dup_rate_mean": None, "n": 0}


# -------------------------
# 7. the noise law
# -------------------------


def _noise_cell(n, c, rng, n_groups=100, ps=None):
    """Two independent replay sets of the same groups."""
    ps = ps if ps is not None else [0.5] * n
    a = bernoulli_cell(ps, c, rng, n_groups)
    b = bernoulli_cell(ps, c, rng, n_groups)
    return a, b


@pytest.mark.parametrize("n", [2, 4, 8])
@pytest.mark.parametrize("c", [2, 4, 8])
def test_noise_law_simulation_matches_the_formula(n, c):
    """Returns generated exactly under the model the prediction assumes (one common
    return variance, independent replays): the measured over predicted ratio has to
    land inside the preregistered probe band. This checks the implementation against
    the formula, not the real data.

    No convention is passed in: the module defaults are the ones preregistration
    revision 10 fixed, and this is the self-test of those defaults.
    """
    rng = np.random.default_rng([2026, n, c])
    a, b = _noise_cell(n, c, rng)
    rep = nl.cell_noise_ratio(a, b, n_boot=500, seed=0)
    assert rep["n_groups"] >= 50
    assert 0.7 <= rep["ratio"] <= 1.4


@pytest.mark.parametrize("n", [2, 4, 8])
def test_noise_law_ddof_one_scales_the_ratio_by_n_over_n_minus_one(n):
    """A record of the retired convention, kept because it says what changed.

    An unbiased sample variance (ddof=1) multiplies every group's measured value
    by exactly n / (n - 1), because the difference vector's mean is zero by
    construction. At two alternatives that doubles the ratio and puts the cell
    outside the [0.7, 1.4] band whatever the data says, which is why the default
    is now ddof=0 (driver ruling A1).
    """
    rng = np.random.default_rng([2027, n])
    a, b = _noise_cell(n, 4, rng, n_groups=60)
    at0 = nl.cell_noise_ratio(a, b, n_boot=0, seed=0, ddof=0)
    at1 = nl.cell_noise_ratio(a, b, n_boot=0, seed=0, ddof=1)
    assert nl.DDOF == 0
    assert at1["n_groups"] == at0["n_groups"]
    assert at1["ratio"] == pytest.approx(at0["ratio"] * n / (n - 1.0))


def test_noise_law_holds_when_the_alternatives_differ_in_quality():
    """The point of the sigma^2 convention (driver ruling A2).

    sigma^2 is the return variance of one alternative, so alternatives of very
    different quality leave the ratio inside the band. Pooling every return of
    the group instead, which is the retired convention, adds the spread between
    alternatives to the denominator and pushes the same data out of the band
    with nothing wrong in it. Both numbers are computed here from the same
    groups.
    """
    rng = np.random.default_rng(2028)
    ps = [0.02, 0.35, 0.65, 0.98]
    a, b = _noise_cell(4, 4, rng, n_groups=100, ps=ps)
    rep = nl.cell_noise_ratio(a, b, n_boot=0)
    assert rep["n_groups"] >= 90
    assert 0.7 <= rep["ratio"] <= 1.4

    n, c = len(ps), 4
    pooled = []
    for ba, bb in zip(a, b):
        rets_a = np.asarray([cand["returns"] for cand in ba["candidates"]], dtype=float)
        rets_b = np.asarray([cand["returns"] for cand in bb["candidates"]], dtype=float)
        d = sh.loo_adv(rets_a.mean(axis=1)) - sh.loo_adv(rets_b.mean(axis=1))
        measured = float(np.mean(d ** 2)) / 2.0
        sigma2 = float(np.var(np.concatenate([rets_a.ravel(), rets_b.ravel()]), ddof=1))
        if sigma2 > 0:
            pooled.append(measured / (sigma2 * n * n / (n * c * (n - 1))))
    assert float(np.mean(pooled)) < 0.7


def test_noise_law_group_skips_and_pairing():
    a0 = make_bucket([[1, 0], [0, 1], [1, 1]], bucket_id="b0", question_id="q0")
    b0 = make_bucket([[0, 1], [1, 1], [0, 0]], bucket_id="b0", question_id="q0")
    # All returns identical: the prediction is zero, so the group carries no ratio.
    dead_a = make_bucket([[1, 1], [1, 1], [1, 1]], bucket_id="b1", question_id="q1")
    dead_b = make_bucket([[1, 1], [1, 1], [1, 1]], bucket_id="b1", question_id="q1")
    # Different alternative counts on the two seeds: not comparable.
    wide_a = make_bucket([[1, 0], [0, 1], [1, 1], [0, 0]], bucket_id="b2", question_id="q2")
    wide_b = make_bucket([[1, 0], [0, 1], [1, 1]], bucket_id="b2", question_id="q2")
    rep = nl.cell_noise_ratio([a0, dead_a, wide_a], [b0, dead_b, wide_b], n_boot=100)
    assert rep["pair_field"] == "bucket_id"
    assert rep["n_pairs"] == 3 and rep["n_groups"] == 1 and rep["n_skipped"] == 2
    assert nl.group_noise(dead_a, dead_b) is None
    assert nl.group_noise(wide_a, wide_b) is None
    # Falling back to question_id when the bucket ids do not line up.
    shifted = make_bucket([[0, 1], [1, 1], [0, 0]], bucket_id="other", question_id="q0")
    field, pairs = nl.pair_buckets([a0], [shifted])
    assert field == "question_id" and len(pairs) == 1
    with pytest.raises(ValueError):
        nl.pair_buckets([a0], [make_bucket([[1, 0]], bucket_id="zz", question_id="zz")])


def test_noise_law_group_measured_and_predicted_by_hand():
    """Both defaults on one hand-computed group.

    Half means (1, 0, 0.5) against (0.5, 0.5, 0). The measured side is the mean
    of the squared difference of the two advantage vectors, halved. The three
    alternatives pool their two seeds into [1, 1, 1, 0], [0, 0, 0, 1] and
    [1, 0, 0, 0], each of sample variance 0.25, so sigma^2 = 0.25 and the
    prediction is 0.25 * 9 / (6 * 2) = 0.1875.
    """
    a = make_bucket([[1, 1], [0, 0], [1, 0]], bucket_id="b0", question_id="q0")
    b = make_bucket([[1, 0], [0, 1], [0, 0]], bucket_id="b0", question_id="q0")
    gn = nl.group_noise(a, b)
    adv_a = sh.loo_adv([1.0, 0.0, 0.5])
    adv_b = sh.loo_adv([0.5, 0.5, 0.0])
    measured = float(np.mean((adv_a - adv_b) ** 2)) / 2.0
    per_alt = [float(np.var(v, ddof=1))
               for v in ([1, 1, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0])]
    assert per_alt == [pytest.approx(0.25)] * 3
    predicted = float(np.mean(per_alt)) * 9 / (6 * 2)
    assert predicted == pytest.approx(0.1875)
    assert gn.n == 3 and gn.c == pytest.approx(2.0)
    assert gn.measured == pytest.approx(measured)
    assert gn.predicted == pytest.approx(predicted)
    assert gn.ratio == pytest.approx(measured / predicted)


def test_grid_report_picks_the_worst_cell():
    rng = np.random.default_rng(61)
    cells = {(4, 4): _noise_cell(4, 4, rng, n_groups=40),
             (8, 4): _noise_cell(8, 4, rng, n_groups=40)}
    grid = nl.grid_report(cells, n_boot=200, seed=0, ddof=0)
    assert set(grid["ratio"]) == {(4, 4), (8, 4)}
    devs = {k: abs(v - 1.0) for k, v in grid["ratio"].items()}
    worst = max(devs, key=lambda k: devs[k])
    assert grid["worst_cell"] == worst
    assert grid["worst_ratio"] == pytest.approx(grid["ratio"][worst])
    assert grid["max_dev_pct"] == pytest.approx(100.0 * devs[worst])
    assert grid["n_cells"] == 2


# -------------------------
# 8. the aggregators
# -------------------------


def fake_manifest(keys):
    """A manifest shaped like the paper's: key -> entry with a unit.

    The units follow 10_paper/04_rebuild/results/manifest.json: the branching-2
    sweep rows are an agreement rate, the binding-workflow key is a name, and
    every other reliability row is a correlation.
    """
    out = {}
    for key in keys:
        if key.endswith("n_groups"):
            unit, fmt = "groups", "{:d}"
        elif key.endswith("excluded_pct") or key.endswith("max_dev_pct"):
            unit, fmt = "%", "{:.0f}"
        elif key.endswith("wgv"):
            unit, fmt = "variance", "{:.3f}"
        elif ".ratio." in key or key.endswith("worst_ratio"):
            unit, fmt = "ratio", "{:.2f}"
        elif key.endswith("dup_rate"):
            unit, fmt = "rate", "{:.2f}"
        elif key.endswith(".n2.rel"):
            unit, fmt = "agreement", "{:.2f}"
        elif key.endswith("_arm"):
            unit, fmt = "name", "{}"
        else:
            unit, fmt = "spearman", "{:.2f}"
        out[key] = {"value": None, "fmt": fmt, "unit": unit, "verdict": None,
                    "source": None, "n": None, "experiment": key.split(".")[0], "note": ""}
    out["E1.reliability.budget_not_depth"] = {
        "value": None, "fmt": "{}", "unit": "verdict", "verdict": None, "source": None,
        "n": None, "experiment": "E1", "note": "a verdict key, never written by a script"}
    return out


E1_MANIFEST_KEYS = (
    [".".join(["E1.sweep_n", wf, "4b", "n%d" % n, "rel"])
     for wf in ("a2", "a3") for n in (2, 3, 4, 6, 8)]
    + ["E1.sweep_n.a2.4b.delta_n8_n3", "E1.sweep_n.a2.4b.delta_n8_n3_ci_lo"]
    + ["E1.per_decision_n4.%s.4b.%s" % (wf, q)
       for wf in ("a2", "a3")
       for q in ("rel", "rel_ci_lo", "rel_ci_hi", "wgv", "n_groups", "excluded_pct")]
    + ["E1.fixed_b8.%s.4b.%s" % (wf, q)
       for wf in ("a2", "a3")
       for q in ("rel", "rel_ci_lo", "rel_ci_hi", "wgv", "n_groups", "excluded_pct")]
    + ["E1.per_decision_n4.4b.rel_min", "E1.per_decision_n4.4b.rel_max",
       "E1.per_decision_n4.4b.rel_range", "E1.sweep_n.4b.delta_n8_n3_min",
       "E1.sweep_n.4b.delta_n8_n3_min_ci_lo", "E1.sweep_n.4b.delta_n8_n3_min_arm"]
)


def build_e1_tree(root):
    """A small E1 tree: two workflows, sweep cells only, plus two directories the
    scanner has to ignore."""
    rng = np.random.default_rng(71)
    plans = {
        ("a2", 2): [0.2, 0.8],
        ("a2", 3): [0.2, 0.5, 0.8],
        ("a2", 4): [0.1, 0.4, 0.6, 0.9],
        ("a2", 8): [0.02, 0.15, 0.3, 0.45, 0.6, 0.72, 0.85, 0.98],
        ("a3", 4): [0.3, 0.45, 0.55, 0.7],
    }
    made = {}
    for (wf, n), ps in plans.items():
        buckets = bernoulli_cell(ps, 4, rng, 30, prefix="%s_n%d_" % (wf, n))
        path = os.path.join(root, wf, "4b", "sweep_n%d" % n, "buckets.jsonl")
        write_jsonl(path, buckets)
        made[(wf, n)] = buckets
    os.makedirs(os.path.join(root, "a2", "4b", "fixed_b8"), exist_ok=True)
    os.makedirs(os.path.join(root, "not_a_workflow", "4b", "sweep_n4"), exist_ok=True)
    return made


def test_aggregate_e1_on_a_minimal_tree(tmp_path, capsys):
    root = os.path.join(str(tmp_path), "E1")
    made = build_e1_tree(root)
    manifest_keys = [k for k in E1_MANIFEST_KEYS if k != "E1.per_decision_n4.a3.4b.wgv"]
    manifest_path = os.path.join(str(tmp_path), "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(fake_manifest(manifest_keys), fh)
    out_path = os.path.join(str(tmp_path), "summary.json")

    assert aggregate_e1.main(["--results", root, "--manifest", manifest_path,
                              "--out", out_path]) == 0
    with open(out_path, "r", encoding="utf-8") as fh:
        summary = json.load(fh)
    err = capsys.readouterr().err

    assert summary["experiment"] == "E1"
    assert summary["generated_by"]["script"] == "c3.analysis.rebuild.aggregate_e1"
    assert set(summary["generated_by"]) == {"script", "git_sha", "when"}

    keys = summary["keys"]
    assert set(keys) <= set(fake_manifest(manifest_keys))
    assert "E1.reliability.budget_not_depth" not in keys
    assert not any(k.endswith("_arm") for k in keys)      # only two of six workflows
    assert keys["E1.sweep_n.a2.4b.n2.rel"]["unit"] == "agreement"
    assert "E1.fixed_b8.a3.4b.rel" not in keys            # its source cell (sweep_n3 of a3) is absent
    assert "E1.per_decision_n4.a3.4b.wgv" not in keys     # not a manifest key here
    assert "E1.per_decision_n4.4b.rel_range" not in keys  # only two of six workflows

    # n and value agree with a direct recomputation of the same cell.
    cell = sh.cell_reliability(made[("a2", 4)], n_random_splits=aggregate_e1.N_RANDOM_SPLITS,
                               n_boot=aggregate_e1.N_BOOT, seed=aggregate_e1.SEED)
    for key in ("E1.sweep_n.a2.4b.n4.rel", "E1.per_decision_n4.a2.4b.rel",
                "E1.fixed_b8.a2.4b.rel"):
        assert keys[key]["value"] == pytest.approx(cell.rel_evenodd)
        assert keys[key]["n"] == cell.n_groups
        assert keys[key]["unit"] == "spearman"
    assert keys["E1.per_decision_n4.a2.4b.wgv"]["value"] == pytest.approx(cell.wgv)
    assert keys["E1.per_decision_n4.a2.4b.wgv"]["n"] == cell.n_var_groups
    assert keys["E1.per_decision_n4.a2.4b.n_groups"]["value"] == cell.n_groups
    assert keys["E1.per_decision_n4.a2.4b.excluded_pct"]["value"] == pytest.approx(
        cell.excluded_pct)

    # the derived rows say where they come from
    src = "%s/a2/4b/sweep_n4/buckets.jsonl" % root.replace("\\", "/")
    assert keys["E1.fixed_b8.a2.4b.rel"]["source"] == [src]
    assert "derived from sweep_n4 (fixed budget B=8, K=2)" in keys["E1.fixed_b8.a2.4b.rel"]["note"]
    assert "derived from sweep_n4" in keys["E1.per_decision_n4.a2.4b.rel"]["note"]

    # the paired difference
    delta = sh.paired_cell_delta(made[("a2", 8)], made[("a2", 3)],
                                 n_boot=aggregate_e1.N_BOOT, seed=aggregate_e1.SEED)
    assert keys["E1.sweep_n.a2.4b.delta_n8_n3"]["value"] == pytest.approx(delta.delta)
    assert keys["E1.sweep_n.a2.4b.delta_n8_n3"]["n"] == delta.n_pairs
    assert keys["E1.sweep_n.a2.4b.delta_n8_n3_ci_lo"]["value"] == pytest.approx(delta.ci_lo)
    assert len(keys["E1.sweep_n.a2.4b.delta_n8_n3"]["source"]) == 2

    # every ignored directory and every refusal is reported once
    assert "not_a_workflow" in err
    assert "fixed_b8 is retired" in err or "is retired by the contract" in err
    assert "E1.per_decision_n4.a3.4b.wgv: not a manifest key" in err
    assert err.count("ignored:") >= 2


def test_aggregate_e1_runs_as_a_module(tmp_path):
    root = os.path.join(str(tmp_path), "E1")
    build_e1_tree(root)
    manifest_path = os.path.join(str(tmp_path), "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(fake_manifest(E1_MANIFEST_KEYS), fh)
    out_path = os.path.join(str(tmp_path), "summary.json")
    proc = subprocess.run(
        [sys.executable, "-m", "c3.analysis.rebuild.aggregate_e1",
         "--results", root, "--manifest", manifest_path, "--out", out_path],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr
    with open(out_path, "r", encoding="utf-8") as fh:
        summary = json.load(fh)
    assert summary["keys"], proc.stderr
    assert set(summary["keys"]) <= set(fake_manifest(E1_MANIFEST_KEYS))


def test_aggregate_e1b_on_a_minimal_tree(tmp_path, capsys):
    rng = np.random.default_rng(81)
    root = os.path.join(str(tmp_path), "E1b")
    for n, c in ((4, 4), (8, 2)):
        a, b = _noise_cell(n, c, rng, n_groups=25)
        write_jsonl(os.path.join(root, "a2", "4b", "n%d_c%d" % (n, c), "seedA",
                                 "buckets.jsonl"), a)
        write_jsonl(os.path.join(root, "a2", "4b", "n%d_c%d" % (n, c), "seedB",
                                 "buckets.jsonl"), b)
    # a cell whose second seed never landed, and a cell claimed by two models
    write_jsonl(os.path.join(root, "a2", "4b", "n2_c2", "seedA", "buckets.jsonl"),
                _noise_cell(2, 2, rng, n_groups=5)[0])
    dupe_a, dupe_b = _noise_cell(4, 4, rng, n_groups=5)
    write_jsonl(os.path.join(root, "a2", "m2", "n4_c4", "seedA", "buckets.jsonl"), dupe_a)
    write_jsonl(os.path.join(root, "a2", "m2", "n4_c4", "seedB", "buckets.jsonl"), dupe_b)

    keys = ["E1b.noise_law.ratio.n%d.c%d" % (n, c) for n in (2, 4, 8) for c in (2, 4, 8)]
    keys += ["E1b.noise_law.max_dev_pct", "E1b.noise_law.worst_ratio"]
    manifest_path = os.path.join(str(tmp_path), "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(fake_manifest(keys), fh)
    out_path = os.path.join(str(tmp_path), "summary.json")

    assert aggregate_e1b.main(["--results", root, "--manifest", manifest_path,
                               "--out", out_path]) == 0
    with open(out_path, "r", encoding="utf-8") as fh:
        summary = json.load(fh)
    err = capsys.readouterr().err

    assert summary["experiment"] == "E1b"
    written = summary["keys"]
    assert "E1b.noise_law.ratio.n8.c2" in written
    assert "E1b.noise_law.ratio.n4.c4" not in written      # two directories claim it
    assert "E1b.noise_law.ratio.n2.c2" not in written      # seedB missing
    assert written["E1b.noise_law.ratio.n8.c2"]["n"] > 0
    assert written["E1b.noise_law.max_dev_pct"]["value"] == pytest.approx(
        100.0 * abs(written["E1b.noise_law.worst_ratio"]["value"] - 1.0))
    assert "seedB" in err and "carry this cell" in err


def test_aggregate_e1c_temp_on_a_minimal_tree(tmp_path, capsys):
    rng = np.random.default_rng(91)
    root = os.path.join(str(tmp_path), "E1c")
    texts = ["same answer", "same answer", "a different answer", "yet another"]
    for name, ps in (("t070_n4", [0.2, 0.4, 0.6, 0.8]), ("t100_n8", [0.5] * 8)):
        buckets = bernoulli_cell(ps, 4, rng, 20, prefix=name)
        if name == "t070_n4":
            for b in buckets:
                for cand, text in zip(b["candidates"], texts):
                    cand["action_text"] = text
        write_jsonl(os.path.join(root, "temp", name, "buckets.jsonl"), buckets)
    for name in ("calib", "holdout"):
        write_jsonl(os.path.join(root, "band", name, "buckets.jsonl"),
                    bernoulli_cell([0.2, 0.5, 0.8], 4, rng, 15, prefix=name))
    os.makedirs(os.path.join(root, "temp", "t999_n4"), exist_ok=True)

    keys = ["E1c.temp.%s.%s" % (cond, q)
            for cond in ("t070", "t085", "t100", "t100_n8") for q in ("dup_rate", "rel")]
    keys.append("E1c.band.trigger_rel")
    manifest_path = os.path.join(str(tmp_path), "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(fake_manifest(keys), fh)
    out_path = os.path.join(str(tmp_path), "summary.json")

    assert aggregate_e1c_temp.main(["--results", root, "--manifest", manifest_path,
                                    "--out", out_path]) == 0
    with open(out_path, "r", encoding="utf-8") as fh:
        summary = json.load(fh)
    err = capsys.readouterr().err

    written = summary["keys"]
    assert set(written) <= set(keys)
    assert written["E1c.temp.t070.dup_rate"]["value"] == pytest.approx(0.5)
    assert written["E1c.temp.t070.dup_rate"]["n"] == 20
    assert written["E1c.temp.t070.rel"]["unit"] == "spearman"
    assert "E1c.temp.t100_n8.rel" in written
    assert "E1c.band.trigger_rel" not in written
    assert "band subset calib: rel=" in err
    assert "band subset holdout: rel=" in err
    assert "t999_n4 is not a sweep condition" in err


def _two_alternative_hand_reading(buckets):
    """The branching-2 cell, computed from the raw returns without the module.

    With two alternatives the leave-one-out advantages are (q0 - q1, q1 - q0),
    so a half is flat exactly when its two half means are equal, and the
    split-half correlation is +1 when the two halves order the pair the same way
    and -1 when they do not. The within-group variance of the advantages has the
    closed form 2 (m0 - m1)^2 with m the full means.
    """
    agree, usable, variances = 0, 0, []
    for b in buckets:
        rets = [c["returns"] for c in b["candidates"]]
        assert len(rets) == 2
        means = [sum(r) / len(r) for r in rets]
        variances.append(2.0 * (means[0] - means[1]) ** 2)
        a = [sum(r[0::2]) / len(r[0::2]) for r in rets]
        z = [sum(r[1::2]) / len(r[1::2]) for r in rets]
        da, dz = a[0] - a[1], z[0] - z[1]
        if da == 0 or dz == 0:
            continue
        usable += 1
        if (da > 0) == (dz > 0):
            agree += 1
    return {"agreement": agree / usable, "n_groups": usable,
            "n_total": len(buckets), "wgv": float(np.mean(variances))}


def test_two_alternative_rows_are_the_agreement_rate_and_its_correlation(tmp_path, capsys):
    """Driver ruling A3: the sweep row of a branching-2 cell is the direction
    agreement rate, the fixed-budget row of mt4 is the mean correlation of the
    same pool, and the two are the same reading (mean rho = 2 x agreement - 1).
    """
    rng = np.random.default_rng(101)
    root = os.path.join(str(tmp_path), "E1")
    buckets = bernoulli_cell([0.35, 0.65], 4, rng, 40, prefix="mt4_n2_")
    write_jsonl(os.path.join(root, "mt4", "4b", "sweep_n2", "buckets.jsonl"), buckets)

    manifest_keys = (["E1.sweep_n.mt4.4b.n2.rel"]
                     + ["E1.fixed_b8.mt4.4b.%s" % q
                        for q in ("rel", "rel_ci_lo", "rel_ci_hi", "wgv", "n_groups",
                                  "excluded_pct")])
    manifest_path = os.path.join(str(tmp_path), "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(fake_manifest(manifest_keys), fh)
    out_path = os.path.join(str(tmp_path), "summary.json")
    assert aggregate_e1.main(["--results", root, "--manifest", manifest_path,
                              "--out", out_path]) == 0
    with open(out_path, "r", encoding="utf-8") as fh:
        keys = json.load(fh)["keys"]

    hand = _two_alternative_hand_reading(buckets)
    assert 0 < hand["agreement"] < 1                      # both directions occur
    assert hand["n_groups"] < hand["n_total"]             # some halves are flat

    n2 = keys["E1.sweep_n.mt4.4b.n2.rel"]
    assert n2["value"] == pytest.approx(hand["agreement"])
    assert n2["n"] == hand["n_groups"]
    assert n2["unit"] == "agreement"

    mt4 = keys["E1.fixed_b8.mt4.4b.rel"]
    assert mt4["value"] == pytest.approx(2 * n2["value"] - 1)
    assert mt4["value"] == pytest.approx(2 * hand["agreement"] - 1)
    assert mt4["unit"] == "spearman"
    assert mt4["n"] == hand["n_groups"]

    # the rest of the fixed-budget row reads the same min_cands=2 pool
    assert keys["E1.fixed_b8.mt4.4b.n_groups"]["value"] == hand["n_groups"]
    assert keys["E1.fixed_b8.mt4.4b.excluded_pct"]["value"] == pytest.approx(
        100.0 * (hand["n_total"] - hand["n_groups"]) / hand["n_total"])
    assert keys["E1.fixed_b8.mt4.4b.wgv"]["value"] == pytest.approx(hand["wgv"])
    assert keys["E1.fixed_b8.mt4.4b.wgv"]["n"] == hand["n_total"]
    assert (keys["E1.fixed_b8.mt4.4b.rel_ci_lo"]["value"]
            <= mt4["value"] <= keys["E1.fixed_b8.mt4.4b.rel_ci_hi"]["value"])

    for key in ("E1.sweep_n.mt4.4b.n2.rel", "E1.fixed_b8.mt4.4b.rel"):
        assert aggregate_e1.N2_NOTE in keys[key]["note"]
    assert "derived from sweep_n2 (fixed budget B=8, K=4)" in mt4["note"]
    assert "the driver decides that convention" not in capsys.readouterr().err


def test_arm_display_names_are_the_six_printed_names():
    """Driver ruling B7: the name table the manifest's name keys print."""
    assert aggregate_e1.ARM_DISPLAY_NAMES == {
        "a2": "two-agent",
        "a3": "three-agent",
        "mt4": "four-turn",
        "branch": "branching",
        "c5": "five-agent chain",
        "c10": "ten-agent chain",
    }
    assert sorted(aggregate_e1.ARM_DISPLAY_NAMES) == sorted(aggregate_e1.WORKFLOWS)
    assert sorted(aggregate_e1.ARM_DISPLAY_NAMES) == sorted(aggregate_e1.PD_WORKFLOWS)


def test_delta_min_arm_writes_the_printed_workflow_name(tmp_path):
    """The binding workflow of the branching criterion is written as its printed
    name, and it is the workflow whose own delta key holds the smallest rise."""
    rng = np.random.default_rng(111)
    root = os.path.join(str(tmp_path), "E1")
    for wf in aggregate_e1.PD_WORKFLOWS:
        for n, ps in ((3, [0.25, 0.5, 0.75]),
                      (8, [0.02, 0.15, 0.3, 0.45, 0.6, 0.72, 0.85, 0.98])):
            write_jsonl(os.path.join(root, wf, "4b", "sweep_n%d" % n, "buckets.jsonl"),
                        bernoulli_cell(ps, 4, rng, 12, prefix="%s_n%d_" % (wf, n)))

    manifest_keys = (["E1.sweep_n.%s.4b.n%d.rel" % (wf, n)
                      for wf in aggregate_e1.PD_WORKFLOWS for n in (3, 8)]
                     + ["E1.sweep_n.%s.4b.delta_n8_n3%s" % (wf, tail)
                        for wf in aggregate_e1.PD_WORKFLOWS for tail in ("", "_ci_lo")]
                     + ["E1.sweep_n.4b.delta_n8_n3_min",
                        "E1.sweep_n.4b.delta_n8_n3_min_ci_lo",
                        "E1.sweep_n.4b.delta_n8_n3_min_arm"])
    manifest_path = os.path.join(str(tmp_path), "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(fake_manifest(manifest_keys), fh)
    out_path = os.path.join(str(tmp_path), "summary.json")
    assert aggregate_e1.main(["--results", root, "--manifest", manifest_path,
                              "--out", out_path]) == 0
    with open(out_path, "r", encoding="utf-8") as fh:
        keys = json.load(fh)["keys"]

    per_wf = {wf: keys["E1.sweep_n.%s.4b.delta_n8_n3" % wf]["value"]
              for wf in aggregate_e1.PD_WORKFLOWS}
    assert len(per_wf) == 6
    binding = min(sorted(per_wf), key=lambda wf: per_wf[wf])
    arm = keys["E1.sweep_n.4b.delta_n8_n3_min_arm"]
    assert arm["value"] == aggregate_e1.ARM_DISPLAY_NAMES[binding]
    assert arm["unit"] == "name"
    assert arm["n"] == 6
    assert keys["E1.sweep_n.4b.delta_n8_n3_min"]["value"] == pytest.approx(per_wf[binding])
    assert binding in arm["note"]


def test_summary_builder_refuses_what_the_contract_refuses(tmp_path, capsys):
    manifest = fake_manifest(["E1.sweep_n.a2.4b.n4.rel"])
    builder = aggregate_e1.SummaryBuilder("E1", "test", manifest)
    assert builder.add("E1.sweep_n.a2.4b.n4.rel", 0.5, n=3, source="x/buckets.jsonl")
    assert not builder.add("E1.not.a.key", 0.5, n=3, source="x")
    assert not builder.add("E1.reliability.budget_not_depth", "confirmed", n=3, source="x")
    assert not builder.add("E1.sweep_n.a2.4b.n4.rel", None, n=3, source="x")
    assert not builder.add("E1.sweep_n.a2.4b.n4.rel", float("nan"), n=3, source="x")
    assert not builder.add("E1.sweep_n.a2.4b.n4.rel", 0.5, n=0, source="x")
    assert list(builder.keys) == ["E1.sweep_n.a2.4b.n4.rel"]
    assert builder.keys["E1.sweep_n.a2.4b.n4.rel"]["source"] == ["x/buckets.jsonl"]
    assert len(builder.refused) == 5
    path = builder.write(os.path.join(str(tmp_path), "deep", "summary.json"))
    with open(path, "r", encoding="utf-8") as fh:
        assert json.load(fh)["keys"]["E1.sweep_n.a2.4b.n4.rel"]["value"] == 0.5
    assert "not a manifest key" in capsys.readouterr().err


def test_load_buckets_reports_a_broken_line(tmp_path):
    path = os.path.join(str(tmp_path), "buckets.jsonl")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(make_bucket([[1, 0], [0, 1], [1, 1]])) + "\n")
        fh.write("\n")
        fh.write("{not json\n")
    with pytest.raises(ValueError) as exc:
        sh.load_buckets(path)
    assert "line 3" in str(exc.value)
