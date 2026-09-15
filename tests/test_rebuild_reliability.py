# tests/test_rebuild_reliability.py
"""
Unit tests for the rebuild-experiment reliability family:
c3.analysis.rebuild.{splithalf, duplicates, noise_law, aggregate_e1,
aggregate_e1b, aggregate_e1c_temp}.

Everything here is synthetic: no bucket file from a real run is read, and no
number in these tests is a claim about the experiments. Several cases are the
offline-reproducible half of the self-tests of the reference split-half script,
which is the authoritative definition of the split-half estimator.
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
    land inside the probe band the analysis plan fixes. This checks the
    implementation against the formula, not the real data.

    No convention is passed in: the module defaults are the ones revision 10 of
    the analysis plan fixed, and this is the self-test of those defaults.
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
    is now ddof=0 (the maintainers' decision of 2026-09-15).
    """
    rng = np.random.default_rng([2027, n])
    a, b = _noise_cell(n, 4, rng, n_groups=60)
    at0 = nl.cell_noise_ratio(a, b, n_boot=0, seed=0, ddof=0)
    at1 = nl.cell_noise_ratio(a, b, n_boot=0, seed=0, ddof=1)
    assert nl.DDOF == 0
    assert at1["n_groups"] == at0["n_groups"]
    assert at1["ratio"] == pytest.approx(at0["ratio"] * n / (n - 1.0))


def test_noise_law_holds_when_the_alternatives_differ_in_quality():
    """The point of the sigma^2 convention (the maintainers' decision of 2026-09-15).

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

    The units follow the results manifest of the paper build: the branching-2
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
        elif ".noise_ratio." in key:
            # The halves reading of a sweep cell: a ratio, except the rank
            # correlation of that ratio against branching.
            if "vs_n_rho" in key:
                unit, fmt = "spearman", "{:+.2f}"
            else:
                unit, fmt = "ratio", "{:.2f}"
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
    for verdict_key in ("E1.reliability.budget_not_depth", "E1.noise_ratio.n4_all_in_band"):
        out[verdict_key] = {
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
    # the halves reading of every sweep cell, and its three cross-cell readings
    + ["E1.noise_ratio.%s.4b.n%d%s" % (wf, n, tail)
       for wf in ("a2", "a3") for n in (2, 3, 4, 6, 8)
       for tail in ("", "_ci_lo", "_ci_hi")]
    + ["E1.noise_ratio.median", "E1.noise_ratio.vs_n_rho",
       "E1.noise_ratio.vs_n_rho_ci_lo", "E1.noise_ratio.vs_n_rho_ci_hi"]
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
    assert "fixed_b8 is retired" in err or "is retired by the results layout" in err
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


# -------------------------
# the two E1 trees (2026-09-15)
#
# `E1` is the question-pool study the paper reports and `E1_math500all` the
# full-suite appendix comparison. They have the same layout and are different
# measurements, so the appendix tree writes `E1app.*`.
# -------------------------


def _appendix_manifest_keys():
    return [k.replace("E1.", "E1app.", 1) for k in E1_MANIFEST_KEYS]


def test_the_appendix_tree_under_the_default_prefix_is_refused(tmp_path, capsys):
    root = os.path.join(str(tmp_path), "E1_math500all")
    build_e1_tree(root)
    manifest_path = os.path.join(str(tmp_path), "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(fake_manifest(E1_MANIFEST_KEYS), fh)
    out_path = os.path.join(str(tmp_path), "summary.json")

    assert aggregate_e1.main(["--results", root, "--manifest", manifest_path,
                              "--out", out_path]) == 2
    assert not os.path.exists(out_path)
    err = capsys.readouterr().err
    assert "appendix tree E1_math500all" in err
    assert "--key_prefix E1app" in err
    # the refusal is the same whether it is reached through the CLI or the library
    assert aggregate_e1.key_prefix_refusal(root, "E1") is not None
    assert aggregate_e1.key_prefix_refusal(root, "E1app") is None
    assert aggregate_e1.key_prefix_refusal(root + os.sep, "E1") is not None
    with pytest.raises(ValueError):
        aggregate_e1.build_e1_summary(root, fake_manifest(E1_MANIFEST_KEYS))


def test_the_appendix_prefix_renames_every_key(tmp_path):
    root = os.path.join(str(tmp_path), "E1_math500all")
    build_e1_tree(root)
    manifest_path = os.path.join(str(tmp_path), "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(fake_manifest(_appendix_manifest_keys()), fh)
    out_path = os.path.join(str(tmp_path), "summary.json")

    assert aggregate_e1.main(["--results", root, "--manifest", manifest_path,
                              "--out", out_path, "--key_prefix", "E1app"]) == 0
    with open(out_path, "r", encoding="utf-8") as fh:
        summary = json.load(fh)

    assert summary["keys"]
    assert all(k.startswith("E1app.") for k in summary["keys"])
    assert not any(k.startswith("E1.") for k in summary["keys"])
    # the experiment name is the measurement's, the prefix is the tree's
    assert summary["experiment"] == "E1"
    assert summary["key_prefix"] == "E1app"
    assert summary["results_root"].endswith("E1_math500all")
    assert "\\" not in summary["results_root"]


def test_the_two_trees_write_the_same_values_under_different_names(tmp_path):
    """Nothing but the key names changes: the same buckets under the two prefixes
    give the same numbers, which is what makes the rename safe."""
    pool = os.path.join(str(tmp_path), "E1")
    appendix = os.path.join(str(tmp_path), "E1_math500all")
    build_e1_tree(pool)
    build_e1_tree(appendix)
    man = fake_manifest(E1_MANIFEST_KEYS + _appendix_manifest_keys())

    a = aggregate_e1.build_e1_summary(pool, man).payload()
    b = aggregate_e1.build_e1_summary(appendix, man, key_prefix="E1app").payload()

    assert set(b["keys"]) == {k.replace("E1.", "E1app.", 1) for k in a["keys"]}
    for key, entry in a["keys"].items():
        other = b["keys"][key.replace("E1.", "E1app.", 1)]
        assert other["value"] == entry["value"]
        assert other["n"] == entry["n"]
        assert other["unit"] == entry["unit"]


def test_the_prefix_only_renames_the_builders_own_experiment_segment():
    man = fake_manifest(["E1app.sweep_n.a2.4b.n4.rel", "E1b.noise_law.max_dev_pct"])
    builder = aggregate_e1.SummaryBuilder("E1", "script", man, key_prefix="E1app")
    assert builder.key("E1.sweep_n.a2.4b.n4.rel") == "E1app.sweep_n.a2.4b.n4.rel"
    assert builder.key("E1b.noise_law.max_dev_pct") == "E1b.noise_law.max_dev_pct"
    assert builder.key("no_dot") == "no_dot"
    # a builder with no prefix is the old behaviour exactly
    plain = aggregate_e1.SummaryBuilder("E1", "script", man)
    assert plain.key("E1.sweep_n.a2.4b.n4.rel") == "E1.sweep_n.a2.4b.n4.rel"


def test_the_skip_lines_name_the_keys_the_tree_would_have_written(tmp_path, capsys):
    """An operator reading stderr of the appendix run must not be sent looking
    for `E1.*` keys that run never writes."""
    root = os.path.join(str(tmp_path), "E1_math500all")
    build_e1_tree(root)
    aggregate_e1.build_e1_summary(root, fake_manifest(_appendix_manifest_keys()),
                                  key_prefix="E1app")
    err = capsys.readouterr().err
    assert "skip E1app.per_decision_n4.4b.rel_{min,max,range}" in err
    assert "skip E1app.sweep_n.4b.delta_n8_n3_min{,_ci_lo,_arm}" in err
    assert "E1app: " in err
    assert "skip E1." not in err


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
    """The maintainers' decision of 2026-09-15: the sweep row of a branching-2 cell is the direction
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
    assert "the maintainers decide that convention" not in capsys.readouterr().err


def test_arm_display_names_are_the_six_printed_names():
    """The name table the manifest's name keys print."""
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


def test_summary_builder_refuses_what_the_results_layout_refuses(tmp_path, capsys):
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


# -------------------------
# 9. the noise ratio read off each sweep cell's own halves
#
# Revision 19 of the preregistration (2026-09-15) reads the noise law off the E1
# sweep cells as well as off the E1b grid: the even and the odd replays of one
# group are two independent estimates of the same advantage vector at the same
# decision point, so a cell collected once already carries a noise measurement
# and nothing is rerun for it. The estimator is the one E1b calls; the first two
# tests here are the guard that factoring it out moved no E1b number.
# -------------------------


def _hand_e1b_reading(bucket_a, bucket_b, ddof=0):
    """One group's E1b reading, transcribed from noise_law.group_noise as it stood
    before `returns_noise` was factored out of it (2026-09-15).

    A second copy rather than a pinned constant: it can be read against the
    implementation line by line, and it catches a moved number on any data, not
    just on the one group a constant would cover.
    """
    rets_a = [list(c["returns"]) for c in bucket_a["candidates"]]
    rets_b = [list(c["returns"]) for c in bucket_b["candidates"]]
    n = len(rets_a)
    if n < 2 or len(rets_b) != n:
        return None
    if any(len(r) == 0 for r in rets_a) or any(len(r) == 0 for r in rets_b):
        return None
    adv_a = sh.loo_adv([float(np.mean(r)) for r in rets_a])
    adv_b = sh.loo_adv([float(np.mean(r)) for r in rets_b])
    measured = float(np.var(adv_a - adv_b, ddof=ddof)) / 2.0
    per_alt = []
    for ra, rb in zip(rets_a, rets_b):
        vals = [float(v) for v in ra] + [float(v) for v in rb]
        if len(vals) >= 2:
            per_alt.append(float(np.var(vals, ddof=1)))
    if not per_alt:
        return None
    sigma2 = float(np.mean(per_alt))
    counts = [len(r) for r in rets_a] + [len(r) for r in rets_b]
    c = float(np.mean(counts))
    budget = n * c
    predicted = sigma2 * n * n / (budget * (n - 1))
    if not (predicted > 0):
        return None
    return {"n": n, "c": c, "measured": measured, "predicted": predicted,
            "ratio": measured / predicted}


def _hand_halves_reading(bucket, ddof=0):
    """One group's halves reading, written out from the raw returns.

    Independent of the module on purpose: truncate every alternative to the
    smallest replay count, split the replay indices into even and odd, take each
    half's leave-one-out advantages, halve the mean of their squared difference,
    and predict from the within-alternative sample variance of all the replays
    the halves used at the budget of ONE half.
    """
    rets = [list(c["returns"]) for c in bucket["candidates"]]
    n = len(rets)
    c_min = min(len(r) for r in rets)
    used = [r[:c_min] for r in rets]
    even = [r[0::2] for r in used]
    odd = [r[1::2] for r in used]
    adv_even = sh.loo_adv([float(np.mean(r)) for r in even])
    adv_odd = sh.loo_adv([float(np.mean(r)) for r in odd])
    measured = float(np.var(adv_even - adv_odd, ddof=ddof)) / 2.0
    sigma2 = float(np.mean([float(np.var(r, ddof=1)) for r in used]))
    k = (len(even[0]) + len(odd[0])) / 2.0
    predicted = sigma2 * n * n / ((n * k) * (n - 1))
    return {"n": n, "c": k, "measured": measured, "predicted": predicted,
            "ratio": measured / predicted if predicted > 0 else None}


def test_the_e1b_reading_is_unchanged_by_the_shared_estimator():
    """The regression the refactor owes: every E1b number, group by group, is the
    one the pre-refactor body produced. Exact equality, not approximate: the
    operations and their order are meant to be the same ones."""
    rng = np.random.default_rng(1801)
    a, b = _noise_cell(4, 4, rng, n_groups=40)
    checked = 0
    for ba, bb in zip(a, b):
        got = nl.group_noise(ba, bb)
        hand = _hand_e1b_reading(ba, bb)
        assert (got is None) == (hand is None)
        if got is None:
            continue
        checked += 1
        assert got.n == hand["n"]
        assert got.c == hand["c"]
        assert got.measured == hand["measured"]
        assert got.predicted == hand["predicted"]
        assert got.ratio == hand["ratio"]
    assert checked >= 30

    # the wrapper adds nothing but a choice of which returns are the two sides
    core = nl.returns_noise(sh.candidate_returns(a[0]), sh.candidate_returns(b[0]))
    direct = nl.group_noise(a[0], b[0])
    assert (core.measured, core.predicted, core.ratio, core.n, core.c) == (
        direct.measured, direct.predicted, direct.ratio, direct.n, direct.c)


def test_the_e1b_cell_numbers_are_unchanged_by_the_shared_estimator():
    """The same guard one level up, including the bootstrap interval, which is
    what the manifest key prints."""
    rng = np.random.default_rng(1802)
    a, b = _noise_cell(4, 4, rng, n_groups=30)
    rep = nl.cell_noise_ratio(a, b, n_boot=500, seed=0)
    hand = [_hand_e1b_reading(ba, bb) for ba, bb in zip(a, b)]
    hand = [h["ratio"] for h in hand if h is not None]

    assert rep["n_groups"] == len(hand)
    assert sorted(rep["ratios"]) == pytest.approx(sorted(hand))
    assert rep["ratio"] == pytest.approx(float(np.mean(hand)))
    assert rep["ci_lo"] <= rep["ratio"] <= rep["ci_hi"]
    # the E1b cell reading does not see the halves flag at all: its two sides are
    # the two seeds, and a flat half is not a thing it can have
    assert "n_flat_half" not in rep


@pytest.mark.parametrize("n,c", [(2, 4), (3, 4), (4, 4), (6, 4), (8, 4), (4, 8), (8, 8)])
def test_the_halves_reading_matches_the_law_on_data_built_to_it(n, c):
    """The anchor of the whole reading, and the reason this work package has one.

    Returns are drawn exactly under the model the prediction assumes: every
    alternative has the same return variance (Bernoulli(0.5), so sigma^2 = 0.25)
    and replays are independent. A half then holds k = c / 2 replays per
    alternative, the leave-one-out advantage of a half has variance
    sigma^2 n / (k (n - 1)) per element, and the difference of the two halves
    measures exactly that. Both sides of the ratio have to land on that closed
    form.

    Read with the shipped defaults, which since 2026-09-15 keep the groups whose
    half came back flat. That matters to this check as much as to the data: the
    clause is a selection on the quantity being measured, so switching it on
    would move the mean away from the closed form with nothing wrong in the
    estimator (test_the_flat_half_clause_lifts_the_halves_ratio measures it).

    The tolerance is Monte Carlo, not a claim about the estimator: a thousand
    groups of a statistic whose own spread is larger than its mean leave the mean
    good to a few percent, and 15 percent is several times that. What the check
    really pins is the factor of two: using the whole cell's budget n c, instead
    of one half's n c / 2, would halve the prediction and no tolerance of this
    size would hide that.
    """
    rng = np.random.default_rng([2029, n, c])
    buckets = bernoulli_cell([0.5] * n, c, rng, 1000)
    min_cands = 2 if n == 2 else sh.DEFAULT_MIN_CANDS
    readings = [nl.group_noise_halves(b, min_cands=min_cands) for b in buckets]
    readings = [g for g in readings if g is not None]
    assert len(readings) >= 900

    k = c / 2.0
    closed_form = 0.25 * n / (k * (n - 1))
    assert float(np.mean([g.measured for g in readings])) == pytest.approx(
        closed_form, rel=0.15)
    assert float(np.mean([g.predicted for g in readings])) == pytest.approx(
        closed_form, rel=0.15)
    assert all(g.c == k for g in readings)
    assert all(g.n == n for g in readings)

    rep = nl.cell_noise_ratio_halves(buckets, n_boot=200, seed=0, min_cands=min_cands)
    assert rep["n_groups"] == len(readings)
    assert rep["replays_per_half"] == pytest.approx(k)
    assert rep["n_half_uneven"] == 0
    assert nl.in_band(rep["ratio"]), rep["ratio"]
    assert rep["ci_lo"] <= rep["ratio"] <= rep["ci_hi"]


def test_the_halves_prediction_is_read_at_one_half_s_budget():
    """The budget of the prediction, on one hand-built group and with no
    randomness: four replays per alternative, so two per half, so the prediction
    is at n x 2 and not at n x 4. Read against the E1b reading of the same
    returns split as two seeds, which is the same arithmetic at twice the budget.
    """
    returns = [[1.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 1.0]]
    bucket = make_bucket(returns)
    got = nl.group_noise_halves(bucket)
    hand = _hand_halves_reading(bucket)

    assert got.n == 3
    assert got.c == pytest.approx(2.0)                    # replays per HALF
    assert got.measured == pytest.approx(hand["measured"])
    assert got.predicted == pytest.approx(hand["predicted"])
    assert got.ratio == pytest.approx(hand["ratio"])

    sigma2 = float(np.mean([float(np.var(r, ddof=1)) for r in returns]))
    assert got.predicted == pytest.approx(sigma2 * 9 / (6 * 2))   # budget 3 x 2, not 3 x 4

    # Handing the same two halves to the E1b entry point as if they were two
    # seeds gives the same numbers, which is the claim that the two readings are
    # one estimator: same pooled sigma^2, same two replays per side, same budget.
    even = make_bucket([r[0::2] for r in returns])
    odd = make_bucket([r[1::2] for r in returns])
    as_two_seeds = nl.group_noise(even, odd)
    assert as_two_seeds.measured == pytest.approx(got.measured)
    assert as_two_seeds.predicted == pytest.approx(got.predicted)
    assert as_two_seeds.c == pytest.approx(got.c)


def test_the_halves_reading_takes_the_count_clauses_and_counts_the_flat_one():
    """Which groups the two statistics share, and the two places they part.

    The count clauses are shared: too few alternatives and too few replays drop
    a group from both. The two asymmetries are deliberate and go in opposite
    directions. A group with a flat half has no correlation and does have a noise
    reading, so it stays here and is counted (the maintainers' decision of
    2026-09-15). A group whose replays never vary has a perfectly good
    correlation and no noise reading at all, because its predicted noise is zero
    and the ratio would divide by it.
    """
    short = make_bucket([[1.0, 0.0], [0.0, 1.0]], bucket_id="b_short")
    one_replay = make_bucket([[1.0], [0.0], [1.0]], bucket_id="b_one")
    flat = make_bucket([[1.0, 1.0], [1.0, 1.0], [0.0, 1.0]], bucket_id="b_flat")
    frozen = make_bucket([[1.0, 1.0], [0.0, 0.0], [1.0, 1.0]], bucket_id="b_frozen")
    good = make_bucket([[1.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 0.0],
                        [1.0, 1.0, 0.0, 1.0]], bucket_id="b_good")

    assert sh.bucket_splithalf(short) is None
    assert nl.group_noise_halves(short) is None
    assert sh.bucket_splithalf(one_replay) is None
    assert nl.group_noise_halves(one_replay) is None
    # the flat half: the correlation cannot exist, the noise reading can, and by
    # default it is taken
    assert sh.bucket_splithalf(flat) is None
    assert nl.group_noise_halves(flat) is not None
    assert nl.group_noise_halves(flat, drop_flat_halves=True) is None
    # the frozen group: the correlation exists, the noise reading cannot
    assert sh.bucket_splithalf(frozen) == pytest.approx(1.0)
    assert nl.group_noise_halves(frozen) is None
    assert nl.group_noise_halves(frozen, drop_flat_halves=True) is None

    cell = [short, one_replay, flat, frozen, good]
    rep = nl.cell_noise_ratio_halves(cell, n_boot=0)
    assert rep["n_groups"] == 2 and rep["n_total"] == 5
    assert rep["excluded_pct"] == pytest.approx(60.0)
    assert rep["n_flat_half"] == 1
    hand_flat = _hand_halves_reading(flat)["ratio"]
    hand_good = _hand_halves_reading(good)["ratio"]
    assert rep["ratio"] == pytest.approx((hand_flat + hand_good) / 2.0)
    # the other pool rides along: the same cell without the flat-half group
    assert rep["n_groups_flat_dropped"] == 1
    assert rep["ratio_flat_dropped"] == pytest.approx(hand_good)
    assert nl.cell_noise_ratio_halves(cell, n_boot=0, drop_flat_halves=True)["ratio"] == (
        pytest.approx(rep["ratio_flat_dropped"]))
    # the reliability of the same cell keeps the frozen group and drops the flat one
    assert sh.cell_reliability(cell, n_random_splits=5, n_boot=0).n_groups == 2


def test_the_flat_half_clause_lifts_the_halves_ratio():
    """What the flat-half clause costs, measured rather than argued.

    A group whose even half comes back flat is a group whose even estimate is
    exactly zero. Under the null used here (every alternative the same quality)
    its squared difference is one side's variance instead of two, so it sits in
    the low half of the measured noise, and dropping it lifts the cell mean. The
    clause therefore lifts the ratio most where flat halves are common, which is
    at few alternatives: with two alternatives and four replays about six groups
    in ten come back with a flat half, against roughly one in eight at four
    alternatives.

    This is the measurement the maintainers decided on (2026-09-15): the noise
    reading keeps the flat-half groups, because that lift falls with the number
    of alternatives and would plant a trend in exactly the quantity secondary
    criterion a' tests. The clause stays reachable as a keyword, and every cell
    reports the other pool's reading, so the sensitivity is never lost.

    The assertion is only on the direction and only at two alternatives, where
    the effect is far larger than the Monte Carlo spread.
    """
    rng = np.random.default_rng(1803)
    buckets = bernoulli_cell([0.5, 0.5], 4, rng, 800)
    kept = nl.cell_noise_ratio_halves(buckets, n_boot=0, min_cands=2,
                                      drop_flat_halves=False)
    dropped = nl.cell_noise_ratio_halves(buckets, n_boot=0, min_cands=2,
                                         drop_flat_halves=True)

    # the measured noise itself, where the selection acts and the Monte Carlo
    # spread is smallest: the surviving groups are the noisier ones
    def measured(flag):
        got = [nl.group_noise_halves(b, min_cands=2, drop_flat_halves=flag)
               for b in buckets]
        return float(np.mean([g.measured for g in got if g is not None]))

    assert measured(True) > measured(False)
    assert dropped["n_flat_half"] == kept["n_flat_half"] > 0
    # the pool only grows by flat-half groups, and by at most all of them (a flat
    # group whose replays never varied has no prediction either way)
    assert dropped["n_groups"] < kept["n_groups"] <= dropped["n_groups"] + dropped["n_flat_half"]
    assert dropped["ratio"] > kept["ratio"]
    # the sensitivity every cell reports is exactly the other pool's reading
    assert kept["ratio_flat_dropped"] == pytest.approx(dropped["ratio"])
    assert kept["n_groups_flat_dropped"] == dropped["n_groups"]
    assert dropped["ratio_flat_dropped"] == pytest.approx(dropped["ratio"])


def test_the_cross_cell_readings():
    """The three statements revision 19 makes over the cells: a median, a rank
    correlation against branching with an interval over cells, and a band test.
    """
    assert nl.RATIO_BAND == (0.7, 1.4)
    assert nl.in_band(0.7) and nl.in_band(1.4) and nl.in_band(1.0)
    assert not nl.in_band(0.69) and not nl.in_band(1.41)
    assert not nl.in_band(None) and not nl.in_band(float("nan"))
    assert nl.all_in_band([0.8, 1.2]) is True
    assert nl.all_in_band([0.8, 1.5]) is False
    assert nl.all_in_band([]) is None
    assert nl.ratio_median([]) is None
    assert nl.ratio_median([1.0, 2.0, 3.0]) == pytest.approx(2.0)
    assert nl.ratio_median([1.0, 2.0]) == pytest.approx(1.5)

    # a set that rises with branching: the correlation is exactly one, and the
    # interval over cells sits above zero
    rising = [(n, 0.8 + 0.05 * n) for n in (2, 3, 4, 6, 8) for _ in range(6)]
    up = nl.ratio_vs_n(rising, n_boot=500, seed=0)
    assert up["rho"] == pytest.approx(1.0)
    assert up["n_cells"] == 30
    assert up["n_draws"] + up["n_degenerate"] == 500
    assert up["ci_lo"] > 0

    # the same branchings with no trend, built so the correlation is exactly zero
    # rather than zero on average: every branching carries the same six ratios, so
    # the interval has to contain zero, which is what criterion a' asks of the
    # real cells
    plateau = [(n, r) for n in (2, 3, 4, 6, 8)
               for r in (0.90, 0.95, 1.00, 1.05, 1.10, 1.15)]
    none = nl.ratio_vs_n(plateau, n_boot=500, seed=0)
    assert none["rho"] == pytest.approx(0.0, abs=1e-9)
    assert none["ci_lo"] < 0 < none["ci_hi"]

    # the point estimate is the tie-corrected Spearman scipy computes
    ns = np.asarray([p[0] for p in plateau], dtype=float)
    rs = np.asarray([p[1] for p in plateau], dtype=float)
    assert none["rho"] == pytest.approx(stats.spearmanr(ns, rs).statistic, abs=1e-9)

    # degenerate inputs say so instead of inventing a correlation
    assert nl.ratio_vs_n([(4, 1.0)])["rho"] is None
    assert nl.ratio_vs_n([(4, 1.0), (4, 1.2), (4, 0.9)])["rho"] is None
    assert nl.ratio_vs_n(rising, n_boot=0)["ci_lo"] is None


def test_aggregate_e1_writes_the_halves_noise_ratio(tmp_path, capsys):
    """The keys the sweep gains, on the same minimal tree the other aggregator
    tests use: one ratio and two bounds per cell, the median and the rank
    correlation over cells, and the verdict of criterion b' printed rather than
    written.
    """
    root = os.path.join(str(tmp_path), "E1")
    made = build_e1_tree(root)
    manifest_path = os.path.join(str(tmp_path), "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(fake_manifest(E1_MANIFEST_KEYS), fh)
    out_path = os.path.join(str(tmp_path), "summary.json")

    assert aggregate_e1.main(["--results", root, "--manifest", manifest_path,
                              "--out", out_path]) == 0
    with open(out_path, "r", encoding="utf-8") as fh:
        keys = json.load(fh)["keys"]
    err = capsys.readouterr().err

    expected = {}
    for (wf, n), buckets in made.items():
        expected[(wf, n)] = nl.cell_noise_ratio_halves(
            buckets, n_boot=aggregate_e1.N_BOOT, seed=aggregate_e1.SEED,
            min_cands=aggregate_e1.MIN_CANDS_N2 if n == 2 else aggregate_e1.MIN_CANDS)

    written = [k for k in keys if k.startswith("E1.noise_ratio.")]
    assert written, "no halves noise key was written at all"
    for (wf, n), rep in sorted(expected.items()):
        base = "E1.noise_ratio.%s.4b.n%d" % (wf, n)
        if rep["ratio"] is None:
            assert base not in keys
            continue
        assert keys[base]["value"] == pytest.approx(rep["ratio"])
        assert keys[base]["n"] == rep["n_groups"]
        assert keys[base]["unit"] == "ratio"
        assert keys[base]["source"] == [
            "%s/%s/4b/sweep_n%d/buckets.jsonl" % (root.replace("\\", "/"), wf, n)]
        assert keys[base + "_ci_lo"]["value"] == pytest.approx(rep["ci_lo"])
        assert keys[base + "_ci_hi"]["value"] == pytest.approx(rep["ci_hi"])
        assert (keys[base + "_ci_lo"]["value"] <= keys[base]["value"]
                <= keys[base + "_ci_hi"]["value"])
        # a cell holding flat-half groups keeps them and says what dropping them
        # would have read (the maintainers' decision of 2026-09-15)
        if rep["n_flat_half"]:
            assert "came back with a flat half and are kept" in keys[base]["note"]
            assert ("dropping them instead would read %.4f" % rep["ratio_flat_dropped"]
                    in keys[base]["note"])
    assert "replays per half" in keys["E1.noise_ratio.a2.4b.n4"]["note"]

    # the three cross-cell readings
    usable = {cell: rep for cell, rep in expected.items() if rep["ratio"] is not None}
    ordered = sorted((wf, "4b", n) for (wf, n) in usable)
    values = [usable[(wf, n)]["ratio"] for (wf, _, n) in ordered]
    assert keys["E1.noise_ratio.median"]["value"] == pytest.approx(nl.ratio_median(values))
    assert keys["E1.noise_ratio.median"]["n"] == len(values)
    assert keys["E1.noise_ratio.median"]["unit"] == "ratio"
    vs_n = nl.ratio_vs_n([(n, usable[(wf, n)]["ratio"]) for (wf, _, n) in ordered],
                         n_boot=aggregate_e1.N_BOOT, seed=aggregate_e1.SEED)
    assert keys["E1.noise_ratio.vs_n_rho"]["value"] == pytest.approx(vs_n["rho"])
    assert keys["E1.noise_ratio.vs_n_rho_ci_lo"]["value"] == pytest.approx(vs_n["ci_lo"])
    assert keys["E1.noise_ratio.vs_n_rho_ci_hi"]["value"] == pytest.approx(vs_n["ci_hi"])
    assert keys["E1.noise_ratio.vs_n_rho"]["unit"] == "spearman"
    # both cross-cell keys say that they pool the substrates together
    for key in ("E1.noise_ratio.median", "E1.noise_ratio.vs_n_rho"):
        assert "pooled across substrates" in keys[key]["note"]

    # the verdict of criterion b' is printed and never written, and this tree has
    # only two of the six workflows, so the line says it cannot be read yet
    assert "E1.noise_ratio.n4_all_in_band" not in keys
    assert "E1.noise_ratio.n4_all_in_band cannot be read yet" in err
    assert "2 of the six workflows" in err


def test_the_appendix_tree_carries_the_cell_ratios_and_decides_nothing(tmp_path, capsys):
    """The halves reading is measured on both trees and decided on neither by a
    script. The per-cell keys are renamed like every other cell key; the three
    cross-cell keys belong to the question-pool tree, so the appendix run refuses
    them the way it refuses any key its manifest does not define, and the verdict
    line is not printed there at all.
    """
    root = os.path.join(str(tmp_path), "E1_math500all")
    build_e1_tree(root)
    manifest = fake_manifest(_appendix_manifest_keys())
    for key in ("E1app.noise_ratio.median", "E1app.noise_ratio.vs_n_rho",
                "E1app.noise_ratio.vs_n_rho_ci_lo", "E1app.noise_ratio.vs_n_rho_ci_hi"):
        manifest.pop(key, None)

    payload = aggregate_e1.build_e1_summary(root, manifest, key_prefix="E1app").payload()
    err = capsys.readouterr().err

    ratios = [k for k in payload["keys"] if k.startswith("E1app.noise_ratio.")]
    assert ratios
    assert all(".4b.n" in k for k in ratios)
    assert "skip E1app.noise_ratio.median: not a manifest key" in err
    assert "n4_all_in_band" not in err
    assert "skip E1." not in err
