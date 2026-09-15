# c3/analysis/rebuild/coupling.py
"""E5: the paired contrast between two policies, with the estimators frozen.

Two arms (the frozen SFT policy and the seed-0 C3 policy) run the same question
set with the same workflow and the same budget, so every quantity is paired on
`question_id` and the difficulty of the questions cancels. Three quantities per
decision point:

    rel          split-half Spearman reliability of the measured advantages
    influence    answer-level plug-in mutual information, nats
    correction   among the alternatives whose upstream message is wrong, the
                 share of downstream replays that still answer correctly

The claim the experiment tests is a coupling: training should lower influence
(the downstream agent leans on the upstream message less) while raising the
correction rate, with reliability unchanged.

What changed against `30_analysis/server_tierA/e11_paired.py`, the script this
is moved from:
- influence is answer-level with the Miller-Madow correction and is reported in
  nats without clipping at zero, as the preregistration froze it. The old script
  used the uncorrected plug-in value clipped at zero.
- correction counts a downstream replay as repaired when its ANSWER is correct.
  The old script used the recorded return of that replay instead.
- no tertile view. The tertile split conflates influence with difficulty, which
  is exactly what the pairing is here to remove.
- no AUC and no OLS. Neither is preregistered, so neither is computed; nothing
  in this module partials difficulty or reliability out of anything.

Answer extraction and correctness are injected (`extractor`, `is_correct`), so
this module never parses mathematics and never decides equivalence.

Module-level imports are limited to numpy, scipy and the standard library.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

from c3.analysis.rebuild.influence import (
    answer_symbol,
    bucket_influence,
    format_p,
)

__all__ = [
    "MIN_CANDS",
    "arm_scan",
    "bucket_correction",
    "coupling_report",
    "e5_key_values",
]

#: Fewer alternatives than this and the bucket is dropped from the reliability
#: estimate (the exclusion rule of the reference split-half script).
MIN_CANDS = 3

#: Tolerance for "this difference is exactly zero"; see `_paired_t`.
_ZERO_TOL = 1e-12


# -----------------------------------------------------------------------------
# split-half reliability (the minimal even/odd estimator of WP-R2a section 1.1)
# -----------------------------------------------------------------------------


def _loo_adv(q: Sequence[float]) -> Optional[np.ndarray]:
    """Leave-one-out advantage: adv[j] = q[j] - (sum(q) - q[j]) / (n - 1)."""
    arr = np.asarray(list(q), dtype=float)
    n = arr.size
    if n < 2:
        return None
    return arr - (arr.sum() - arr) / (n - 1)


def _splithalf_evenodd(bucket: Mapping[str, Any], *, min_cands: int = MIN_CANDS) -> Optional[float]:
    """Per-bucket split-half Spearman on the even and odd replay indices.

    Definition, word for word the one WP-R2a section 1.1 froze from
    `30_analysis/local/splithalf_bootstrap_local.py`: c_min is the smallest
    replay count over the alternatives; q_A[j] is alternative j's mean return
    over the even replay indices below c_min and q_B[j] over the odd ones;
    adv(q)[j] = q[j] - (sum(q) - q[j]) / (n - 1); rho is the Spearman
    correlation of adv(q_A) with adv(q_B). The bucket is excluded (None) when it
    has fewer than `min_cands` alternatives, when c_min < 2, or when either
    half's advantage vector is constant.

    This is a private copy on purpose: WP-R2a owns the full reliability module
    and the two work packages share no files, so the driver can check the two
    implementations against each other.
    """
    cands = bucket.get("candidates", []) or []
    rets = [list(c.get("returns", []) or []) for c in cands]
    if len(rets) < min_cands:
        return None
    c_min = min(len(r) for r in rets)
    if c_min < 2:
        return None

    qa = [float(np.mean([r[i] for i in range(c_min) if i % 2 == 0])) for r in rets]
    qb = [float(np.mean([r[i] for i in range(c_min) if i % 2 == 1])) for r in rets]
    adv_a, adv_b = _loo_adv(qa), _loo_adv(qb)
    if adv_a is None or adv_b is None:
        return None
    if not (np.std(adv_a) > 0 and np.std(adv_b) > 0):
        return None
    rho = stats.spearmanr(adv_a, adv_b)[0]
    if rho is None or np.isnan(rho):
        return None
    return float(rho)


# -----------------------------------------------------------------------------
# correction rate
# -----------------------------------------------------------------------------


def bucket_correction(
    bucket: Mapping[str, Any],
    extractor: Callable[[str], Optional[str]],
    is_correct: Callable[[str, Any], bool],
) -> Tuple[float, int, int]:
    """(rate, n_wrong_alternatives, n_replays) for one decision point.

    An alternative is "wrong upstream" when its own `action_text`, read through
    the same extractor, does not answer the question correctly. The rate is the
    share of the downstream replays of those alternatives whose answer is
    correct, pooled over the alternatives of this bucket.

    The rate is nan when the bucket has no wrong alternative with replays: the
    quantity is undefined there, and such a question drops out of the pairing.
    `n_wrong_alternatives` counts ALTERNATIVES, as the specification requires,
    including a wrong one that happens to carry no replay.
    """
    qid = bucket.get("question_id")
    cands = bucket.get("candidates", []) or []
    n_wrong = 0
    hits = 0
    trials = 0
    for c in cands:
        upstream = answer_symbol(c.get("action_text", ""), extractor)
        if is_correct(upstream, qid):
            continue
        n_wrong += 1
        nexts = c.get("next_actions", []) or []
        if not isinstance(nexts, list):
            nexts = [nexts]
        for t in nexts:
            trials += 1
            if is_correct(answer_symbol(t, extractor), qid):
                hits += 1
    rate = (hits / trials) if trials else float("nan")
    return rate, n_wrong, trials


# -----------------------------------------------------------------------------
# one arm
# -----------------------------------------------------------------------------


def arm_scan(
    buckets: Iterable[Mapping[str, Any]],
    extractor: Callable[[str], Optional[str]],
    is_correct: Callable[[str, Any], bool],
) -> Tuple[Dict[Any, Dict[str, Any]], Dict[str, int]]:
    """Scan one arm's bucket file into per-question quantities.

    Returns (by_question, stats). A question id seen twice keeps its first
    bucket and raises the `duplicate_question_id` counter, so an accidental
    concatenation shows up instead of silently changing a mean.
    """
    by_q: Dict[Any, Dict[str, Any]] = {}
    counters = {"buckets": 0, "duplicate_question_id": 0, "no_question_id": 0, "rel_excluded": 0}
    for b in buckets:
        counters["buckets"] += 1
        qid = b.get("question_id")
        if qid is None:
            counters["no_question_id"] += 1
            continue
        if qid in by_q:
            counters["duplicate_question_id"] += 1
            continue
        rel = _splithalf_evenodd(b)
        if rel is None:
            counters["rel_excluded"] += 1
        rate, n_wrong, n_trials = bucket_correction(b, extractor, is_correct)
        by_q[qid] = {
            "rel": rel,
            "influence": float(bucket_influence(b, extractor)),
            "correction": rate,
            "n_wrong": n_wrong,
            "n_trials": n_trials,
        }
    return by_q, counters


# -----------------------------------------------------------------------------
# the paired contrast
# -----------------------------------------------------------------------------


def _paired_t(a: Sequence[float], b: Sequence[float]) -> Dict[str, Any]:
    """Paired contrast b minus a: mean difference, 95% t interval, two-sided p.

    Degenerate cases, the convention this work package fixed: with every
    difference exactly zero the two arms are identical, so p = 1.0 and the
    interval is [0, 0] (scipy returns nan, because the standard error is zero).
    With a constant non-zero difference scipy's own answer is kept, p = 0.0, and
    the interval collapses onto that difference. Fewer than two pairs gives nan
    everywhere.
    """
    x = np.asarray(list(a), dtype=float)
    y = np.asarray(list(b), dtype=float)
    d = y - x
    n = int(d.size)
    out: Dict[str, Any] = {"n": n, "mean_a": float("nan"), "mean_b": float("nan"),
                           "diff": float("nan"), "ci_lo": float("nan"),
                           "ci_hi": float("nan"), "p": float("nan"),
                           "wilcoxon_p": float("nan")}
    if n == 0:
        return out
    out["mean_a"] = float(np.mean(x))
    out["mean_b"] = float(np.mean(y))
    out["diff"] = float(np.mean(d))
    if n < 2:
        return out

    sd = float(np.std(d, ddof=1))
    if sd <= 0.0:
        if abs(out["diff"]) < _ZERO_TOL:
            out["ci_lo"] = out["ci_hi"] = 0.0
            out["p"] = 1.0
        else:
            out["ci_lo"] = out["ci_hi"] = out["diff"]
            out["p"] = 0.0
    else:
        half = float(stats.t.ppf(0.975, n - 1)) * sd / np.sqrt(n)
        out["ci_lo"] = out["diff"] - half
        out["ci_hi"] = out["diff"] + half
        out["p"] = float(stats.ttest_rel(y, x).pvalue)

    if sd <= 0.0 and abs(out["diff"]) < _ZERO_TOL:
        # Same convention as the t test above. Calling scipy here would divide
        # by a zero standard error and only produce a warning and a nan.
        out["wilcoxon_p"] = 1.0
    else:
        try:
            with np.errstate(invalid="ignore", divide="ignore"):
                out["wilcoxon_p"] = float(stats.wilcoxon(y, x).pvalue)
        except (ValueError, ZeroDivisionError):
            out["wilcoxon_p"] = float("nan")
    return out


def coupling_report(
    scan_a: Mapping[Any, Mapping[str, Any]],
    scan_b: Mapping[Any, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Pair the two arms on the question id. Arm a is the reference (SFT).

    Every difference is b minus a, matching the manifest's "C3 minus SFT".
    Each quantity uses the questions on which it is defined in both arms:
    reliability drops the questions excluded by the split-half rule, correction
    drops the questions where an arm has no wrong upstream alternative with
    replays. Those counts are reported, they are not silently absorbed.
    """
    common = sorted(set(scan_a) & set(scan_b), key=lambda q: str(q))

    rel_q = [q for q in common
             if scan_a[q]["rel"] is not None and scan_b[q]["rel"] is not None]
    rel = _paired_t([scan_a[q]["rel"] for q in rel_q], [scan_b[q]["rel"] for q in rel_q])

    infl = _paired_t([scan_a[q]["influence"] for q in common],
                     [scan_b[q]["influence"] for q in common])

    corr_q = [q for q in common
              if not np.isnan(scan_a[q]["correction"]) and not np.isnan(scan_b[q]["correction"])]
    corr = _paired_t([scan_a[q]["correction"] for q in corr_q],
                     [scan_b[q]["correction"] for q in corr_q])

    return {
        "n_questions": len(common),
        "rel": dict(rel, n_excluded=len(common) - len(rel_q)),
        "influence": infl,
        "correction": dict(
            corr,
            n_excluded=len(common) - len(corr_q),
            n_upstream_wrong_a=int(sum(scan_a[q]["n_wrong"] for q in corr_q)),
            n_upstream_wrong_b=int(sum(scan_b[q]["n_wrong"] for q in corr_q)),
            n_upstream_wrong_all_a=int(sum(scan_a[q]["n_wrong"] for q in common)),
            n_upstream_wrong_all_b=int(sum(scan_b[q]["n_wrong"] for q in common)),
        ),
    }


def e5_key_values(report: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """The E5 manifest keys with {value, n, note}, ready for `write_summary`.

    Arm a is written to the `.sft` keys and arm b to the `.c3` keys. A value
    that is nan, a p value that is not computable, or a key whose own n is zero
    is left out rather than written as null. The Wilcoxon p values the
    specification asks for have no manifest key of their own, so they ride in
    the note of the t-test key.
    """
    rel = report["rel"]
    infl = report["influence"]
    corr = report["correction"]
    out: Dict[str, Dict[str, Any]] = {}

    def put(key: str, value: Any, n: Optional[int], note: str = "") -> None:
        if value is None or (n is not None and n <= 0):
            return
        if isinstance(value, float) and not np.isfinite(value):
            return
        out[key] = {"value": value, "n": n, "note": note}

    def put_p(key: str, p: float, n: Optional[int], note: str = "") -> None:
        text = format_p(p)
        if text is None or (n is not None and n <= 0):
            return
        raw = "raw p = %.6g" % p
        out[key] = {"value": text, "n": n, "note": (note + "; " + raw) if note else raw}

    put("E5.n_questions", int(report["n_questions"]), int(report["n_questions"]),
        "question ids present in both arms")

    rel_note = "mean over the %d paired questions where both arms pass the split-half rule (%d excluded)" % (
        rel["n"], rel["n_excluded"])
    put("E5.rel.sft", rel["mean_a"], rel["n"], rel_note)
    put("E5.rel.c3", rel["mean_b"], rel["n"], rel_note)
    put_p("E5.rel.diff_p", rel["p"], rel["n"],
          "paired t, two-sided; Wilcoxon p = %.6g" % rel["wilcoxon_p"])

    infl_note = "answer-level plug-in MI, Miller-Madow corrected, nats, not clipped at zero"
    put("E5.influence.sft", infl["mean_a"], infl["n"], infl_note)
    put("E5.influence.c3", infl["mean_b"], infl["n"], infl_note)
    put("E5.influence.diff", infl["diff"], infl["n"], "C3 minus SFT, paired on the question")
    put("E5.influence.ci_lo", infl["ci_lo"], infl["n"], "95% t interval of the paired difference")
    put("E5.influence.ci_hi", infl["ci_hi"], infl["n"], "95% t interval of the paired difference")
    put_p("E5.influence.p", infl["p"], infl["n"],
          "paired t, two-sided; Wilcoxon p = %.6g" % infl["wilcoxon_p"])

    corr_note = "mean over the %d questions where both arms have a wrong upstream alternative with replays (%d excluded)" % (
        corr["n"], corr["n_excluded"])
    put("E5.correction.sft", corr["mean_a"], corr["n"], corr_note)
    put("E5.correction.c3", corr["mean_b"], corr["n"], corr_note)
    put("E5.correction.diff", corr["diff"], corr["n"], "C3 minus SFT, paired on the question")
    put("E5.correction.ci_lo", corr["ci_lo"], corr["n"], "95% t interval of the paired difference")
    put("E5.correction.ci_hi", corr["ci_hi"], corr["n"], "95% t interval of the paired difference")
    put_p("E5.correction.p", corr["p"], corr["n"],
          "paired t, two-sided; Wilcoxon p = %.6g" % corr["wilcoxon_p"])
    put("E5.correction.n_upstream_wrong.sft", corr["n_upstream_wrong_a"], corr["n"],
        "wrong upstream alternatives over the paired questions; %d over every shared question"
        % corr["n_upstream_wrong_all_a"])
    put("E5.correction.n_upstream_wrong.c3", corr["n_upstream_wrong_b"], corr["n"],
        "wrong upstream alternatives over the paired questions; %d over every shared question"
        % corr["n_upstream_wrong_all_b"])

    return out
