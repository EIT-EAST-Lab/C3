# c3/analysis/rebuild/bias_map.py
"""E3a: the ablation bias map, decision point by decision point.

One bucket is one decision point. One alternative in it is not a sampled
message but an injected null action (`meta.null_arm`, realised either as an
empty message or as a placeholder message, `meta.null_form`). Against that arm:

    barR_j         mean return of alternative j over its replays
    bias           mean_{j != j0} barR_j - barR_{j0}
    delta_q        max_{j != j0} barR_j - min_{j != j0} barR_j
    influence      answer-level plug-in MI over the real alternatives (nats)
    baseline_level barR_{j0}

Sign convention: bias is "with the message" minus "without it", so a positive
bias means removing the message costs return.

Stratification, with the thresholds fixed before any bias is looked at
(preregistration E3): split at the median influence, then split the high half
by whether the real alternatives differed in value at all (delta_q > 0).

Red-team checks carried over from `30_analysis/server_tierA/p1rb_analysis.py`:
R1 the baseline level is always reported, because an injection that collapses
the downstream agent would manufacture a large bias for a reason that has
nothing to do with credit; R2 the two realisations of the null action are
compared as a paired difference (`null_forms_paired`); R3 the bias-influence
correlation is deliberately NOT computed here, since it needs a partial that
the preregistration does not claim.

Module-level imports are limited to numpy, scipy and the standard library.
"""

from __future__ import annotations

import sys
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

from c3.analysis.rebuild.influence import (
    bucket_influence,
    format_p,
)

__all__ = [
    "ZERO_TOL",
    "CUTOFF_PCTS",
    "MIN_STRATUM_N",
    "STRATUM_NAMES",
    "decision_points",
    "bias_map_report",
    "null_forms_paired",
    "e3a_key_values",
]

#: A bias counts as "exactly zero" below this. Returns are means of a handful of
#: 0/1 replays, so a true zero is exact in binary floating point and this only
#: guards the subtraction.
ZERO_TOL = 1e-12

#: Robustness cutoffs for `E3a.cutoffs.max_mw_p`: keep the top X% by influence.
CUTOFF_PCTS = (60, 50, 40, 30)

#: Tie handling at a cutoff. "ge" keeps every point whose influence is at or
#: above the threshold; with "gt" the threshold itself is excluded. "ge" is the
#: default because it makes the 50% cutoff identical to the primary median
#: split, and because influence has heavy ties at zero (many decision points
#: produce one single downstream answer), where "gt" would empty the high half.
DEFAULT_HIGH_SIDE = "ge"

#: The three strata of the preregistered split, in the order they are reported.
STRATUM_NAMES = ("high_infl_diff", "high_infl_nodiff", "low_infl")

#: Below this many points a stratum is reported as degenerate. It is a REPORTING
#: threshold only: the split rule itself is preregistered and is not touched
#: here, and no key is dropped because of it. What it catches is the failure mode
#: seen on the first real E3a cell, where the influence estimator returned
#: exactly zero on most decision points, so the median split put almost every
#: point on the high side and left the low stratum with a handful.
MIN_STRATUM_N = 10


# -----------------------------------------------------------------------------
# small statistics helpers
# -----------------------------------------------------------------------------


def _t_p(differences: Sequence[float]) -> float:
    """Two-sided t p value of a difference vector against zero.

    A paired t on (a, b) is the one-sample t on a - b, so this covers both the
    bias-against-zero test and the null-form paired test.

    Convention for the degenerate case, which the spec left to this work
    package: if every difference is exactly zero the two samples are identical,
    so there is nothing to reject and p = 1.0 (scipy returns nan there, because
    the standard error is zero). A constant non-zero difference keeps scipy's
    own answer, p = 0.0. Fewer than two values gives nan.
    """
    d = np.asarray(list(differences), dtype=float)
    if d.size < 2:
        return float("nan")
    if np.all(np.abs(d) < ZERO_TOL):
        return 1.0
    return float(stats.ttest_1samp(d, 0.0).pvalue)


def _mw_p(a: Sequence[float], b: Sequence[float]) -> float:
    """Two-sided Mann-Whitney U p value, nan when it is not computable.

    scipy returns nan when both samples are constant and equal (the variance of
    the statistic is zero). That nan is passed through: "not computable" is not
    the same as "no difference", and the caller drops the key instead of
    inventing a number.
    """
    a = list(a)
    b = list(b)
    if not a or not b:
        return float("nan")
    try:
        return float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except ValueError:
        return float("nan")


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if len(values) else float("nan")


def _stratum_stats(biases: Sequence[float]) -> Dict[str, Any]:
    """Descriptive statistics of one stratum's bias vector."""
    arr = np.asarray(list(biases), dtype=float)
    n = int(arr.size)
    zero_n = int(np.sum(np.abs(arr) < ZERO_TOL)) if n else 0
    return {
        "n": n,
        "mean_bias": float(np.mean(arr)) if n else float("nan"),
        "median": float(np.median(arr)) if n else float("nan"),
        "iqr_lo": float(np.percentile(arr, 25)) if n else float("nan"),
        "iqr_hi": float(np.percentile(arr, 75)) if n else float("nan"),
        "zero_n": zero_n,
        "zero_share_pct": (100.0 * zero_n / n) if n else float("nan"),
        "mean_abs_bias": float(np.mean(np.abs(arr))) if n else float("nan"),
    }


# -----------------------------------------------------------------------------
# buckets -> decision points
# -----------------------------------------------------------------------------


def decision_points(
    buckets: Iterable[Mapping[str, Any]],
    extractor: Callable[[str], Optional[str]],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Scan one bucket file into decision points.

    A bucket is usable when `meta.null_arm` points at an existing alternative
    that has at least one return, and at least one other alternative also has
    returns. Anything else is counted in the returned reason table and left out:
    a decision point without a landed injection is not a measurement.

    Returns (points, skipped) where each point carries question_id, bias,
    delta_q, influence, baseline_level, n_real and null_form.
    """
    points: List[Dict[str, Any]] = []
    skipped = {"no_null_arm": 0, "null_arm_out_of_range": 0, "no_null_returns": 0, "no_real_returns": 0}

    for b in buckets:
        meta = b.get("meta", {}) or {}
        if not isinstance(meta, Mapping):
            meta = {}
        cands = b.get("candidates", []) or []
        raw = meta.get("null_arm", None)
        if raw is None:
            skipped["no_null_arm"] += 1
            continue
        try:
            j0 = int(raw)
        except (TypeError, ValueError):
            skipped["no_null_arm"] += 1
            continue
        if not (0 <= j0 < len(cands)):
            skipped["null_arm_out_of_range"] += 1
            continue

        r0 = cands[j0].get("returns", []) or []
        if not r0:
            skipped["no_null_returns"] += 1
            continue
        real = [float(np.mean(c.get("returns", [])))
                for k, c in enumerate(cands) if k != j0 and (c.get("returns", []) or [])]
        if not real:
            skipped["no_real_returns"] += 1
            continue

        q0 = float(np.mean(r0))
        points.append({
            "question_id": b.get("question_id"),
            "bias": float(np.mean(real)) - q0,
            "delta_q": float(np.max(real) - np.min(real)),
            "influence": float(bucket_influence(b, extractor)),
            "baseline_level": q0,
            "n_real": len(real),
            "null_form": meta.get("null_form"),
        })
    return points, skipped


# -----------------------------------------------------------------------------
# decision points -> the map
# -----------------------------------------------------------------------------


def _degenerate_strata(strata: Mapping[str, Any], *, stream=None) -> Dict[str, str]:
    """Which strata came out below :data:`MIN_STRATUM_N`, and what to say of them.

    One line per thin stratum on stderr, and the same sentence returned so it can
    ride into the note of every key read off that stratum. Nothing is dropped and
    no threshold moves: the split rule is preregistered, and whether a degenerate
    split means the measurement has to be rerun is the driver's call, taken on a
    summary that says so out loud instead of on a number that looks ordinary.
    """
    out = stream if stream is not None else sys.stderr
    messages: Dict[str, str] = {}
    for name in STRATUM_NAMES:
        n = int(strata[name]["n"])
        if n >= MIN_STRATUM_N:
            continue
        messages[name] = ("stratum %s has n=%d points; the median split is degenerate "
                          "on this data" % (name, n))
        print("[bias_map] %s" % messages[name], file=out)
    return messages


def bias_map_report(
    points: Sequence[Mapping[str, Any]],
    *,
    high_side: str = DEFAULT_HIGH_SIDE,
    stream=None,
) -> Dict[str, Any]:
    """Every quantity the E3a keys are read off, plus the red-team R1 level.

    The top stratum is "high influence and the alternatives differed in value";
    the rest is everything else, which is the denominator of the concentration
    factor. A concentration factor with a zero denominator is reported as
    infinity when the top stratum is non-zero and as nan when both vanish; it is
    not silently replaced by a finite number.

    `strata["degenerate"]` maps each stratum thinner than :data:`MIN_STRATUM_N`
    to the sentence `e3a_key_values` appends to the notes of the keys read off
    it; `stream` is where the same sentences are warned about (stderr by
    default).
    """
    if high_side not in ("ge", "gt"):
        raise ValueError("high_side must be 'ge' or 'gt', got %r" % (high_side,))

    n_points = len(points)
    bias = np.asarray([float(p["bias"]) for p in points], dtype=float)
    delta_q = np.asarray([float(p["delta_q"]) for p in points], dtype=float)
    infl = np.asarray([float(p["influence"]) for p in points], dtype=float)
    base = np.asarray([float(p["baseline_level"]) for p in points], dtype=float)
    abs_bias = np.abs(bias)

    def _high_mask(threshold: float) -> np.ndarray:
        return infl >= threshold if high_side == "ge" else infl > threshold

    def _split_p(mask: np.ndarray) -> float:
        diff = abs_bias[mask & (delta_q > 0.0)]
        nodiff = abs_bias[mask & ~(delta_q > 0.0)]
        return _mw_p(diff.tolist(), nodiff.tolist())

    median_infl = float(np.median(infl)) if n_points else float("nan")
    high = _high_mask(median_infl) if n_points else np.zeros(0, dtype=bool)
    low = ~high if n_points else np.zeros(0, dtype=bool)
    has_diff = delta_q > 0.0

    top_mask = high & has_diff
    nodiff_mask = high & ~has_diff
    rest_mask = ~top_mask

    strata: Dict[str, Any] = {
        "high_infl_diff": _stratum_stats(bias[top_mask].tolist()),
        "high_infl_nodiff": _stratum_stats(bias[nodiff_mask].tolist()),
        "low_infl": _stratum_stats(bias[low].tolist()),
        "mw_p": _mw_p(abs_bias[top_mask].tolist(), abs_bias[nodiff_mask].tolist()),
    }
    strata["degenerate"] = _degenerate_strata(strata, stream=stream)

    top_abs = abs_bias[top_mask]
    rest_abs = abs_bias[rest_mask]
    total_abs = float(np.sum(abs_bias)) if n_points else 0.0
    top_mean = float(np.mean(top_abs)) if top_abs.size else float("nan")
    rest_mean = float(np.mean(rest_abs)) if rest_abs.size else float("nan")
    if np.isnan(top_mean) or np.isnan(rest_mean):
        concentration = float("nan")
    elif rest_mean > 0.0:
        concentration = top_mean / rest_mean
    elif top_mean > 0.0:
        concentration = float("inf")
    else:
        concentration = float("nan")

    per_cutoff: Dict[int, float] = {}
    for pct in CUTOFF_PCTS:
        if not n_points:
            per_cutoff[pct] = float("nan")
            continue
        thr = float(np.percentile(infl, 100 - pct))
        per_cutoff[pct] = _split_p(_high_mask(thr))
    finite = [v for v in per_cutoff.values() if not np.isnan(v)]
    max_mw_p = max(finite) if finite else float("nan")

    return {
        "n_points": n_points,
        "high_side": high_side,
        "bias": {
            "mean": float(np.mean(bias)) if n_points else float("nan"),
            "median": float(np.median(bias)) if n_points else float("nan"),
            "p": _t_p(bias.tolist()),
            "total_abs": total_abs,
        },
        "influence": {
            "mean_nats": float(np.mean(infl)) if n_points else float("nan"),
            "median_nats": median_infl,
            "negative_n": int(np.sum(infl < 0.0)) if n_points else 0,
        },
        "baseline_level": {
            "mean": float(np.mean(base)) if n_points else float("nan"),
            "median": float(np.median(base)) if n_points else float("nan"),
            "zero_n": int(np.sum(base <= 0.0)) if n_points else 0,
        },
        "strata": strata,
        "top_stratum": {
            "n": int(top_abs.size),
            "mean_abs_bias": top_mean,
            "share_of_points_pct": (100.0 * top_abs.size / n_points) if n_points else float("nan"),
            "share_of_bias_pct": (100.0 * float(np.sum(top_abs)) / total_abs) if total_abs > 0.0 else float("nan"),
        },
        "rest": {"n": int(rest_abs.size), "mean_abs_bias": rest_mean},
        "concentration_factor": concentration,
        "cutoffs": {"per_cutoff": per_cutoff, "max_mw_p": max_mw_p},
    }


def null_forms_paired(
    points_a: Sequence[Mapping[str, Any]],
    points_b: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Red-team R2: the two realisations of the null action, paired by question.

    Only question ids present on both sides take part. Returns the paired mean
    difference (b minus a), its two-sided t p value and the number of pairs.
    """
    a_by_q = {p["question_id"]: p for p in points_a if p.get("question_id") is not None}
    b_by_q = {p["question_id"]: p for p in points_b if p.get("question_id") is not None}
    common = sorted(set(a_by_q) & set(b_by_q), key=lambda q: str(q))
    diffs = [float(b_by_q[q]["bias"]) - float(a_by_q[q]["bias"]) for q in common]
    return {
        "n_pairs": len(common),
        "mean_diff": _mean(diffs),
        "p": _t_p(diffs),
    }


# -----------------------------------------------------------------------------
# the map -> manifest keys
# -----------------------------------------------------------------------------


def _strata_of(key: str) -> Tuple[str, ...]:
    """Which strata a key is read off, for the degeneracy note.

    A prefix rule rather than a table, so a key added later inherits the right
    warning instead of silently losing it. `E3a.top_stratum.*` and
    `E3a.concentration_factor` are readings of `high_infl_diff` under another
    name; `E3a.n_points` stands for the whole point set and takes every warning.
    """
    if key == "E3a.n_points":
        return STRATUM_NAMES
    if key == "E3a.strata.mw_p":
        return ("high_infl_diff", "high_infl_nodiff")
    if key == "E3a.concentration_factor" or key.startswith("E3a.top_stratum."):
        return ("high_infl_diff",)
    for name in STRATUM_NAMES:
        if key.startswith("E3a.strata.%s." % name):
            return (name,)
    return ()


def e3a_key_values(
    report: Mapping[str, Any],
    *,
    null_forms: Optional[Mapping[str, Any]] = None,
    extra_strata_keys: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """The E3a manifest keys with {value, n, note}, ready for `write_summary`.

    Only the keys the specification enumerates are produced. Nine further E3a
    keys exist in the manifest (per-stratum mean_bias / median / iqr_lo / iqr_hi
    and high_infl_nodiff.zero_n); they are computed in `bias_map_report` and
    `extra_strata_keys=True` emits them, but the default stays with the
    enumerated set, since which keys an aggregation fills is the driver's call.

    A key whose value is nan, whose p value is not computable, or whose own n is
    zero is left out entirely rather than written as null: a statistic with no
    unit behind it is not a measurement.

    Every key read off a stratum that `bias_map_report` found thinner than
    :data:`MIN_STRATUM_N` carries that finding at the end of its note, so a
    degenerate split is visible in the summary and not only on the stderr of the
    run that produced it.
    """
    strata = report["strata"]
    top = report["top_stratum"]
    rest = report["rest"]
    n_points = int(report["n_points"])
    degenerate = dict(strata.get("degenerate") or {})
    out: Dict[str, Dict[str, Any]] = {}

    def full_note(key: str, note: str) -> str:
        """The note plus the degeneracy sentence of every stratum the key reads."""
        parts = [note] if note else []
        parts += [degenerate[name] for name in _strata_of(key) if name in degenerate]
        return "; ".join(parts)

    def put(key: str, value: Any, n: Optional[int], note: str = "") -> None:
        if value is None or (n is not None and n <= 0):
            return
        if isinstance(value, float) and not np.isfinite(value):
            return
        out[key] = {"value": value, "n": n, "note": full_note(key, note)}

    def put_p(key: str, p: float, n: Optional[int], note: str = "") -> None:
        text = format_p(p)
        if text is None or (n is not None and n <= 0):
            return
        raw = "raw p = %.6g" % p
        head = (note + "; " + raw) if note else raw
        out[key] = {"value": text, "n": n, "note": full_note(key, head)}

    put("E3a.n_points", n_points, n_points,
        "decision points with a landed null arm and at least one real alternative")
    # Red-team R1 has no manifest key of its own, so the level of the injected
    # arm rides in this note: a bias read without it cannot be judged.
    put("E3a.bias.mean", report["bias"]["mean"], n_points,
        "mean over decision points of (mean real return) minus (null arm return); "
        "R1 null arm return: mean %.4f, median %.4f, zero at %d of %d points"
        % (report["baseline_level"]["mean"], report["baseline_level"]["median"],
           report["baseline_level"]["zero_n"], n_points))
    put_p("E3a.bias.p", report["bias"]["p"], n_points, "one-sample t against zero, two-sided")

    put("E3a.strata.high_infl_diff.n", strata["high_infl_diff"]["n"], n_points)
    put("E3a.strata.high_infl_nodiff.n", strata["high_infl_nodiff"]["n"], n_points)
    put("E3a.strata.high_infl_nodiff.zero_share_pct",
        strata["high_infl_nodiff"]["zero_share_pct"], strata["high_infl_nodiff"]["n"],
        "share with |bias| < %g" % ZERO_TOL)
    put("E3a.strata.high_infl_diff.zero_n", strata["high_infl_diff"]["zero_n"],
        strata["high_infl_diff"]["n"], "points with |bias| < %g" % ZERO_TOL)
    put_p("E3a.strata.mw_p", strata["mw_p"],
          strata["high_infl_diff"]["n"] + strata["high_infl_nodiff"]["n"],
          "Mann-Whitney U on |bias|, the two high-influence strata, two-sided")
    put("E3a.strata.low_infl.mean_abs_bias", strata["low_infl"]["mean_abs_bias"],
        strata["low_infl"]["n"])

    put("E3a.top_stratum.share_of_points_pct", top["share_of_points_pct"], n_points)
    put("E3a.top_stratum.share_of_bias_pct", top["share_of_bias_pct"], n_points,
        "share of the summed |bias| over all decision points")
    put("E3a.top_stratum.mean_abs_bias", top["mean_abs_bias"], top["n"])
    put("E3a.rest.n", rest["n"], n_points)
    put("E3a.rest.mean_abs_bias", rest["mean_abs_bias"], rest["n"])
    put("E3a.concentration_factor", report["concentration_factor"], n_points,
        "top stratum mean |bias| over the rest")
    put("E3a.influence.mean_nats", report["influence"]["mean_nats"], n_points,
        "Miller-Madow corrected, not clipped at zero; %d of %d points are negative"
        % (report["influence"]["negative_n"], n_points))
    put_p("E3a.cutoffs.max_mw_p", report["cutoffs"]["max_mw_p"], n_points,
          "largest of the top %s%% cutoffs" % "/".join(str(c) for c in CUTOFF_PCTS))

    if null_forms is not None and null_forms.get("n_pairs", 0) > 0:
        put_p("E3a.null_forms.paired_p", null_forms["p"], int(null_forms["n_pairs"]),
              "paired t on the per-question bias difference, two-sided; mean difference %.6g"
              % null_forms["mean_diff"])

    if extra_strata_keys:
        for name in ("high_infl_diff", "high_infl_nodiff"):
            st = strata[name]
            for field in ("mean_bias", "median", "iqr_lo", "iqr_hi"):
                put("E3a.strata.%s.%s" % (name, field), st[field], st["n"])
        put("E3a.strata.high_infl_nodiff.zero_n", strata["high_infl_nodiff"]["zero_n"],
            strata["high_infl_nodiff"]["n"], "points with |bias| < %g" % ZERO_TOL)

    return out
