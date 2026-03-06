# c3/analysis/metrics.py
"""
Pure, reusable metrics for Credit (Var/Fidelity) and Influence (conditional MI).

Bucket schema expectations (see `c3.analysis.buckets`):
- bucket["candidates"][j]["returns"] : list[float]
- (influence) bucket["candidates"][j]["next_actions"] : list[str]
- bucket["meta"] may contain segmentation hints:
    - real_j: int (index of the "observed/real" action among credit candidates)
    - credit_n: int (number of candidates that participate in credit/var/influence)
    - v_extra_start: int (start index of V-extra samples)
    - v_extra_n: int (length of V-extra segment)

Conventions:
- Credit operates on per-candidate mean return \bar R_j (proxy for Q(h,a_j)).
- Influence uses natural log (nats). Convert to bits via /log(2) if needed.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .buckets import aggregate_candidate_returns, validate_bucket

Bucket = Mapping[str, Any]

__all__ = [
    "credit_var",
    "credit_var_report",
    "credit_fidelity",
    "build_fidelity_pairs",
    "canonicalize_for_influence",
    "hash_symbol",
    "influence_mi",
    "influence_report",
]


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------


def _as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    return [x]


def _safe_float(x: Any) -> float:
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"Non-finite float: {x}")
    return v


def _mean_std_stderr(x: np.ndarray) -> Tuple[float, float, float]:
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1)) if x.size >= 2 else 0.0
    stderr = std / math.sqrt(x.size) if x.size >= 2 else 0.0
    return mean, std, stderr


def _percentile_ci(samples: np.ndarray, alpha: float) -> Tuple[float, float]:
    lo = float(np.percentile(samples, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(samples, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    iters: int,
    ci: float,
    seed: int,
) -> Optional[Tuple[float, float]]:
    if iters <= 0 or values.size == 0:
        return None
    rng = np.random.default_rng(seed)
    n = values.size
    means = np.empty((iters,), dtype=np.float64)
    for t in range(iters):
        idx = rng.integers(0, n, size=n, endpoint=False)
        means[t] = float(np.mean(values[idx]))
    alpha = 1.0 - ci
    return _percentile_ci(means, alpha)


def _meta_of(bucket: Bucket) -> Mapping[str, Any]:
    m = bucket.get("meta", {})
    return m if isinstance(m, Mapping) else {}


def _clip_int(x: Any, lo: int, hi: int, default: int) -> int:
    try:
        v = int(x)
    except Exception:
        return default
    return max(lo, min(hi, v))


def _credit_action_indices(meta: Mapping[str, Any], n_actions: int) -> np.ndarray:
    """
    Returns indices of actions that participate in credit/var/influence.

    Default: all actions are credit actions.
    """
    if n_actions <= 0:
        return np.zeros((0,), dtype=np.int64)

    credit_n = meta.get("credit_n", n_actions)
    credit_n = _clip_int(credit_n, 0, n_actions, n_actions)
    return np.arange(credit_n, dtype=np.int64)


def _v_extra_action_indices(meta: Mapping[str, Any], n_actions: int) -> np.ndarray:
    """
    Returns indices of V-extra actions (used only for estimating V(h) in fidelity).

    Default: empty.
    """
    if n_actions <= 0:
        return np.zeros((0,), dtype=np.int64)

    start = meta.get("v_extra_start", n_actions)
    n = meta.get("v_extra_n", 0)

    start = _clip_int(start, 0, n_actions, n_actions)
    n = _clip_int(n, 0, n_actions, 0)

    end = min(n_actions, start + n)
    if end <= start:
        return np.zeros((0,), dtype=np.int64)
    return np.arange(start, end, dtype=np.int64)


def _as_weights(counts: Optional[np.ndarray], n: int) -> np.ndarray:
    """
    Convert optional counts -> nonnegative finite weights of shape (n,).

    Robustness rules:
      - None / shape mismatch / non-finite -> uniform weights
      - negative -> clipped to 0
      - all-zero -> uniform weights
    """
    if counts is None:
        return np.ones((n,), dtype=np.float64)

    w = np.asarray(counts, dtype=np.float64)
    if w.shape != (n,):
        w = np.ones((n,), dtype=np.float64)

    if not np.all(np.isfinite(w)):
        w = np.ones((n,), dtype=np.float64)

    # Robustness: clip negatives and avoid all-zero weights.
    w = np.maximum(w, 0.0)
    if float(np.sum(w)) <= 0.0:
        w = np.ones((n,), dtype=np.float64)

    return w


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    denom = float(np.sum(weights))
    # If all weights are zero (or invalid), fall back to unweighted mean to keep metrics finite.
    if denom <= 0.0 or not np.isfinite(denom):
        return float(np.mean(values)) if values.size else 0.0
    num = float(np.sum(values * weights))
    if not np.isfinite(num):
        return float(np.mean(values)) if values.size else 0.0
    return num / denom


# -----------------------------------------------------------------------------
# Credit: baselines + A_j
# -----------------------------------------------------------------------------


def _baseline_full_mean(values: np.ndarray, *, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """
    b_j = E[barR] (weighted if weights provided).
    """
    n = int(values.size)
    if n == 0:
        return values.astype(np.float64, copy=True)
    w = _as_weights(weights, n)
    m = _weighted_mean(values, w)
    return np.full((n,), m, dtype=np.float64)


def _baseline_loo(values: np.ndarray, *, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Leave-one-out baseline. Weighted version matches the paper's "restart & replay count" semantics:

      b_j = (sum_k w_k v_k - w_j v_j) / (sum_k w_k - w_j)

    Fallbacks:
    - If n<=1, baseline=value (=> advantage 0).
    - If denom<=0 for some j, baseline=value for that j (=> advantage 0).
    """
    n = int(values.size)
    if n <= 1:
        return values.astype(np.float64, copy=True)

    w = _as_weights(weights, n)
    sw = float(np.sum(w))
    sv = float(np.sum(w * values))

    out = np.empty((n,), dtype=np.float64)
    for j in range(n):
        denom = sw - float(w[j])
        if denom <= 0.0 or not np.isfinite(denom):
            out[j] = float(values[j])
        else:
            out[j] = (sv - float(w[j]) * float(values[j])) / denom
    return out


def _credit_from_barR(
    barR: np.ndarray,
    mode: str,
    *,
    counts: Optional[np.ndarray] = None,
    v_critic: Any = None,
    bucket: Optional[Bucket] = None,
) -> np.ndarray:
    """
    Returns per-candidate scalar credit A_j.

    Modes (paper-facing, stable):
      - c3_loo / magrpo_rloo:  A_j = barR_j - LOO(barR)_j
      - c3_full_mean / magrpo_mean: A_j = barR_j - mean(barR)
      - mappo_v: A_j = barR_j - V(h)  (V(h) from v_critic callable or scalar)
    """
    mode = str(mode)
    barR = np.asarray(barR, dtype=np.float64)
    n = int(barR.size)

    if n == 0:
        return barR

    if mode in ("c3_loo", "magrpo_rloo"):
        base = _baseline_loo(barR, weights=counts)
        return barR - base

    if mode in ("c3_full_mean", "magrpo_mean"):
        base = _baseline_full_mean(barR, weights=counts)
        return barR - base

    if mode == "mappo_v":
        if v_critic is None:
            raise ValueError("mode='mappo_v' requires v_critic")
        if callable(v_critic):
            if bucket is None:
                raise ValueError("mode='mappo_v' with callable v_critic requires bucket")
            V = v_critic(bucket)
        else:
            V = v_critic
        V = _safe_float(V)
        return barR - V

    raise ValueError(f"Unknown credit mode: {mode}")


# -----------------------------------------------------------------------------
# Credit: Var
# -----------------------------------------------------------------------------


def credit_var(bucket: Bucket, mode: str, *, v_critic: Any = None) -> float:
    """
    Var_j(A_j) within a single context bucket, based on per-candidate mean return.

    IMPORTANT: computed only over credit candidates (meta.credit_n segment).
    Returns population variance (ddof=0) for stability across small N.
    """
    validate_bucket(bucket)
    meta = _meta_of(bucket)

    barR_all, counts_all = aggregate_candidate_returns(bucket)
    n_actions = int(barR_all.size)
    if n_actions <= 1:
        return 0.0

    credit_idx = _credit_action_indices(meta, n_actions)
    if credit_idx.size <= 1:
        return 0.0

    barR = barR_all[credit_idx]
    counts = counts_all[credit_idx] if counts_all is not None else None

    A = _credit_from_barR(barR, mode, counts=counts, v_critic=v_critic, bucket=bucket)
    if A.size <= 1:
        return 0.0
    return float(np.var(A, ddof=0))


def credit_var_report(
    buckets_iter: Iterable[Bucket],
    mode: str,
    *,
    v_critic: Any = None,
    bootstrap_iters: int = 0,
    ci: float = 0.95,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Aggregate Var across buckets: mean/std/stderr and optional bootstrap CI over buckets.
    """
    vals: List[float] = []
    for b in buckets_iter:
        vals.append(credit_var(b, mode, v_critic=v_critic))

    arr = np.asarray(vals, dtype=np.float64)
    mean, std, stderr = _mean_std_stderr(arr)
    out: Dict[str, Any] = {
        "metric": "credit_var",
        "mode": str(mode),
        "n_buckets": int(arr.size),
        "mean": mean,
        "std": std,
        "stderr": stderr,
    }
    ci_pair = _bootstrap_mean_ci(arr, iters=bootstrap_iters, ci=ci, seed=seed)
    if ci_pair is not None:
        out["bootstrap_ci"] = {
            "ci": float(ci),
            "low": ci_pair[0],
            "high": ci_pair[1],
            "iters": int(bootstrap_iters),
            "seed": int(seed),
        }
    return out


# -----------------------------------------------------------------------------
# Credit: Fidelity (Spearman)
# -----------------------------------------------------------------------------


def _rankdata_average_ties(x: np.ndarray) -> np.ndarray:
    """
    Average-rank for ties, 1..n (like scipy.stats.rankdata(method='average')).
    """
    n = int(x.size)
    if n == 0:
        return x

    order = np.argsort(x, kind="mergesort")
    ranks = np.empty((n,), dtype=np.float64)

    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def _pearsonr(a: np.ndarray, b: np.ndarray) -> float:
    # Degenerate cases: not enough points / mismatch -> define correlation as 0
    # to keep downstream aggregation and plotting finite.
    if a.size != b.size or a.size < 2:
        return 0.0

    a0 = a - np.mean(a)
    b0 = b - np.mean(b)

    da = float(np.dot(a0, a0))
    db = float(np.dot(b0, b0))

    # If either side has zero variance (or numerical issues), define correlation as 0.
    if (not np.isfinite(da)) or (not np.isfinite(db)) or da <= 0.0 or db <= 0.0:
        return 0.0

    r = float(np.dot(a0, b0) / math.sqrt(da * db))

    # Numerical safety: clamp and avoid NaN/inf.
    if not np.isfinite(r):
        return 0.0
    return max(-1.0, min(1.0, r))


def credit_fidelity(pairs: Sequence[Tuple[float, float]]) -> float:
    """
    Spearman(A_u, Δ*_u) over pairs.

    Robust definition:
      - If undefined (n < 2 or constant), return 0.0 instead of NaN.
    """
    if not pairs or len(pairs) < 2:
        return 0.0

    A = np.asarray([_safe_float(p[0]) for p in pairs], dtype=np.float64)
    D = np.asarray([_safe_float(p[1]) for p in pairs], dtype=np.float64)

    # If less than 2 points (should be covered above), define fidelity as 0.
    if A.size < 2 or D.size < 2:
        return 0.0

    rA = _rankdata_average_ties(A)
    rD = _rankdata_average_ties(D)
    return _pearsonr(rA, rD)


def build_fidelity_pairs(
    buckets_iter: Iterable[Bucket],
    mode: str,
    *,
    v_critic: Any = None,
    variant: str = "both",
    real_j_default: int = 0,
    estimate_v_by_extra_samples: bool = False,
    strict_real: bool = True,
) -> Dict[str, List[Tuple[float, float]]]:
    """
    Build (A, Δ*) pairs from buckets.

    Δ*_j := barR_j - V_hat(h)

    V_hat(h) estimation:
      - If estimate_v_by_extra_samples=True and meta has V-extra segment:
            V_hat = mean(barR over v_extra_idx)   (weighted by counts)
        Else:
            V_hat = mean(barR over credit_idx)    (weighted by counts)

    variant:
      - "real_only": use only j = meta.real_j per bucket (paper-closest)
      - "all_candidates": include all credit candidates as separate points
      - "both": compute both lists (default)

    Returns:
      {"real_only": [...], "all_candidates": [...]}
      (keys omitted if not requested).
    """
    variant = str(variant)
    if variant not in ("real_only", "all_candidates", "both"):
        raise ValueError(f"Unknown fidelity variant: {variant}")

    want_real = variant in ("real_only", "both")
    want_all = variant in ("all_candidates", "both")

    out: Dict[str, List[Tuple[float, float]]] = {}
    if want_real:
        out["real_only"] = []
    if want_all:
        out["all_candidates"] = []

    for b in buckets_iter:
        validate_bucket(b)
        meta = _meta_of(b)

        barR_all, counts_all = aggregate_candidate_returns(b)
        n_actions = int(barR_all.size)
        if n_actions == 0:
            continue

        credit_idx = _credit_action_indices(meta, n_actions)
        if credit_idx.size == 0:
            continue

        # V-hat: prefer V-extra segment if requested and available.
        if estimate_v_by_extra_samples:
            v_extra_idx = _v_extra_action_indices(meta, n_actions)
        else:
            v_extra_idx = np.zeros((0,), dtype=np.int64)

        if v_extra_idx.size > 0:
            vals = barR_all[v_extra_idx]
            wts = counts_all[v_extra_idx] if counts_all is not None else None
            V_hat = _weighted_mean(vals, _as_weights(wts, int(vals.size)))
        else:
            vals = barR_all[credit_idx]
            wts = counts_all[credit_idx] if counts_all is not None else None
            V_hat = _weighted_mean(vals, _as_weights(wts, int(vals.size)))

        # Credits and deltas: ONLY for credit candidates.
        barR = barR_all[credit_idx]
        counts = counts_all[credit_idx] if counts_all is not None else None
        A = _credit_from_barR(barR, mode, counts=counts, v_critic=v_critic, bucket=b)
        delta = barR - float(V_hat)

        # Index mapping: original action index -> position in the sliced arrays.
        # (credit_idx is strictly increasing arange in our default semantics, but keep generic.)
        pos_of: Dict[int, int] = {int(j): int(i) for i, j in enumerate(credit_idx.tolist())}

        if want_real:
            rj = meta.get("real_j", real_j_default)
            try:
                real_j = int(rj)
            except Exception:
                real_j = int(real_j_default)

            if real_j not in pos_of:
                if strict_real:
                    raise ValueError(
                        f"Bucket real_j={real_j} is not in credit candidates. "
                        f"credit_idx=[{credit_idx[0]}..{credit_idx[-1]}], n_actions={n_actions}, meta={dict(meta)}"
                    )
                # fallback: clamp into credit range
                real_j = int(credit_idx[min(max(real_j, 0), int(credit_idx.size - 1))])

            p = pos_of[real_j]
            out["real_only"].append((float(A[p]), float(delta[p])))

        if want_all:
            for p in range(int(A.size)):
                out["all_candidates"].append((float(A[p]), float(delta[p])))

    return out


# -----------------------------------------------------------------------------
# Influence: conditional mutual information I(J;Y|h)
# -----------------------------------------------------------------------------

# Optional: reuse your existing sanitizer if present.
try:
    from c3.text_sanitize import sanitize_math_solution_text as _sanitize_math_solution_text  # type: ignore
except Exception:  # pragma: no cover
    _sanitize_math_solution_text = None  # type: ignore


_FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]+)?\n|\n```", flags=re.MULTILINE)
_WS_IN_LINE_RE = re.compile(r"[ \t]+")


def canonicalize_for_influence(text: str) -> str:
    """
    Low-regret canonicalization:
    - optional math-solution sanitizer (if available)
    - strip markdown fences, normalize newlines
    - collapse runs of spaces/tabs inside lines
    - collapse multiple blank lines
    """
    if text is None:
        return ""

    s = str(text)

    if _sanitize_math_solution_text is not None:
        try:
            s = _sanitize_math_solution_text(s)
        except Exception:
            pass

    s = _FENCE_RE.sub("\n", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")

    lines: List[str] = []
    for line in s.split("\n"):
        line = _WS_IN_LINE_RE.sub(" ", line.rstrip())
        lines.append(line)

    out_lines: List[str] = []
    blank = 0
    for line in lines:
        if line == "":
            blank += 1
            if blank <= 1:
                out_lines.append("")
        else:
            blank = 0
            out_lines.append(line)

    return "\n".join(out_lines).strip()


def hash_symbol(text: str) -> str:
    """
    Stable symbol hash (16 hex chars) for discrete MI.
    """
    s = canonicalize_for_influence(text)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def _extract_next_actions(bucket: Bucket, *, indices: np.ndarray) -> List[List[str]]:
    """
    Returns next_actions grouped by candidate position in `indices`.
    """
    validate_bucket(bucket)
    cands = bucket["candidates"]
    out: List[List[str]] = []

    for j in indices.tolist():
        ys = cands[j].get("next_actions", [])
        ys = ys if isinstance(ys, list) else _as_list(ys)
        out.append([str(y) for y in ys])

    return out


def _build_vocab_from_counts(counts: Counter, top_k: int) -> List[str]:
    if top_k <= 0:
        return []
    return [sym for sym, _ in counts.most_common(top_k)]


def influence_mi(
    bucket: Bucket,
    *,
    top_k: int = 64,
    alpha: float = 1.0,
    canonicalize_fn: Callable[[str], str] = canonicalize_for_influence,
    vocab: Optional[Sequence[str]] = None,
) -> float:
    """
    Per-bucket I(J;Y|h), with h fixed per bucket.

    IMPORTANT: computed only over credit candidates (meta.credit_n).
    - J: candidate index within the *credit set* (renumbered 0..K-1)
    - Y: hashed/canonicalized next action symbol (top-K + OTHER)
    """
    validate_bucket(bucket)
    meta = _meta_of(bucket)

    barR_all, _counts_all = aggregate_candidate_returns(bucket)
    n_actions = int(barR_all.size)
    if n_actions <= 1:
        return 0.0

    credit_idx = _credit_action_indices(meta, n_actions)
    if credit_idx.size <= 1:
        return 0.0

    ys_by_j = _extract_next_actions(bucket, indices=credit_idx)
    n_j = len(ys_by_j)
    if n_j <= 1:
        return 0.0

    sym_by_j: List[List[str]] = []
    total_counts: Counter = Counter()
    n_samples_by_j = np.zeros((n_j,), dtype=np.float64)

    for j, ys in enumerate(ys_by_j):
        syms: List[str] = []
        for y in ys:
            s = canonicalize_fn(y)
            sy = hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]
            syms.append(sy)
        sym_by_j.append(syms)
        total_counts.update(syms)
        n_samples_by_j[j] = float(len(syms))

    total_samples = float(np.sum(n_samples_by_j))
    if total_samples <= 0.0:
        return 0.0

    vocab_list = list(vocab) if vocab is not None else _build_vocab_from_counts(total_counts, top_k=top_k)
    vocab_set = set(vocab_list)

    OTHER = "OTHER"
    y_bins = vocab_list + [OTHER]
    y_index = {y: i for i, y in enumerate(y_bins)}
    V = int(len(y_bins))

    # p(j) from empirical sample mass
    p_j = n_samples_by_j / total_samples

    counts_y_given_j = np.zeros((n_j, V), dtype=np.float64)
    for j in range(n_j):
        for sy in sym_by_j[j]:
            yb = sy if sy in vocab_set else OTHER
            counts_y_given_j[j, y_index[yb]] += 1.0

    denom = n_samples_by_j[:, None] + float(alpha) * float(V)
    p_y_given_j = (counts_y_given_j + float(alpha)) / np.maximum(denom, 1e-12)

    p_y = np.sum(p_j[:, None] * p_y_given_j, axis=0)
    p_y = np.maximum(p_y, 1e-300)

    ratio = np.maximum(p_y_given_j, 1e-300) / p_y[None, :]
    mi = float(np.sum(p_j[:, None] * p_y_given_j * np.log(ratio)))

    return max(0.0, mi)


def influence_report(
    buckets_iter: Iterable[Bucket],
    *,
    top_k: int = 64,
    alpha: float = 1.0,
    canonicalize_fn: Callable[[str], str] = canonicalize_for_influence,
    bootstrap_iters: int = 0,
    ci: float = 0.95,
    seed: int = 0,
    use_global_vocab: bool = True,
) -> Dict[str, Any]:
    """
    Aggregate Influence across buckets (mean/std/stderr + optional bootstrap CI).

    If use_global_vocab:
      - first pass collects symbol frequencies across all buckets (credit candidates only)
      - second pass computes MI using the same top-K vocab (more comparable across methods)
    """
    buckets = list(buckets_iter)

    vocab: Optional[List[str]] = None
    if use_global_vocab:
        counts: Counter = Counter()
        for b in buckets:
            validate_bucket(b)
            meta = _meta_of(b)
            barR_all, _ = aggregate_candidate_returns(b)
            n_actions = int(barR_all.size)
            if n_actions <= 0:
                continue
            credit_idx = _credit_action_indices(meta, n_actions)
            if credit_idx.size <= 0:
                continue
            ys_by_j = _extract_next_actions(b, indices=credit_idx)
            for ys in ys_by_j:
                for y in ys:
                    s = canonicalize_fn(str(y))
                    sym = hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]
                    counts[sym] += 1
        vocab = _build_vocab_from_counts(counts, top_k=top_k)

    vals: List[float] = []
    for b in buckets:
        vals.append(
            influence_mi(
                b,
                top_k=top_k,
                alpha=alpha,
                canonicalize_fn=canonicalize_fn,
                vocab=vocab,
            )
        )

    arr = np.asarray(vals, dtype=np.float64)
    mean, std, stderr = _mean_std_stderr(arr)

    out: Dict[str, Any] = {
        "metric": "influence_mi",
        "n_buckets": int(arr.size),
        "mean": mean,
        "std": std,
        "stderr": stderr,
        "top_k": int(top_k),
        "alpha": float(alpha),
        "use_global_vocab": bool(use_global_vocab),
        "vocab_size": int(len(vocab) if vocab is not None else 0),
        "units": "nats",
    }

    ci_pair = _bootstrap_mean_ci(arr, iters=bootstrap_iters, ci=ci, seed=seed)
    if ci_pair is not None:
        out["bootstrap_ci"] = {
            "ci": float(ci),
            "low": ci_pair[0],
            "high": ci_pair[1],
            "iters": int(bootstrap_iters),
            "seed": int(seed),
        }

    if vocab is not None:
        out["vocab"] = vocab  # reproducibility (top-K only; OTHER is implicit)

    return out
