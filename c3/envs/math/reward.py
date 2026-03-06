# -*- coding: utf-8 -*-
"""
MathEnv reward function (C3-compatible).

Reward modes (meta["task_env_cfg"]["reward_mode"]):
  - strict: score a single answer (usually meta["answer_text"] / prediction), reward ∈ {0,1}
  - vote_binary: strict-majority vote across answer roles, reward ∈ {0,1}
  - avg_per_answer: average correctness across answer roles, reward ∈ [0,1]

Backends (meta["task_env_cfg"]["math_backend"]):
  - simple: parsing + exact compare (default)
  - marft: optional backend; failures fall back to simple (best-effort)

API:
  score_math(*, prediction, label, meta) -> (reward, info)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .parsing import normalize_math_answer, parse_math_answer
from c3.text_sanitize import sanitize_math_solution_text


# -----------------------------------------------------------------------------
# Config helpers
# -----------------------------------------------------------------------------


def _env_cfg(meta: Optional[dict]) -> Dict[str, Any]:
    if not isinstance(meta, dict):
        return {}
    cfg = meta.get("task_env_cfg", {})
    return cfg if isinstance(cfg, dict) else {}


def _canon_reward_mode(x: Any) -> str:
    s = str(x or "").strip().lower()
    if s in {"vote", "vote_binary", "majority", "majority_vote"}:
        return "vote_binary"
    if s in {"avg", "average", "avg_per_answer", "mean"}:
        return "avg_per_answer"
    return "strict"


def _canon_backend(x: Any) -> str:
    return "marft" if str(x or "").strip().lower() == "marft" else "simple"


def _s(text: Any) -> str:
    return "" if text is None else str(text)


# -----------------------------------------------------------------------------
# Scoring backends
# -----------------------------------------------------------------------------


def _compare_simple(pred_text: str, label_text: str) -> Tuple[float, Dict[str, Any]]:
    pred_raw, pred_method = parse_math_answer(pred_text or "")
    gold_raw, gold_method = parse_math_answer(label_text or "")

    info: Dict[str, Any] = {
        "env": "MathEnv",
        "backend": "simple",
        "pred_method": pred_method,
        "gold_method": gold_method,
    }

    if pred_raw is None or gold_raw is None:
        info.update(
            {
                "pred_raw": pred_raw,
                "gold_raw": gold_raw,
                "match": False,
                "reason": "missing_pred_or_label",
            }
        )
        return 0.0, info

    pred_norm, pred_frac = normalize_math_answer(pred_raw)
    gold_norm, gold_frac = normalize_math_answer(gold_raw)

    match = (pred_frac == gold_frac) if (pred_frac is not None and gold_frac is not None) else (pred_norm == gold_norm)

    info.update(
        {
            "pred_raw": pred_raw,
            "gold_raw": gold_raw,
            "pred_norm": pred_norm,
            "gold_norm": gold_norm,
            "match": bool(match),
        }
    )
    return (1.0 if match else 0.0), info


def _compare_marft(
    pred_text: str, label_text: str, meta: Dict[str, Any], use_math_verify: bool
) -> Tuple[float, Dict[str, Any]]:
    # Optional backend; import here so "simple" path has zero dependency cost.
    from c3.envs.math.backends.marft.scorer import score_math_marft  # type: ignore

    r, info = score_math_marft(prediction=pred_text, label=label_text, meta=meta, use_math_verify=use_math_verify)
    det = info if isinstance(info, dict) else {}
    det.setdefault("env", "MathEnv")
    det.setdefault("backend", "marft")
    return float(r), det


def _score_one(
    *,
    pred_text: str,
    label_text: str,
    backend_requested: str,
    meta: Dict[str, Any],
    use_math_verify: bool,
) -> Tuple[float, Dict[str, Any]]:
    """
    Score one (role, prediction) with backend selection and best-effort fallback.
    Inputs should already be sanitized.
    """
    if backend_requested == "marft":
        try:
            r, det = _compare_marft(pred_text, label_text, meta, use_math_verify=use_math_verify)
            det = dict(det)
            det["backend_used"] = "marft"
            return float(r), det
        except Exception as e:
            # Fall back to simple; record why.
            r, det = _compare_simple(pred_text, label_text)
            det = dict(det)
            det["backend_used"] = "simple"
            det["backend_fallback_reason"] = f"{type(e).__name__}: {e}"
            return float(r), det

    r, det = _compare_simple(pred_text, label_text)
    det = dict(det)
    det["backend_used"] = "simple"
    return float(r), det


# -----------------------------------------------------------------------------
# Prediction extraction + aggregation
# -----------------------------------------------------------------------------


def _extract_role_predictions(*, meta: Dict[str, Any], prediction: str) -> Tuple[List[str], List[str]]:
    """
    Returns (roles, texts). Multi-role if meta provides answer_roles + traj_role_outputs.
    """
    traj_role_outputs = meta.get("traj_role_outputs", None)
    answer_roles = meta.get("answer_roles", None)

    roles: List[str] = []
    texts: List[str] = []

    if isinstance(answer_roles, (list, tuple)) and isinstance(traj_role_outputs, dict) and len(answer_roles) > 0:
        for r in answer_roles:
            rr = str(r)
            roles.append(rr)
            texts.append(_s(traj_role_outputs.get(rr, "")))
        return roles, texts

    # Single prediction fallback.
    roles = [str(meta.get("answer_role", "answer"))]
    texts = [_s(meta.get("answer_text", prediction or ""))]
    return roles, texts


def _aggregate(mode: str, scores: Sequence[float], correct: Sequence[bool]) -> float:
    if not scores:
        return 0.0
    if mode == "avg_per_answer":
        return float(sum(scores) / float(len(scores)))
    if mode == "vote_binary":
        c = sum(1 for x in correct if x)
        return 1.0 if c > (len(correct) / 2.0) else 0.0
    # strict
    return float(scores[0])


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def score_math(*, prediction: str, label: Optional[str] = None, meta: Optional[dict] = None) -> Tuple[float, Dict[str, Any]]:
    meta_d: Dict[str, Any] = meta if isinstance(meta, dict) else {}
    cfg = _env_cfg(meta_d)

    mode = _canon_reward_mode(cfg.get("reward_mode", "strict"))
    backend_req = _canon_backend(cfg.get("math_backend", "simple"))
    use_math_verify = bool(cfg.get("use_math_verify", False))

    info: Dict[str, Any] = {
        "env": "MathEnv",
        "reward_mode": mode,
        "backend_requested": backend_req,
        "use_math_verify": use_math_verify,
        "answer_role": meta_d.get("answer_role"),
        "answer_roles": meta_d.get("answer_roles"),
        "question_id": meta_d.get("question_id"),
        "k_id": meta_d.get("k_id"),
    }

    label_str = _s(label).strip()
    if not label_str:
        info.update({"match": False, "reason": "missing_label"})
        return 0.0, info

    # Sanitize only for judging (do not mutate original tokens/logprobs).
    label_clean = sanitize_math_solution_text(label_str)

    roles, pred_texts = _extract_role_predictions(meta=meta_d, prediction=_s(prediction))
    per_role: List[Dict[str, Any]] = []
    scores: List[float] = []
    correct: List[bool] = []

    for rr, pt in zip(roles, pred_texts):
        pt_clean = sanitize_math_solution_text(pt)
        r, det = _score_one(
            pred_text=pt_clean,
            label_text=label_clean,
            backend_requested=backend_req,
            meta=meta_d,
            use_math_verify=use_math_verify,
        )

        det = dict(det)
        det["role"] = rr
        per_role.append(det)
        scores.append(float(r))
        correct.append(bool(float(r) >= 0.999))

    reward = _aggregate(mode, scores, correct)

    info["backend_used"] = per_role[0].get("backend_used", "simple") if per_role else "simple"
    info["per_role"] = per_role
    info["reward"] = float(reward)
    return float(reward), info
