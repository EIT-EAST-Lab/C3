# -*- coding: utf-8 -*-
"""
CodeEnv reward function.

Registry contract (see envs/registry.py):
  score_code(*, prediction: str, label: str | None = None, meta: dict | None = None) -> (reward, info)

We implement MBPP/MBPP+ style judging:
- tests/meta are expected in meta["dataset_meta"] (dict) or meta["dataset_meta_json"] (json string).
- label is optional for CodeEnv.

Extra knobs (env):
- C3_CODE_SANDBOX_MEM_MB: override per-call RLIMIT_AS for executor (default: 4096)
- C3_CODE_SANDBOX_CPU_S:  override per-call RLIMIT_CPU for executor (default: 0 => executor auto)
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple

from c3.envs.code.executor import run_mbpp_tests


def _maybe_load_dataset_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    dm = meta.get("dataset_meta", None)
    if isinstance(dm, dict):
        return dm

    dmj = meta.get("dataset_meta_json", None)
    if isinstance(dmj, str) and dmj.strip():
        try:
            obj = json.loads(dmj)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


def _coerce_timeout(meta: Dict[str, Any], default: int = 15) -> int:
    cfg = meta.get("task_env_cfg", {})
    if isinstance(cfg, dict):
        t = cfg.get("code_timeout", None)
        if isinstance(t, (int, float)) and t > 0:
            return int(t)
    return int(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.environ.get(name, "")).strip() or default)
    except Exception:
        return default


def compute_code_reward(
    *,
    prediction: str,
    label: Optional[str],
    meta: Optional[Dict[str, Any]] = None,
) -> Tuple[float, Dict[str, Any]]:
    """Core implementation for CodeEnv reward."""
    meta = meta or {}
    dataset_meta = _maybe_load_dataset_meta(meta)

    # Optional reference code (not executed by default; kept for compatibility/debug).
    ref_code = None
    for k in ("reference_code", "ref_code", "canonical_solution", "solution"):
        v = dataset_meta.get(k, None)
        if isinstance(v, str) and v.strip():
            ref_code = v
            break

    timeout = _coerce_timeout(meta, default=15)

    # Per-call sandbox limits (override env-based defaults inside executor).
    mem_mb = _env_int("C3_CODE_SANDBOX_MEM_MB", 4096)
    cpu_s = _env_int("C3_CODE_SANDBOX_CPU_S", 0)
    cpu_override = cpu_s if cpu_s > 0 else None

    score, info = run_mbpp_tests(
        candidate_code=str(prediction or ""),
        ref_code=ref_code if isinstance(ref_code, str) else None,
        sample_meta=dataset_meta,
        timeout=timeout,
        mem_mb=mem_mb,
        cpu_s=cpu_override,
    )

    info = dict(info or {})
    info["reward_mode"] = "mbpp_pass_rate"
    info["env_name"] = "CodeEnv"
    info["answer_role"] = meta.get("answer_role")
    info["question_id"] = meta.get("question_id")
    info["k_id"] = meta.get("k_id")

    if info.get("total", 0) == 0:
        info["errors"] = (info.get("errors", []) or []) + ["MissingTests"]
        return 0.0, info

    return float(score), info


def score_code(
    *,
    prediction: str,
    label: Optional[str] = None,
    meta: Optional[dict] = None,
) -> Tuple[float, Dict[str, Any]]:
    """Registry entrypoint."""
    return compute_code_reward(prediction=prediction, label=label, meta=meta if isinstance(meta, dict) else None)
