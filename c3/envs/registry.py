"""Env reward registry for C3 graft.

Keep this module lightweight and import-safe.

API:
  - get_env_reward_fn(env_name) -> callable

Reward function signature:
  score(*, prediction: str, label: str | None = None, meta: dict | None = None)
    -> tuple[float, dict]

Where:
  - reward is a float (MathEnv typically 0/1; CodeEnv typically pass-rate in [0,1])
  - info is a JSON-serializable dict (best-effort)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

RewardFn = Callable[..., Tuple[float, Dict[str, Any]]]

SUPPORTED_ENVS = ("MathEnv", "CodeEnv")


def _normalize_env_name(env_name: str) -> str:
    s = (env_name or "").strip()
    if s.lower() in {"math", "mathenv", "c3_math"}:
        return "MathEnv"
    if s.lower() in {"code", "codeenv", "c3_code"}:
        return "CodeEnv"
    return s


def get_env_reward_fn(env_name: str) -> RewardFn:
    """Return the reward function for the given env_name."""
    name = _normalize_env_name(env_name)

    if name == "MathEnv":
        from .math.reward import score_math as _fn
        return _fn

    if name == "CodeEnv":
        # MBPP/MBPP+-style scoring (pass-rate in [0, 1])
        from .code.reward import score_code as _fn
        return _fn

    raise KeyError(f"Unknown env_name={env_name!r}. Supported: {SUPPORTED_ENVS}")
