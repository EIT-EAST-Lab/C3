# -*- coding: utf-8 -*-
"""
C3 credit provider registry (Rule-B).

This module does two things:
1) Builds the C3 credit provider instance for the trainer pipeline.
2) Extracts a *stable* provider cfg dict from CLI args (fail-fast, no silent fallback).

Rule-B policy:
- Legacy C3 knobs (cf_mode / regenerate / all-k / etc.) are intentionally unsupported.
  If any legacy arg is present (non-None), we raise to avoid silent misconfiguration.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from c3.integration.marl_specs import RoleSpec

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ALG_C3 = "c3"

_ALLOWED_CREDIT_VARIANTS = ("reward_only", "value_assisted", "value_only")
_ALLOWED_BASELINE_MODES = ("loo", "full_mean")

# Legacy knobs explicitly removed from Rule-B. Presence => fail-fast.
_LEGACY_C3_ARG_KEYS = (
    "c3_cf_mode",
    "c3_variance_threshold",
    "c3_credit_mode",
    "c3_return_all_k",
    "c3_regen_do_sample",
    "c3_regen_temperature",
    "c3_regen_top_p",
    "c3_regen_top_k",
    "c3_regen_max_new_tokens",
    "c3_regen_min_new_tokens",
    "c3_regen_num_beams",
    "c3_regen_repetition_penalty",
    "c3_regen_seed",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _norm_lower_str(x: Any) -> str:
    return str(x or "").strip().lower()


def _coerce_float(x: Any, *, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _coerce_optional_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        return None


def _fail_fast_if_legacy_args_present(args: Any) -> None:
    for k in _LEGACY_C3_ARG_KEYS:
        if getattr(args, k, None) is not None:
            raise ValueError(
                f"[C3][FAIL-FAST] Legacy C3 arg {k!r} is not supported in Rule-B nested rollouts."
            )


def _require_choice(*, name: str, value: str, allowed: tuple[str, ...]) -> str:
    if value not in allowed:
        raise ValueError(
            f"[C3][FAIL-FAST] Invalid {name}={value!r}. Must be one of {set(allowed)}."
        )
    return value


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_credit_provider(
    *,
    marl_algorithm: str,
    args: Any,
    roles: tuple[RoleSpec, ...],
    critic: Any,
    generate_for_roles=None,
    critic_preamble_path: str = "",
):
    """
    Create the credit provider for the given MARL algorithm.

    Returns:
      - C3 provider instance if marl_algorithm == "c3"
      - None otherwise

    Note:
      Rule-B removes regenerate/per-role callbacks. If a caller still wires it,
      we fail-fast to prevent accidental legacy behavior.
    """
    if _norm_lower_str(marl_algorithm) != _ALG_C3:
        return None

    _fail_fast_if_legacy_args_present(args)

    if generate_for_roles is not None:
        raise ValueError(
            "[C3][FAIL-FAST] generate_for_roles is not supported in Rule-B mainline. "
            "Remove regenerate/per-role wiring from the caller."
        )

    from c3.credit.c3.provider import C3CreditProvider

    return C3CreditProvider(
        args=args,
        roles=roles,
        q_critic=critic,
        generate_for_roles=None,
        critic_preamble_path=str(critic_preamble_path or ""),
    )


def build_credit_cfg_from_args(args: Any) -> Dict[str, Any]:
    """
    Convert CLI args into a provider cfg dict (stable keys, fail-fast).

    Keys consumed by Rule-B provider.compute(...):
      - credit_variant: {"reward_only","value_assisted","value_only"}
      - va_alpha: float in [0,1] (only used by value_assisted)
      - baseline_mode: {"loo","full_mean"}  (ablation: No LOO uses full_mean)
      - critic_ctx_limit: optional int
      - critic_forward_bs: optional int
    """
    _fail_fast_if_legacy_args_present(args)

    cfg: Dict[str, Any] = {}

    # ---- credit variant ----
    cv = _require_choice(
        name="c3_credit_variant",
        value=_norm_lower_str(getattr(args, "c3_credit_variant", None)),
        allowed=_ALLOWED_CREDIT_VARIANTS,
    )
    cfg["credit_variant"] = cv

    # ---- value-assisted alpha ----
    alpha = _coerce_float(getattr(args, "c3_va_alpha", 1.0), default=1.0)
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"[C3][FAIL-FAST] c3_va_alpha must be in [0,1], got {alpha}")
    cfg["va_alpha"] = float(alpha)

    # ---- baseline mode (Step3) ----
    # Backward-compatible default: loo, but still written as a stable cfg key.
    bm_raw = getattr(args, "c3_baseline_mode", None)
    bm = _norm_lower_str(bm_raw) or "loo"
    bm = _require_choice(
        name="c3_baseline_mode",
        value=bm,
        allowed=_ALLOWED_BASELINE_MODES,
    )
    cfg["baseline_mode"] = bm

    # ---- optional critic limits ----
    ctx = _coerce_optional_int(getattr(args, "critic_ctx_limit", None))
    if ctx is not None:
        cfg["critic_ctx_limit"] = ctx

    fbs = _coerce_optional_int(getattr(args, "critic_forward_bs", None))
    if fbs is not None:
        cfg["critic_forward_bs"] = fbs

    return cfg
