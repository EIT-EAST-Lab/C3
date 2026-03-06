# -*- coding: utf-8 -*-
"""C3 advantage/return calculator (compat + fallback).

Important (M3):
- The *real* C3 credit assignment (counterfactual with centralized Q-critic)
  is computed upstream in `openrlhf/trainer/ppo_utils/experience_maker.py`
  via `c3.credit.c3.*` (C3CreditProvider + materialize/routing).

Why keep this module?
- ExperienceMaker / registry may still call a token-level calculator by name.
- When the C3 credit path is unavailable (K<=1, missing critic scorer, incomplete K groups),
  we need a safe fallback that keeps training runnable and stable.

Current behavior:
- Actor advantages:
    Use MAGRPO-style group baseline on scalar rewards (per question_id group if provided).
- Critic returns:
    If `values` are provided, compute GAE returns (stable critic training).
    Otherwise, return token rewards masked by action_mask.

This matches the "C3 is enabled, but credit assignment is computed elsewhere" architecture.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch

from .magrpo import compute_magrpo
from .utils import expand_scalar_rewards_to_tokens, gae_advantages

logger = logging.getLogger(__name__)


def compute_c3(
    *,
    action_mask: torch.Tensor,  # [B, T]
    rewards: torch.Tensor,  # [B] or [B, T]
    values: Optional[torch.Tensor] = None,  # [B, T] optional (for critic returns only)
    group_ids: Optional[torch.Tensor] = None,  # [B] question_id for grouping
    gamma: float = 1.0,
    lambd: float = 1.0,
    normalize_adv: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Token-level C3 calculator (fallback).

    NOTE:
      - This function does NOT implement C3 counterfactual credit.
        That is handled upstream (ExperienceMaker) when `marl_alg == "c3"`.
      - Here we provide:
          * adv: MAGRPO-style group baseline advantage (safe fallback)
          * ret: GAE returns if values provided, else token rewards
    """
    # 1) Fallback actor advantages via group baseline (stable and K-friendly)
    adv, _ = compute_magrpo(
        action_mask=action_mask,
        rewards=rewards,
        values=None,
        group_ids=group_ids,
        gamma=gamma,
        lambd=lambd,
        normalize_adv=normalize_adv,
    )

    # 2) Critic returns: prefer GAE if values exist (for critic training stability)
    token_rewards = expand_scalar_rewards_to_tokens(rewards, action_mask)
    if values is not None:
        # We intentionally ignore the GAE advantage here; actor advantage should come from C3 credit.
        _, ret = gae_advantages(
            rewards=token_rewards,
            values=values,
            action_mask=action_mask,
            gamma=float(gamma),
            lambd=float(lambd),
        )
        # Helpful debug note (do not spam info-level logs)
        logger.debug(
            "C3 fallback compute_c3(): using MAGRPO-style advantages; using values only for GAE returns. "
            "Full C3 credit is computed upstream via C3CreditProvider."
        )
        return adv, ret

    # No critic values: returns are just episodic token rewards on action positions
    logger.debug(
        "C3 fallback compute_c3(): values=None, returning token rewards as returns. "
        "Full C3 credit may be unavailable upstream (e.g., K<=1 or missing critic scorer)."
    )
    ret = token_rewards * action_mask.to(token_rewards.dtype)
    return adv, ret
