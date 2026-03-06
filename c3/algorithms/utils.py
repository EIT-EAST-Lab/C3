"""Shared utilities for MARL advantage calculators (token-level)."""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def _ensure_2d(x: torch.Tensor, name: str) -> torch.Tensor:
    if x.dim() != 2:
        raise ValueError(f"{name} must be 2D [B, T], got shape={tuple(x.shape)}")
    return x


def _ensure_1d(x: torch.Tensor, name: str) -> torch.Tensor:
    if x.dim() != 1:
        raise ValueError(f"{name} must be 1D [B], got shape={tuple(x.shape)}")
    return x


def expand_scalar_rewards_to_tokens(rewards: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    """Expand scalar rewards into token-level rewards.

    If rewards is:
      - [B] : put the scalar reward on the *last* action token for each sample.
      - [B,T]: assume already token-level.
    """

    action_mask = _ensure_2d(action_mask, "action_mask")
    B, T = action_mask.shape

    if rewards.dim() == 2:
        if rewards.shape != (B, T):
            raise ValueError(f"rewards [B,T] shape mismatch: rewards={tuple(rewards.shape)} mask={(B,T)}")
        return rewards

    rewards = _ensure_1d(rewards, "rewards")
    if rewards.shape[0] != B:
        raise ValueError(f"rewards [B] length mismatch: rewards={tuple(rewards.shape)} B={B}")

    out = torch.zeros((B, T), device=action_mask.device, dtype=rewards.dtype)
    # last action index per row
    # If a row has no actions (shouldn't happen), keep zeros.
    with torch.no_grad():
        idx = action_mask.long().argmax(dim=1)  # first 1
        # But we want last 1 -> reverse search
        rev = torch.flip(action_mask.long(), dims=[1])
        last_from_end = rev.argmax(dim=1)  # index from end
        last_idx = (T - 1) - last_from_end
        # For rows with no actions, argmax returns 0; detect via sum
        has_any = action_mask.sum(dim=1) > 0
        last_idx = torch.where(has_any, last_idx, idx)
    out[torch.arange(B, device=action_mask.device), last_idx] = rewards
    return out


def discount_cumsum(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    """Compute discounted cumulative sums along the last dim (time).

    rewards: [B, T]
    returns: [B, T]
    """

    rewards = _ensure_2d(rewards, "rewards")
    B, T = rewards.shape
    out = torch.zeros_like(rewards)
    running = torch.zeros((B,), device=rewards.device, dtype=rewards.dtype)
    for t in range(T - 1, -1, -1):
        running = rewards[:, t] + gamma * running
        out[:, t] = running
    return out


def gae_advantages(
    *,
    rewards: torch.Tensor,  # [B, T]
    values: torch.Tensor,  # [B, T]
    action_mask: torch.Tensor,  # [B, T]
    gamma: float,
    lambd: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE advantages and returns (token-level).

    This is a standard GAE implementation over token steps.
    """

    rewards = _ensure_2d(rewards, "rewards")
    values = _ensure_2d(values, "values")
    action_mask = _ensure_2d(action_mask, "action_mask")
    if rewards.shape != values.shape or rewards.shape != action_mask.shape:
        raise ValueError(
            f"shape mismatch: rewards={tuple(rewards.shape)} values={tuple(values.shape)} mask={tuple(action_mask.shape)}"
        )

    B, T = rewards.shape
    advantages = torch.zeros_like(rewards)
    lastgaelam = torch.zeros((B,), device=rewards.device, dtype=rewards.dtype)

    # Next values: shift left, last next_value = 0
    next_values = torch.zeros_like(values)
    next_values[:, :-1] = values[:, 1:]

    for t in range(T - 1, -1, -1):
        mask_t = action_mask[:, t].to(rewards.dtype)
        delta = rewards[:, t] + gamma * next_values[:, t] - values[:, t]
        lastgaelam = delta + gamma * lambd * lastgaelam
        # Only keep on action positions
        advantages[:, t] = lastgaelam * mask_t
        lastgaelam = lastgaelam * mask_t  # reset across non-action pads

    returns = advantages + values
    returns = returns * action_mask.to(returns.dtype)
    return advantages, returns


def normalize_advantages(advantages: torch.Tensor, action_mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize advantages across *action tokens only*."""

    advantages = _ensure_2d(advantages, "advantages")
    action_mask = _ensure_2d(action_mask, "action_mask")
    mask = action_mask.to(advantages.dtype)
    denom = mask.sum().clamp_min(1.0)
    mean = (advantages * mask).sum() / denom
    var = ((advantages - mean) ** 2 * mask).sum() / denom
    std = torch.sqrt(var + eps)
    out = (advantages - mean) / std
    out = out * mask
    return out


def group_mean_std(x: torch.Tensor, group_ids: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-group mean/std for 1D values.

    x: [B]
    group_ids: [B] int-like
    Returns:
      mean_per_row: [B]
      std_per_row: [B]
    """

    x = _ensure_1d(x, "x")
    group_ids = _ensure_1d(group_ids, "group_ids")
    if x.shape[0] != group_ids.shape[0]:
        raise ValueError("x and group_ids must have same length")

    # Make group ids contiguous for scatter ops.
    uniq, inv = torch.unique(group_ids.to(torch.long), sorted=True, return_inverse=True)
    G = uniq.shape[0]

    sums = torch.zeros((G,), device=x.device, dtype=x.dtype)
    cnts = torch.zeros((G,), device=x.device, dtype=x.dtype)
    sums.scatter_add_(0, inv, x)
    cnts.scatter_add_(0, inv, torch.ones_like(x))
    means = sums / cnts.clamp_min(1.0)

    # var
    diffs = x - means[inv]
    vsums = torch.zeros((G,), device=x.device, dtype=x.dtype)
    vsums.scatter_add_(0, inv, diffs * diffs)
    vars_ = vsums / cnts.clamp_min(1.0)
    stds = torch.sqrt(vars_ + eps)

    return means[inv], stds[inv]
