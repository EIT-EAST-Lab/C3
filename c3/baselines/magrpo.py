"""MAGRPO (multi-agent GRPO-style) advantage/return computation.

We compute *scalar* advantages per sample and broadcast them to token positions.

Grouping:
  - Caller provides `group_ids` with shape [B].
  - In MAS/C3, a common and recommended choice is grouping per (question_id, role_id):
      group_id = question_id * num_roles + role_id
    so different roles do NOT share a baseline.

Baselines (per group):
  - "group_mean":  A = r - mean_group(r)
  - "rloo":        A = r - mean_{others in group}(r)  (leave-one-out)

Notes:
  - This module does not require a critic (`values` is ignored; kept for API compatibility).
  - Token-level normalization is OFF by default to avoid length bias.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from .utils import expand_scalar_rewards_to_tokens, normalize_advantages


# ----------------------------- grouping helpers -----------------------------


def _inverse_and_counts(group_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (inv[B], counts[G]) for group_ids[B]."""
    g = group_ids.to(torch.long).view(-1)
    if g.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=g.device), torch.empty((0,), dtype=torch.long, device=g.device)
    _, inv = torch.unique(g, sorted=True, return_inverse=True)
    counts = torch.bincount(inv)
    return inv, counts


def _require_min_group_size(group_ids: torch.Tensor, *, require_group_k: bool) -> None:
    """If require_group_k=True, ensure every group has at least 2 samples."""
    if not require_group_k:
        return

    g = group_ids.to(torch.long).view(-1)
    if g.numel() == 0:
        raise RuntimeError("MAGRPO require_group_k=True but got empty group_ids.")

    _, counts = _inverse_and_counts(g)
    if counts.numel() == 0 or int(counts.min().item()) < 2:
        mn = int(counts.min().item()) if counts.numel() else 0
        mx = int(counts.max().item()) if counts.numel() else 0
        ng = int(counts.numel())
        raise RuntimeError(
            "MAGRPO require_group_k=True violated: each group must have >=2 rollouts "
            f"(num_groups={ng}, min_group_size={mn}, max_group_size={mx}). "
            "This usually means K<=1 or grouping/routing is broken."
        )


def _group_stats(
    x: torch.Tensor, group_ids: torch.Tensor, *, eps: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute per-group sums/cnts and per-row mean/std.

    Returns:
      sums[G], cnts[G] (float), mean_per_row[B], std_per_row[B], inv[B]
    """
    x = x.view(-1)
    inv, counts = _inverse_and_counts(group_ids)

    G = int(counts.numel())
    sums = torch.zeros((G,), device=x.device, dtype=x.dtype)
    sums.scatter_add_(0, inv, x)

    cnts = counts.to(dtype=x.dtype, device=x.device)
    means_g = sums / cnts.clamp_min(1.0)
    mean_per_row = means_g[inv]

    diffs = x - mean_per_row
    vsums = torch.zeros((G,), device=x.device, dtype=x.dtype)
    vsums.scatter_add_(0, inv, diffs * diffs)

    vars_g = vsums / cnts.clamp_min(1.0)
    stds_g = torch.sqrt(vars_g + eps)
    std_per_row = stds_g[inv]

    return sums, cnts, mean_per_row, std_per_row, inv


# ---------------------------------- main -----------------------------------


def compute_magrpo(
    *,
    action_mask: torch.Tensor,  # [B, T]
    rewards: torch.Tensor,  # [B] or [B, T]
    values: Optional[torch.Tensor] = None,  # ignored (kept for API compatibility)
    group_ids: Optional[torch.Tensor] = None,  # [B]
    baseline: str = "group_mean",
    gamma: float = 1.0,  # unused (kept for API compatibility)
    lambd: float = 1.0,  # unused (kept for API compatibility)
    normalize_adv: bool = True,
    require_group_k: bool = False,
    token_normalize: bool = False,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute MAGRPO advantages/returns (token-level tensors)."""
    _ = values, gamma, lambd  # API compatibility; not used in MAGRPO route1

    if action_mask.dim() != 2:
        raise ValueError(f"action_mask must be [B,T], got shape={tuple(action_mask.shape)}")

    # Expand scalar reward onto token grid, then summarize to a scalar per sample.
    # (We keep `.sum(dim=1)` to preserve existing semantics of expand_scalar_rewards_to_tokens.)
    token_rewards = expand_scalar_rewards_to_tokens(rewards, action_mask)  # [B, T]
    scalar_r = token_rewards.sum(dim=1)  # [B]

    # ------------------------ scalar advantages per sample ------------------------
    if group_ids is None:
        # No grouping: batch baseline as a safe fallback.
        adv_scalar = scalar_r - scalar_r.mean()
        if normalize_adv:
            adv_scalar = adv_scalar / scalar_r.std(unbiased=False).clamp_min(float(eps))
    else:
        g = group_ids.to(torch.long).view(-1)
        if g.numel() != scalar_r.numel():
            raise ValueError(
                f"group_ids must have shape [B] with B={int(scalar_r.numel())}, got numel={int(g.numel())}"
            )

        bmode = (baseline or "group_mean").strip().lower()
        if bmode not in ("group_mean", "rloo"):
            raise ValueError(f"invalid magrpo baseline={baseline!r}, expected 'group_mean'|'rloo'")

        _require_min_group_size(g, require_group_k=require_group_k)

        sums, cnts, mean_per_row, std_per_row, inv = _group_stats(scalar_r, g, eps=float(eps))

        if bmode == "rloo":
            # Leave-one-out mean (exclude self). If a group has size 1 and require_group_k=False,
            # denom clamps to 1 and the LOO mean becomes 0 (legacy behavior).
            denom = (cnts[inv] - 1.0).clamp_min(1.0)
            mean_per_row = (sums[inv] - scalar_r) / denom

        adv_scalar = scalar_r - mean_per_row
        if normalize_adv:
            adv_scalar = adv_scalar / std_per_row.clamp_min(float(eps))

    # ------------------------------ broadcast to tokens ------------------------------
    am = action_mask.to(token_rewards.dtype)
    adv = adv_scalar.unsqueeze(1) * am
    ret = token_rewards * am  # no critic: returns are token rewards on action positions

    if token_normalize:
        adv = normalize_advantages(adv, action_mask, eps=float(eps))

    return adv, ret
