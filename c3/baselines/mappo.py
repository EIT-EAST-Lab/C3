from __future__ import annotations

from typing import Tuple

import torch

__all__ = ["compute_mappo_step_gae"]


def _validate_1d(name: str, x: torch.Tensor, n: int, device: torch.device) -> None:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(x).__name__}")
    if x.device != device:
        raise ValueError(f"{name} must be on device={device}, got {x.device}")
    if x.dim() != 1 or int(x.numel()) != int(n):
        raise ValueError(f"{name} must be [N] with N={n}, got shape={tuple(x.shape)}")


def _lexsort_episode_step(episode_ids: torch.Tensor, step_ids: torch.Tensor) -> torch.Tensor:
    """Stable lexicographic sort by (episode_id, step_id) without composite-key overflow."""
    ep = episode_ids.to(torch.long)
    st = step_ids.to(torch.long)

    # Stable by secondary key, then stable by primary key.
    order = torch.argsort(st, stable=True)
    order = order[torch.argsort(ep[order], stable=True)]
    return order


def compute_mappo_step_gae(
    *,
    rewards: torch.Tensor,  # [N] per-step reward (scalar)
    values: torch.Tensor,  # [N] V(s_t)
    terminals: torch.Tensor,  # [N] 1 only at episode last step
    episode_ids: torch.Tensor,  # [N]
    step_ids: torch.Tensor,  # [N] strictly increasing within an episode (after sorting)
    gamma: float = 1.0,
    lambd: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    MAPPO-style step-GAE over macro-steps.

    Invariants (fail-fast):
      - rows can be globally unordered
      - within each episode: step_ids must be strictly increasing after sorting
      - terminals must be 1 only at the last step of each episode
    """
    if rewards.dim() != 1:
        raise ValueError(f"rewards must be [N], got {tuple(rewards.shape)}")

    n = int(rewards.numel())
    device = rewards.device

    _validate_1d("values", values, n, device)
    _validate_1d("terminals", terminals, n, device)
    _validate_1d("episode_ids", episode_ids, n, device)
    _validate_1d("step_ids", step_ids, n, device)

    if n == 0:
        return rewards, rewards

    g = float(gamma)
    lam = float(lambd)
    if g < 0.0 or lam < 0.0:
        raise ValueError(f"gamma/lambd must be >= 0, got gamma={g}, lambd={lam}")

    order = _lexsort_episode_step(episode_ids, step_ids)

    ep_s = episode_ids[order].to(torch.long)
    st_s = step_ids[order].to(torch.long)
    r_s = rewards[order]
    v_s = values[order]
    t_s = terminals[order].to(torch.long)  # checks only; arithmetic uses float mask below

    adv_s = torch.zeros_like(r_s)
    ret_s = torch.zeros_like(r_s)

    # Segment by episode after sorting.
    _, counts = torch.unique_consecutive(ep_s, return_counts=True)

    cursor = 0
    for c in counts.tolist():
        if c <= 0:
            continue

        sl = slice(cursor, cursor + c)
        sid = st_s[sl]
        rr = r_s[sl]
        vv = v_s[sl]
        tt = t_s[sl]
        epi = int(ep_s[sl][0].item())

        if sid.numel() > 1 and torch.any(sid[1:] <= sid[:-1]):
            raise RuntimeError(
                f"[MAPPO][FAIL-FAST] Non-increasing step_ids within episode: {sid.tolist()} "
                f"(episode_id={epi}, T={int(sid.numel())})"
            )

        if int(tt[-1].item()) != 1:
            raise RuntimeError(
                "[MAPPO][FAIL-FAST] episode last step is not terminal. "
                f"episode_id={epi}, step_ids={sid.tolist()}, terminals={tt.tolist()}"
            )
        if tt.numel() > 1 and torch.any(tt[:-1] != 0):
            raise RuntimeError(f"[MAPPO][FAIL-FAST] Terminal appears before last step. episode_id={epi}")

        not_done = (1.0 - tt.to(rr.dtype)).clamp_(0.0, 1.0)

        # next_v[t] = v[t+1], last next_v = 0 (terminal already handled by not_done)
        next_v = torch.zeros_like(vv)
        if vv.numel() > 1:
            next_v[:-1] = vv[1:]

        delta = rr + (g * next_v * not_done) - vv

        gae = rr.new_zeros(())
        for i in range(int(rr.numel()) - 1, -1, -1):
            gae = delta[i] + (g * lam * not_done[i] * gae)
            adv_s[cursor + i] = gae
            ret_s[cursor + i] = gae + vv[i]

        cursor += c

    inv = torch.empty_like(order)
    inv[order] = torch.arange(n, device=device, dtype=order.dtype)
    return adv_s[inv], ret_s[inv]
