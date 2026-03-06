"""
MARL advantage/return algorithm registry (C3 graft).

This module is intentionally dependency-light:
- No OpenRLHF Experience imports.
- Callers pass plain tensors (token-aligned) and optional group ids.

Contract for returned functions:
  fn(
      *,
      action_mask: torch.Tensor,              # [B, T] float/bool (zeros mark padding)
      rewards: torch.Tensor,                  # [B] or [B, T]
      values: torch.Tensor | None = None,     # [B, T] optional (often scalar baseline broadcast)
      group_ids: torch.Tensor | None = None,  # [B] grouping ids (e.g., question_id)
      gamma: float = 1.0,
      lambd: float = 1.0,
      normalize_adv: bool = True,
      **kwargs,
  ) -> tuple[torch.Tensor, torch.Tensor]

Returns:
  - advantages: [B, T] (zeros where action_mask == 0)
  - returns:    [B, T] (zeros where action_mask == 0)

Notes:
- "mappo" is *not* dispatched here in current OpenRLHF-C3 integration.
  MAPPO step-GAE is computed in the training pipeline (ExperienceMaker) because it
  needs episode/step structure across roles, which is not representable by the
  token-only contract above.
"""

from __future__ import annotations

from typing import Callable, Dict

import torch

# Callable returning (advantages, returns), both token-aligned tensors.
MarlAdvFn = Callable[..., tuple[torch.Tensor, torch.Tensor]]

# Canonical names and historical aliases.
# Keep aliases stable to avoid breaking configs; unsupported names can still be canonicalized.
_ALIASES: Dict[str, str] = {
    "none": "none",
    "auto": "auto",
    # Project naming: GRPO variant used in C3 is multi-agent -> MAGRPO.
    "grpo": "magrpo",
    "magrpo": "magrpo",
    "mappo": "mappo",
    "c3": "c3",
}

_SUPPORTED_CANONICAL = ("magrpo", "c3")  # registry-dispatchable algorithms


def canonical_name(name: str) -> str:
    """Normalize an algorithm name to its canonical form."""
    s = (name or "none").strip().lower()
    return _ALIASES.get(s, s)


def get(name: str) -> MarlAdvFn:
    """Resolve a token-aligned advantage/return calculator.

    Dispatch rules:
      - "none"/"auto": returns MAGRPO (safe default for token-only dispatch).
      - "magrpo": MAGRPO group-baseline advantages.
      - "c3": C3 placeholder baseline (only if implemented in algorithms.c3).
      - "mappo": intentionally NOT supported here (computed in ExperienceMaker).
    """
    cname = canonical_name(name)

    if cname in {"none", "auto", "magrpo"}:
        from .magrpo import compute_magrpo

        return compute_magrpo

    if cname == "c3":
        from .c3 import compute_c3

        return compute_c3

    if cname == "mappo":
        raise KeyError(
            "marl_algorithm='mappo' is not dispatched via algorithms.registry.get(). "
            "MAPPO advantages/returns are computed in ExperienceMaker using step-GAE over role steps."
        )

    raise KeyError(
        f"Unknown marl_algorithm={name!r} (canonical={cname!r}). "
        f"Supported: {sorted(set(_ALIASES.values()))}"
    )


def list_supported() -> Dict[str, str]:
    """Human-readable descriptions of algorithms relevant to this codebase."""
    return {
        "magrpo": (
            "Multi-agent GRPO-style group-baseline advantages (no critic required; K-rollouts recommended)."
        ),
        "mappo": (
            "Multi-agent PPO with centralized V-critic + step-GAE over role steps. "
            "Computed in ExperienceMaker (not dispatched via this registry)."
        ),
        "c3": (
            "C3 algorithm (centralized Q-critic). Registry entry may point to a placeholder baseline "
            "depending on phase/integration."
        ),
    }


__all__ = ["MarlAdvFn", "canonical_name", "get", "list_supported"]
