# -*- coding: utf-8 -*-
"""
Minimal types for C3 credit assignment within OpenRLHF.

We intentionally keep these lightweight and dependency-free, so the rest of the
training stack can import them safely.

Terminology:
- "trajectory" (traj): one episode rollout for a given question_id and k_id
- "role outputs": a mapping role_name -> generated text for that role
- "team reward": scalar reward computed from environment or RM for that traj
- "credit": per-role scalar advantage (counterfactual) derived from team reward and critic signals
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple


@dataclass(frozen=True)
class TrajectoryKey:
    question_id: int
    k_id: int


@dataclass(frozen=True)
class Trajectory:
    key: TrajectoryKey
    prompt: str
    label: Optional[str]
    role_outputs: Mapping[str, str]
    team_reward: float
    meta: Mapping[str, Any]


@dataclass(frozen=True)
class CreditOutput:
    """
    Per-role scalar advantages/credits to be broadcast to token-level advantages.
    """
    key: TrajectoryKey
    per_role_adv: Mapping[str, float]
    info: Mapping[str, Any]


@dataclass(frozen=True)
class CriticRequest:
    """
    A single critic scoring request.

    For a centralized Q-critic, we typically score:
      Q(prompt, role_outputs)
    and for counterfactual:
      Q(prompt, role_outputs with role_i replaced by baseline)

    We represent the critic input as a text `query` plus optional structured meta.
    """
    key: TrajectoryKey
    query: str
    meta: Mapping[str, Any]


@dataclass(frozen=True)
class CriticResponse:
    key: TrajectoryKey
    value: float
    info: Mapping[str, Any]
