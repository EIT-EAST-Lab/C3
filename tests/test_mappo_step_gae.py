from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="MAPPO step-GAE runtime tests require torch.")
import torch
from c3.algorithms.mappo import compute_mappo_step_gae


def test_compute_mappo_step_gae_supports_k8_multi_episode_batches() -> None:
    rewards = torch.tensor(
        [0.0, 1.0] * 8,
        dtype=torch.float32,
    )
    values = torch.tensor(
        [0.1, 0.2] * 8,
        dtype=torch.float32,
    )
    terminals = torch.tensor(
        [0, 1] * 8,
        dtype=torch.long,
    )
    episode_ids = torch.tensor(
        [kid for kid in range(8) for _ in range(2)],
        dtype=torch.long,
    )
    step_ids = torch.tensor(
        [0, 1] * 8,
        dtype=torch.long,
    )

    advantages, returns = compute_mappo_step_gae(
        rewards=rewards,
        values=values,
        terminals=terminals,
        episode_ids=episode_ids,
        step_ids=step_ids,
        gamma=1.0,
        lambd=1.0,
    )

    assert advantages.shape == rewards.shape
    assert returns.shape == rewards.shape
    assert torch.isfinite(advantages).all()
    assert torch.isfinite(returns).all()
    assert torch.all(returns > 0)


def test_compute_mappo_step_gae_rejects_non_increasing_step_ids() -> None:
    rewards = torch.tensor([0.0, 1.0], dtype=torch.float32)
    values = torch.tensor([0.1, 0.2], dtype=torch.float32)
    terminals = torch.tensor([0, 1], dtype=torch.long)
    episode_ids = torch.tensor([0, 0], dtype=torch.long)
    step_ids = torch.tensor([1, 1], dtype=torch.long)

    try:
        compute_mappo_step_gae(
            rewards=rewards,
            values=values,
            terminals=terminals,
            episode_ids=episode_ids,
            step_ids=step_ids,
            gamma=1.0,
            lambd=1.0,
        )
    except RuntimeError as exc:
        assert "Non-increasing step_ids" in str(exc)
    else:
        raise AssertionError("Expected MAPPO step-GAE to reject non-increasing step_ids.")


def test_compute_mappo_step_gae_rejects_terminal_before_last_step() -> None:
    rewards = torch.tensor([0.0, 1.0], dtype=torch.float32)
    values = torch.tensor([0.1, 0.2], dtype=torch.float32)
    terminals = torch.tensor([1, 0], dtype=torch.long)
    episode_ids = torch.tensor([0, 0], dtype=torch.long)
    step_ids = torch.tensor([0, 1], dtype=torch.long)

    try:
        compute_mappo_step_gae(
            rewards=rewards,
            values=values,
            terminals=terminals,
            episode_ids=episode_ids,
            step_ids=step_ids,
            gamma=1.0,
            lambd=1.0,
        )
    except RuntimeError as exc:
        assert "episode last step is not terminal" in str(exc)
    else:
        raise AssertionError("Expected MAPPO step-GAE to reject non-terminal episode endings.")
