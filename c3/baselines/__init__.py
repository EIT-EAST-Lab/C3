"""The comparison baselines: MARL advantage and return calculators.

These are the methods the paper compares against, not the paper's own credit
assignment, which lives in `c3.credit`. `openrlhf/trainer/ppo_utils/
experience_maker.py` picks one of them by the name the registry resolves.

Keep imports light: this package must be safe to import without initializing
distributed, ray or vLLM.
"""

__all__ = [
    "registry",
    "magrpo",
    "mappo",
    "group_baseline",
]
