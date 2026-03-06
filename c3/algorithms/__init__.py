"""C3 algorithm graft (MARL advantage/return calculators).

Phase 5a scope:
  - Add algorithm modules + a registry (no OpenRLHF core wiring yet).
  - Keep imports light: this package should be safe to import without
    initializing distributed / ray / vLLM.

Phase 5b will integrate these calculators into:
  openrlhf/trainer/ppo_utils/experience_maker.py
"""

__all__ = [
    "registry",
    "magrpo",
    "mappo",
    "c3",
]
