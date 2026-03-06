"""C3 environment reward providers (extension-only).

Phase 4a scope:
  - Provide deterministic, dependency-light reward functions for C3 envs
    (MathEnv / CodeEnv), without touching OpenRLHF core trainer stack.
"""

from .registry import SUPPORTED_ENVS, get_env_reward_fn

__all__ = [
    "SUPPORTED_ENVS",
    "get_env_reward_fn",
]
