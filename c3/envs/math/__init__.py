"""MathEnv reward (deterministic).

Phase 4a:
  - Parsing helpers (boxed / hash / last-line + numeric normalization)
  - score_math(prediction, label) -> (reward, info)
"""

from .reward import score_math

__all__ = ["score_math"]
