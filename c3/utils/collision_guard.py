"""Runtime collision guard for 63-bit context keys.

Even with a strong hash, collisions are *theoretically* possible. For research
code that claims reproducibility, we prefer to fail fast if a collision is ever
observed during a run, rather than silently merging unrelated contexts.

The guard maps:
  key (int) -> fingerprint (str)

If the same key is observed with a different fingerprint, an exception is
raised with actionable diagnostics.
"""

from __future__ import annotations

import threading
from typing import Dict, Optional


class ContextKeyCollisionError(RuntimeError):
    """Raised when two different texts map to the same context key."""


class CollisionGuard:
    """In-memory collision guard.

    This class is intentionally simple and dependency-free.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._seen: Dict[int, str] = {}

    def observe(self, key: int, fp: str, *, where: str) -> None:
        """Record an observation.

        Args:
            key: 63-bit integer hash.
            fp: Short fingerprint of the underlying text.
            where: Human-readable location tag (module/function/role/depth).
        """
        k = int(key)
        f = str(fp)
        loc = str(where)
        with self._lock:
            old = self._seen.get(k)
            if old is None:
                self._seen[k] = f
                return
            if old != f:
                raise ContextKeyCollisionError(
                    f"Context-key collision at {loc}: key={k} old_fp={old} new_fp={f}"
                )

    def reset(self) -> None:
        """Clear all recorded observations."""
        with self._lock:
            self._seen.clear()

    def size(self) -> int:
        """Number of unique keys observed."""
        with self._lock:
            return len(self._seen)


_GLOBAL_GUARD: Optional[CollisionGuard] = None
_GLOBAL_LOCK = threading.Lock()


def global_guard() -> CollisionGuard:
    """Return a process-global guard instance."""
    global _GLOBAL_GUARD
    if _GLOBAL_GUARD is not None:
        return _GLOBAL_GUARD
    with _GLOBAL_LOCK:
        if _GLOBAL_GUARD is None:
            _GLOBAL_GUARD = CollisionGuard()
        return _GLOBAL_GUARD
