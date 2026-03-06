"""C3 minimal graft for OpenRLHF.

This package is intentionally isolated under openrlhf.ext.* so the OpenRLHF
core (cli/trainer/ray/utils) can remain the single source of truth.

Phase 1 provides config assets + loaders only.
"""

from importlib import resources as _resources

__all__ = ["configs", "integration", "mas", "envs", "algorithms"]


def package_root() -> str:
    """Return the filesystem path of this package root (best-effort)."""
    return str(_resources.files(__package__))
