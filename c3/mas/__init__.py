"""Multi-agent system (MAS) utilities for C3.

Keep this package import-light.

Running `python -m c3.mas.rollout_generator` imports
`c3.mas` first. If we eagerly import rollout_generator here,
Python will emit a RuntimeWarning about the module being already imported.
"""

__all__ = ["MASRolloutGenerator"]


def __getattr__(name: str):
    """Lazy re-export to avoid importing rollout_generator at package import time."""

    if name in __all__:
        from . import rollout_generator as _m

        return getattr(_m, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
