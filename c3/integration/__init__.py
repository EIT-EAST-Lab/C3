"""Integration utilities for C3 extensions.

Keep this module dependency-light.

Important:
  We intentionally avoid importing ``marl_specs`` eagerly at package import
  time. Running ``python -m c3.integration.marl_specs`` would
  otherwise trigger a RuntimeWarning about the module already being imported.
"""

__all__ = ["RoleSpec", "TaskSpec", "load_roles", "load_task", "topo_sort_roles"]


def __getattr__(name: str):
    """Lazy re-export.

    This avoids importing marl_specs at package import time.
    """
    if name in __all__:
        from . import marl_specs as _m
        return getattr(_m, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
