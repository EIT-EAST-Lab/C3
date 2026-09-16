"""Task assets for C3: the loaders for the files under ``configs/``.

``config`` reads a task file and its role file into runtime specs;
``datasets`` loads the datasets a task declares.

Keep this module dependency-light.

Important:
  We intentionally avoid importing ``config`` eagerly at package import
  time. Running ``python -m c3.task.config`` would
  otherwise trigger a RuntimeWarning about the module already being imported.
"""

__all__ = ["RoleSpec", "TaskSpec", "load_roles", "load_task", "topo_sort_roles"]


def __getattr__(name: str):
    """Lazy re-export.

    This avoids importing config at package import time.
    """
    if name in __all__:
        from . import config as _m
        return getattr(_m, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
