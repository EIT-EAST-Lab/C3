# -*- coding: utf-8 -*-
"""Deprecated import alias for :mod:`c3.credit.counterfactual`.

The package was renamed in release 0.2.0 because the doubled name
``c3.credit.c3`` was a recurring source of confusion. Importing this module
re-exports the new package and registers the old submodule names, so existing
code keeps working; new code should import ``c3.credit.counterfactual``.
"""

from __future__ import annotations

import sys as _sys
import warnings as _warnings
from importlib import import_module as _import_module

_NEW_PACKAGE = "c3.credit.counterfactual"
_SUBMODULES = (
    "baselines",
    "materialize",
    "prompts",
    "provider",
    "registry",
    "scoring",
    "types",
)

_warnings.warn(
    "c3.credit.c3 is deprecated and will be removed in a future release; "
    "import c3.credit.counterfactual instead.",
    DeprecationWarning,
    stacklevel=2,
)

_new = _import_module(_NEW_PACKAGE)
for _name in _SUBMODULES:
    _module = _import_module(f"{_NEW_PACKAGE}.{_name}")
    _sys.modules[f"{__name__}.{_name}"] = _module
    globals()[_name] = _module

__all__ = list(_SUBMODULES)
