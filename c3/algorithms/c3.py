# -*- coding: utf-8 -*-
"""Deprecated import alias for :mod:`c3.algorithms.group_baseline`.

The module was renamed in release 0.2.0: it never implemented the paper's C3
credit assignment, only the MAGRPO-style group-baseline fallback calculator.
The registered algorithm name ``"c3"`` is unchanged.
"""

from __future__ import annotations

import warnings as _warnings

from .group_baseline import compute_c3

_warnings.warn(
    "c3.algorithms.c3 is deprecated and will be removed in a future release; "
    "import c3.algorithms.group_baseline instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["compute_c3"]
