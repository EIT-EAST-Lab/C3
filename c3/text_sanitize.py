# -*- coding: utf-8 -*-
"""Deprecated import alias for :mod:`c3.utils.text_sanitize`.

The module moved out of the package root in release 0.2.0.
"""

from __future__ import annotations

import warnings as _warnings

from .utils.text_sanitize import sanitize_math_solution_text

_warnings.warn(
    "c3.text_sanitize is deprecated and will be removed in a future release; "
    "import c3.utils.text_sanitize instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["sanitize_math_solution_text"]
