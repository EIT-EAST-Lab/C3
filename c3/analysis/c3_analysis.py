# -*- coding: utf-8 -*-
"""Deprecated import alias for :mod:`c3.analysis.analysis`.

The module was renamed in release 0.2.0: inside a package already named ``c3``
the ``c3_`` prefix was redundant. The subcommands, arguments and output format
are unchanged, and ``python -m c3.analysis.c3_analysis`` keeps working through
this shim.
"""

from __future__ import annotations

import warnings as _warnings

from .analysis import main

_warnings.warn(
    "c3.analysis.c3_analysis is deprecated and will be removed in a future release; "
    "use c3.analysis.analysis instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["main"]


if __name__ == "__main__":
    main()
