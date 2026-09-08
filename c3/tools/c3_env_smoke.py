# -*- coding: utf-8 -*-
"""Deprecated import alias for :mod:`c3.tools.env_smoke`.

The module was renamed in release 0.2.0: inside a package already named ``c3``
the ``c3_`` prefix was redundant. The command line interface is unchanged, and
``python -m c3.tools.c3_env_smoke`` keeps working through this shim.
"""

from __future__ import annotations

import warnings as _warnings

from .env_smoke import main

_warnings.warn(
    "c3.tools.c3_env_smoke is deprecated and will be removed in a future release; "
    "use c3.tools.env_smoke instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
