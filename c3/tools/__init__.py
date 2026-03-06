# -*- coding: utf-8 -*-
"""
c3.tools

Paper/analysis tooling namespace for C3.

This package exists primarily to:
1) Provide a stable `c3.tools.*` import path (used by examples/c3 scripts).
2) Define a minimal, explicit public surface for analysis CLIs.

Design:
- Keep this file tiny and dependency-free (fast import, no side effects).
- Export modules (not functions) so callers can use `python -m ...` entrypoints.
"""

from __future__ import annotations

# Public modules (kept minimal on purpose).
__all__ = [
    "c3_env_smoke",
    "main_results",
    "analysis_results",
    "plot_paper_figures",
]
