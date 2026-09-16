# -*- coding: utf-8 -*-
"""
c3.reporting

The CLIs that turn run artifacts into the tables and figures of the paper.

This package exists primarily to:
1) Provide a stable `c3.reporting.*` import path (used by the paper scripts).
2) Define a minimal, explicit public surface for those CLIs.

Design:
- Keep this file tiny and dependency-free (fast import, no side effects).
- Export modules (not functions) so callers can use `python -m ...` entrypoints.
"""

from __future__ import annotations

# Public modules (kept minimal on purpose).
__all__ = [
    "main_results",
    "analysis_results",
    "plot_paper_figures",
]
