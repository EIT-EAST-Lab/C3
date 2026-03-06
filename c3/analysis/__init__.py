# c3/analysis/__init__.py
"""
C3 analysis package.

This subpackage hosts offline-analysis utilities for paper experiments:
- replay: run counterfactual continuations from a fixed restart state
- buckets: JSONL schema + IO helpers for replay-produced buckets
- metrics: credit fidelity/variance and influence (conditional MI)

Keep this package dependency-light and side-effect-free.
"""

from __future__ import annotations

__all__ = ["replay", "buckets", "metrics"]
