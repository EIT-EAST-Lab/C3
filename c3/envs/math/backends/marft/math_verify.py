# -*- coding: utf-8 -*-
"""Optional Math-Verify backend (lazy import, Ray-safe)."""

from __future__ import annotations


def compute_score(model_output: str, ground_truth: str, timeout_score: float = 0.0) -> float:
    """Return score in [0,1]. If math-verify unavailable, raise ImportError."""
    try:
        from math_verify.errors import TimeoutException
        from math_verify.metric import math_metric
        from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
    except Exception as e:
        raise ImportError("math-verify is not installed or failed to import") from e

    try:
        verify_func = math_metric(
            gold_extraction_target=(LatexExtractionConfig(),),
            pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
        )
    except Exception:
        return 0.0

    ground_truth_boxed = "\\boxed{" + (ground_truth or "") + "}"
    try:
        ret_score, _ = verify_func([ground_truth_boxed], [model_output])
        return float(ret_score or 0.0)
    except TimeoutException:
        return float(timeout_score)
    except Exception:
        return 0.0
