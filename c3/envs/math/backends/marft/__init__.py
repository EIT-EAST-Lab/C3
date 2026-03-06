# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Callable

from .normalize import normalize_expr
from .scorer import score_math_marft


def _missing(name: str, err: Exception) -> Callable[..., Any]:
    def _f(*args: Any, **kwargs: Any) -> Any:
        raise ImportError(f"{name} is unavailable: {type(err).__name__}: {err}") from err
    return _f


try:
    from .parse_utils_qwen import extract_answer  # type: ignore
except Exception as e:
    extract_answer = _missing("parse_utils_qwen.extract_answer", e)  # type: ignore

try:
    from .verify_utils import grade_answer  # type: ignore
except Exception as e:
    grade_answer = _missing("verify_utils.grade_answer", e)  # type: ignore

try:
    from .math_verify import compute_score as compute_score_verify  # type: ignore
except Exception as e:
    compute_score_verify = _missing("math_verify.compute_score", e)  # type: ignore

try:
    from .grader import math_equal  # type: ignore
except Exception as e:
    math_equal = _missing("grader.math_equal", e)  # type: ignore


__all__ = [
    "normalize_expr",
    "score_math_marft",
    "extract_answer",
    "grade_answer",
    "compute_score_verify",
    "math_equal",
]
