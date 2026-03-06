# -*- coding: utf-8 -*-
"""
MARFT backend scorer (MathEnv).

Contract:
  score_math_marft(prediction: str, label: str, meta: dict, use_math_verify: bool) -> (reward, info)
  reward in [0, 1].

Policy:
- If use_math_verify=False => raise ImportError (caller falls back).
- If SymPy missing OR verification fails (parse/simplify/timeout/guard) => raise ImportError (caller falls back).
"""

from __future__ import annotations

import logging
import os
import re
import signal
import threading
from contextlib import contextmanager
from typing import Any, Dict, List, Tuple

from .normalize import normalize_expr

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Tunables (env-overridable)
# ---------------------------------------------------------------------

_MAX_EXPR_CHARS = int(os.environ.get("C3_MARFT_MAX_EXPR_CHARS", "4000"))
_MAX_INT_DIGITS = int(os.environ.get("C3_MARFT_MAX_INT_DIGITS", "300"))
_MAX_POW_OPS = int(os.environ.get("C3_MARFT_MAX_POW_OPS", "8"))
_SYM_TIMEOUT_S = float(os.environ.get("C3_MARFT_SYMPY_TIMEOUT_S", "1.5"))
_FAIL_LOG_BUDGET = int(os.environ.get("C3_MARFT_FAIL_LOG_BUDGET", "20"))

# ---------------------------------------------------------------------
# Regexes / parsing helpers
# ---------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")
_INTERVAL_RE = re.compile(r"^\s*([\[\(])\s*(.+?)\s*,\s*(.+?)\s*([\]\)])\s*$")
_DIGITS_RE = re.compile(r"\d{" + str(_MAX_INT_DIGITS + 1) + r",}")

# ---------------------------------------------------------------------
# Defensive SymPy guards
# ---------------------------------------------------------------------


class _SympyTimeout(Exception):
    """SymPy verification timed out."""


class _SympyGuard(Exception):
    """Expression rejected before SymPy (too risky)."""


@contextmanager
def _time_limit(seconds: float):
    """Best-effort wall-clock timeout using SIGALRM (Linux main-thread only)."""
    if seconds <= 0:
        yield
        return
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    old_handler = signal.getsignal(signal.SIGALRM)
    old_timer = signal.getitimer(signal.ITIMER_REAL)

    def _handler(signum, frame):
        raise _SympyTimeout(f"timeout>{seconds}s")

    try:
        signal.signal(signal.SIGALRM, _handler)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        yield
    finally:
        try:
            signal.setitimer(signal.ITIMER_REAL, old_timer[0], old_timer[1])
        except Exception:
            signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def _guard_expr_str(s: str) -> None:
    """Cheap circuit breaker to prevent SymPy hangs/OOM."""
    if not s:
        raise _SympyGuard("empty")
    if len(s) > _MAX_EXPR_CHARS:
        raise _SympyGuard(f"expr_too_long:{len(s)}")
    if _DIGITS_RE.search(s):
        raise _SympyGuard(f"int_too_long:>{_MAX_INT_DIGITS}")
    pow_ops = s.count("**") + s.count("^")
    if pow_ops > _MAX_POW_OPS:
        raise _SympyGuard(f"pow_ops_too_many:{pow_ops}")


# ---------------------------------------------------------------------
# String utilities
# ---------------------------------------------------------------------


def _ws_eq(a: str, b: str) -> bool:
    return _WS_RE.sub("", a or "") == _WS_RE.sub("", b or "")


def _strip_outer_parens(s: str) -> str:
    """Strip a single outer (...) only if it wraps whole string (balanced)."""
    s = (s or "").strip()
    if len(s) < 2 or s[0] != "(" or s[-1] != ")":
        return s

    depth = 0
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and i != len(s) - 1:
                return s
    return s[1:-1].strip() if depth == 0 else s


def _split_top_level_commas(s: str) -> List[str]:
    """Split on commas not inside (), [], {}."""
    s = (s or "").strip()
    if not s:
        return []

    out: List[str] = []
    buf: List[str] = []
    d_paren = d_brack = d_brace = 0

    for ch in s:
        if ch == "(":
            d_paren += 1
        elif ch == ")":
            d_paren = max(0, d_paren - 1)
        elif ch == "[":
            d_brack += 1
        elif ch == "]":
            d_brack = max(0, d_brack - 1)
        elif ch == "{":
            d_brace += 1
        elif ch == "}":
            d_brace = max(0, d_brace - 1)

        if ch == "," and d_paren == 0 and d_brack == 0 and d_brace == 0:
            out.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)

    tail = "".join(buf).strip()
    if tail:
        out.append(tail)
    return out


# ---------------------------------------------------------------------
# SymPy equivalence (best-effort)
# ---------------------------------------------------------------------


def _try_sympy_equiv(pred: str, gold: str) -> Tuple[bool, str]:
    """
    Best-effort symbolic equivalence via SymPy.

    Returns:
      (ok, detail)
      detail starts with:
        - "sympy_unavailable: ..."
        - "sympy_parse_or_simplify_failed: ..."
        - or a short method tag
    """
    p = ("" if pred is None else str(pred)).strip()
    g = ("" if gold is None else str(gold)).strip()

    if _ws_eq(p, g):
        return True, "norm_str_eq"

    try:
        import sympy  # type: ignore
        from sympy.parsing.sympy_parser import (  # type: ignore
            convert_xor,
            implicit_multiplication_application,
            parse_expr,
            standard_transformations,
        )
    except Exception as e:
        return False, f"sympy_unavailable: {type(e).__name__}: {e}"

    transformations = standard_transformations + (implicit_multiplication_application, convert_xor)

    def _parse(expr: str):
        # Avoid catastrophic eager eval during parsing (e.g., 2**99999999).
        expr = ("" if expr is None else str(expr)).strip()
        _guard_expr_str(expr)
        try:
            return parse_expr(expr, transformations=transformations, evaluate=False)
        except TypeError:
            # Older sympy may not support evaluate=.
            return parse_expr(expr, transformations=transformations)

    def _is_const_nonzero(x) -> bool:
        try:
            return x != 0 and getattr(x, "free_symbols", set()) == set()
        except Exception:
            return False

    def _sympy_clear_cache() -> None:
        try:
            from sympy.core.cache import clear_cache  # type: ignore

            clear_cache()
        except Exception:
            pass

    def _log_fail(kind: str, msg: str) -> None:
        global _FAIL_LOG_BUDGET
        if _FAIL_LOG_BUDGET <= 0:
            return
        _FAIL_LOG_BUDGET -= 1
        logger.warning(
            "[marft][sympy] %s: %s | pred(len=%d,pow=%d,head=%r) gold(len=%d,pow=%d,head=%r)",
            kind,
            msg,
            len(p),
            p.count("**") + p.count("^"),
            p[:200],
            len(g),
            g.count("**") + g.count("^"),
            g[:200],
        )

    def _impl() -> Tuple[bool, str]:
        # 1) Tuple / multi-value answers
        p_parts = _split_top_level_commas(_strip_outer_parens(p))
        g_parts = _split_top_level_commas(_strip_outer_parens(g))
        if len(p_parts) > 1 or len(g_parts) > 1:
            if len(p_parts) != len(g_parts):
                return False, "tuple_len_mismatch"
            for i, (ps, gs) in enumerate(zip(p_parts, g_parts)):
                ps, gs = ps.strip(), gs.strip()
                if _ws_eq(ps, gs):
                    continue
                pe = _parse(ps)
                ge = _parse(gs)
                if sympy.simplify(pe - ge) != 0:
                    return False, f"tuple_elem{i}_neq"
            return True, "tuple_all"

        # 2) Interval notation: (a,b], [a,b), etc
        mp = _INTERVAL_RE.match(p)
        mg = _INTERVAL_RE.match(g)
        if mp and mg:
            lp, ap, bp, rp = mp.groups()
            lg, ag, bg, rg = mg.groups()

            ap_expr = _parse(ap)
            bp_expr = _parse(bp)
            ag_expr = _parse(ag)
            bg_expr = _parse(bg)

            if sympy.simplify(ap_expr - ag_expr) != 0:
                return False, "interval_left_mismatch"
            if sympy.simplify(bp_expr - bg_expr) != 0:
                return False, "interval_right_mismatch"
            if lp != lg or rp != rg:
                return False, "interval_open_closed_mismatch"
            return True, "interval"

        # 3) Equations (single '='): compare (lhs-rhs), allow non-zero const scaling
        if p.count("=") == 1 and g.count("=") == 1:
            pl, pr = [x.strip() for x in p.split("=", 1)]
            gl, gr = [x.strip() for x in g.split("=", 1)]

            p_expr = _parse(f"({pl})-({pr})")
            g_expr = _parse(f"({gl})-({gr})")

            if sympy.simplify(p_expr - g_expr) == 0:
                return True, "eq_same"

            sp = sympy.simplify(p_expr)
            sg = sympy.simplify(g_expr)

            if sg == 0 and sp == 0:
                return True, "eq_both_zero"
            if sg == 0 or sp == 0:
                return False, "eq_one_zero"

            try:
                ratio = sympy.simplify(sp / sg)
                if _is_const_nonzero(ratio):
                    return True, "eq_scaled"
            except Exception:
                pass

            return False, "eq_neq"

        # 4) Scalar expressions
        pe = _parse(p)
        ge = _parse(g)
        return (sympy.simplify(pe - ge) == 0), "sympy_eq"

    try:
        with _time_limit(_SYM_TIMEOUT_S):
            return _impl()
    except _SympyTimeout as e:
        _sympy_clear_cache()
        _log_fail("timeout", str(e))
        return False, "sympy_parse_or_simplify_failed:timeout"
    except _SympyGuard as e:
        _sympy_clear_cache()
        _log_fail("guard", str(e))
        return False, f"sympy_parse_or_simplify_failed:{e}"
    except Exception as e:
        _sympy_clear_cache()
        msg = str(e)
        if len(msg) > 200:
            msg = msg[:200] + "...(truncated)"
        _log_fail(type(e).__name__, msg)
        return False, f"sympy_parse_or_simplify_failed:{type(e).__name__}:{msg}"


# ---------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------

def _maybe_extract_with_qwen(pred: str, *, data_name: str = "math") -> str:
    try:
        from .parse_utils_qwen import extract_answer as _extract  # type: ignore
    except Exception:
        return ""
    try:
        # 对 MATH 类任务默认不启用 last-number（符号答案更多）
        return str(_extract(pred or "", data_name, use_last_number=False) or "").strip()
    except Exception:
        return ""


def _strip_obvious_noise(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.replace("<|im_start|>", "").replace("<|im_end|>", "")
    s = s.replace("im_start", " ").replace("im_end", " ")
    s = re.sub(r"\*\*\s*(actor|reasoner|assistant|system)\s*\*\*", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\b(actor|reasoner)\s*:\s*", " ", s, flags=re.IGNORECASE)
    return s.strip()


def score_math_marft(
    *,
    prediction: str,
    label: str,
    meta: Dict[str, Any],
    use_math_verify: bool = False,
) -> Tuple[float, Dict[str, Any]]:
    from c3.envs.math.parsing import parse_math_answer

    pred_tok, pred_method = parse_math_answer(prediction or "")
    gold_tok, gold_method = parse_math_answer(label or "")

    pred_raw = _strip_obvious_noise((pred_tok or "").strip())
    gold_raw = _strip_obvious_noise((gold_tok or "").strip())

    info: Dict[str, Any] = {
        "env": "MathEnv",
        "backend": "marft",
        "use_math_verify": bool(use_math_verify),
        "pred_method": pred_method,
        "gold_method": gold_method,
        "pred_raw": pred_raw,
        "gold_raw": gold_raw,
    }

    if not pred_raw or not gold_raw:
        info.update({"match": False, "reason": "empty_extracted_pred_or_label"})
        return 0.0, info

    # 若输出包含 boxed / 明确模板答案，优先用成熟 extractor
    pred_qwen = _maybe_extract_with_qwen(prediction or "", data_name="math")
    gold_qwen = _maybe_extract_with_qwen(label or "", data_name="math")
    if pred_qwen:
        info["pred_qwen"] = pred_qwen
        pred_raw = pred_qwen
    if gold_qwen:
        info["gold_qwen"] = gold_qwen
        gold_raw = gold_qwen

    # 1) 鲁棒判等（tuple/浮点/latex->text 等）
    try:
        from .verify_utils import grade_answer  # type: ignore
        ok_grade = bool(grade_answer(pred_raw, gold_raw))
        info["grade_answer"] = ok_grade
        if ok_grade:
            info.update({"match": True, "method": "grade_answer"})
            return 1.0, info
    except Exception as e:
        info["grade_answer_error"] = f"{type(e).__name__}: {e}"

    # 2) 可选 math-verify
    if use_math_verify:
        try:
            from .math_verify import compute_score  # type: ignore
            mv = float(compute_score(prediction or "", gold_raw))
            info["math_verify_score"] = mv
            if mv >= 0.999:
                info.update({"match": True, "method": "math_verify"})
                return 1.0, info
        except ImportError as e:
            info["math_verify_unavailable"] = str(e)
        except Exception as e:
            info["math_verify_error"] = f"{type(e).__name__}: {e}"

    # 3) SymPy 等价（你现有的 guard + timeout 兜底）
    pred_norm = normalize_expr(pred_raw)
    gold_norm = normalize_expr(gold_raw)
    info["pred_norm"] = pred_norm
    info["gold_norm"] = gold_norm

    if not pred_norm.strip() or not gold_norm.strip():
        raise ImportError("marft_normalize_empty")

    ok, detail = _try_sympy_equiv(pred_norm, gold_norm)
    info["method"] = detail
    info["match"] = bool(ok)

    if (not ok) and (detail.startswith("sympy_unavailable") or detail.startswith("sympy_parse_or_simplify_failed")):
        # 让 reward.py 回退 simple
        raise ImportError(detail)

    return (1.0 if ok else 0.0), info
