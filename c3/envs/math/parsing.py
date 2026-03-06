# -*- coding: utf-8 -*-
"""
MathEnv parsing helpers (migrated from C3).

Public (C3-compatible):
- parse_numeric_answer(text) -> Optional[str]
- parse_answer(text) -> Optional[str]
- numeric_equal(gt_raw, pred_token) -> bool
- compute_accuracy(ground_truth, predicted_answer) -> float
- most_frequent(lst) -> Optional[str]      # tie-break: lexicographically smallest
- enforce_final_numeric_line(text, style="hash"|"boxed", only_if_missing=True) -> str

Compatibility (used by current math/reward.py before FS-008):
- parse_math_answer(text) -> (Optional[str], method: str)
- normalize_math_answer(ans) -> (normalized_str, Optional[Fraction])
- extract_last_boxed(text) / extract_hash_answer(text)
"""

from __future__ import annotations

import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import List, Optional, Tuple

__all__ = [
    "parse_numeric_answer",
    "parse_answer",
    "numeric_equal",
    "compute_accuracy",
    "most_frequent",
    "enforce_final_numeric_line",
    # compatibility
    "parse_math_answer",
    "normalize_math_answer",
    "extract_last_boxed",
    "extract_hash_answer",
]

# -----------------------------------------------------------------------------
# Numeric token patterns
# -----------------------------------------------------------------------------

# Allow thousand separators: 1,234.56; also allow plain 1234.56.
# IMPORTANT: do not match the numerator of a fraction "1/2" as a decimal token.
_NUM_DECIMAL = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?!\s*/)"

# Fraction: 11/2 (no mixed numbers); allow spaces around '/'
_NUM_FRACTION = r"\d+\s*/\s*\d+"

# Mixed number: 1 1/2 or -1 1/2 (sign could be at the front)
_NUM_MIXED = r"([-+]?\d+)\s+(\d+)\s*/\s*(\d+)"

# IMPORTANT: fraction first, then decimal, to avoid "1/2" => "2"
_NUM_TOKEN = rf"[-+]?(?:{_NUM_FRACTION}|{_NUM_DECIMAL})"

_RE_NUM_TOKEN = re.compile(_NUM_TOKEN)
_RE_HASH_TOKEN = re.compile(rf"####\s*({_NUM_TOKEN})")

# Common unicode minus chars
_MINUS_CHARS = "\u2212\u2012\u2013\u2014\u2010"  # − ‒ – — ‐

# Whole #### line (not numeric-only); we will take the LAST occurrence.
_RE_HASH_LINE = re.compile(r"(?m)^\s*####\s*(.+?)\s*$")

# "Final Answer:" style anchors (take LAST occurrence)
_RE_FINAL_ANCHOR_LINE = re.compile(
    r"(?im)^\s*(?:final answer|the answer is|answer|答案是|最后答案|最终答案)\s*[:：]\s*(.+?)\s*$"
)
_RE_FINAL_ANCHOR_INLINE = re.compile(
    r"(?i)(?:final answer|the answer is|answer|答案是|最后答案|最终答案)\s*[:：]\s*(.+)$"
)

# -----------------------------------------------------------------------------
# Fraction-like normalization (LaTeX / parenthesized integer forms)
# -----------------------------------------------------------------------------

_RE_LATEX_FRAC_INT_BRACE = re.compile(
    r"\\(?:d?frac|tfrac)\s*\{\s*([+-]?\d+)\s*\}\s*\{\s*([+-]?\d+)\s*\}"
)
_RE_LATEX_FRAC_INT_SPACE = re.compile(
    r"\\(?:d?frac|tfrac)\s+([+-]?\d+)\s+([+-]?\d+)"
)
_RE_PAREN_FRAC_INT = re.compile(
    r"\(\s*([+-]?\d+)\s*\)\s*/\s*\(\s*([+-]?\d+)\s*\)"
)
_RE_PAREN_NUM = re.compile(r"\(\s*([+-]?\d+)\s*\)")

# -----------------------------------------------------------------------------
# Tail cleanup (interval-safe)
# -----------------------------------------------------------------------------

_TRAIL_PUNCT = " \t\r\n.。；;，,"


def _strip_trailing_junk(s: str) -> str:
    """
    Trim benign trailing punctuation/spaces, and ONLY remove obviously-unmatched
    closing delimiters.

    IMPORTANT: keep valid interval closers like (a,b] and [a,b).
    """
    s = (s or "").strip()
    if not s:
        return s

    # 1) strip trailing punctuation/spaces, then strip again
    s = s.rstrip(_TRAIL_PUNCT).strip()
    if not s:
        return s

    def _has_any_left_interval_delim(t: str) -> bool:
        # Interval notation may be "(a,b]" or "[a,b)" etc.
        return ("(" in t) or ("[" in t)

    # 2) strip only *obviously* unmatched trailing closers
    changed = True
    while changed and s:
        changed = False

        # Unmatched '}' (common wrapper artifact like "\\boxed{18}}")
        if s.endswith("}") and ("{" not in s):
            s = s[:-1].rstrip(_TRAIL_PUNCT).strip()
            changed = True
            continue

        # Unmatched ']' — DO NOT remove if we have '(' or '[' anywhere (intervals!)
        if s.endswith("]") and (not _has_any_left_interval_delim(s)):
            s = s[:-1].rstrip(_TRAIL_PUNCT).strip()
            changed = True
            continue

        # Unmatched ')' — DO NOT remove if we have '(' or '[' anywhere (intervals!)
        if s.endswith(")") and (not _has_any_left_interval_delim(s)):
            s = s[:-1].rstrip(_TRAIL_PUNCT).strip()
            changed = True
            continue

    return s


# -----------------------------------------------------------------------------
# Low-level cleanup
# -----------------------------------------------------------------------------

def _normalize_fraction_like(s: str) -> str:
    """Best-effort normalization so \\frac{1}{2}, (1)/(2) become 1/2 (integer-only)."""
    if not s:
        return s
    s = _RE_LATEX_FRAC_INT_BRACE.sub(r"\1/\2", s)
    s = _RE_LATEX_FRAC_INT_SPACE.sub(r"\1/\2", s)
    s = _RE_PAREN_FRAC_INT.sub(r"\1/\2", s)
    # (12) -> 12 (avoid "(1)/(2)" being extracted as trailing "2")
    s = _RE_PAREN_NUM.sub(r"\1", s)
    return s


def _strip_commas(s: str) -> str:
    """Remove thousands separators and normalize unicode minus to '-'."""
    if not s:
        return s
    s = s.replace(",", "")
    for ch in _MINUS_CHARS:
        s = s.replace(ch, "-")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _sanitize_text(text: str) -> str:
    """Minimal cleaning to improve regex matching."""
    if not text:
        return ""
    return _normalize_fraction_like(_strip_commas(text))


# -----------------------------------------------------------------------------
# Boxed / hash / anchor extractors
# -----------------------------------------------------------------------------

def extract_last_boxed(text: str) -> Optional[str]:
    """Extract content of the last \\boxed{...} (supports nested braces via counting)."""
    if not text:
        return None

    marker = r"\boxed{"
    idx = text.rfind(marker)
    if idx < 0:
        return None

    i = idx + len(marker)
    depth = 1
    out: List[str] = []
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
            out.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                val = _strip_trailing_junk("".join(out).strip())
                return val if val else None
            out.append(ch)
        else:
            out.append(ch)
        i += 1
    return None


def extract_hash_answer(text: str) -> Optional[str]:
    """Take the LAST #### line (not the first)."""
    ms = _RE_HASH_LINE.findall(text or "")
    if not ms:
        return None
    ans = _strip_trailing_junk((ms[-1] or "").strip())
    return ans if ans else None


def extract_final_anchored_answer(text: str) -> Optional[str]:
    """Take the LAST occurrence of 'Final Answer: ...' / 'Answer: ...' anchors."""
    if not text:
        return None
    t = str(text)

    ms = _RE_FINAL_ANCHOR_LINE.findall(t)
    if ms:
        val = _strip_trailing_junk(ms[-1] or "")
        return val if val else None

    ms2 = _RE_FINAL_ANCHOR_INLINE.findall(t)
    if ms2:
        val = _strip_trailing_junk(ms2[-1] or "")
        return val if val else None

    return None


# -----------------------------------------------------------------------------
# Numeric parsing core
# -----------------------------------------------------------------------------

def _extract_candidate_tokens(text: str) -> List[str]:
    """
    Extract likely numeric token candidates from free-form text.

    Priority:
      1) last "#### <num>"
      2) last \\boxed{<...>} (simple regex)
      3) anchors (e.g. "Answer: 3")
      4) last numeric on last 2 non-empty lines
      5) last numeric in full text
    """
    text = _sanitize_text(text)
    if not text:
        return []

    ms = _RE_HASH_TOKEN.findall(text)
    if ms:
        return [_strip_trailing_junk(ms[-1].strip())]

    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        return [_strip_trailing_junk(boxed[-1].strip())]

    anchors = [
        re.compile(rf"(?i)(?:final answer|答案是|最后答案|the answer is|answer:)\s*({_NUM_TOKEN})"),
        re.compile(rf"(?i)(?:equals|=)\s*({_NUM_TOKEN})"),
    ]
    for pat in anchors:
        m = pat.search(text)
        if m:
            return [_strip_trailing_junk(m.group(1).strip())]

    tail = "\n".join([ln for ln in text.strip().splitlines() if ln][-2:])
    candidates = _RE_NUM_TOKEN.findall(tail)
    if candidates:
        return [_strip_trailing_junk(candidates[-1].strip())]

    all_nums = _RE_NUM_TOKEN.findall(text)
    return [_strip_trailing_junk(all_nums[-1].strip())] if all_nums else []


def _to_fraction(token: str) -> Optional[Fraction]:
    """
    Convert token string to Fraction. Supports:
      - mixed: 1 1/2, -1 1/2
      - fraction: 11/2, -11/2
      - decimal/integer: Decimal -> Fraction
    """
    if token is None:
        return None
    s = _strip_commas(token)

    m = re.fullmatch(_NUM_MIXED, s)
    if m:
        sign = -1 if s.startswith("-") else 1
        a, b, c = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return sign * (abs(a) + Fraction(b, c))

    if re.fullmatch(rf"[-+]?{_NUM_FRACTION}", s):
        sign = -1 if s.startswith("-") else 1
        num, den = re.split(r"/", s.replace("+", ""))
        try:
            return sign * Fraction(int(num.strip().lstrip("+-")), int(den.strip()))
        except Exception:
            return None

    try:
        return Fraction(Decimal(s))
    except (InvalidOperation, ValueError):
        return None


def parse_numeric_answer(text: str) -> Optional[str]:
    """
    Extract a normalized numeric token from text. Returns None if extraction fails.
    Token can be integer/decimal/fraction (with optional sign) or mixed number.
    """
    cands = _extract_candidate_tokens(text)
    if not cands:
        return None

    cand = _strip_commas(cands[0])

    inner = _RE_NUM_TOKEN.findall(cand)
    if inner:
        return _strip_trailing_junk(inner[-1].strip()) or None

    m = re.fullmatch(_NUM_MIXED, cand)
    if m:
        return _strip_trailing_junk(cand.strip()) or None

    return None


def parse_answer(text: str) -> Optional[str]:
    return parse_numeric_answer(text)


def numeric_equal(gt_raw: str, pred_token: Optional[str]) -> bool:
    """Exact numeric comparison using Fraction; fallback to float equality."""
    if pred_token is None:
        return False
    gt_tok = parse_numeric_answer(gt_raw)
    if gt_tok is None:
        return False

    gt = _to_fraction(gt_tok)
    pr = _to_fraction(pred_token)

    if gt is None or pr is None:
        try:
            return float(_strip_commas(gt_tok)) == float(_strip_commas(pred_token))
        except Exception:
            return False
    return gt == pr


def compute_accuracy(ground_truth: str, predicted_answer: Optional[str]) -> float:
    """Return 1.0 or 0.0. predicted_answer can be token or free text."""
    if predicted_answer is None:
        return 0.0

    pr = str(predicted_answer).strip()
    pr_norm = _strip_commas(pr)

    if not (re.fullmatch(_NUM_TOKEN, pr_norm) or re.fullmatch(_NUM_MIXED, pr_norm)):
        pr = parse_numeric_answer(pr) or ""
    if not pr:
        return 0.0

    return 1.0 if numeric_equal(ground_truth, pr) else 0.0


def most_frequent(lst: List[Optional[str]]) -> Optional[str]:
    """Majority vote; in tie, lexicographically smallest non-None."""
    valid = [x for x in lst if x is not None]
    if not valid:
        return None
    cnt = Counter(valid)
    max_c = max(cnt.values())
    cands = [k for k, v in cnt.items() if v == max_c]
    return sorted(cands)[0]


def enforce_final_numeric_line(text: str, style: str = "hash", only_if_missing: bool = True) -> str:
    """Append a standardized final answer line if a numeric token is parsable."""
    token = parse_numeric_answer(text)
    if not token:
        return text

    if only_if_missing:
        if style == "boxed":
            if re.search(rf"\\boxed\{{\s*{re.escape(token)}\s*\}}", text):
                return text
        else:
            if re.search(rf"(?m)^\s*####\s*{re.escape(token)}\s*$", text):
                return text

    suffix = f"\\boxed{{{token}}}" if style == "boxed" else f"#### {token}"
    needs_nl = "" if text.endswith("\n") else "\n"
    return f"{text.rstrip()}{needs_nl}{suffix}"


# -----------------------------------------------------------------------------
# OpenRLHF compatibility wrappers (pre-FS-008)
# -----------------------------------------------------------------------------

def parse_math_answer(text: str) -> Tuple[Optional[str], str]:
    """
    Extract a final answer string from possibly long model outputs.

    Notes:
    - This returns a *string answer* (may be expression/LaTeX), not necessarily numeric.
    - Trailing punctuation noise is cleaned via _strip_trailing_junk().
    """
    if not text:
        return None, "empty"

    t = str(text)

    hashed = extract_hash_answer(t)
    if hashed:
        return _strip_trailing_junk(hashed), "hash"

    boxed = extract_last_boxed(t)
    if boxed is not None and str(boxed).strip():
        return _strip_trailing_junk(str(boxed)), "boxed"

    anchored = extract_final_anchored_answer(t)
    if anchored is not None and anchored.strip():
        return _strip_trailing_junk(anchored), "anchor"

    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if not lines:
        return None, "empty_lines"

    last = lines[-1]
    m = _RE_FINAL_ANCHOR_INLINE.search(last)
    if m:
        last2 = _strip_trailing_junk((m.group(1) or "").strip())
        return last2, "anchor_inline"

    return _strip_trailing_junk(last), "last_line"


def normalize_math_answer(ans: str) -> Tuple[str, Optional[Fraction]]:
    """
    Best-effort normalization:
      - If a pure option letter (A/B/C/D), normalize to lowercase letter.
      - If numeric-ish, try Fraction; else return whitespace-insensitive lowercase string.
    """
    s = (ans or "").strip()
    s = s.strip("$ ").strip()
    s = s.rstrip(" \t\r\n.。；;，,").strip()  # IMPORTANT: rstrip then strip

    m = re.fullmatch(r"\(?\s*([A-Za-z])\s*\)?", s)
    if m:
        return m.group(1).lower(), None

    numericish = bool(re.fullmatch(r"[0-9\s,./()+\-]+", s))
    tok = (parse_numeric_answer(s) or _strip_commas(s)) if numericish else _strip_commas(s)

    frac = _to_fraction(tok)
    if frac is not None:
        if frac.denominator == 1:
            return str(frac.numerator), frac
        return f"{frac.numerator}/{frac.denominator}", frac

    s2 = re.sub(r"\s+", "", s)
    return s2.lower(), None
