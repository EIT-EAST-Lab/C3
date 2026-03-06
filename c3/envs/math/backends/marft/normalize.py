# -*- coding: utf-8 -*-
"""
MARFT backend: lightweight, dependency-free math expression normalization.

Goals:
- stdlib-only
- best-effort LaTeX -> "sympy-ish" string
- conservative rewrites (avoid SymPy parse traps)

Main:
  normalize_expr(text: str) -> str
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

_RE_WS = re.compile(r"\s+")
_RE_DOLLAR = re.compile(r"^\$+|\$+$")
_RE_LEFT_RIGHT = re.compile(r"\\left|\\right")
_RE_TEX_SPACES = re.compile(r"(\\[,;:!])|(\\(?:quad|qquad)\b)")

_RE_MULT = re.compile(r"(\\cdot|\\times)\b")
_RE_DIV = re.compile(r"(\\div)\b")
_RE_PI = re.compile(r"\\pi\b")
_RE_INFTY = re.compile(r"\\infty\b")
_RE_PM = re.compile(r"\\pm\b")
_RE_MP = re.compile(r"\\mp\b")
_RE_EQ = re.compile(r"\\,=\\,|\\=")

_RE_LATEX_FUNCS = re.compile(r"\\(sin|cos|tan|cot|sec|csc|log|ln)\b")
_RE_FUNC_NEEDS_PARENS = re.compile(
    r"\b(sin|cos|tan|cot|sec|csc|log|ln)\s+(?!\()([0-9A-Za-z_+\-*/.]+)"
)

_RE_BRACES_AROUND_SINGLE = re.compile(r"\{([A-Za-z0-9_]+)\}")
_RE_LETTER_DOT_LETTER = re.compile(r"([A-Za-z])\.([A-Za-z])")
_RE_NUM_WITH_UNITS = re.compile(
    r"^([+-]?\d+(?:\.\d+)?)(?:\s+[A-Za-z][A-Za-z0-9_]*\.?)+\s*$"
)

_RE_QUOTES = re.compile(r"[\"'`“”‘’′″]")
_RE_ALLOWED = re.compile(r"[^0-9A-Za-z_+\-*/().=^,\[\] \t]")

_TEX_TEXT_BLOCK_CMDS = ("text", "mathrm", "mathbf", "textbf", "textit", "operatorname")

_TRAIL_NOISE = " \t\r\n.。；;，,"


def _squeeze_ws(s: str) -> str:
    return _RE_WS.sub(" ", s).strip()


def _strip_math_wrappers(s: str) -> str:
    s = s.strip()
    s = _RE_DOLLAR.sub("", s).strip()
    s = s.replace(r"\(", "").replace(r"\)", "")
    s = s.replace(r"\[", "").replace(r"\]", "")
    return s.strip()


def _find_braced(s: str, start: int) -> Optional[Tuple[str, int]]:
    if start < 0 or start >= len(s) or s[start] != "{":
        return None
    depth = 0
    i = start
    out = []
    while i < len(s):
        ch = s[i]
        if ch == "{":
            depth += 1
            if depth > 1:
                out.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out), i + 1
            out.append(ch)
        else:
            out.append(ch)
        i += 1
    return None


def _strip_tex_text_blocks(s: str) -> str:
    # Unwrap \text{...} etc by keeping their body (best-effort, no nested brace guarantee).
    for cmd in _TEX_TEXT_BLOCK_CMDS:
        needle = "\\" + cmd
        for _ in range(100):
            idx = s.find(needle)
            if idx < 0:
                break
            j = idx + len(needle)
            while j < len(s) and s[j].isspace():
                j += 1
            if j < len(s) and s[j] == "{":
                br = _find_braced(s, j)
                if br is not None:
                    body, end = br
                    s = s[:idx] + " " + body + " " + s[end:]
                    continue
            s = s[:idx] + " " + s[j:]
    return s


def _replace_frac_once(s: str) -> Tuple[str, bool]:
    idx = s.find(r"\frac")
    if idx < 0:
        return s, False
    j = idx + len(r"\frac")
    while j < len(s) and s[j].isspace():
        j += 1
    if j >= len(s) or s[j] != "{":
        return s, False
    a = _find_braced(s, j)
    if a is None:
        return s, False
    A, k = a
    while k < len(s) and s[k].isspace():
        k += 1
    if k >= len(s) or s[k] != "{":
        return s, False
    b = _find_braced(s, k)
    if b is None:
        return s, False
    B, end = b
    return s[:idx] + f"({A})/({B})" + s[end:], True


def _replace_sqrt_once(s: str) -> Tuple[str, bool]:
    idx = s.find(r"\sqrt")
    if idx < 0:
        return s, False
    j = idx + len(r"\sqrt")
    while j < len(s) and s[j].isspace():
        j += 1

    if j < len(s) and s[j] == "[":
        rb = s.find("]", j + 1)
        if rb < 0:
            return s, False
        root_idx = s[j + 1 : rb].strip()
        j = rb + 1
        while j < len(s) and s[j].isspace():
            j += 1
        if j >= len(s) or s[j] != "{":
            return s, False
        a = _find_braced(s, j)
        if a is None:
            return s, False
        A, end = a
        rep = f"({A})**(1/({root_idx}))" if root_idx else f"sqrt({A})"
        return s[:idx] + rep + s[end:], True

    if j >= len(s) or s[j] != "{":
        return s, False
    a = _find_braced(s, j)
    if a is None:
        return s, False
    A, end = a
    return s[:idx] + f"sqrt({A})" + s[end:], True


def _replace_power(s: str) -> str:
    s = re.sub(r"\^\{([^{}]+)\}", r"**(\1)", s)
    s = re.sub(r"\^([A-Za-z0-9_]+)", r"**(\1)", s)
    return s


def normalize_expr(text: str) -> str:
    """
    Normalize math expression for SymPy parsing (best-effort).

    Highlights:
    - unwraps \text{...}-style blocks (prevents stray numbers)
    - converts \frac, \sqrt, \cdot, \times, ^{...}
    - conservative charset filtering
    """
    s = "" if text is None else str(text)
    s = _strip_math_wrappers(s)

    # trailing noise: rstrip then strip (Patch2)
    s = s.strip()
    s = s.rstrip(_TRAIL_NOISE).strip()

    s = _strip_tex_text_blocks(s)
    s = _RE_LATEX_FUNCS.sub(r"\1", s)

    s = _RE_LEFT_RIGHT.sub("", s)
    s = _RE_TEX_SPACES.sub(" ", s)

    s = _RE_MULT.sub("*", s)
    s = _RE_DIV.sub("/", s)
    s = _RE_PI.sub("pi", s)
    s = _RE_INFTY.sub("oo", s)
    s = _RE_PM.sub("+-", s)
    s = _RE_MP.sub("-+", s)
    s = _RE_EQ.sub("=", s)

    s = _squeeze_ws(s)
    s = _RE_LETTER_DOT_LETTER.sub(r"\1 \2", s)
    s = _RE_FUNC_NEEDS_PARENS.sub(r"\1(\2)", s)

    for _ in range(50):
        s, c1 = _replace_frac_once(s)
        s, c2 = _replace_sqrt_once(s)
        if not (c1 or c2):
            break

    s = _replace_power(s)
    s = _RE_BRACES_AROUND_SINGLE.sub(r"\1", s)
    s = s.replace("{", "(").replace("}", ")")
    s = _squeeze_ws(s)

    s = _RE_QUOTES.sub("", s)
    s = _RE_ALLOWED.sub(" ", s)
    s = _squeeze_ws(s)

    m = _RE_NUM_WITH_UNITS.match(s)
    if m:
        s = m.group(1)

    # final trailing noise insurance
    s = s.rstrip(_TRAIL_NOISE).strip()
    return s
