# -*- coding: utf-8 -*-
"""
C3 text sanitizers (stdlib-only).

Purpose:
- Remove hallucinated tool execution transcripts (```python/```output blocks),
  and obvious error/traceback lines that can pollute MathEnv context/state/reward parsing.
- Remove common chat template special tokens / role markers that frequently leak into model outputs
  (e.g., <|im_end|>, **Actor**, "Actor:", etc.) without touching math content.

Notes:
- We intentionally DO NOT rewrite LaTeX fractions (e.g., \\frac{1}{2}) here.
  Fraction/LaTeX normalization belongs to MathEnv parsing / MARFT normalization.
"""

from __future__ import annotations

import re
from typing import Optional

# -----------------------------------------------------------------------------
# Patterns
# -----------------------------------------------------------------------------

_RE_FENCE = re.compile(
    r"```(?P<lang>[A-Za-z0-9_-]*)\s*\n(?P<body>[\s\S]*?)```",
    re.MULTILINE,
)

# Keep this intentionally broad: if a line looks like stack trace / exception noise, drop it.
_RE_ERROR_LINE = re.compile(r"\b(?:Traceback|Exception|[A-Za-z_]*Error)\b")

# Unwrap common LaTeX text wrappers (units / prose) by keeping their body.
# (Do not aggressively delete content here; math-level normalization belongs to MathEnv parsing / MARFT.)
# Best-effort only (no nested braces).
_RE_LATEX_TEXT_BLOCK = re.compile(
    r"\\(?:text|mathrm|mathbf|textbf|textit|operatorname)\s*\{\s*([^{}]*?)\s*\}"
)

# Special tokens commonly seen in chat templates / tokenizer outputs.
# We remove them as standalone markers; we do NOT try to strip arbitrary angle-bracket text.
_SPECIAL_TOKENS = (
    "<|im_start|>",
    "<|im_end|>",
    "<|assistant|>",
    "<|user|>",
    "<|system|>",
    "<|tool|>",
    "<|endoftext|>",
    "</s>",
    "<s>",
)

# Markdown/role headers that frequently precede answers.
# Example: "**Actor** 411 im_end"  -> "411"
_RE_MD_ROLE_HEADER = re.compile(r"\*\*\s*(actor|reasoner|assistant|system|user)\s*\*\*", re.IGNORECASE)

# Simple "Role:" prefixes.
_RE_ROLE_PREFIX = re.compile(r"^\s*(actor|reasoner|assistant|system|user)\s*:\s*", re.IGNORECASE)

# Some models emit "Final Answer:" or similar; removing the prefix is safe.
_RE_ANSWER_PREFIX = re.compile(r"^\s*(final\s+answer|answer|final)\s*:\s*", re.IGNORECASE)

# Some templates emit separators.
_RE_SEP_LINE = re.compile(r"^\s*(?:={3,}|-{3,}|_{3,}|\*{3,})\s*$")


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def sanitize_math_solution_text(text: Optional[str]) -> str:
    """Remove tool transcripts + obvious error lines + template tokens from math solution text (best-effort).

    Philosophy:
    - Prefer removing *markers* (tokens/role headers/tracebacks) while keeping the rest.
    - Avoid aggressive regexes that could delete legitimate math (e.g., parentheses, brackets, LaTeX).
    """
    s = "" if text is None else str(text)

    # 1) Unwrap \text{...} etc early to reduce stray words in prose that can confuse downstream parsers.
    s = _RE_LATEX_TEXT_BLOCK.sub(lambda m: f" {m.group(1)} ", s)

    # 2) Remove common chat special tokens (exact matches only).
    for tok in _SPECIAL_TOKENS:
        s = s.replace(tok, " ")

    # Some tokenizers leak bare "im_start"/"im_end" without brackets.
    # Only remove these as whole words to avoid harming legitimate substrings.
    s = re.sub(r"\bim_start\b", " ", s)
    s = re.sub(r"\bim_end\b", " ", s)

    # 3) Remove markdown role headers anywhere they appear (often injected in the middle).
    s = _RE_MD_ROLE_HEADER.sub(" ", s)

    # 4) Remove fenced tool/output blocks that are very likely irrelevant to MathEnv scoring.
    def _fence_repl(m: "re.Match[str]") -> str:
        lang = (m.group("lang") or "").strip().lower()
        body = m.group("body") or ""

        # Drop tool-like blocks (python/output) completely.
        if lang in {"python", "py", "output", "bash", "sh"}:
            return "\n"

        # Drop blocks that look like tracebacks/errors.
        if _RE_ERROR_LINE.search(body):
            return "\n"

        # Otherwise keep (could contain useful derivations).
        return m.group(0)

    s = _RE_FENCE.sub(_fence_repl, s)

    # 5) Line-wise cleanup: remove obvious error lines; trim role prefixes only at line start.
    kept = []
    for raw_line in s.splitlines():
        line = raw_line

        # Drop pure separator lines (often from templates/logs).
        if _RE_SEP_LINE.match(line):
            continue

        # Drop traceback/exception noise lines.
        if _RE_ERROR_LINE.search(line):
            continue

        # Remove leading role prefixes safely.
        line = _RE_ROLE_PREFIX.sub("", line)

        # Remove leading "Final Answer:" / "Answer:" prefixes (safe).
        line = _RE_ANSWER_PREFIX.sub("", line)

        kept.append(line)

    s = "\n".join(kept)

    # 6) Normalize whitespace a bit (without collapsing newlines too aggressively).
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()

    return s
