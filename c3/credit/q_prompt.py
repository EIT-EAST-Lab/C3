# -*- coding: utf-8 -*-
"""c3.credit.q_prompt

The text the centralized Q critic reads.

`extract_question` pulls the question out of an observation, and `format_for_q`
renders that question followed by the role answers in DAG-layer order, either
whole or cut to a prefix, optionally keeping only the ancestors of one role. The
layer order itself comes from `role_dag`.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .role_dag import _collect_ancestors

# Capture content after "Question:" (case-insensitive, dot matches newline).
_Q_BLOCK = re.compile(r"(?is)question\s*:\s*(.+)")
# Split off common trailing delimiters/headings that often follow question statements.
_SPLIT_TAIL = re.compile(r"\n\s*(?:---+|###|\*\*|Assistant|System|User|[A-Za-z _]*Answer[:：])\s*|\Z")


def _extract_question_from_text(text: str) -> str:
    """
    Best-effort extraction for legacy prompt formats that embed:
      "Question: ...\n<other sections>"
    """
    m = _Q_BLOCK.search(text)
    if not m:
        return text
    q = m.group(1).strip()
    q = _SPLIT_TAIL.split(q, maxsplit=1)[0].strip()
    return q if q else text


def extract_question(observation: Any) -> str:
    """
    Extract question text from an observation.

    Supported inputs:
      - New format: {"question": "..."} -> direct
      - Legacy dict: {role: prompt_with_question} -> regex capture after "Question:"
      - Otherwise: str(observation)
    """
    if isinstance(observation, dict):
        if "question" in observation:
            return str(observation["question"])

        if observation:
            # Legacy: dict mapping roles -> prompts; take the first value.
            first = str(next(iter(observation.values())))
            return _extract_question_from_text(first)

    return str(observation)


def format_for_q(
    question: str,
    actions: Dict[str, str],
    *,
    mode: str = "full",
    up_to_role: Optional[str] = None,
    layers: Optional[List[List[str]]] = None,
    parents: Optional[Dict[str, List[str]]] = None,
    prefix_scope: str = "topo_prefix",
    strict: bool = True,
) -> str:
    """
    Build critic/Q input text. Order is stable by DAG layers.

    mode:
      - "full": include all roles present in `actions` in DAG order
      - "prefix": include roles up to `up_to_role` (inclusive)

    prefix_scope (only for mode="prefix"):
      - "topo_prefix" (default): include all roles encountered before up_to_role in DAG-layer order
        (may include parallel roles not on the ancestor chain).
      - "ancestors_only": include only ancestors of up_to_role (and up_to_role itself), excluding unrelated parallel roles.
        Requires `parents` when strict=True.
    """
    mode_s = str(mode or "").lower().strip()
    if mode_s not in ("full", "prefix"):
        raise ValueError(f"format_for_q.mode must be 'full' or 'prefix', got {mode!r}")

    if strict and layers is None:
        raise ValueError("format_for_q requires DAG layers when strict=True (provide layers).")

    ps = str(prefix_scope or "topo_prefix").lower().strip()
    if ps not in ("topo_prefix", "ancestors_only"):
        raise ValueError(f"format_for_q.prefix_scope must be 'topo_prefix' or 'ancestors_only', got {prefix_scope!r}")

    if mode_s == "prefix":
        if up_to_role is None:
            raise ValueError("format_for_q(mode='prefix') requires up_to_role.")

        if strict and layers is not None:
            all_roles = {r for layer in layers for r in layer}
            if up_to_role not in all_roles:
                raise ValueError(
                    f"format_for_q(mode='prefix') up_to_role={up_to_role!r} not found in layers roles={sorted(all_roles)}"
                )

        if ps == "ancestors_only" and strict and parents is None:
            raise ValueError("format_for_q(prefix_scope='ancestors_only') requires `parents` when strict=True.")

    # ancestors_only is only meaningful for prefix mode.
    ancestors: Optional[set] = None
    if mode_s == "prefix" and ps == "ancestors_only":
        ancestors = _collect_ancestors(str(up_to_role), parents)

    parts: List[str] = [f"Question: {question}\n\n"]

    # strict=False fallback: no DAG ordering provided; use sorted keys (not recommended).
    if layers is None:
        for r in sorted(actions.keys()):
            parts.append(f"--- {r}'s Answer ---\n{actions.get(r, '')}\n\n")
            if mode_s == "prefix" and up_to_role == r:
                break
        return "".join(parts)

    # Layered DAG order.
    for layer in layers:
        for r in layer:
            if mode_s == "prefix" and ancestors is not None:
                # Keep only ancestors + the target role itself.
                if (r != up_to_role) and (r not in ancestors):
                    continue

            if r in actions:
                parts.append(f"--- {r}'s Answer ---\n{actions[r]}\n\n")

            if mode_s == "prefix" and up_to_role == r:
                return "".join(parts)

    return "".join(parts)
