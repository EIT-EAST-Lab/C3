"""Prompt rendering helpers for MAS.

C3 roles prompts often reference:
  - the original question
  - previous roles' outputs

We intentionally keep rendering conservative and predictable:
  - Supports Python format placeholders like {question}, {context}
  - Also exposes each role's output as {<role>} and {<role>_output}
  - Missing keys are replaced with empty string (no KeyError), so prompts stay usable.
"""

from __future__ import annotations

from typing import Dict, Mapping, List


class _SafeDict(dict):
    def __missing__(self, key):  # type: ignore[override]
        return ""


def build_render_context(
    *,
    question: str,
    role_outputs: Mapping[str, str],
    topo_so_far: List[str],
) -> Dict[str, str]:
    """Build a rendering context for a role prompt."""

    ctx: Dict[str, str] = {"question": question}

    # A simple concatenation of previous role outputs, in topo order so far.
    ctx["context"] = "\n\n".join([role_outputs.get(r, "") for r in topo_so_far if role_outputs.get(r, "")])

    for role, out in role_outputs.items():
        ctx[role] = out
        ctx[f"{role}_output"] = out
    return ctx


def render_role_prompt(role_prompt: str, *, ctx: Mapping[str, str]) -> str:
    """Render a role prompt with best-effort placeholder substitution."""

    # Most C3 prompts are plain text. If they contain braces, treat as format string.
    if "{" in role_prompt and "}" in role_prompt:
        try:
            return role_prompt.format_map(_SafeDict(ctx))
        except Exception:
            # If formatting fails due to unmatched braces etc., fall back to raw.
            return role_prompt
    return role_prompt
