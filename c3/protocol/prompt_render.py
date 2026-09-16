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

from typing import Dict, Mapping, List, Sequence, Set


class _SafeDict(dict):
    def __missing__(self, key):  # type: ignore[override]
        return ""


def ancestors_in_topo_order(
    role: str,
    roles_topo: Sequence[str],
    depends_on: Mapping[str, Sequence[str]],
) -> List[str]:
    """Return the transitive dependencies of `role`, ordered by `roles_topo`.

    This is the context scope used at generation time: a role reads what it
    depends on, directly or indirectly, and nothing else. For a chain the
    result is exactly the topological prefix, so chains are unaffected; for a
    branching workflow the two parallel roles drop out of each other's context.

    Conventions:
      - `role` itself is never included.
      - A name with no entry in `depends_on` is treated as having no parents.
      - An ancestor absent from `roles_topo` is dropped: it has no position in
        the order, and the topological prefix this replaces could not have
        carried it either.
      - Visited parents are remembered, so a malformed cyclic map terminates
        instead of looping forever.
    """

    order = [str(r) for r in roles_topo]
    target = str(role)

    seen: Set[str] = set()
    stack: List[str] = [target]
    while stack:
        cur = stack.pop()
        for parent in depends_on.get(cur, ()) or ():
            p = str(parent)
            if p not in seen:
                seen.add(p)
                stack.append(p)
    seen.discard(target)

    return [r for r in order if r in seen]


def build_render_context(
    *,
    question: str,
    role_outputs: Mapping[str, str],
    topo_so_far: List[str],
) -> Dict[str, str]:
    """Build a rendering context for a role prompt.

    Callers pass the roles whose outputs this role is allowed to read, in
    order. Every generation-time caller passes `ancestors_in_topo_order(...)`.
    """

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
