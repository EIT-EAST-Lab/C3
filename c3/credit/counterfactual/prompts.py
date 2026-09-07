# -*- coding: utf-8 -*-
"""
Prompt builders for C3 critic.

We standardize how to serialize:
- user prompt (question)
- role outputs (trajectory)
into a single text query for the critic model.

This keeps the implementation simple and compatible with existing reward/critic APIs.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


def build_critic_query(
    *,
    prompt: str,
    role_outputs: Mapping[str, str],
    roles_topo: Optional[Sequence[str]] = None,
    task_name: Optional[str] = None,
    env_name: Optional[str] = None,
    extra_header: Optional[str] = None,
) -> str:
    """
    Build a plain-text critic query.

    Format is intentionally explicit and stable:
      [Task=...][Env=...]
      # Question
      ...
      # Roles (topological order)
      - roleA:
        <text>
      - roleB:
        <text>

    Args:
      roles_topo: ordering of roles. If None, uses sorted(role_outputs.keys()).
    """
    lines = []

    hdr = []
    if task_name:
        hdr.append(f"Task={task_name}")
    if env_name:
        hdr.append(f"Env={env_name}")
    if hdr:
        lines.append("[" + "][".join(hdr) + "]")

    if extra_header:
        lines.append(str(extra_header).strip())

    lines.append("# Question")
    lines.append(str(prompt).rstrip())

    order = list(roles_topo) if roles_topo is not None else sorted(role_outputs.keys())

    lines.append("# Roles")
    for r in order:
        txt = role_outputs.get(r, "")
        lines.append(f"- {r}:")
        # indent role content for readability
        for ln in str(txt).splitlines():
            lines.append(f"  {ln}")
        if not str(txt).endswith("\n"):
            # keep a visible separation between roles
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def replace_role_output(
    role_outputs: Mapping[str, str],
    role: str,
    new_text: str,
) -> Dict[str, str]:
    """Return a shallow-copied dict with one role replaced."""
    d = dict(role_outputs)
    d[str(role)] = str(new_text)
    return d
