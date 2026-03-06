# -*- coding: utf-8 -*-
"""c3.credit.c3.baselines

C3 formatting & dependency helpers.

NOTE:
  Legacy diagonal counterfactual group construction (prepare_cf_groups / regenerate / variance gating)
  was removed in the Rule-B nested rollout refactor. Rule-B credit assignment uses sibling-group
  leave-one-out baselines (see provider.py + materialize.materialize_c3_tree_groups).
"""

from __future__ import annotations

import re
from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple

from c3.integration.marl_specs import RoleSpec

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


def _collect_ancestors(up_to_role: str, parents: Optional[Dict[str, List[str]]]) -> Optional[set]:
    """
    Collect all ancestors of `up_to_role` via `parents` adjacency (excluding `up_to_role` itself).
    Returns None if parents is None.
    """
    if parents is None:
        return None

    ancestors: set = set()
    stack = [up_to_role]
    while stack:
        cur = stack.pop()
        for p in parents.get(cur, []) or []:
            if p not in ancestors:
                ancestors.add(p)
                stack.append(p)
    ancestors.discard(up_to_role)
    return ancestors


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


def build_dependency_from_roles(
    roles: Sequence[RoleSpec],
) -> Tuple[Dict[str, List[str]], List[List[str]], List[str], Dict[str, List[str]], Dict[str, int], Dict[str, set]]:
    """
    Build C3 dependency helpers from RoleSpec list.

    Returns:
      parents:        role -> list[parent_role]
      layers:         List[List[role]] topological layers (Kahn layering)
      topo_order:     List[role] topological order
      children:       role -> list[child_role]
      role_to_layer:  role -> layer index
      descendants:    role -> set(descendant roles)
    """
    if not roles:
        raise ValueError("roles is empty")

    names = [r.name for r in roles]
    if len(set(names)) != len(names):
        seen = set()
        dup = []
        for n in names:
            if n in seen:
                dup.append(n)
            seen.add(n)
        raise ValueError(f"duplicate role names passed to build_dependency_from_roles: {sorted(set(dup))}")

    by = {r.name: r for r in roles}

    # parents map (preserve declared depends_on ordering).
    parents: Dict[str, List[str]] = {}
    for n in names:
        deps = getattr(by[n], "depends_on", None) or []
        parents[n] = list(deps)

    # Graph edges + indegree.
    indeg: Dict[str, int] = {n: 0 for n in names}
    out_edges: Dict[str, List[str]] = {n: [] for n in names}
    for child in names:
        for p in parents[child]:
            if p not in by:
                raise ValueError(f"role {child} depends_on unknown role {p}")
            out_edges[p].append(child)
            indeg[child] += 1

    # Topological order (Kahn). Use deque for O(1) pops.
    indeg2 = dict(indeg)
    q = deque([n for n in names if indeg2[n] == 0])  # preserve initial order from `names`
    topo: List[str] = []
    while q:
        cur = q.popleft()
        topo.append(cur)
        for nxt in out_edges[cur]:
            indeg2[nxt] -= 1
            if indeg2[nxt] == 0:
                q.append(nxt)

    if len(topo) != len(names):
        stuck = [n for n in names if indeg2[n] > 0]
        raise ValueError(f"cycle detected in roles depends_on graph: {stuck}")

    # Layered topo (Kahn layering).
    indeg3 = dict(indeg)
    cur_layer: List[str] = [n for n in names if indeg3[n] == 0]
    layers: List[List[str]] = []
    while cur_layer:
        layers.append(list(cur_layer))
        nxt_layer: List[str] = []
        for n in cur_layer:
            for nxt in out_edges[n]:
                indeg3[nxt] -= 1
                if indeg3[nxt] == 0:
                    nxt_layer.append(nxt)
        cur_layer = nxt_layer

    # children map (role -> direct children).
    children: Dict[str, List[str]] = {n: list(out_edges.get(n, [])) for n in names}

    # role_to_layer map.
    role_to_layer: Dict[str, int] = {}
    for i, layer in enumerate(layers):
        for r in layer:
            role_to_layer[r] = i

    # descendants map (role -> transitive descendants set).
    def _collect_desc(root: str) -> set:
        vis: set = set()
        qq = deque([root])
        while qq:
            u = qq.popleft()
            for v in children.get(u, []) or []:
                if v not in vis:
                    vis.add(v)
                    qq.append(v)
        vis.discard(root)
        return vis

    descendants: Dict[str, set] = {r: _collect_desc(r) for r in names}

    return parents, layers, topo, children, role_to_layer, descendants
