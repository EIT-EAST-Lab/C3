# -*- coding: utf-8 -*-
"""c3.credit.role_dag

The role dependency graph the credit path reads.

`build_dependency_from_roles` turns a list of RoleSpec into the shapes the
trainer and the provider ask for: parents, Kahn layers, a topological order,
children, the layer index of each role, and the descendants of each role. It is
pure graph structure: no reward, no critic, no counterfactual quantity.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

from c3.task.config import RoleSpec


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
