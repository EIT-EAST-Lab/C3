"""Role DAG helpers for C3 MAS.

We keep this module dependency-light so it can be imported from both training
and standalone tooling without pulling in heavy ML dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from c3.integration.marl_specs import RoleSpec


@dataclass(frozen=True)
class RoleNode:
    name: str
    depends_on: Tuple[str, ...]
    with_answer: bool


class RoleGraph:
    """A directed acyclic graph of roles.

    Notes:
      - Supports N=1 (single role) as a valid special case.
      - Validates missing dependencies and cycles.
    """

    def __init__(self, roles: Sequence[RoleSpec]):
        if not roles:
            raise ValueError("roles list is empty")

        # Validate uniqueness
        names = [r.name for r in roles]
        if len(set(names)) != len(names):
            dup = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"duplicate role names: {dup}")

        self.nodes: Dict[str, RoleNode] = {
            r.name: RoleNode(name=r.name, depends_on=tuple(r.depends_on), with_answer=bool(r.with_answer))
            for r in roles
        }

        # Validate dependencies exist
        for r in roles:
            for dep in r.depends_on:
                if dep not in self.nodes:
                    raise ValueError(f"role {r.name} depends_on unknown role {dep}")

        self._topo: List[str] = self._topo_sort()
        self._layers: List[List[str]] = self._build_layers()

    def topo_order(self) -> List[str]:
        return list(self._topo)

    def layers(self) -> List[List[str]]:
        """Return a layered topological order.

        Each layer contains roles whose dependencies are in previous layers.
        For a simple chain (reasoner->actor), layers will be [[reasoner],[actor]].
        For N=1, layers will be [[role0]].
        """

        return [list(layer) for layer in self._layers]

    def parents(self, role: str) -> List[str]:
        return list(self.nodes[role].depends_on)

    def _topo_sort(self) -> List[str]:
        indeg: Dict[str, int] = {n: 0 for n in self.nodes}
        out_edges: Dict[str, List[str]] = {n: [] for n in self.nodes}

        for n, node in self.nodes.items():
            for dep in node.depends_on:
                out_edges[dep].append(n)
                indeg[n] += 1

        q = [n for n in self.nodes if indeg[n] == 0]
        out: List[str] = []
        while q:
            cur = q.pop(0)
            out.append(cur)
            for nxt in out_edges[cur]:
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    q.append(nxt)

        if len(out) != len(self.nodes):
            stuck = [n for n, d in indeg.items() if d > 0]
            raise ValueError(f"cycle detected in roles depends_on graph: {stuck}")

        return out

    def _build_layers(self) -> List[List[str]]:
        # Kahn-style layering
        indeg: Dict[str, int] = {n: 0 for n in self.nodes}
        out_edges: Dict[str, List[str]] = {n: [] for n in self.nodes}
        for n, node in self.nodes.items():
            for dep in node.depends_on:
                out_edges[dep].append(n)
                indeg[n] += 1

        layers: List[List[str]] = []
        cur = [n for n in self.nodes if indeg[n] == 0]
        while cur:
            layers.append(list(cur))
            nxt_layer: List[str] = []
            for n in cur:
                for nxt in out_edges[n]:
                    indeg[nxt] -= 1
                    if indeg[nxt] == 0:
                        nxt_layer.append(nxt)
            cur = nxt_layer
        return layers
