from __future__ import annotations

import pytest

from c3.integration.marl_specs import RoleSpec
from c3.mas.role_graph import RoleGraph


def test_role_graph_topo_order_for_chain() -> None:
    roles = [
        RoleSpec(name='reasoner', prompt='r', with_answer=False),
        RoleSpec(name='actor', prompt='a', with_answer=True, depends_on=('reasoner',)),
    ]
    graph = RoleGraph(roles)
    assert graph.topo_order() == ['reasoner', 'actor']
    assert graph.layers() == [['reasoner'], ['actor']]
    assert graph.parents('actor') == ['reasoner']


def test_role_graph_rejects_missing_dependency() -> None:
    roles = [RoleSpec(name='actor', prompt='a', with_answer=True, depends_on=('reasoner',))]
    with pytest.raises(ValueError, match='unknown role'):
        RoleGraph(roles)


def test_role_graph_rejects_cycle() -> None:
    roles = [
        RoleSpec(name='reasoner', prompt='r', with_answer=False, depends_on=('actor',)),
        RoleSpec(name='actor', prompt='a', with_answer=True, depends_on=('reasoner',)),
    ]
    with pytest.raises(ValueError, match='cycle'):
        RoleGraph(roles)
