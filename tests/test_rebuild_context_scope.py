"""Contract tests for the generation-time context scope.

Appendix 04 says the two Solvers of the branching workflow "work from the task
text and the plan in parallel, neither seeing the other". That sentence is a
claim about what the model reads, so it has to hold in the code that assembles
the prompt, not only in the Q-critic view helper.

The assembly now passes `ancestors_in_topo_order(...)` wherever it used to pass
the topological prefix. For the five chain workflows the two are identical, so
those rows are untouched; only the branching row changes, and it changes into
what the paper already says.

These tests use the dependency-light loaders plus `c3.protocol.prompt_render`,
`c3.protocol.rollout_generator` (import only, no torch call path) and
`c3.analysis.replay`'s prompt renderer with `tokenizer=None`, so they run
without the training stack.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pytest

from c3.task.config import load_task
from c3.protocol.prompt_render import ancestors_in_topo_order, build_render_context
from c3.protocol.role_graph import RoleGraph


REPO_ROOT = Path(__file__).resolve().parents[1]

# The five workflows whose wiring is a chain: for them ancestors == topological prefix.
CHAIN_WORKFLOWS = {
    "a2": "configs/tasks/math.yaml",
    "a3": "configs/tasks/math_a3.yaml",
    "mt4": "configs/tasks/math_mt4.yaml",
    "c5": "configs/tasks/math_c5.yaml",
    "c10": "configs/tasks/math_c10.yaml",
}

CHAIN_IDS = sorted(CHAIN_WORKFLOWS)

BRANCH_TASK = "configs/tasks/math_branch.yaml"


def _spec(task_yaml: str):
    return load_task(str(REPO_ROOT / task_yaml))


def _topo_and_depends(task_yaml: str) -> Tuple[List[str], Dict[str, Tuple[str, ...]]]:
    spec = _spec(task_yaml)
    topo = RoleGraph(spec.roles).topo_order()
    depends_on = {r.name: tuple(r.depends_on) for r in spec.roles}
    return topo, depends_on


# ---------------------------------------------------------------------------
# 1. The helper itself
# ---------------------------------------------------------------------------


def test_ancestors_follow_the_dependency_edges_transitively() -> None:
    topo = ["a", "b", "c"]
    depends_on = {"a": (), "b": ("a",), "c": ("b",)}

    assert ancestors_in_topo_order("a", topo, depends_on) == []
    assert ancestors_in_topo_order("b", topo, depends_on) == ["a"]
    assert ancestors_in_topo_order("c", topo, depends_on) == ["a", "b"]


def test_ancestors_never_include_the_role_itself() -> None:
    topo = ["a", "b"]
    # A self edge is malformed input; it must not put the role in its own context.
    assert ancestors_in_topo_order("b", topo, {"b": ("a", "b")}) == ["a"]


def test_ancestors_are_ordered_by_roles_topo_not_by_the_dependency_map() -> None:
    topo = ["first", "second", "third", "target"]
    depends_on = {"target": ("third", "first", "second")}

    assert ancestors_in_topo_order("target", topo, depends_on) == ["first", "second", "third"]


def test_a_role_missing_from_the_dependency_map_has_no_ancestors() -> None:
    assert ancestors_in_topo_order("unknown", ["a", "b"], {"a": (), "b": ("a",)}) == []


def test_an_ancestor_absent_from_roles_topo_is_dropped() -> None:
    # It has no position in the order, and the prefix this replaces could not
    # have carried it either.
    assert ancestors_in_topo_order("b", ["b"], {"b": ("ghost",)}) == []


def test_a_cyclic_dependency_map_terminates() -> None:
    depends_on = {"a": ("b",), "b": ("a",)}
    assert ancestors_in_topo_order("a", ["a", "b"], depends_on) == ["b"]


# ---------------------------------------------------------------------------
# 2. Chains are untouched
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("workflow", CHAIN_IDS)
def test_chain_workflow_ancestors_equal_the_topological_prefix(workflow: str) -> None:
    """The five chain rows keep byte-identical prompts under the new scope."""
    topo, depends_on = _topo_and_depends(CHAIN_WORKFLOWS[workflow])

    for index, role in enumerate(topo):
        ancestors = ancestors_in_topo_order(role, topo, depends_on)
        assert set(ancestors) == set(topo[:index]), f"{workflow}:{role}"
        assert ancestors == list(topo[:index]), f"{workflow}:{role}"


@pytest.mark.parametrize("workflow", CHAIN_IDS)
def test_chain_workflow_is_really_a_chain(workflow: str) -> None:
    """Guards the test above: it would pass trivially on a mis-specified graph."""
    topo, depends_on = _topo_and_depends(CHAIN_WORKFLOWS[workflow])

    assert depends_on[topo[0]] == ()
    for previous, role in zip(topo, topo[1:]):
        assert depends_on[role] == (previous,), f"{workflow}:{role}"


def test_branching_is_the_only_workflow_whose_scope_changes() -> None:
    topo, depends_on = _topo_and_depends(BRANCH_TASK)

    differing = [
        role
        for index, role in enumerate(topo)
        if ancestors_in_topo_order(role, topo, depends_on) != list(topo[:index])
    ]
    assert differing == ["solver_b"]


# ---------------------------------------------------------------------------
# 3. Branching: the two solvers are apart, the integrator joins them
# ---------------------------------------------------------------------------


def test_branching_ancestors_keep_the_solvers_apart() -> None:
    topo, depends_on = _topo_and_depends(BRANCH_TASK)

    assert ancestors_in_topo_order("planner", topo, depends_on) == []
    assert ancestors_in_topo_order("solver_a", topo, depends_on) == ["planner"]
    assert ancestors_in_topo_order("solver_b", topo, depends_on) == ["planner"]
    assert ancestors_in_topo_order("integrator", topo, depends_on) == [
        "planner",
        "solver_a",
        "solver_b",
    ]


def _branch_outputs() -> Dict[str, str]:
    return {
        "planner": "PLAN-TEXT",
        "solver_a": "SOLVER-A-TEXT",
        "solver_b": "SOLVER-B-TEXT",
        "integrator": "INTEGRATOR-TEXT",
    }


def _generation_context(role: str, topo: Sequence[str], depends_on) -> str:
    """The `ctx["context"]` string as the generation path now builds it."""
    return build_render_context(
        question="Q",
        role_outputs=_branch_outputs(),
        topo_so_far=ancestors_in_topo_order(role, topo, depends_on),
    )["context"]


def test_generation_context_of_solver_b_excludes_solver_a() -> None:
    topo, depends_on = _topo_and_depends(BRANCH_TASK)
    context = _generation_context("solver_b", topo, depends_on)

    assert "PLAN-TEXT" in context
    assert "SOLVER-A-TEXT" not in context


def test_generation_context_of_the_integrator_carries_both_solvers() -> None:
    topo, depends_on = _topo_and_depends(BRANCH_TASK)
    context = _generation_context("integrator", topo, depends_on)

    assert "PLAN-TEXT" in context
    assert "SOLVER-A-TEXT" in context
    assert "SOLVER-B-TEXT" in context


# ---------------------------------------------------------------------------
# 4. The replay-side prompt renderer
# ---------------------------------------------------------------------------


def _replay_renderer(task_yaml: str):
    from c3.analysis.replay import _OpenRLHFPromptRenderer

    # tokenizer=None makes the composer fall back to plain text, which is enough
    # to see which role outputs reached the prompt.
    return _OpenRLHFPromptRenderer(task_spec=_spec(task_yaml), tokenizer=None)


def _replay_prompt(renderer, topo: Sequence[str], role: str) -> str:
    return renderer.render_role_prompt(
        question="Q",
        roles_topo=list(topo),
        role_outputs=_branch_outputs(),
        target_role=role,
    )


def test_replay_prompt_for_solver_b_excludes_solver_a() -> None:
    topo, _depends_on = _topo_and_depends(BRANCH_TASK)
    prompt = _replay_prompt(_replay_renderer(BRANCH_TASK), topo, "solver_b")

    assert "PLAN-TEXT" in prompt
    assert "SOLVER-A-TEXT" not in prompt


def test_replay_prompt_for_the_integrator_carries_both_solvers() -> None:
    topo, _depends_on = _topo_and_depends(BRANCH_TASK)
    prompt = _replay_prompt(_replay_renderer(BRANCH_TASK), topo, "integrator")

    assert "PLAN-TEXT" in prompt
    assert "SOLVER-A-TEXT" in prompt
    assert "SOLVER-B-TEXT" in prompt


def test_replay_prompt_for_a_chain_still_carries_the_whole_prefix() -> None:
    """The scope change must not quietly shorten a chain's context."""
    topo, _depends_on = _topo_and_depends(CHAIN_WORKFLOWS["c5"])
    renderer = _replay_renderer(CHAIN_WORKFLOWS["c5"])

    outputs = {role: f"{role.upper()}-TEXT" for role in topo}
    prompt = renderer.render_role_prompt(
        question="Q",
        roles_topo=list(topo),
        role_outputs=outputs,
        target_role=topo[-1],
    )

    for role in topo[:-1]:
        assert outputs[role] in prompt


# ---------------------------------------------------------------------------
# 5. Every call site actually asks for ancestors
# ---------------------------------------------------------------------------

HELPER = "ancestors_in_topo_order"
CONTEXT_BUILDERS = {"build_render_context", "_build_render_context"}

# file -> how many context-building calls it is expected to contain
CALL_SITE_FILES = {
    "c3/protocol/rollout_generator.py": 3,
    "c3/analysis/replay.py": 1,
}


def _callee_name(node: ast.AST) -> str:
    if not isinstance(node, ast.Call):
        return ""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _scan_call_sites(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))

    bound_to_helper = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and _callee_name(node.value) == HELPER:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    bound_to_helper.add(target.id)

    sites = []
    for node in ast.walk(tree):
        if _callee_name(node) in CONTEXT_BUILDERS:
            keywords = {kw.arg: kw.value for kw in node.keywords}
            sites.append((node.lineno, keywords.get("topo_so_far")))

    return sites, bound_to_helper


@pytest.mark.parametrize("relative_path", sorted(CALL_SITE_FILES))
def test_every_context_call_site_passes_ancestors(relative_path: str) -> None:
    """A source-level tripwire.

    The tree-expansion call site cannot be exercised here (it needs the training
    stack), so the guard against a silent revert to the topological prefix is to
    read the call sites.
    """
    sites, bound_to_helper = _scan_call_sites(REPO_ROOT / relative_path)

    assert len(sites) == CALL_SITE_FILES[relative_path], f"{relative_path}: {sites}"

    for lineno, value in sites:
        assert value is not None, f"{relative_path}:{lineno} passes no topo_so_far"
        if isinstance(value, ast.Name):
            assert value.id in bound_to_helper, f"{relative_path}:{lineno} passes {value.id}"
        else:
            assert _callee_name(value) == HELPER, f"{relative_path}:{lineno}"
