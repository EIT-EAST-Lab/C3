"""Contract tests for the task configurations added for the rebuild study: the
five deeper workflows, plus the two-agent copy the depth study reads.

The depth study measures one decision point per workflow, so the wiring of each
workflow is a paper-facing fact: the topological order fixes which role is the
measured position, and the dependency edges fix what that role reads. These
tests pin both, plus the context assembly of the one workflow whose wiring is
not a chain.

They use only the dependency-light loaders (`c3.task.config`,
`c3.protocol.role_graph`, `c3.credit.role_dag`, `c3.credit.q_prompt`), so they
run without the training stack.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

import pytest
import yaml

from c3.credit.q_prompt import format_for_q
from c3.credit.role_dag import _collect_ancestors, build_dependency_from_roles
from c3.task.config import load_task
from c3.protocol.prompt_render import build_render_context
from c3.protocol.role_graph import RoleGraph


REPO_ROOT = Path(__file__).resolve().parents[2]

# workflow -> (task yaml, roles json, expected topological order)
WORKFLOWS = {
    "a3": (
        "configs/tasks/math_a3.yaml",
        "configs/roles/math/roles_trio.json",
        ["reasoner", "actor", "verifier"],
    ),
    "mt4": (
        "configs/tasks/math_mt4.yaml",
        "configs/roles/math/roles_mt4.json",
        ["reasoner_1", "actor_1", "reasoner_2", "actor_2"],
    ),
    "branch": (
        "configs/tasks/math_branch.yaml",
        "configs/roles/math/roles_branch.json",
        ["planner", "solver_a", "solver_b", "integrator"],
    ),
    "c5": (
        "configs/tasks/math_c5.yaml",
        "configs/roles/math/roles_c5.json",
        ["planner", "solver", "critic", "reviser", "verifier"],
    ),
    "c10": (
        "configs/tasks/math_c10.yaml",
        "configs/roles/math/roles_c10.json",
        [
            "reader",
            "planner",
            "decomposer",
            "solver_a",
            "solver_b",
            "integrator",
            "critic",
            "reviser",
            "checker",
            "verifier",
        ],
    ),
}

WORKFLOW_IDS = sorted(WORKFLOWS)

ROLES_DIR = REPO_ROOT / "configs" / "roles" / "math"
ROLE_FILES = sorted(path.name for path in ROLES_DIR.glob("*.json"))

# The team size every prompt states about itself, as a word.
TEAM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "ten": 10}
_TEAM_PHRASE = re.compile(r"\b(One|Two|Three|Four|Five|Ten) LLM agents?\b")

MANIFEST = REPO_ROOT / "configs" / "data_manifest.yaml"

# The one evaluation suite the depth-study task files carry and the paper task
# does not.
MATHPOOL_SUITE = {"name": "MATHPOOL", "path": "data/MATH/pool_informative.jsonl", "limit": None}

# The two-agent arm of the depth study. It is not in WORKFLOWS because it has no
# wiring of its own: it is configs/tasks/math.yaml with the pool suite appended,
# same role file and all. It exists so that the depth study can measure the
# two-agent arm on the pool without putting a post-screening suite into the task
# file a reader of the paper runs.
A2_TASK_YAML = "configs/tasks/math_a2.yaml"

# Every task file the depth study reads, which is every file that declares
# MATHPOOL. The cell drivers default to --split MATHPOOL, so a file missing here
# cannot be measured at all.
POOL_TASK_YAMLS = (A2_TASK_YAML,) + tuple(WORKFLOWS[workflow][0] for workflow in WORKFLOW_IDS)

# What configs/tasks/math.yaml evaluates: the five suites of the main table, which
# is what the evaluation that runs during training measures. The two saturation
# controls (GSM8K-test, CMATH-test) are reported once, after training, and live in
# configs/tasks/math_final_eval.yaml with the rest of the final protocol.
MAIN_TABLE_SUITES = ["MATH500", "Minerva-Math", "AMC23", "AIME24", "AIME25"]

# Every math task file whose evaluation suites have to resolve to a prepared
# artifact. The probe task and the final-evaluation task have their own equivalent
# checks in tests/unit/test_rebuild_eval_probe.py and tests/unit/test_rebuild_final_eval.py.
MATH_TASK_YAMLS = (
    "configs/tasks/math.yaml",
    "configs/tasks/math_screen.yaml",
) + POOL_TASK_YAMLS


def _task(workflow: str):
    task_yaml, _roles_json, _topo = WORKFLOWS[workflow]
    return load_task(str(REPO_ROOT / task_yaml))


def _dependency(workflow: str):
    spec = _task(workflow)
    parents, layers, topo, _children, _role_to_layer, _descendants = build_dependency_from_roles(spec.roles)
    return parents, layers, topo


# ---------------------------------------------------------------------------
# 1. Role graph: load, topological order, single answering role
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("workflow", WORKFLOW_IDS)
def test_workflow_roles_build_a_graph_with_the_declared_topological_order(workflow: str) -> None:
    _task_yaml, _roles_json, expected_topo = WORKFLOWS[workflow]
    spec = _task(workflow)
    graph = RoleGraph(spec.roles)

    assert graph.topo_order() == expected_topo
    assert spec.topo_role_names() == expected_topo


@pytest.mark.parametrize("workflow", WORKFLOW_IDS)
def test_only_the_last_role_of_the_workflow_emits_the_answer(workflow: str) -> None:
    _task_yaml, _roles_json, expected_topo = WORKFLOWS[workflow]
    spec = _task(workflow)

    answering = [r.name for r in spec.roles if r.with_answer]
    assert answering == [expected_topo[-1]]


@pytest.mark.parametrize("workflow", WORKFLOW_IDS)
def test_only_the_answering_role_asks_for_a_boxed_answer(workflow: str) -> None:
    """A stray boxed instruction upstream would let a non-final role end the episode.

    The four-turn workflow is the deliberate exception: its two Actor turns are
    the same agent and therefore share one system prompt, so the boxed
    instruction is present at both Actor positions and scoped by wording to the
    last decision point.
    """
    _task_yaml, _roles_json, expected_topo = WORKFLOWS[workflow]
    spec = _task(workflow)

    boxed = [r.name for r in spec.roles if "\\boxed" in r.prompt]
    if workflow == "mt4":
        assert boxed == ["actor_1", "actor_2"]
    else:
        assert boxed == [expected_topo[-1]]


@pytest.mark.parametrize("workflow", WORKFLOW_IDS)
def test_roles_json_follows_the_roles_duo_schema(workflow: str) -> None:
    _task_yaml, roles_json, expected_topo = WORKFLOWS[workflow]
    raw = json.loads((REPO_ROOT / roles_json).read_text(encoding="utf-8"))

    assert isinstance(raw, list) and len(raw) == len(expected_topo)
    for item in raw:
        assert set(item) <= {"role", "prompt", "with_answer", "depends_on"}
        assert isinstance(item["role"], str) and item["role"]
        assert isinstance(item["prompt"], str) and item["prompt"]
        assert isinstance(item["with_answer"], bool)
        assert item["prompt"].startswith("<|im_start|>system: ")
        assert item["prompt"].endswith("<|im_end|>\n")
        # A backspace here means a JSON file wrote \b instead of \\b before "oxed".
        assert "\b" not in item["prompt"]


@pytest.mark.parametrize("roles_json", ROLE_FILES)
def test_every_prompt_states_the_size_of_its_own_team(roles_json: str) -> None:
    """The team a prompt describes has to be the team the file actually wires.

    This is model-facing text, not a comment: the Reasoner of the three-agent
    file opened with "Two LLM agents (Reasoner -> Actor)", copied from the
    two-agent file, so that role was told it was working alone with the Actor
    while the Verifier behind it was never mentioned.

    The number is the number of agents, which is the number of distinct prompts,
    and not the number of decision points. In every file but one those are the
    same. The four-turn workflow is the exception by design: two agents take four
    turns, so its four roles share two prompts and say "Two LLM agents".
    """
    raw = json.loads((ROLES_DIR / roles_json).read_text(encoding="utf-8"))
    prompts = [str(item["prompt"]) for item in raw]
    agents = len(set(prompts))

    if roles_json == "roles_solo.json":
        # One agent, so there is no team to name and no prompt claims one.
        assert len(raw) == 1
        assert [p for p in prompts if _TEAM_PHRASE.search(p)] == []
        return

    stated = []
    for item in raw:
        match = _TEAM_PHRASE.search(str(item["prompt"]))
        assert match is not None, f"{roles_json}: role {item['role']} never names the team"
        stated.append(TEAM_WORDS[match.group(1).lower()])

    assert set(stated) == {agents}, (
        f"{roles_json}: the prompts claim {sorted(set(stated))} agents, "
        f"the file holds {agents} distinct prompts for {len(raw)} roles"
    )

    if roles_json == "roles_mt4.json":
        assert (agents, len(raw)) == (2, 4)
    else:
        assert agents == len(raw)


def test_the_four_turn_workflow_reuses_one_prompt_per_agent() -> None:
    """The four-turn row holds the agent count at two while depth rises to four.

    That control only holds if the two Reasoner turns are the same agent and the
    two Actor turns are the same agent, which here means one identical system
    prompt per agent.
    """
    spec = _task("mt4")
    by_name = {r.name: r for r in spec.roles}

    assert by_name["reasoner_1"].prompt == by_name["reasoner_2"].prompt
    assert by_name["actor_1"].prompt == by_name["actor_2"].prompt


# ---------------------------------------------------------------------------
# 2. Branching: parallel solvers, joined integrator
# ---------------------------------------------------------------------------


def test_branching_dependency_edges_are_not_a_chain() -> None:
    parents, layers, topo = _dependency("branch")

    assert parents["solver_a"] == ["planner"]
    assert parents["solver_b"] == ["planner"]
    assert parents["integrator"] == ["solver_a", "solver_b"]
    assert layers == [["planner"], ["solver_a", "solver_b"], ["integrator"]]
    assert topo == ["planner", "solver_a", "solver_b", "integrator"]


def test_branching_ancestors_keep_the_two_solvers_apart() -> None:
    parents, _layers, _topo = _dependency("branch")

    assert _collect_ancestors("solver_a", parents) == {"planner"}
    assert _collect_ancestors("solver_b", parents) == {"planner"}
    assert _collect_ancestors("integrator", parents) == {"planner", "solver_a", "solver_b"}


def _branch_actions() -> Dict[str, str]:
    return {
        "planner": "PLAN-TEXT",
        "solver_a": "SOLVER-A-TEXT",
        "solver_b": "SOLVER-B-TEXT",
        "integrator": "INTEGRATOR-TEXT",
    }


def test_branching_context_assembly_under_ancestors_only() -> None:
    """`format_for_q` is the assembly that respects the branching wiring.

    It needs `prefix_scope="ancestors_only"`; the default `topo_prefix` includes
    every role that precedes the target in topological order, siblings included.
    """
    parents, layers, _topo = _dependency("branch")
    actions = _branch_actions()

    solver_b_ctx = format_for_q(
        "Q",
        actions,
        mode="prefix",
        up_to_role="solver_b",
        layers=layers,
        parents=parents,
        prefix_scope="ancestors_only",
    )
    assert "PLAN-TEXT" in solver_b_ctx
    assert "SOLVER-B-TEXT" in solver_b_ctx
    assert "SOLVER-A-TEXT" not in solver_b_ctx

    integrator_ctx = format_for_q(
        "Q",
        actions,
        mode="prefix",
        up_to_role="integrator",
        layers=layers,
        parents=parents,
        prefix_scope="ancestors_only",
    )
    assert "PLAN-TEXT" in integrator_ctx
    assert "SOLVER-A-TEXT" in integrator_ctx
    assert "SOLVER-B-TEXT" in integrator_ctx


def test_the_q_critic_view_is_the_context_scope_left_on_the_topological_prefix() -> None:
    """The Q-critic view is a second assembly, and it was left behind.

    The generation prompt now carries ancestors only. `format_for_q`, which
    builds the critic's view of a prefix, still defaults to `topo_prefix`, so
    SolverB's critic view still shows SolverA. The PPO trainer already declares
    `ancestors_only`, but the critic actor reads its own module constant
    instead, so that declaration has no effect yet. Both trainer files import
    torch, so the two constants are read from source rather than imported. This
    test exists so the remaining gap is visible rather than assumed away;
    change it together with the constants when the gap is closed.
    """
    import ast

    parents, layers, _topo = _dependency("branch")
    actions = _branch_actions()

    solver_b_view = format_for_q(
        "Q",
        actions,
        mode="prefix",
        up_to_role="solver_b",
        layers=layers,
        parents=parents,
    )
    assert "SOLVER-A-TEXT" in solver_b_view

    def _constant(relative_path: str, name: str):
        tree = ast.parse((REPO_ROOT / relative_path).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, ast.Assign):
                targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                targets = [node.target.id]
            if name in targets and node.value is not None:
                return ast.literal_eval(node.value)
        raise AssertionError(f"{relative_path}: no assignment to {name}")

    trainer_cfg = _constant("openrlhf/trainer/ppo_trainer.py", "_Q_CRITIC_VIEW_CFG")
    assert trainer_cfg["prefix_scope"] == "ancestors_only"
    assert _constant("openrlhf/trainer/ray/ppo_critic.py", "_Q_PREFIX_SCOPE") == "topo_prefix"


def test_generation_time_context_hides_the_sibling_solver() -> None:
    """The generation path assembles context from the ancestors of the role.

    `c3.protocol.prompt_render.build_render_context` is called by both the rollout
    generator and the replay prompt renderer with
    `topo_so_far = ancestors_in_topo_order(role, topo, depends_on)`, so
    SolverB's rendered context carries the plan and not SolverA's attempt. That
    is what appendix 04 claims the branching workflow does.
    """
    from c3.protocol.prompt_render import ancestors_in_topo_order

    parents, _layers, topo = _dependency("branch")
    actions = _branch_actions()

    ctx = build_render_context(
        question="Q",
        role_outputs=actions,
        topo_so_far=ancestors_in_topo_order("solver_b", topo, parents),
    )
    assert "PLAN-TEXT" in ctx["context"]
    assert "SOLVER-A-TEXT" not in ctx["context"]


# ---------------------------------------------------------------------------
# 3. Four turns: the second Reasoner sees both earlier turns
# ---------------------------------------------------------------------------


def test_four_turn_second_reasoner_sees_the_first_two_turns() -> None:
    parents, layers, _topo = _dependency("mt4")
    actions = {
        "reasoner_1": "REASONER-1-TEXT",
        "actor_1": "ACTOR-1-TEXT",
        "reasoner_2": "REASONER-2-TEXT",
        "actor_2": "ACTOR-2-TEXT",
    }

    assert _collect_ancestors("reasoner_2", parents) == {"reasoner_1", "actor_1"}

    ctx = format_for_q(
        "Q",
        actions,
        mode="prefix",
        up_to_role="reasoner_2",
        layers=layers,
        parents=parents,
        prefix_scope="ancestors_only",
    )
    assert "REASONER-1-TEXT" in ctx
    assert "ACTOR-1-TEXT" in ctx
    assert "ACTOR-2-TEXT" not in ctx


# ---------------------------------------------------------------------------
# 4. Task yaml wiring
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("workflow", WORKFLOW_IDS)
def test_task_yaml_loads_and_points_at_an_existing_roles_file(workflow: str) -> None:
    task_yaml, roles_json, _topo = WORKFLOWS[workflow]
    spec = load_task(str(REPO_ROOT / task_yaml))

    assert Path(spec.roles_path).is_file()
    assert Path(spec.roles_path).resolve() == (REPO_ROOT / roles_json).resolve()
    assert spec.env_name == "MathEnv"
    assert spec.experiment_name == f"c3_math_{workflow}"


def _environment_without_suites(environment) -> Dict[str, object]:
    return {key: value for key, value in dict(environment).items() if key != "eval_suites"}


@pytest.mark.parametrize("workflow", WORKFLOW_IDS)
def test_task_yaml_differs_from_math_yaml_only_in_name_roles_path_and_the_pool_suite(workflow: str) -> None:
    """Same data, same reward conventions; the identity, the wiring and MATHPOOL move.

    MATHPOOL is the screened pool the depth study measures on. It is an
    evaluation suite of the six depth-study task files and of nothing else: the
    paper task trains, the pool is not a training set, and an evaluation suite
    that exists only after the E1 screening pass has no business in the task file
    a reader of the paper runs.
    """
    task_yaml, _roles_json, _topo = WORKFLOWS[workflow]
    base = load_task(str(REPO_ROOT / "configs/tasks/math.yaml"))
    spec = load_task(str(REPO_ROOT / task_yaml))

    assert _environment_without_suites(spec.environment) == _environment_without_suites(base.environment)
    assert spec.train_datasets == base.train_datasets
    assert list(spec.eval_suites) == list(base.eval_suites) + [MATHPOOL_SUITE]
    assert set(spec.mas) == set(base.mas)
    assert spec.experiment_name != base.experiment_name
    assert spec.roles_path != base.roles_path


def test_the_paper_task_evaluates_the_main_table_suites_and_not_the_screened_pool() -> None:
    """The evaluation that runs during training is the five main-table suites.

    The two saturation controls are not among them: they are reported once, from
    the evaluation of record after training, and carrying them here would pay for
    1,098 plus 1,319 problems at every monitoring evaluation for a curve nobody
    reads. The screened pool is absent for the reason the neighbouring test gives.
    """
    base = load_task(str(REPO_ROOT / "configs/tasks/math.yaml"))
    assert [str(s.get("name", "")) for s in base.eval_suites] == MAIN_TABLE_SUITES


@pytest.mark.parametrize("task_yaml", POOL_TASK_YAMLS)
def test_every_depth_study_task_evaluates_on_the_screened_pool(task_yaml: str) -> None:
    """Six task files, the two-agent arm included.

    `--split` is one value for a whole sweep of the cell drivers, so a task file
    of the depth study that did not declare MATHPOOL would make the default
    sweep unrunnable rather than merely incomplete.
    """
    spec = load_task(str(REPO_ROOT / task_yaml))
    suites = {str(s.get("name", "")): dict(s) for s in spec.eval_suites}

    assert "MATHPOOL" in suites, sorted(suites)
    assert suites["MATHPOOL"] == MATHPOOL_SUITE


def test_the_two_agent_task_is_the_paper_task_plus_the_pool_suite() -> None:
    """Same data, same reward conventions, same role file; only the identity and
    the pool suite move. The two-agent arm of the depth study has to be the
    paper's two-agent workflow, or the arm is measuring something else."""
    base = load_task(str(REPO_ROOT / "configs/tasks/math.yaml"))
    spec = load_task(str(REPO_ROOT / A2_TASK_YAML))

    assert _environment_without_suites(spec.environment) == _environment_without_suites(base.environment)
    assert spec.train_datasets == base.train_datasets
    assert list(spec.eval_suites) == list(base.eval_suites) + [MATHPOOL_SUITE]
    assert set(spec.mas) == set(base.mas)
    assert Path(spec.roles_path).resolve() == Path(base.roles_path).resolve()
    assert spec.experiment_name == "c3_math_a2"
    assert spec.experiment_name != base.experiment_name


def test_every_evaluation_suite_of_every_math_task_is_a_prepared_artifact() -> None:
    """A task file may not name a data file the manifest cannot produce.

    Both halves of this bug shipped once: `MATHPOOL` and `MATHTEST` pointed at
    files that were in no manifest entry and that no script under
    `scripts/10_data/` wrote, so a reader who ran the preparation exactly as the
    README says ended up with task files pointing at nothing.
    """
    outputs = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))["outputs"]
    prepared = {str(entry["output_path"]) for entry in outputs}

    unresolved: List[str] = []
    for rel_path in MATH_TASK_YAMLS:
        spec = load_task(str(REPO_ROOT / rel_path))
        for suite in spec.eval_suites:
            if str(suite.get("path", "")) not in prepared:
                unresolved.append(f"{rel_path}: {suite.get('name')} -> {suite.get('path')}")

    assert not unresolved, "evaluation suites with no manifest entry:\n" + "\n".join(unresolved)


def test_every_depth_study_task_has_a_math500_evaluation_suite() -> None:
    """The appendix comparison runs the same sweep at `--split MATH500`, so every
    task file of the depth study must expose that suite as well as the pool."""
    names: List[str] = []
    for task_yaml in POOL_TASK_YAMLS:
        spec = load_task(str(REPO_ROOT / task_yaml))
        suites = [str(s.get("name", "")) for s in spec.eval_suites]
        assert "MATH500" in suites, f"{task_yaml}: {suites}"
        names.extend(suites)
    assert names
