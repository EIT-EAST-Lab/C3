"""CPU mechanism test for the paper's counterfactual credit assignment.

The module needs torch, and is skipped without it, so an environment that has
not installed the CPU tier reports a skip instead of a collection error.

It builds a rollout tree by hand, annotated exactly the way
`c3/mas/rollout_generator.py` annotates a real one, and checks the advantages
that `C3CreditProvider.compute` produces against values derived independently
with plain Python arithmetic. No GPU, no model, no data files.
"""

from __future__ import annotations

import random
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest

torch = pytest.importorskip("torch", reason="the credit mechanism is defined over torch tensors")

from c3.credit.counterfactual.materialize import materialize_c3_tree_groups  # noqa: E402
from c3.credit.counterfactual.provider import C3CreditProvider  # noqa: E402
from c3.integration.marl_specs import RoleSpec  # noqa: E402


# ---------------------------------------------------------------------------
# The paper's Duo protocol and the fixed scenario
# ---------------------------------------------------------------------------

DUO_ROLES: Tuple[RoleSpec, ...] = (
    RoleSpec(
        name="reasoner",
        prompt="You are the reasoner. Question: {question}",
        with_answer=False,
    ),
    RoleSpec(
        name="actor",
        prompt="You are the actor. Context: {context}",
        with_answer=True,
        depends_on=("reasoner",),
    ),
)
SOLO_ROLES: Tuple[RoleSpec, ...] = (DUO_ROLES[0],)

# Two reasoner alternatives, four actor alternatives under each: eight leaves,
# which is the paper's evaluation budget B = 8 at one rollout per alternative.
PAPER_FANOUT: List[int] = [2, 4]

# Leaf terminal rewards are binary. Alternative A scores 2 of 4 and alternative
# B scores 1 of 4, so the internal subtree means are 0.5 and 0.25.
LEAVES_UNDER_A: List[float] = [1.0, 0.0, 0.0, 1.0]
LEAVES_UNDER_B: List[float] = [0.0, 0.0, 1.0, 0.0]
SCENARIO_LEAVES: List[float] = LEAVES_UNDER_A + LEAVES_UNDER_B

QUESTION = "A train leaves at 09:00 and arrives at 11:30. How long is the trip?"


# ---------------------------------------------------------------------------
# The baselines, written out again in plain Python
# ---------------------------------------------------------------------------


def loo_baseline(values: Sequence[float]) -> List[float]:
    total = float(sum(values))
    n = len(values)
    return [(total - float(v)) / float(n - 1) for v in values]


def full_mean_baseline(values: Sequence[float]) -> List[float]:
    mean = float(sum(values)) / float(len(values))
    return [mean for _ in values]


def advantages_of(values: Sequence[float], baseline_mode: str) -> List[float]:
    baseline = loo_baseline(values) if baseline_mode == "loo" else full_mean_baseline(values)
    return [float(v) - b for v, b in zip(values, baseline)]


def expected_by_path(
    fanout: Sequence[int], leaf_rewards: Sequence[float], baseline_mode: str
) -> Dict[Tuple[int, ...], float]:
    """Expected advantage per node, keyed by the node's path in the tree."""
    first, second = int(fanout[0]), int(fanout[1])
    subtree_means = [
        float(sum(leaf_rewards[i * second : (i + 1) * second])) / float(second) for i in range(first)
    ]

    expected: Dict[Tuple[int, ...], float] = {}
    for i, value in enumerate(advantages_of(subtree_means, baseline_mode)):
        expected[(i,)] = value
    for i in range(first):
        siblings = list(leaf_rewards[i * second : (i + 1) * second])
        for j, value in enumerate(advantages_of(siblings, baseline_mode)):
            expected[(i, j)] = value
    return expected


# ---------------------------------------------------------------------------
# Hand-built rollout tree, following the generator's annotation contract
# ---------------------------------------------------------------------------


def build_tree_rows(
    *,
    fanout: Sequence[int],
    leaf_rewards: Sequence[float],
    roles: Sequence[str] = ("reasoner", "actor"),
    question: str = QUESTION,
    question_id: int = 0,
    no_replay: bool = False,
) -> List[Dict[str, Any]]:
    """One row per node, in the order the generator emits them.

    Mirrors c3/mas/rollout_generator.py: node ids are handed out layer by layer
    starting at 0 with the root state at -1, leaf_start and leaf_size come from
    the prefix encoding, an internal node's reward is the mean of its leaf
    subtree, and adv_group_id encodes (question_id, role_id, parent_id).
    """
    num_roles = len(roles)
    if len(fanout) != num_roles:
        raise ValueError("fanout must have one entry per role")

    total_leaves = 1
    for width in fanout:
        total_leaves *= int(width)
    if len(leaf_rewards) != total_leaves:
        raise ValueError(f"expected {total_leaves} leaf rewards, got {len(leaf_rewards)}")

    # total nodes per question: sum over depths of prod(fanout[: depth + 1])
    nodes_per_question = 0
    running = 1
    for width in fanout:
        running *= int(width)
        nodes_per_question += running
    parent_base = nodes_per_question + 1
    question_base = num_roles * parent_base

    def leaf_size(depth: int) -> int:
        size = 1
        for t in range(depth + 1, num_roles):
            size *= int(fanout[t])
        return size

    def encode_prefix(path: Tuple[int, ...]) -> int:
        value = 0
        for t, digit in enumerate(path):
            value = value * int(fanout[t]) + int(digit)
        return value

    rows: List[Dict[str, Any]] = []
    next_node_id = 0
    frontier: List[Tuple[Tuple[int, ...], int]] = [((), -1)]  # the root state is node -1

    for depth, role in enumerate(roles):
        children: List[Tuple[Tuple[int, ...], int]] = []
        for parent_path, parent_node_id in frontier:
            for sibling in range(int(fanout[depth])):
                path = parent_path + (sibling,)
                node_id = next_node_id
                next_node_id += 1

                prefix_id = encode_prefix(path)
                size = leaf_size(depth)
                start = prefix_id * size
                is_leaf = int(depth == num_roles - 1)
                k_id = prefix_id if is_leaf else -1
                reward = float(sum(leaf_rewards[start : start + size])) / float(size)

                stored_parent = -1 if (no_replay and depth > 0) else parent_node_id
                adv_group_id = (
                    question_id * question_base + depth * parent_base + (stored_parent + 1)
                )

                outputs = {
                    str(roles[t]): f"{roles[t]} output for prefix {path[: t + 1]}"
                    for t in range(depth + 1)
                }

                rows.append(
                    {
                        "path": path,
                        "question": question,
                        "question_id": int(question_id),
                        "k_id": int(k_id),
                        "role": str(role),
                        "role_id": int(depth),  # topo index, the depth on a chain
                        "adv_group_id": int(adv_group_id),
                        "c3_parent_id": int(stored_parent),
                        "c3_node_id": int(node_id),
                        "c3_depth": int(depth),
                        "is_leaf": int(is_leaf),
                        "c3_leaf_start": int(start),
                        "c3_leaf_size": int(size),
                        "reward": float(reward),
                        "traj_role_outputs": outputs,
                    }
                )
                children.append((path, node_id))
        frontier = children

    return rows


INFO_KEYS = (
    "question_id",
    "k_id",
    "role",
    "role_id",
    "adv_group_id",
    "c3_parent_id",
    "c3_node_id",
    "c3_depth",
    "is_leaf",
    "c3_leaf_start",
    "c3_leaf_size",
    "question",
)


def make_experiences(
    rows: Sequence[Dict[str, Any]],
    *,
    roles: Sequence[str] = ("reasoner", "actor"),
    rows_per_experience: int = 0,
) -> List[Any]:
    """Pack rows into duck-typed Experience objects (one info list per key)."""
    size = int(rows_per_experience) if rows_per_experience else len(rows)
    chunks = [list(rows[i : i + size]) for i in range(0, len(rows), size)]

    experiences: List[Any] = []
    for chunk in chunks:
        info: Dict[str, Any] = {key: [row[key] for row in chunk] for key in INFO_KEYS}
        info["traj_role_outputs"] = {
            str(role): [str(row["traj_role_outputs"].get(role, "")) for row in chunk]
            for role in roles
        }
        experiences.append(
            SimpleNamespace(
                prompts=[str(row["question"]) for row in chunk],
                reward=torch.tensor([row["reward"] for row in chunk], dtype=torch.float32),
                info=info,
            )
        )
    return experiences


def make_provider(
    *,
    fanout: Optional[Sequence[int]],
    roles: Sequence[RoleSpec] = DUO_ROLES,
    no_replay: bool = False,
) -> C3CreditProvider:
    args = SimpleNamespace(
        c3_fanout_list=list(fanout) if fanout is not None else None,
        c3_no_replay=bool(no_replay),
        c3_credit_variant="reward_only",
        c3_baseline_mode="loo",
        c3_va_alpha=1.0,
    )
    return C3CreditProvider(args=args, roles=list(roles), q_critic=None)


def run_mechanism(
    rows: Sequence[Dict[str, Any]],
    *,
    fanout: Optional[Sequence[int]],
    baseline_mode: str,
    roles: Sequence[RoleSpec] = DUO_ROLES,
    no_replay: bool = False,
    rows_per_experience: int = 0,
) -> Tuple[List[dict], List[float], Dict[str, float]]:
    """Materialize, compute, and flatten the advantages back into row order."""
    role_names = [role.name for role in roles]
    experiences = make_experiences(rows, roles=role_names, rows_per_experience=rows_per_experience)
    groups, _gid_to_index, _tree_diag = materialize_c3_tree_groups(experiences, roles=role_names)
    provider = make_provider(fanout=fanout, roles=roles, no_replay=no_replay)
    advantages, diag = provider.compute(
        groups,
        experiences=experiences,
        cfg={"credit_variant": "reward_only", "baseline_mode": baseline_mode},
    )

    for tensor in advantages:
        assert not bool(torch.isnan(tensor).any()), "an Experience row was left without a value"

    flat = [float(value) for tensor in advantages for value in tensor.tolist()]
    assert len(flat) == len(rows)
    return groups, flat, diag


def experience_offsets(
    rows: Sequence[Dict[str, Any]], *, rows_per_experience: int
) -> Dict[Tuple[int, int], int]:
    """(experience index, row index) to position in the flattened row order."""
    size = int(rows_per_experience) if rows_per_experience else len(rows)
    return {(index // size, index % size): index for index in range(len(rows))}


# ---------------------------------------------------------------------------
# 1. The fixed scenario, against numbers derived by hand
# ---------------------------------------------------------------------------

# Derived by hand from the formula and asserted literally, so that a slip in the
# formula written above cannot hide behind the same slip in the provider.
HAND_DERIVED: Dict[str, Dict[Tuple[int, ...], float]] = {
    "loo": {
        (0,): 0.25,
        (1,): -0.25,
        (0, 0): 2.0 / 3.0,
        (0, 1): -2.0 / 3.0,
        (0, 2): -2.0 / 3.0,
        (0, 3): 2.0 / 3.0,
        (1, 0): -1.0 / 3.0,
        (1, 1): -1.0 / 3.0,
        (1, 2): 1.0,
        (1, 3): -1.0 / 3.0,
    },
    "full_mean": {
        (0,): 0.125,
        (1,): -0.125,
        (0, 0): 0.5,
        (0, 1): -0.5,
        (0, 2): -0.5,
        (0, 3): 0.5,
        (1, 0): -0.25,
        (1, 1): -0.25,
        (1, 2): 0.75,
        (1, 3): -0.25,
    },
}


@pytest.mark.parametrize("baseline_mode", ["loo", "full_mean"])
def test_paper_scenario_matches_the_hand_derived_advantages(baseline_mode: str) -> None:
    rows = build_tree_rows(fanout=PAPER_FANOUT, leaf_rewards=SCENARIO_LEAVES)
    assert [row["reward"] for row in rows[:2]] == [0.5, 0.25]  # the subtree means

    # Spreading the ten nodes over three Experience objects puts the routing
    # from (group, node) back to (experience, row) under test as well.
    _groups, flat, _diag = run_mechanism(
        rows, fanout=PAPER_FANOUT, baseline_mode=baseline_mode, rows_per_experience=4
    )

    from_formula = expected_by_path(PAPER_FANOUT, SCENARIO_LEAVES, baseline_mode)
    literal = HAND_DERIVED[baseline_mode]
    assert torch.allclose(
        torch.tensor([from_formula[path] for path in literal], dtype=torch.float64),
        torch.tensor([literal[path] for path in literal], dtype=torch.float64),
        atol=1e-6,
    )

    actual = torch.tensor(flat, dtype=torch.float64)
    expected = torch.tensor([literal[row["path"]] for row in rows], dtype=torch.float64)
    assert torch.allclose(actual, expected, atol=1e-6), f"{flat} != {expected.tolist()}"


# ---------------------------------------------------------------------------
# 2 and 3. Within-group invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("baseline_mode", ["loo", "full_mean"])
def test_advantages_sum_to_zero_within_every_sibling_group(baseline_mode: str) -> None:
    rows = build_tree_rows(fanout=PAPER_FANOUT, leaf_rewards=SCENARIO_LEAVES)
    groups, flat, _diag = run_mechanism(
        rows, fanout=PAPER_FANOUT, baseline_mode=baseline_mode, rows_per_experience=4
    )

    offsets = experience_offsets(rows, rows_per_experience=4)
    assert len(groups) == 3
    for group in groups:
        values = [flat[offsets[(ref.exp_idx, ref.row_idx)]] for ref in group["node_refs"]]
        assert sum(values) == pytest.approx(0.0, abs=1e-6)


def test_loo_is_the_full_mean_advantage_scaled_by_n_over_n_minus_one() -> None:
    rows = build_tree_rows(fanout=PAPER_FANOUT, leaf_rewards=SCENARIO_LEAVES)
    groups, loo_flat, _ = run_mechanism(
        rows, fanout=PAPER_FANOUT, baseline_mode="loo", rows_per_experience=4
    )
    _groups, mean_flat, _ = run_mechanism(
        rows, fanout=PAPER_FANOUT, baseline_mode="full_mean", rows_per_experience=4
    )

    offsets = experience_offsets(rows, rows_per_experience=4)
    for group in groups:
        indices = [offsets[(ref.exp_idx, ref.row_idx)] for ref in group["node_refs"]]
        n = len(indices)
        scale = float(n) / float(n - 1)
        for index in indices:
            assert loo_flat[index] == pytest.approx(scale * mean_flat[index], abs=1e-6)


# ---------------------------------------------------------------------------
# 4. Singleton group
# ---------------------------------------------------------------------------


def test_a_singleton_group_gets_exactly_zero() -> None:
    rows = build_tree_rows(fanout=[1], leaf_rewards=[1.0], roles=("reasoner",))
    assert len(rows) == 1

    _groups, flat, diag = run_mechanism(rows, fanout=[1], baseline_mode="loo", roles=SOLO_ROLES)
    assert flat == [0.0]
    assert diag["c3/rule_b_groups"] == 1.0


# ---------------------------------------------------------------------------
# 5. Fanout sizing
# ---------------------------------------------------------------------------


def test_a_sibling_group_of_the_wrong_size_is_rejected() -> None:
    rows = build_tree_rows(fanout=PAPER_FANOUT, leaf_rewards=SCENARIO_LEAVES)
    truncated = [row for row in rows if row["path"] != (0, 3)]  # three actors under A
    assert len(truncated) == 9

    with pytest.raises(RuntimeError, match="Sibling group size mismatch"):
        run_mechanism(truncated, fanout=PAPER_FANOUT, baseline_mode="loo")


def test_no_replay_accepts_the_collapsed_group_of_eight() -> None:
    rows = build_tree_rows(fanout=PAPER_FANOUT, leaf_rewards=SCENARIO_LEAVES, no_replay=True)
    depth_one = {row["adv_group_id"] for row in rows if row["c3_depth"] == 1}
    assert len(depth_one) == 1, "no-replay collapses the parent, so all actors share one group"

    groups, flat, diag = run_mechanism(
        rows, fanout=PAPER_FANOUT, baseline_mode="loo", no_replay=True
    )
    assert sorted(len(group["node_refs"]) for group in groups) == [2, 8]
    assert diag["c3/no_replay"] == 1.0

    # The eight leaves are one group now: rewards [1,0,0,1,0,0,1,0], mean 3/8.
    assert flat[2:] == pytest.approx(advantages_of(SCENARIO_LEAVES, "loo"), abs=1e-6)


# ---------------------------------------------------------------------------
# 6. Annotation contract
# ---------------------------------------------------------------------------


def test_a_row_without_adv_group_id_is_rejected() -> None:
    rows = build_tree_rows(fanout=PAPER_FANOUT, leaf_rewards=SCENARIO_LEAVES)
    experiences = make_experiences(rows)
    del experiences[0].info["adv_group_id"]

    with pytest.raises(RuntimeError, match="adv_group_id"):
        materialize_c3_tree_groups(experiences, roles=["reasoner", "actor"])


def test_two_questions_sharing_an_adv_group_id_are_rejected() -> None:
    rows = build_tree_rows(fanout=PAPER_FANOUT, leaf_rewards=SCENARIO_LEAVES)
    experiences = make_experiences(rows)
    experiences[0].info["question_id"][1] = 7  # same group, different question

    with pytest.raises(RuntimeError, match="collision"):
        materialize_c3_tree_groups(experiences, roles=["reasoner", "actor"])


# ---------------------------------------------------------------------------
# 7. Property test over random trees
# ---------------------------------------------------------------------------


def test_random_trees_agree_with_plain_python_arithmetic() -> None:
    rng = random.Random(20260907)
    shapes = ([2, 4], [3, 2], [4, 2])

    for trial in range(20):
        fanout = list(rng.choice(shapes))
        leaf_rewards = [float(rng.randint(0, 1)) for _ in range(fanout[0] * fanout[1])]
        rows = build_tree_rows(fanout=fanout, leaf_rewards=leaf_rewards)

        for baseline_mode in ("loo", "full_mean"):
            _groups, flat, _diag = run_mechanism(
                rows, fanout=fanout, baseline_mode=baseline_mode, rows_per_experience=3
            )
            expected = expected_by_path(fanout, leaf_rewards, baseline_mode)
            for row, value in zip(rows, flat):
                assert value == pytest.approx(expected[row["path"]], abs=1e-6), (
                    f"trial={trial} fanout={fanout} rewards={leaf_rewards} "
                    f"mode={baseline_mode} path={row['path']}"
                )


# ---------------------------------------------------------------------------
# 8. Diagnostics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("baseline_mode", ["loo", "full_mean"])
def test_diagnostics_describe_the_constructed_tree(baseline_mode: str) -> None:
    rows = build_tree_rows(fanout=PAPER_FANOUT, leaf_rewards=SCENARIO_LEAVES)
    _groups, _flat, diag = run_mechanism(
        rows, fanout=PAPER_FANOUT, baseline_mode=baseline_mode, rows_per_experience=4
    )

    assert diag["c3/rule_b_groups"] == 3.0  # one reasoner group, two actor groups
    assert diag["c3/rule_b_nodes"] == 10.0  # two internal nodes and eight leaves
    assert diag["c3/rule_b_group_size_min"] == 2.0
    assert diag["c3/rule_b_group_size_max"] == 4.0
    assert diag["c3/baseline_mode_is_loo"] == (1.0 if baseline_mode == "loo" else 0.0)
    assert diag["c3/baseline_mode_is_full_mean"] == (0.0 if baseline_mode == "loo" else 1.0)
    assert diag["c3/variant_reward_only"] == 1.0
    assert diag["c3/q_used"] == 0.0
