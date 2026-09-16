"""Tests for the cross bucket batched replay path.

What is pinned here.

First, equivalence. `c3.analysis.replay_batched.run_buckets_batched` must build
the same buckets as a loop over `ReplayRunner.run_bucket`: same number of
alternatives, same number of replays per alternative, same recorded downstream
actions, same meta keys apart from the seed scheme it adds. With a policy whose
text ignores the seed the two paths must agree byte for byte, including the
bucket identifier, which is the strongest form this can take. With a policy
whose text depends on the seed only the structure can agree, because the two
paths derive seeds differently on purpose.

Second, the batching itself: how many calls reach the policy, and that cutting a
stage into chunks changes nothing but the number of calls.

Third, the seed function, which has to be a pure function of the slot it names,
stable across processes, or a replay cannot be reproduced.

Fourth, that the flag is off by default, so the sequential path is untouched.
`LEGACY_BUCKETS_SHA256` was captured from `build-buckets` before the flag
existed and is compared against the output of the current code.

Everything runs on fake policies, so no model, no GPU and no torch are needed.
The two policy adapters that do need an engine (`_VLLMPolicy`, `_HFPolicy`) are
exercised through their own call shape only: a stub engine records the sampling
parameters, which is as deep as this machine can go.
"""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

from c3.analysis import analysis as analysis_cli  # noqa: E402
from c3.analysis import replay as replay_mod  # noqa: E402
from c3.analysis import replay_batched as batched_mod  # noqa: E402
from c3.analysis.buckets import read_buckets_jsonl, write_buckets_jsonl  # noqa: E402


# Output of `build-buckets` on the fixture below, taken with the code that had
# no --batched flag (2026-09-15). The flag must not move these bytes.
LEGACY_BUCKETS_SHA256 = "e78e914b338c8d3e149a2d15d3d436c8a4412af6ac4fe4848d1fb37189168df4"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _digest(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=6).hexdigest()


class _CountingPolicy:
    """Deterministic policy whose text is a hash of the prompt and the seed.

    One call returns n distinct texts, so nothing is lost to de-duplication and
    the call counts stay clean.
    """

    def __init__(self) -> None:
        self.sample_calls = 0
        self.sample_many_calls = 0
        self.sequences = 0

    def texts(self, prompt: str, n: int, seed: Any) -> List[str]:
        head = _digest(str(prompt))
        return [f"[{head}|s{seed}|{k}]" for k in range(int(n))]

    def sample(self, prompt: str, n: int = 1, **decoding: Any) -> List[str]:
        self.sample_calls += 1
        self.sequences += int(n)
        return self.texts(prompt, n, decoding.get("seed"))

    def sample_many(
        self,
        prompts: Sequence[str],
        n: int = 1,
        seeds: Optional[Sequence[int]] = None,
        **decoding: Any,
    ) -> List[List[str]]:
        self.sample_many_calls += 1
        out: List[List[str]] = []
        for i, prompt in enumerate(prompts):
            seed = seeds[i] if seeds is not None else decoding.get("seed")
            self.sequences += int(n)
            out.append(self.texts(prompt, n, seed))
        return out


class _PromptOnlyPolicy(_CountingPolicy):
    """Same, except the text ignores the seed.

    Under this policy the batched path and the sequential path must produce
    identical bytes, which isolates prompt assembly and ordering from the seed
    scheme.
    """

    def texts(self, prompt: str, n: int, seed: Any) -> List[str]:
        head = _digest(str(prompt))
        return [f"[{head}|{k}]" for k in range(int(n))]


class _SmallPoolPolicy(_CountingPolicy):
    """Policy that runs out of distinct texts, so de-duplication has to retry.

    The size of the pool depends on the prompt, so different buckets end up
    needing different numbers of extra alternatives, which is the case the
    batched stage has to group.
    """

    def texts(self, prompt: str, n: int, seed: Any) -> List[str]:
        head = _digest(str(prompt))
        pool = 1 + int(head[:2], 16) % 3
        return [f"[{head}|{k % pool}]" for k in range(int(n))]


class _FailingPolicy(_CountingPolicy):
    """Raises as soon as a prompt asks a named role to speak."""

    def __init__(self, role: str) -> None:
        super().__init__()
        self.role = role

    def texts(self, prompt: str, n: int, seed: Any) -> List[str]:
        if f"ROLE:{self.role}\n" in prompt or prompt.endswith(f"ROLE:{self.role}"):
            raise RuntimeError(f"engine refused role {self.role}")
        return super().texts(prompt, n, seed)


class _Renderer:
    """Prompt = question, role under the pen, then every earlier role output."""

    def render_role_prompt(
        self,
        *,
        question: str,
        roles_topo: Sequence[str],
        role_outputs: Mapping[str, str],
        target_role: str,
        meta: Optional[Mapping[str, Any]] = None,
    ) -> str:
        parts = [f"Q:{question}", f"ROLE:{target_role}"]
        for role in roles_topo:
            if role in role_outputs:
                parts.append(f"{role}={role_outputs[role]}")
        return "\n".join(parts)


class _Evaluator:
    """Binary return derived from the whole set of role outputs."""

    def __init__(self) -> None:
        self.calls = 0

    def evaluate(
        self,
        *,
        restart: Any,
        role_outputs: Mapping[str, str],
        meta: Optional[Mapping[str, Any]] = None,
    ) -> float:
        self.calls += 1
        blob = "|".join(f"{k}={role_outputs[k]}" for k in sorted(role_outputs))
        return float(int(_digest(blob)[:4], 16) % 2)


class _FailingEvaluator(_Evaluator):
    def evaluate(self, **kwargs: Any) -> float:
        raise RuntimeError("reward function exploded")


# ---------------------------------------------------------------------------
# Runner and config builders
# ---------------------------------------------------------------------------


def _roles(k: int) -> List[str]:
    return [f"r{i}" for i in range(k)]


def _make_runner(
    policy: Any,
    *,
    roles_topo: Sequence[str],
    n_questions: int,
    tag: str,
    evaluator: Any = None,
    seed: int = 0,
) -> replay_mod.ReplayRunner:
    """A real ReplayRunner wired to fakes, so the batched path sees the real thing."""
    dataset = [
        {"id": f"{tag}-q{i}", "input": f"question {i} of {tag}", "answer": "42"} for i in range(n_questions)
    ]
    return replay_mod.ReplayRunner(
        task_spec=f"wp-r7-{tag}",
        policy=policy,
        evaluator=evaluator if evaluator is not None else _Evaluator(),
        prompt_renderer=_Renderer(),
        runner_meta={"task": "fake-task", "split": "FAKE", "method": "sft", "seed": int(seed)},
        dataset=dataset,
        eval_suite_name="FAKE",
        roles_topo=list(roles_topo),
    )


def _restart_states(runner: replay_mod.ReplayRunner, target_role: str, limit: int) -> List[Any]:
    return list(
        runner.iter_restart_states(
            task="fake-task",
            split="FAKE",
            target_role=target_role,
            limit=limit,
            seed=0,
        )
    )


def _make_cfg(
    *,
    target_role: str,
    next_role: Optional[str],
    num_candidates: int,
    num_completions: int,
    record_next: bool = True,
    include_real: bool = False,
    extra_v: int = 0,
    seed: Optional[int] = 0,
) -> replay_mod.ReplayConfig:
    decoding: Dict[str, Any] = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "max_new_tokens": 16}
    if seed is not None:
        decoding["seed"] = int(seed)
    return replay_mod.ReplayConfig(
        target_role=target_role,
        num_candidates=int(num_candidates),
        num_completions_per_candidate=int(num_completions),
        decoding=decoding,
        record_next_teammate=bool(record_next),
        next_role=next_role,
        include_real_as_j0=bool(include_real),
        num_extra_v_samples=int(extra_v),
        prefix_decoding={},
    )


def _as_rows(buckets: Sequence[Any], path: Path) -> List[Dict[str, Any]]:
    """Write buckets through the real writer and read them back as dicts.

    Going through the writer is what gives each row its bucket_id, which is
    exactly the field the two paths have to agree on.
    """
    write_buckets_jsonl(path, list(buckets), overwrite=True)
    return list(read_buckets_jsonl(path))


def _both_paths(
    policy_factory: Callable[[], Any],
    *,
    roles_topo: Sequence[str],
    n_questions: int,
    tag: str,
    cfg: replay_mod.ReplayConfig,
    tmp_path: Path,
    batch_prompts: int = batched_mod.DEFAULT_BATCH_PROMPTS,
) -> Dict[str, Any]:
    """Run the sequential path and the batched path over the same inputs."""
    seq_policy = policy_factory()
    seq_runner = _make_runner(seq_policy, roles_topo=roles_topo, n_questions=n_questions, tag=tag)
    seq_states = _restart_states(seq_runner, cfg.target_role, n_questions)
    seq_buckets = [seq_runner.run_bucket(state, cfg) for state in seq_states]

    bat_policy = policy_factory()
    bat_runner = _make_runner(bat_policy, roles_topo=roles_topo, n_questions=n_questions, tag=tag)
    bat_states = _restart_states(bat_runner, cfg.target_role, n_questions)
    bat_buckets = batched_mod.run_buckets_batched(
        bat_runner,
        bat_states,
        cfg,
        target_role=cfg.target_role,
        next_role=cfg.next_role,
        batch_prompts=batch_prompts,
    )

    return {
        "seq_policy": seq_policy,
        "bat_policy": bat_policy,
        "seq_rows": _as_rows(seq_buckets, tmp_path / "sequential.jsonl"),
        "bat_rows": _as_rows(bat_buckets, tmp_path / "batched.jsonl"),
        "seq_buckets": seq_buckets,
        "bat_buckets": bat_buckets,
    }


# ---------------------------------------------------------------------------
# 1. Equivalence with the sequential path
# ---------------------------------------------------------------------------


def test_structure_matches_the_sequential_path(tmp_path: Path) -> None:
    cfg = _make_cfg(target_role="r0", next_role="r1", num_candidates=3, num_completions=2)
    run = _both_paths(
        _CountingPolicy,
        roles_topo=_roles(4),
        n_questions=4,
        tag="struct",
        cfg=cfg,
        tmp_path=tmp_path,
    )

    seq_rows, bat_rows = run["seq_rows"], run["bat_rows"]
    assert len(seq_rows) == len(bat_rows) == 4

    for before, after in zip(seq_rows, bat_rows):
        assert after["question_id"] == before["question_id"]
        assert after["ctx_hash"] == before["ctx_hash"]
        assert after["target_role"] == before["target_role"]
        assert after["restart"] == before["restart"]

        assert len(after["candidates"]) == len(before["candidates"]) == 3
        for cand_after, cand_before in zip(after["candidates"], before["candidates"]):
            assert cand_after["j"] == cand_before["j"]
            assert len(cand_after["returns"]) == len(cand_before["returns"]) == 2
            assert len(cand_after["next_actions"]) == len(cand_before["next_actions"]) == 2

        # The batched path adds exactly one meta key and changes no other.
        assert set(after["meta"]) == set(before["meta"]) | {"seed_scheme"}
        assert after["meta"]["seed_scheme"] == "batched_v1"
        for key, value in before["meta"].items():
            assert after["meta"][key] == value

        assert isinstance(after["bucket_id"], str) and after["bucket_id"].startswith("bkt_")


def test_a_seed_sensitive_policy_gives_different_text_on_the_two_paths(tmp_path: Path) -> None:
    """The seed schemes differ on purpose, so the sampled text must differ too."""
    cfg = _make_cfg(target_role="r0", next_role="r1", num_candidates=3, num_completions=2)
    run = _both_paths(
        _CountingPolicy,
        roles_topo=_roles(4),
        n_questions=4,
        tag="struct",
        cfg=cfg,
        tmp_path=tmp_path,
    )
    seq_text = [c["action_text"] for row in run["seq_rows"] for c in row["candidates"]]
    bat_text = [c["action_text"] for row in run["bat_rows"] for c in row["candidates"]]
    assert seq_text != bat_text


def test_a_seed_independent_policy_gives_identical_buckets(tmp_path: Path) -> None:
    """Take the seed out of the picture and the two paths agree byte for byte."""
    cfg = _make_cfg(target_role="r0", next_role="r1", num_candidates=3, num_completions=2)
    run = _both_paths(
        _PromptOnlyPolicy,
        roles_topo=_roles(4),
        n_questions=5,
        tag="identical",
        cfg=cfg,
        tmp_path=tmp_path,
    )

    seq_rows, bat_rows = run["seq_rows"], run["bat_rows"]
    assert len(seq_rows) == len(bat_rows) == 5
    for before, after in zip(seq_rows, bat_rows):
        stripped = dict(after)
        meta = dict(after["meta"])
        assert meta.pop("seed_scheme") == "batched_v1"
        stripped["meta"] = meta
        assert stripped == before, f"bucket {before['question_id']} differs"

    # bucket_id is part of that comparison, so state it once on its own.
    assert [row["bucket_id"] for row in bat_rows] == [row["bucket_id"] for row in seq_rows]


def test_two_role_workflow_matches_the_batched_next_branch(tmp_path: Path) -> None:
    """K=2 is the one case the sequential path already batches.

    Structure has to match, but the downstream text cannot be compared byte for
    byte even under a seed independent policy, and that is not a defect: the
    sequential branch draws the c replays of one alternative as n=c inside a
    single call, while the batched path draws them as c calls of n=1 with a seed
    each. On an engine both are c independent samples; on a fake that enumerates
    within a call they differ. The number of samples is the same either way.
    """
    cfg = _make_cfg(target_role="r0", next_role="r1", num_candidates=4, num_completions=3)
    run = _both_paths(
        _PromptOnlyPolicy,
        roles_topo=_roles(2),
        n_questions=3,
        tag="k2",
        cfg=cfg,
        tmp_path=tmp_path,
    )

    assert len(run["bat_rows"]) == len(run["seq_rows"]) == 3
    for before, after in zip(run["seq_rows"], run["bat_rows"]):
        assert after["ctx_hash"] == before["ctx_hash"]
        assert after["restart"] == before["restart"]
        assert set(after["meta"]) == set(before["meta"]) | {"seed_scheme"}
        assert len(after["candidates"]) == len(before["candidates"]) == 4

        for cand_after, cand_before in zip(after["candidates"], before["candidates"]):
            # Stage A is one call of n=4 on both paths, so the alternatives, and
            # with them the bucket identity, are the same text.
            assert cand_after["action_text"] == cand_before["action_text"]
            assert len(cand_after["next_actions"]) == len(cand_before["next_actions"]) == 3
            assert len(cand_after["returns"]) == len(cand_before["returns"]) == 3
        assert after["bucket_id"] == before["bucket_id"]

    assert run["bat_policy"].sequences == run["seq_policy"].sequences


def test_deduplication_shortfall_lands_the_same_way_on_both_paths(tmp_path: Path) -> None:
    """A policy that repeats itself must leave both paths with the same count."""
    cfg = _make_cfg(target_role="r0", next_role="r1", num_candidates=4, num_completions=1)
    run = _both_paths(
        _SmallPoolPolicy,
        roles_topo=_roles(3),
        n_questions=6,
        tag="dedup",
        cfg=cfg,
        tmp_path=tmp_path,
    )

    got = [row["meta"]["candidate_total_got"] for row in run["bat_rows"]]
    assert got == [row["meta"]["candidate_total_got"] for row in run["seq_rows"]]
    # The fixture is only worth anything if it actually starves some buckets and
    # leaves the groups of different sizes.
    assert min(got) < 4
    assert len(set(got)) > 1
    for row in run["bat_rows"]:
        assert len(row["candidates"]) == row["meta"]["candidate_total_got"]
        assert row["meta"]["credit_n"] == min(4, row["meta"]["candidate_total_got"])


def test_extra_v_samples_and_a_real_action_at_j0_behave_as_in_run_bucket(tmp_path: Path) -> None:
    cfg = _make_cfg(
        target_role="r0",
        next_role="r1",
        num_candidates=2,
        num_completions=2,
        include_real=True,
        extra_v=2,
    )
    # A seed independent policy would hand the same text back on the n=1 round
    # and on the top up round, so the seed sensitive one is the honest fixture
    # here: with a real engine those two rounds carry different seeds.
    run = _both_paths(
        _CountingPolicy,
        roles_topo=_roles(3),
        n_questions=3,
        tag="fidelity",
        cfg=cfg,
        tmp_path=tmp_path,
    )
    for before, after in zip(run["seq_rows"], run["bat_rows"]):
        assert after["meta"]["real_j"] == before["meta"]["real_j"] == 0
        assert after["meta"]["credit_n"] == before["meta"]["credit_n"] == 2
        assert after["meta"]["v_extra_n"] == before["meta"]["v_extra_n"] == 2
        assert after["meta"]["candidate_total_req"] == before["meta"]["candidate_total_req"] == 4
        assert len(after["candidates"]) == len(before["candidates"]) == 4


def test_the_target_role_may_be_last_in_the_workflow(tmp_path: Path) -> None:
    """No role after the target means no generation in stage B, only returns."""
    cfg = _make_cfg(target_role="r2", next_role=None, num_candidates=2, num_completions=2, record_next=False)
    run = _both_paths(
        _PromptOnlyPolicy,
        roles_topo=_roles(3),
        n_questions=2,
        tag="tail",
        cfg=cfg,
        tmp_path=tmp_path,
    )
    for before, after in zip(run["seq_rows"], run["bat_rows"]):
        for cand in after["candidates"]:
            assert cand["next_actions"] == []
            assert len(cand["returns"]) == 2
        meta = dict(after["meta"])
        meta.pop("seed_scheme")
        stripped = dict(after)
        stripped["meta"] = meta
        assert stripped == before


def test_the_batched_path_repeats_itself_byte_for_byte(tmp_path: Path) -> None:
    cfg = _make_cfg(target_role="r0", next_role="r1", num_candidates=3, num_completions=2)

    def _one(name: str) -> bytes:
        runner = _make_runner(_CountingPolicy(), roles_topo=_roles(4), n_questions=4, tag="repeat")
        states = _restart_states(runner, "r0", 4)
        buckets = batched_mod.run_buckets_batched(runner, states, cfg, target_role="r0", next_role="r1")
        path = tmp_path / name
        write_buckets_jsonl(path, buckets, overwrite=True)
        return path.read_bytes()

    assert _one("first.jsonl") == _one("second.jsonl")


# ---------------------------------------------------------------------------
# 2. Call counts
# ---------------------------------------------------------------------------


def test_call_counts_on_a_ten_role_workflow(tmp_path: Path) -> None:
    """The whole point of the module: 37 calls where the old path needs 5780."""
    roles = _roles(10)
    cfg = _make_cfg(target_role="r0", next_role="r1", num_candidates=8, num_completions=4)

    bat_policy = _CountingPolicy()
    bat_runner = _make_runner(bat_policy, roles_topo=roles, n_questions=20, tag="counts")
    bat_states = _restart_states(bat_runner, "r0", 20)
    assert bat_policy.sample_calls == 0  # the target role is first, no prefix sampling
    buckets = batched_mod.run_buckets_batched(bat_runner, bat_states, cfg, target_role="r0", next_role="r1")

    roles_after = 9
    assert bat_policy.sample_many_calls == 1 + 4 * roles_after == 37
    assert bat_policy.sample_calls == 0
    assert len(buckets) == 20

    seq_policy = _CountingPolicy()
    seq_runner = _make_runner(seq_policy, roles_topo=roles, n_questions=20, tag="counts")
    seq_states = _restart_states(seq_runner, "r0", 20)
    for state in seq_states:
        seq_runner.run_bucket(state, cfg)

    per_bucket = 1 + 8 * 4 * roles_after
    assert per_bucket == 289
    assert seq_policy.sample_calls == 20 * per_bucket == 5780
    assert seq_policy.sample_many_calls == 0

    # Same work, same number of sequences; only the packaging changed.
    assert bat_policy.sequences == seq_policy.sequences


def test_chunking_changes_the_number_of_calls_but_not_the_result(tmp_path: Path) -> None:
    cfg = _make_cfg(target_role="r0", next_role="r1", num_candidates=3, num_completions=2)

    def _one(batch_prompts: int, name: str) -> Dict[str, Any]:
        policy = _CountingPolicy()
        runner = _make_runner(policy, roles_topo=_roles(4), n_questions=5, tag="chunk")
        states = _restart_states(runner, "r0", 5)
        buckets = batched_mod.run_buckets_batched(
            runner,
            states,
            cfg,
            target_role="r0",
            next_role="r1",
            batch_prompts=batch_prompts,
        )
        path = tmp_path / name
        write_buckets_jsonl(path, buckets, overwrite=True)
        return {"bytes": path.read_bytes(), "calls": policy.sample_many_calls}

    big = _one(batched_mod.DEFAULT_BATCH_PROMPTS, "big.jsonl")
    small = _one(7, "small.jsonl")

    assert small["bytes"] == big["bytes"]
    assert big["calls"] == 1 + 2 * 3 == 7  # 5 prompts per stage, one chunk each
    # Stage A has 5 prompts and every stage B call has 15, so 7 per chunk splits
    # them into one and three calls.
    assert small["calls"] == 1 + 6 * 3 == 19


# ---------------------------------------------------------------------------
# 3. The seed function
# ---------------------------------------------------------------------------


def test_the_seed_function_is_a_pure_function_of_its_slot() -> None:
    first = batched_mod.batched_seed(0, "q7", 3, 1, "actor")
    second = batched_mod.batched_seed(0, "q7", 3, 1, "actor")
    assert first == second

    # Pinned so that a later refactor cannot move the seeds silently, and so
    # that a process dependent hash would be caught.
    assert first == 1472623668


def test_the_seed_function_moves_when_any_component_moves() -> None:
    base = dict(base_seed=0, bucket_key="q7", candidate_index=3, replay_index=1, role_name="actor")

    def _seed(**overrides: Any) -> int:
        args = dict(base)
        args.update(overrides)
        return batched_mod.batched_seed(
            args["base_seed"],
            args["bucket_key"],
            args["candidate_index"],
            args["replay_index"],
            args["role_name"],
        )

    reference = _seed()
    variants = {
        "base_seed": _seed(base_seed=1),
        "bucket_key": _seed(bucket_key="q8"),
        "candidate_index": _seed(candidate_index=4),
        "replay_index": _seed(replay_index=2),
        "role_name": _seed(role_name="verifier"),
    }
    for name, value in variants.items():
        assert value != reference, name
    assert len(set(variants.values())) == len(variants)


def test_seeds_stay_inside_the_engine_range() -> None:
    seeds = [
        batched_mod.batched_seed(seed, f"q{q}", j, r, role)
        for seed in (0, 1, 7)
        for q in range(12)
        for j in (-1, 0, 5)
        for r in (0, 3)
        for role in ("r1", "r9")
    ]
    assert len(seeds) == 3 * 12 * 3 * 2 * 2
    assert all(0 <= s < 2 ** 31 for s in seeds)
    assert batched_mod.SEED_MODULUS == 2 ** 31
    # A degenerate hash would collapse these; they should be spread out.
    assert len(set(seeds)) == len(seeds)


def test_every_generation_slot_of_a_cell_gets_its_own_seed() -> None:
    cfg = _make_cfg(target_role="r0", next_role="r1", num_candidates=3, num_completions=2)
    roles = _roles(4)
    seeds = [
        batched_mod.batched_seed(0, f"seeds-q{q}", j, r, role)
        for q in range(4)
        for j in range(cfg.num_candidates)
        for r in range(cfg.num_completions_per_candidate)
        for role in roles[1:]
    ]
    assert len(set(seeds)) == len(seeds) == 4 * 3 * 2 * 3


# ---------------------------------------------------------------------------
# 4. Failure carries context and writes nothing
# ---------------------------------------------------------------------------


def test_a_failed_generation_names_the_stage_the_role_and_the_replay() -> None:
    cfg = _make_cfg(target_role="r0", next_role="r1", num_candidates=2, num_completions=2)
    runner = _make_runner(_FailingPolicy("r2"), roles_topo=_roles(4), n_questions=2, tag="boom")
    states = _restart_states(runner, "r0", 2)

    with pytest.raises(batched_mod.BatchedReplayError) as excinfo:
        batched_mod.run_buckets_batched(runner, states, cfg, target_role="r0", next_role="r1")

    message = str(excinfo.value)
    assert "stage B" in message
    assert "role r2" in message
    assert "replay 0" in message
    assert "engine refused role r2" in message
    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_a_failed_evaluation_names_the_bucket_the_candidate_and_the_replay() -> None:
    cfg = _make_cfg(target_role="r0", next_role="r1", num_candidates=2, num_completions=2)
    runner = _make_runner(
        _PromptOnlyPolicy(),
        roles_topo=_roles(3),
        n_questions=2,
        tag="boom-eval",
        evaluator=_FailingEvaluator(),
    )
    states = _restart_states(runner, "r0", 2)

    with pytest.raises(batched_mod.BatchedReplayError) as excinfo:
        batched_mod.run_buckets_batched(runner, states, cfg, target_role="r0", next_role="r1")

    message = str(excinfo.value)
    assert "stage C" in message
    assert "candidate 0" in message
    assert "replay 0" in message
    assert "reward function exploded" in message


def test_the_measured_position_must_agree_with_the_config() -> None:
    cfg = _make_cfg(target_role="r0", next_role="r1", num_candidates=2, num_completions=1)
    runner = _make_runner(_PromptOnlyPolicy(), roles_topo=_roles(3), n_questions=1, tag="mismatch")
    states = _restart_states(runner, "r0", 1)

    with pytest.raises(ValueError, match="target_role"):
        batched_mod.run_buckets_batched(runner, states, cfg, target_role="r1", next_role="r1")
    with pytest.raises(ValueError, match="next_role"):
        batched_mod.run_buckets_batched(runner, states, cfg, target_role="r0", next_role="r2")


def test_an_empty_batch_is_not_an_error() -> None:
    cfg = _make_cfg(target_role="r0", next_role="r1", num_candidates=2, num_completions=1)
    runner = _make_runner(_PromptOnlyPolicy(), roles_topo=_roles(3), n_questions=1, tag="empty")
    assert batched_mod.run_buckets_batched(runner, [], cfg, target_role="r0", next_role="r1") == []


# ---------------------------------------------------------------------------
# 5. The policy adapters
# ---------------------------------------------------------------------------


class _FakeSamplingParams:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = dict(kwargs)


class _FakeCompletion:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeRequestOutput:
    def __init__(self, texts: Sequence[str]) -> None:
        self.outputs = [_FakeCompletion(t) for t in texts]


class _FakeLLM:
    """Stands in for vllm.LLM: records what it was asked, answers in order."""

    def __init__(self) -> None:
        self.calls: List[Any] = []

    def generate(self, prompts: Sequence[str], params: Any) -> List[_FakeRequestOutput]:
        self.calls.append((list(prompts), params))
        out: List[_FakeRequestOutput] = []
        for i, prompt in enumerate(prompts):
            p = params[i] if isinstance(params, list) else params
            n = int(p.kwargs["n"])
            seed = p.kwargs["seed"]
            out.append(_FakeRequestOutput([f"{prompt}|seed={seed}|{k} STOPHERE tail" for k in range(n)]))
        return out


def _vllm_policy_stub() -> Any:
    policy = object.__new__(replay_mod._VLLMPolicy)
    policy.SamplingParams = _FakeSamplingParams
    policy.tokenizer = None
    policy.seed = 7
    policy.llm = _FakeLLM()
    return policy


def test_the_sampling_protocol_declares_the_batched_method() -> None:
    assert hasattr(replay_mod.SamplingPolicy, "sample_many")
    assert callable(getattr(replay_mod._VLLMPolicy, "sample_many"))
    assert callable(getattr(replay_mod._HFPolicy, "sample_many"))


def test_the_vllm_adapter_sends_one_call_with_one_params_object_per_prompt() -> None:
    policy = _vllm_policy_stub()
    out = policy.sample_many(
        ["p1", "p2", "p3"],
        n=2,
        seeds=[11, 12, 13],
        temperature=0.1,
        top_p=0.5,
        top_k=3,
        max_new_tokens=9,
        stop=["STOPHERE"],
        seed=999,
    )

    assert len(policy.llm.calls) == 1
    prompts, params = policy.llm.calls[0]
    assert prompts == ["p1", "p2", "p3"]
    assert isinstance(params, list) and len(params) == 3

    for i, seed in enumerate((11, 12, 13)):
        fields = params[i].kwargs
        assert fields["n"] == 2
        assert fields["temperature"] == 0.1
        assert fields["top_p"] == 0.5
        assert fields["top_k"] == 3
        assert fields["max_tokens"] == 9
        assert fields["stop"] == ["STOPHERE"]
        # The per prompt seed wins over the one inside the decoding dict.
        assert fields["seed"] == seed

    assert out == [[f"p{i + 1}|seed={seed}|{k}" for k in range(2)] for i, seed in enumerate((11, 12, 13))]


def test_the_vllm_adapter_falls_back_to_the_decoding_seed_then_to_its_own() -> None:
    policy = _vllm_policy_stub()
    policy.sample_many(["p1"], n=1, seeds=None, temperature=0.7, seed=321)
    assert policy.llm.calls[0][1][0].kwargs["seed"] == 321

    policy = _vllm_policy_stub()
    policy.sample_many(["p1"], n=1, seeds=None, temperature=0.7)
    assert policy.llm.calls[0][1][0].kwargs["seed"] == 7


def test_the_vllm_adapter_builds_the_same_parameters_as_the_single_prompt_path() -> None:
    """Guards the one duplication: sample() was left untouched by contract."""
    decoding = {"temperature": 0.3, "top_p": 0.6, "top_k": 5, "max_new_tokens": 11, "stop": "STOPHERE", "seed": 42}

    policy = _vllm_policy_stub()
    policy.sample("p1", n=3, **decoding)
    single = dict(policy.llm.calls[0][1].kwargs)

    policy = _vllm_policy_stub()
    policy.sample_many(["p1"], n=3, seeds=None, **decoding)
    batched = dict(policy.llm.calls[0][1][0].kwargs)

    assert batched == single


def test_the_vllm_adapter_rejects_a_mismatched_seed_list_and_a_short_answer() -> None:
    policy = _vllm_policy_stub()
    with pytest.raises(ValueError, match="seeds"):
        policy.sample_many(["p1", "p2"], n=1, seeds=[1])

    class _ShortLLM(_FakeLLM):
        def generate(self, prompts: Sequence[str], params: Any) -> List[_FakeRequestOutput]:
            return super().generate(list(prompts)[:1], params)

    policy = _vllm_policy_stub()
    policy.llm = _ShortLLM()
    with pytest.raises(RuntimeError, match="one result per prompt"):
        policy.sample_many(["p1", "p2"], n=1, seeds=[1, 2])


def test_the_hf_adapter_loops_over_sample_with_one_seed_per_prompt() -> None:
    class _Stub:
        def __init__(self) -> None:
            self.calls: List[Any] = []

        def sample(self, prompt: str, n: int = 1, **decoding: Any) -> List[str]:
            self.calls.append((prompt, n, dict(decoding)))
            return [f"{prompt}#{k}#{decoding.get('seed')}" for k in range(n)]

    stub = _Stub()
    out = replay_mod._HFPolicy.sample_many(stub, ["a", "b"], n=2, seeds=[5, 6], temperature=0.3)

    assert [call[0] for call in stub.calls] == ["a", "b"]
    assert [call[1] for call in stub.calls] == [2, 2]
    assert [call[2]["seed"] for call in stub.calls] == [5, 6]
    assert all(call[2]["temperature"] == 0.3 for call in stub.calls)
    assert out == [["a#0#5", "a#1#5"], ["b#0#6", "b#1#6"]]

    with pytest.raises(ValueError, match="seeds"):
        replay_mod._HFPolicy.sample_many(_Stub(), ["a", "b"], n=1, seeds=[1])


def test_a_policy_without_the_batched_method_still_works() -> None:
    """Old policies keep working through a loop over sample()."""

    class _LegacyPolicy:
        def __init__(self) -> None:
            self.sample_calls = 0

        def sample(self, prompt: str, n: int = 1, **decoding: Any) -> List[str]:
            self.sample_calls += 1
            return [f"[{_digest(prompt)}|{k}]" for k in range(int(n))]

    policy = _LegacyPolicy()
    cfg = _make_cfg(target_role="r0", next_role="r1", num_candidates=2, num_completions=2)
    runner = _make_runner(policy, roles_topo=_roles(3), n_questions=2, tag="legacy")
    states = _restart_states(runner, "r0", 2)
    buckets = batched_mod.run_buckets_batched(runner, states, cfg, target_role="r0", next_role="r1")

    assert len(buckets) == 2
    # One call per bucket in stage A, then one per (bucket, candidate, replay,
    # role) in stage B: 2 + 2 * 2 * 2 * 2.
    assert policy.sample_calls == 18


# ---------------------------------------------------------------------------
# 6. The command line
# ---------------------------------------------------------------------------

_FACTORY_MODULE = "c3_wp_r7_fake_runner_factory"

_CLI_ROLES = _roles(3)


def _register_factory(builder: Callable[[], Any]) -> str:
    module = types.ModuleType(_FACTORY_MODULE)
    module.make_runner = lambda _args: builder()  # type: ignore[attr-defined]
    sys.modules[_FACTORY_MODULE] = module
    return f"{_FACTORY_MODULE}:make_runner"


def _cli_runner(policy: Any = None, evaluator: Any = None) -> replay_mod.ReplayRunner:
    return _make_runner(
        policy if policy is not None else _PromptOnlyPolicy(),
        roles_topo=_CLI_ROLES,
        n_questions=4,
        tag="cli",
        evaluator=evaluator,
    )


def _run_cli(out_path: Path, *extra: str, builder: Optional[Callable[[], Any]] = None) -> None:
    spec = _register_factory(builder if builder is not None else _cli_runner)
    analysis_cli.main(
        [
            "build-buckets",
            "--task",
            "fake-task",
            "--split",
            "FAKE",
            "--policy_ckpt",
            "/unused",
            "--method",
            "sft",
            "--target_role",
            "r0",
            "--next_role",
            "r1",
            "--record_next_teammate",
            "--num_candidates",
            "3",
            "--num_completions",
            "2",
            "--limit",
            "4",
            "--seed",
            "0",
            "--decoding_json",
            json.dumps({"temperature": 0.7, "top_p": 0.8, "top_k": 20, "max_new_tokens": 16}, sort_keys=True),
            "--runner_factory",
            spec,
            "--out",
            str(out_path),
            *extra,
        ]
    )


def test_build_buckets_defaults_to_the_sequential_path() -> None:
    parser = analysis_cli._build_parser()
    args = parser.parse_args(
        [
            "build-buckets",
            "--task",
            "t",
            "--split",
            "s",
            "--policy_ckpt",
            "/unused",
            "--target_role",
            "r0",
            "--out",
            "/unused/buckets.jsonl",
        ]
    )
    assert args.batched is False
    assert args.batch_prompts == batched_mod.DEFAULT_BATCH_PROMPTS == 4096

    flagged = parser.parse_args(
        [
            "build-buckets",
            "--task",
            "t",
            "--split",
            "s",
            "--policy_ckpt",
            "/unused",
            "--target_role",
            "r0",
            "--out",
            "/unused/buckets.jsonl",
            "--batched",
            "--batch_prompts",
            "64",
        ]
    )
    assert flagged.batched is True
    assert flagged.batch_prompts == 64


def test_without_the_flag_the_output_is_the_one_the_old_code_wrote(tmp_path: Path) -> None:
    out = tmp_path / "legacy.jsonl"
    _run_cli(out)
    rows = list(read_buckets_jsonl(out))

    assert len(rows) == 4
    assert all("seed_scheme" not in row["meta"] for row in rows)
    assert hashlib.sha256(out.read_bytes()).hexdigest() == LEGACY_BUCKETS_SHA256


def test_the_flag_routes_through_the_batched_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: Dict[str, Any] = {}
    real = batched_mod.run_buckets_batched

    def _spy(*args: Any, **kwargs: Any) -> Any:
        seen["args"] = args
        seen["kwargs"] = kwargs
        return real(*args, **kwargs)

    monkeypatch.setattr(batched_mod, "run_buckets_batched", _spy)

    out = tmp_path / "batched.jsonl"
    _run_cli(out, "--batched", "--batch_prompts", "5")
    rows = list(read_buckets_jsonl(out))

    assert len(rows) == 4
    assert all(row["meta"]["seed_scheme"] == "batched_v1" for row in rows)
    assert seen["kwargs"]["target_role"] == "r0"
    assert seen["kwargs"]["next_role"] == "r1"
    assert seen["kwargs"]["batch_prompts"] == 5
    assert len(seen["args"][1]) == 4  # all restart states handed over at once


def test_the_flag_does_not_disturb_the_rest_of_the_row(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.jsonl"
    batched = tmp_path / "batched.jsonl"
    _run_cli(legacy)
    _run_cli(batched, "--batched")

    legacy_rows = list(read_buckets_jsonl(legacy))
    batched_rows = list(read_buckets_jsonl(batched))
    for before, after in zip(legacy_rows, batched_rows):
        meta = dict(after["meta"])
        meta.pop("seed_scheme")
        stripped = dict(after)
        stripped["meta"] = meta
        assert stripped == before


def test_meta_json_still_reaches_every_bucket_under_the_flag(tmp_path: Path) -> None:
    out = tmp_path / "stamped.jsonl"
    injected = {"workflow": "a3", "model": "4b", "rule": "sweep_n3", "seed": 999}
    _run_cli(out, "--batched", "--meta_json", json.dumps(injected, sort_keys=True))

    for row in read_buckets_jsonl(out):
        assert row["meta"]["workflow"] == "a3"
        assert row["meta"]["model"] == "4b"
        assert row["meta"]["rule"] == "sweep_n3"
        assert row["meta"]["seed_scheme"] == "batched_v1"
        # The runner already recorded a seed, so the injected one must lose.
        assert row["meta"]["seed"] == 0


def test_the_limit_still_bounds_the_batch(tmp_path: Path) -> None:
    """Four questions in the fixture, two asked for. The later --limit wins."""
    out = tmp_path / "two.jsonl"
    _run_cli(out, "--batched", "--limit", "2")
    rows = list(read_buckets_jsonl(out))
    assert len(rows) == 2
    assert [row["question_id"] for row in rows] == ["cli-q0", "cli-q1"]


def test_a_failed_cell_leaves_no_file_behind(tmp_path: Path) -> None:
    out = tmp_path / "never.jsonl"

    def _builder() -> Any:
        return _cli_runner(policy=_FailingPolicy("r1"))

    with pytest.raises(batched_mod.BatchedReplayError):
        _run_cli(out, "--batched", builder=_builder)

    assert not out.exists()


# ---------------------------------------------------------------------------
# 7. The E1 cell driver
# ---------------------------------------------------------------------------


def _e1_cells() -> Any:
    script_dir = REPO_ROOT / "scripts" / "70_rebuild"
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    import e1_cells  # noqa: PLC0415

    return e1_cells


def _e1_commands(capsys: pytest.CaptureFixture, *extra: str) -> List[str]:
    assert _e1_cells().main(["--model_root", "/models", "--dry-run", *extra]) == 0
    out = capsys.readouterr().out
    return [line for line in out.splitlines() if line and not line.startswith("#")]


def test_the_cell_driver_asks_for_batching_by_default(capsys: pytest.CaptureFixture) -> None:
    commands = _e1_commands(capsys)
    assert len(commands) == 60
    assert all(" --batched" in line for line in commands)


def test_the_cell_driver_can_be_told_to_run_the_sequential_path(capsys: pytest.CaptureFixture) -> None:
    commands = _e1_commands(capsys, "--no-batched")
    assert len(commands) == 60
    assert not any("--batched" in line for line in commands)
