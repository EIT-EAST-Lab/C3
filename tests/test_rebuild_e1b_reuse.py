"""Contract tests for reusing the alternatives of an earlier run (WP-R17).

What is pinned here.

First, the point of the flag. E1b reads the noise of a credit vector off two
runs of one cell, element by element, so element j of the two advantage vectors
has to stand for the same action in both. `build-buckets
--reuse_candidates_from <earlier buckets.jsonl>` is how the second continuation
seed gets that: the alternatives come from the file, byte for byte and in the
recorded j order, and only the replays below them are drawn again, under this
run's seed. Both halves are asserted, because either half alone would pass with
a run that changed nothing or a run that changed everything.

Second, the refusals. A decision point the source does not cover is an error
naming it, not a bucket quietly sampled afresh; so is a source built with a
different number of alternatives per bucket, so is a source that holds one
decision point twice, and so is combining reuse with the injected null arm of
E3a. The one case that is not an error is a source bucket that came up short of
its own request, which is how a de-duplication loss while the source ran shows
up: it is reused at the count it has, and the bucket records that count.

Third, that the flag is off by default. With no flag the bucket meta carries
none of the three reuse keys. The byte level guard that the default output has
not moved lives in `test_rebuild_e3a.py` and `test_rebuild_replay_batched.py`,
which pin a hash of `build-buckets` output on their own fixtures.

Fourth, the path shape written into `meta.candidates_from`, pinned against
`c3.analysis.rebuild.summary.source_path`, which is the statement of record for
how a source path is written down.

Everything runs on fake policies, so no model, no GPU and no torch are needed.

    python -m pytest tests/test_rebuild_e1b_reuse.py -q
"""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

from c3.analysis import analysis as analysis_cli  # noqa: E402
from c3.analysis import replay as replay_mod  # noqa: E402
from c3.analysis import replay_batched as batched_mod  # noqa: E402
from c3.analysis.buckets import read_buckets_jsonl, write_buckets_jsonl  # noqa: E402
from c3.analysis.rebuild.summary import source_path as contract_source_path  # noqa: E402


#: Seed of the first run of a cell, and of the second. E1b calls them A and B.
SEED_A = 1001
SEED_B = 2002

#: Three roles: the measured one, the one whose answer is recorded, one after.
ROLES = ["r0", "r1", "r2"]


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _digest(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=6).hexdigest()


class _SeedPolicy:
    """Deterministic policy whose text depends on the prompt and on the seed.

    Seed sensitivity is what makes the assertions here non-vacuous: without
    reuse the two runs of a cell draw different alternatives, which is the flaw
    this flag exists to remove, and the replays must keep differing once the
    alternatives are shared.
    """

    def __init__(self) -> None:
        self.sample_calls = 0
        self.sample_many_calls = 0
        self.prompts: List[str] = []

    def texts(self, prompt: str, n: int, seed: Any) -> List[str]:
        head = _digest(str(prompt))
        return [f"[{head}|s{seed}|{k}]" for k in range(int(n))]

    def sample(self, prompt: str, n: int = 1, **decoding: Any) -> List[str]:
        self.sample_calls += 1
        self.prompts.append(str(prompt))
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
            self.prompts.append(str(prompt))
            seed = seeds[i] if seeds is not None else decoding.get("seed")
            out.append(self.texts(prompt, n, seed))
        return out


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

    def evaluate(
        self,
        *,
        restart: Any,
        role_outputs: Mapping[str, str],
        meta: Optional[Mapping[str, Any]] = None,
    ) -> float:
        blob = "|".join(f"{k}={role_outputs[k]}" for k in sorted(role_outputs))
        return float(int(_digest(blob)[:4], 16) % 2)


def _target_prompts(policy: _SeedPolicy, target_role: str = "r0") -> List[str]:
    """The prompts that asked the measured role to speak, that is stage A."""
    return [p for p in policy.prompts if p.endswith(f"ROLE:{target_role}")]


# ---------------------------------------------------------------------------
# Runner and config builders
# ---------------------------------------------------------------------------


def _make_runner(
    policy: Any,
    *,
    seed: int,
    n_questions: int = 4,
    tag: str = "e1b",
    roles: Sequence[str] = ROLES,
) -> replay_mod.ReplayRunner:
    dataset = [
        {"id": f"{tag}-q{i}", "input": f"question {i} of {tag}", "answer": "42"} for i in range(n_questions)
    ]
    return replay_mod.ReplayRunner(
        task_spec=f"wp-r17-{tag}",
        policy=policy,
        evaluator=_Evaluator(),
        prompt_renderer=_Renderer(),
        runner_meta={"task": "fake-task", "split": "FAKE", "method": "sft", "seed": int(seed)},
        dataset=dataset,
        eval_suite_name="FAKE",
        roles_topo=list(roles),
    )


def _restart_states(runner: replay_mod.ReplayRunner, limit: int = 4) -> List[Any]:
    return list(
        runner.iter_restart_states(
            task="fake-task",
            split="FAKE",
            target_role="r0",
            limit=limit,
            seed=0,
        )
    )


def _make_cfg(
    *,
    num_candidates: int = 3,
    num_completions: int = 2,
    extra_v: int = 0,
    include_real: bool = False,
) -> replay_mod.ReplayConfig:
    return replay_mod.ReplayConfig(
        target_role="r0",
        num_candidates=int(num_candidates),
        num_completions_per_candidate=int(num_completions),
        decoding={"temperature": 0.7, "top_p": 0.8, "top_k": 20, "max_new_tokens": 16},
        record_next_teammate=True,
        next_role="r1",
        include_real_as_j0=bool(include_real),
        num_extra_v_samples=int(extra_v),
        prefix_decoding={},
    )


def _build(
    *,
    seed: int,
    path: Path,
    num_candidates: int = 3,
    num_completions: int = 2,
    extra_v: int = 0,
    limit: int = 4,
    n_questions: int = 4,
    tag: str = "e1b",
    batched: bool = False,
    reuse: Any = None,
) -> Tuple[List[Dict[str, Any]], _SeedPolicy]:
    """Build one cell through the library and write it with the real writer.

    Going through the writer is what gives every row its j field and its
    bucket_id, which is the shape a reuse source actually has on disk.
    """
    policy = _SeedPolicy()
    runner = _make_runner(policy, seed=seed, n_questions=n_questions, tag=tag)
    cfg = _make_cfg(num_candidates=num_candidates, num_completions=num_completions, extra_v=extra_v)
    states = _restart_states(runner, limit=limit)

    if batched:
        buckets = batched_mod.run_buckets_batched(
            runner,
            states,
            cfg,
            target_role=cfg.target_role,
            next_role=cfg.next_role,
            reuse=reuse,
        )
    else:
        buckets = [runner.run_bucket(state, cfg, reuse=reuse) for state in states]

    write_buckets_jsonl(path, buckets, overwrite=True)
    return list(read_buckets_jsonl(path)), policy


def _texts(rows: Sequence[Mapping[str, Any]]) -> List[List[str]]:
    return [[c["action_text"] for c in row["candidates"]] for row in rows]


def _next_actions(rows: Sequence[Mapping[str, Any]]) -> List[List[List[str]]]:
    return [[list(c["next_actions"]) for c in row["candidates"]] for row in rows]


def _rewrite(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write rows back without the schema validator, for damaged source files."""
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


# ---------------------------------------------------------------------------
# 1. The library: the alternatives are repeated, the replays are redrawn
# ---------------------------------------------------------------------------


def test_two_plain_runs_of_a_cell_disagree_on_the_alternatives(tmp_path: Path) -> None:
    """The flaw the flag removes, stated first so the rest is not vacuous."""
    rows_a, _ = _build(seed=SEED_A, path=tmp_path / "a.jsonl")
    rows_b, _ = _build(seed=SEED_B, path=tmp_path / "b.jsonl")

    assert [row["ctx_hash"] for row in rows_a] == [row["ctx_hash"] for row in rows_b]
    assert _texts(rows_a) != _texts(rows_b)


@pytest.mark.parametrize("batched", [False, True])
def test_reuse_repeats_the_source_alternatives_byte_for_byte(tmp_path: Path, batched: bool) -> None:
    source_path = tmp_path / "seedA.jsonl"
    rows_a, _ = _build(seed=SEED_A, path=source_path)

    reuse = replay_mod.load_candidate_reuse(source_path)
    rows_b, policy_b = _build(seed=SEED_B, path=tmp_path / "seedB.jsonl", batched=batched, reuse=reuse)

    assert _texts(rows_b) == _texts(rows_a)
    assert [row["question_id"] for row in rows_b] == [row["question_id"] for row in rows_a]
    assert [row["ctx_hash"] for row in rows_b] == [row["ctx_hash"] for row in rows_a]

    # Stage A did not run: the measured role was never asked to speak.
    assert _target_prompts(policy_b) == []


@pytest.mark.parametrize("batched", [False, True])
def test_reuse_redraws_the_replays_under_this_runs_seed(tmp_path: Path, batched: bool) -> None:
    source_path = tmp_path / "seedA.jsonl"
    rows_a, _ = _build(seed=SEED_A, path=source_path)

    reuse = replay_mod.load_candidate_reuse(source_path)
    rows_b, _ = _build(seed=SEED_B, path=tmp_path / "seedB.jsonl", batched=batched, reuse=reuse)

    before, after = _next_actions(rows_a), _next_actions(rows_b)
    assert len(before) == len(after) == 4
    for row_before, row_after in zip(before, after):
        assert len(row_after) == len(row_before) == 3
        for cand_before, cand_after in zip(row_before, row_after):
            assert len(cand_after) == len(cand_before) == 2
            assert cand_after != cand_before

    # Same shape as the source otherwise.
    for row_before, row_after in zip(rows_a, rows_b):
        for cand_before, cand_after in zip(row_before["candidates"], row_after["candidates"]):
            assert len(cand_after["returns"]) == len(cand_before["returns"]) == 2


@pytest.mark.parametrize("batched", [False, True])
def test_reuse_with_the_same_seed_reproduces_the_source(tmp_path: Path, batched: bool) -> None:
    """Sanity check on the redraw: seed B only differs because the seed differs.

    The batched path derives its seeds differently from the sequential one on
    purpose, so this holds within a path, not across the two.
    """
    source_path = tmp_path / "seedA.jsonl"
    rows_a, _ = _build(seed=SEED_A, path=source_path, batched=batched)

    reuse = replay_mod.load_candidate_reuse(source_path)
    rows_again, _ = _build(seed=SEED_A, path=tmp_path / "again.jsonl", batched=batched, reuse=reuse)

    assert _next_actions(rows_again) == _next_actions(rows_a)
    assert _texts(rows_again) == _texts(rows_a)


# ---------------------------------------------------------------------------
# 2. The library: meta
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batched", [False, True])
def test_meta_records_that_the_alternatives_were_reused(tmp_path: Path, batched: bool) -> None:
    source_path = tmp_path / "seedA.jsonl"
    _build(seed=SEED_A, path=source_path)

    reuse = replay_mod.load_candidate_reuse(source_path, source_label="20_data/results/E1b/a2/4b/n3_c2/seedA/buckets.jsonl")
    rows, _ = _build(seed=SEED_B, path=tmp_path / "seedB.jsonl", batched=batched, reuse=reuse)

    for row in rows:
        meta = row["meta"]
        assert meta["candidate_reuse"] is True
        assert meta["candidates_from"] == "20_data/results/E1b/a2/4b/n3_c2/seedA/buckets.jsonl"
        assert meta["n_alternatives_effective"] == len(row["candidates"]) == 3
        # The keys the runner already wrote keep their meaning.
        assert meta["candidate_total_req"] == 3
        assert meta["candidate_total_got"] == 3
        assert meta["credit_n"] == 3


def test_the_default_source_label_is_the_file_name(tmp_path: Path) -> None:
    source_path = tmp_path / "seedA.jsonl"
    _build(seed=SEED_A, path=source_path)
    assert replay_mod.load_candidate_reuse(source_path).source == "seedA.jsonl"


@pytest.mark.parametrize("batched", [False, True])
def test_without_reuse_the_meta_carries_none_of_the_three_keys(tmp_path: Path, batched: bool) -> None:
    rows, _ = _build(seed=SEED_A, path=tmp_path / "plain.jsonl", batched=batched)
    for row in rows:
        assert "candidate_reuse" not in row["meta"]
        assert "candidates_from" not in row["meta"]
        assert "n_alternatives_effective" not in row["meta"]


# ---------------------------------------------------------------------------
# 3. The library: refusals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batched", [False, True])
def test_a_decision_point_outside_the_source_is_refused_by_name(tmp_path: Path, batched: bool) -> None:
    """Three questions in the source, four in the run: the fourth is an error."""
    source_path = tmp_path / "seedA.jsonl"
    _build(seed=SEED_A, path=source_path, limit=3)

    reuse = replay_mod.load_candidate_reuse(source_path)
    with pytest.raises(replay_mod.CandidateReuseError) as excinfo:
        _build(seed=SEED_B, path=tmp_path / "seedB.jsonl", limit=4, batched=batched, reuse=reuse)

    assert "e1b-q3" in str(excinfo.value)
    assert not (tmp_path / "seedB.jsonl").exists()


def test_a_context_the_source_recorded_under_another_prefix_is_refused(tmp_path: Path) -> None:
    """Matching is on the context hash, not on the question alone."""
    source_path = tmp_path / "seedA.jsonl"
    rows, _ = _build(seed=SEED_A, path=source_path)
    moved = [dict(row) for row in rows]
    moved[2]["ctx_hash"] = int(moved[2]["ctx_hash"]) ^ 1
    _rewrite(source_path, moved)

    reuse = replay_mod.load_candidate_reuse(source_path)
    with pytest.raises(replay_mod.CandidateReuseError) as excinfo:
        _build(seed=SEED_B, path=tmp_path / "seedB.jsonl", reuse=reuse)
    assert "e1b-q2" in str(excinfo.value)


@pytest.mark.parametrize("batched", [False, True])
def test_a_different_number_of_alternatives_is_refused(tmp_path: Path, batched: bool) -> None:
    source_path = tmp_path / "seedA.jsonl"
    _build(seed=SEED_A, path=source_path, num_candidates=3)

    reuse = replay_mod.load_candidate_reuse(source_path)
    with pytest.raises(replay_mod.CandidateReuseError) as excinfo:
        _build(seed=SEED_B, path=tmp_path / "seedB.jsonl", num_candidates=2, batched=batched, reuse=reuse)

    message = str(excinfo.value)
    assert "2 alternatives" in message and "3" in message


def test_the_v_extra_segment_counts_towards_the_number_of_alternatives(tmp_path: Path) -> None:
    """A run asks for num_candidates + num_extra_v_samples, and so did the source.

    Only that total is compared, because `candidate_total_req` is the only
    request a source bucket records: it keeps no separate record of how the
    alternatives were split between the credit segment and the V-extra segment.
    A run that moves the split while keeping the total therefore replays the
    same texts under a different reading of j, which this check does not catch.
    E1b asks for no V-extra samples at all, so its split is the whole total.
    """
    source_path = tmp_path / "seedA.jsonl"
    rows_a, _ = _build(seed=SEED_A, path=source_path, num_candidates=2, extra_v=1)
    assert [len(row["candidates"]) for row in rows_a] == [3, 3, 3, 3]
    assert all(row["meta"]["candidate_total_req"] == 3 for row in rows_a)

    reuse = replay_mod.load_candidate_reuse(source_path)
    rows_b, _ = _build(seed=SEED_B, path=tmp_path / "ok.jsonl", num_candidates=2, extra_v=1, reuse=reuse)
    assert _texts(rows_b) == _texts(rows_a)
    for row in rows_b:
        assert row["meta"]["credit_n"] == 2
        assert row["meta"]["v_extra_n"] == 1

    with pytest.raises(replay_mod.CandidateReuseError):
        _build(seed=SEED_B, path=tmp_path / "bad.jsonl", num_candidates=3, extra_v=1, reuse=reuse)


@pytest.mark.parametrize("batched", [False, True])
def test_reuse_and_a_forced_action_cannot_be_combined(tmp_path: Path, batched: bool) -> None:
    source_path = tmp_path / "seedA.jsonl"
    _build(seed=SEED_A, path=source_path)
    reuse = replay_mod.load_candidate_reuse(source_path)

    policy = _SeedPolicy()
    runner = _make_runner(policy, seed=SEED_B)
    cfg = _make_cfg()
    states = _restart_states(runner)

    with pytest.raises(replay_mod.CandidateReuseError) as excinfo:
        if batched:
            batched_mod.run_buckets_batched(
                runner,
                states,
                cfg,
                target_role=cfg.target_role,
                next_role=cfg.next_role,
                forced_actions=[""],
                reuse=reuse,
            )
        else:
            runner.run_bucket(states[0], cfg, forced_actions=[""], reuse=reuse)

    assert "j=0" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 4. The library: a source bucket that came up short
# ---------------------------------------------------------------------------


def _drop_one_alternative(path: Path, index: int) -> None:
    """Take one alternative off bucket `index`, leaving its request untouched.

    This is what a de-duplication loss looks like on disk: candidate_total_req
    stays at what the run asked for and the bucket holds fewer texts than that.
    """
    rows = [dict(row) for row in read_buckets_jsonl(path)]
    row = rows[index]
    row["candidates"] = [dict(c) for c in row["candidates"]][:-1]
    row["meta"] = dict(row["meta"])
    row["meta"]["candidate_total_got"] = len(row["candidates"])
    _rewrite(path, rows)


@pytest.mark.parametrize("batched", [False, True])
def test_a_short_source_bucket_is_reused_at_the_count_it_has(tmp_path: Path, batched: bool) -> None:
    source_path = tmp_path / "seedA.jsonl"
    _build(seed=SEED_A, path=source_path, num_candidates=3)
    _drop_one_alternative(source_path, index=1)
    short_texts = _texts(list(read_buckets_jsonl(source_path)))

    reuse = replay_mod.load_candidate_reuse(source_path)
    rows, _ = _build(seed=SEED_B, path=tmp_path / "seedB.jsonl", num_candidates=3, batched=batched, reuse=reuse)

    assert _texts(rows) == short_texts
    assert [len(row["candidates"]) for row in rows] == [3, 2, 3, 3]

    short = rows[1]
    assert short["meta"]["n_alternatives_effective"] == 2
    assert short["meta"]["candidate_total_req"] == 3
    assert short["meta"]["candidate_total_got"] == 2
    assert short["meta"]["credit_n"] == 2
    for full in (rows[0], rows[2], rows[3]):
        assert full["meta"]["n_alternatives_effective"] == 3

    # Every alternative that is there is replayed like any other.
    for candidate in short["candidates"]:
        assert len(candidate["returns"]) == 2
        assert len(candidate["next_actions"]) == 2


def test_a_source_without_a_recorded_request_must_match_the_count_it_holds(tmp_path: Path) -> None:
    """With no candidate_total_req there is nothing to tell short from different."""
    source_path = tmp_path / "seedA.jsonl"
    _build(seed=SEED_A, path=source_path, num_candidates=3)
    rows = [dict(row) for row in read_buckets_jsonl(source_path)]
    for row in rows:
        meta = dict(row["meta"])
        meta.pop("candidate_total_req", None)
        row["meta"] = meta
    _rewrite(source_path, rows)

    reuse = replay_mod.load_candidate_reuse(source_path)
    assert reuse.requested_total() is None

    built, _ = _build(seed=SEED_B, path=tmp_path / "ok.jsonl", num_candidates=3, reuse=reuse)
    assert _texts(built) == _texts(rows)

    with pytest.raises(replay_mod.CandidateReuseError):
        _build(seed=SEED_B, path=tmp_path / "bad.jsonl", num_candidates=4, reuse=reuse)


# ---------------------------------------------------------------------------
# 5. Reading the source file
# ---------------------------------------------------------------------------


def test_the_index_is_keyed_on_the_question_and_the_context(tmp_path: Path) -> None:
    source_path = tmp_path / "seedA.jsonl"
    rows, _ = _build(seed=SEED_A, path=source_path)

    reuse = replay_mod.load_candidate_reuse(source_path)
    assert len(reuse) == 4
    assert reuse.requested_total() == 3

    row = rows[0]
    point = reuse.point_for(row["question_id"], row["ctx_hash"])
    assert point.texts == [c["action_text"] for c in row["candidates"]]
    assert point.requested == 3

    # A question id stored as a number still matches a runner that yields an int.
    numeric = [dict(r) for r in rows]
    for item in numeric:
        item["question_id"] = int(str(item["question_id"]).rsplit("q", 1)[1])
    _rewrite(source_path, numeric)
    renumbered = replay_mod.load_candidate_reuse(source_path)
    assert renumbered.point_for(0, rows[0]["ctx_hash"]).texts == point.texts
    assert renumbered.point_for("0", rows[0]["ctx_hash"]).texts == point.texts


def test_the_alternatives_come_back_in_the_recorded_j_order(tmp_path: Path) -> None:
    source_path = tmp_path / "seedA.jsonl"
    rows, _ = _build(seed=SEED_A, path=source_path)
    expected = [c["action_text"] for c in rows[0]["candidates"]]

    shuffled = [dict(row) for row in rows]
    shuffled[0]["candidates"] = list(reversed([dict(c) for c in shuffled[0]["candidates"]]))
    _rewrite(source_path, shuffled)

    reuse = replay_mod.load_candidate_reuse(source_path)
    assert reuse.point_for(rows[0]["question_id"], rows[0]["ctx_hash"]).texts == expected


def test_a_decision_point_recorded_twice_is_refused(tmp_path: Path) -> None:
    source_path = tmp_path / "seedA.jsonl"
    rows, _ = _build(seed=SEED_A, path=source_path)
    _rewrite(source_path, list(rows) + [rows[2]])

    with pytest.raises(replay_mod.CandidateReuseError) as excinfo:
        replay_mod.load_candidate_reuse(source_path)
    assert "twice" in str(excinfo.value)
    assert "e1b-q2" in str(excinfo.value)


def test_an_empty_source_is_refused(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(replay_mod.CandidateReuseError):
        replay_mod.load_candidate_reuse(empty)


def test_a_source_row_without_alternatives_is_refused(tmp_path: Path) -> None:
    source_path = tmp_path / "seedA.jsonl"
    rows, _ = _build(seed=SEED_A, path=source_path)
    damaged = [dict(row) for row in rows]
    damaged[1]["candidates"] = []
    _rewrite(source_path, damaged)

    with pytest.raises(replay_mod.CandidateReuseError) as excinfo:
        replay_mod.load_candidate_reuse(source_path)
    assert "e1b-q1" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 6. The command line
# ---------------------------------------------------------------------------


_FACTORY_MODULE = "c3_wp_r17_fake_runner_factory"


def _register_factory(builder: Callable[[Any], Any]) -> str:
    module = types.ModuleType(_FACTORY_MODULE)
    module.make_runner = builder  # type: ignore[attr-defined]
    sys.modules[_FACTORY_MODULE] = module
    return f"{_FACTORY_MODULE}:make_runner"


def _cli_factory(seen: List[_SeedPolicy], *, n_questions: int = 4) -> Callable[[Any], Any]:
    """Build the runner the CLI asks for, taking the seed from the CLI.

    The real `ReplayRunner.from_cli` puts --seed into runner_meta, and that is
    where both replay paths look when the decoding block carries no seed of its
    own, so the fake has to do the same or the seed under test never arrives.
    """

    def make_runner(args: Any) -> Any:
        policy = _SeedPolicy()
        seen.append(policy)
        return _make_runner(policy, seed=int(args.seed), n_questions=n_questions)

    return make_runner


def _run_cli(
    out_path: Path,
    *extra: str,
    seed: int,
    num_candidates: int = 3,
    limit: int = 4,
    seen: Optional[List[_SeedPolicy]] = None,
    n_questions: int = 4,
) -> None:
    spec = _register_factory(_cli_factory(seen if seen is not None else [], n_questions=n_questions))
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
            str(int(num_candidates)),
            "--num_completions",
            "2",
            "--limit",
            str(int(limit)),
            "--seed",
            str(int(seed)),
            "--decoding_json",
            json.dumps({"temperature": 0.7, "top_p": 0.8, "top_k": 20, "max_new_tokens": 16}, sort_keys=True),
            "--runner_factory",
            spec,
            "--out",
            str(out_path),
            *extra,
        ]
    )


def _rows(path: Path) -> List[Dict[str, Any]]:
    return list(read_buckets_jsonl(path))


@pytest.mark.parametrize("batched", [(), ("--batched",)])
def test_the_cli_reuses_the_alternatives_and_redraws_the_replays(tmp_path: Path, batched: Tuple[str, ...]) -> None:
    source_path = tmp_path / "seedA.jsonl"
    out_path = tmp_path / "seedB.jsonl"
    _run_cli(source_path, *batched, seed=SEED_A)

    seen: List[_SeedPolicy] = []
    _run_cli(out_path, *batched, "--reuse_candidates_from", str(source_path), seed=SEED_B, seen=seen)

    before, after = _rows(source_path), _rows(out_path)
    assert _texts(after) == _texts(before)
    assert _next_actions(after) != _next_actions(before)
    assert _target_prompts(seen[0]) == []

    for row in after:
        assert row["meta"]["candidate_reuse"] is True
        assert row["meta"]["candidates_from"] == str(source_path).replace("\\", "/")
        assert row["meta"]["n_alternatives_effective"] == 3


def test_the_cli_without_the_flag_writes_no_reuse_keys(tmp_path: Path) -> None:
    out_path = tmp_path / "plain.jsonl"
    _run_cli(out_path, seed=SEED_A)
    for row in _rows(out_path):
        assert "candidate_reuse" not in row["meta"]
        assert "candidates_from" not in row["meta"]
        assert "n_alternatives_effective" not in row["meta"]


def test_the_cli_refuses_a_decision_point_the_source_does_not_cover(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source_path = tmp_path / "seedA.jsonl"
    out_path = tmp_path / "seedB.jsonl"
    _run_cli(source_path, seed=SEED_A, limit=3, n_questions=3)

    with pytest.raises(SystemExit) as excinfo:
        _run_cli(
            out_path,
            "--reuse_candidates_from",
            str(source_path),
            seed=SEED_B,
            limit=4,
            n_questions=4,
        )
    assert excinfo.value.code != 0
    assert "e1b-q3" in capsys.readouterr().err


def test_the_cli_refuses_a_count_mismatch_before_it_runs_anything(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source_path = tmp_path / "seedA.jsonl"
    _run_cli(source_path, seed=SEED_A, num_candidates=3)

    seen: List[_SeedPolicy] = []
    with pytest.raises(SystemExit) as excinfo:
        _run_cli(
            tmp_path / "seedB.jsonl",
            "--reuse_candidates_from",
            str(source_path),
            seed=SEED_B,
            num_candidates=2,
            seen=seen,
        )
    assert excinfo.value.code != 0
    assert "alternatives per bucket" in capsys.readouterr().err
    assert not (tmp_path / "seedB.jsonl").exists()
    # Refused before the runner was even built, so nothing was generated.
    assert seen == []


def test_the_cli_refuses_reuse_together_with_the_injected_null_arm(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source_path = tmp_path / "seedA.jsonl"
    _run_cli(source_path, seed=SEED_A)

    with pytest.raises(SystemExit) as excinfo:
        _run_cli(
            tmp_path / "seedB.jsonl",
            "--reuse_candidates_from",
            str(source_path),
            "--inject_literal_candidate",
            "",
            seed=SEED_B,
        )
    assert excinfo.value.code != 0
    assert "--inject_literal_candidate" in capsys.readouterr().err


def test_the_cli_refuses_a_source_file_that_is_not_there(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        _run_cli(tmp_path / "out.jsonl", "--reuse_candidates_from", str(tmp_path / "absent.jsonl"), seed=SEED_B)
    assert excinfo.value.code != 0
    assert "Reuse source not found" in capsys.readouterr().err


def test_the_flag_takes_both_spellings() -> None:
    parser = analysis_cli._build_parser()
    base = [
        "build-buckets",
        "--task",
        "fake-task",
        "--split",
        "FAKE",
        "--policy_ckpt",
        "/unused",
        "--target_role",
        "r0",
        "--out",
        "/tmp/out.jsonl",
    ]
    assert parser.parse_args(base).reuse_candidates_from is None
    for flag in ("--reuse_candidates_from", "--reuse-candidates-from"):
        parsed = parser.parse_args(base + [flag, "seedA/buckets.jsonl"])
        assert parsed.reuse_candidates_from == "seedA/buckets.jsonl"


# ---------------------------------------------------------------------------
# 7. The path shape written into meta.candidates_from
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "given",
    [
        "20_data/results/E1b/a2/4b/n3_c2/seedA/buckets.jsonl",
        "/root/c3/results/E1b/a2/4b/n3_c2/seedA/buckets.jsonl",
        "seedA/buckets.jsonl",
        "C:\\work\\20_data\\results\\E1b\\seedA\\buckets.jsonl",
    ],
)
def test_the_source_label_is_the_path_shape_of_the_results_layout(given: str) -> None:
    """Pinned against the statement of record so the two copies cannot drift."""
    assert analysis_cli._reuse_source_label(given) == contract_source_path(given)


def test_the_source_label_cuts_at_the_last_data_segment() -> None:
    label = analysis_cli._reuse_source_label("20_data/results/E1b/a2/4b/n3_c2/seedA/buckets.jsonl")
    assert label == "20_data/results/E1b/a2/4b/n3_c2/seedA/buckets.jsonl"
    assert "\\" not in analysis_cli._reuse_source_label("seedA\\buckets.jsonl")
