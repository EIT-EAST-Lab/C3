"""Tests for the E3a null action injection.

What is pinned here.

First, the injection itself. `build-buckets --inject_literal_candidate <text>`
hands the text to `ReplayRunner.run_bucket` as a forced action, so it becomes
alternative j=0, and `--num_candidates` keeps meaning the number of sampled
alternatives: the bucket carries one more alternative than it asks for. The
sampled alternatives must be exactly the ones the same run would have drawn
without the injection, because the injected arm is not supposed to move the
sampling.

Second, the record. Every bucket of an injected run carries `meta.null_arm`,
`meta.null_form`, `meta.null_text` and `meta.null_form_note`, and a value
passed in through `--meta_json` cannot contradict them.

Third, the two forms. `empty` and `deleted` are one construction in this code
base: `build_render_context` drops a role whose output is the empty string, and
both prompt composers leave the Context section out when it is empty, so the
downstream prompt is byte for byte the prompt of a team where the role never
spoke. The tests assert that equality on the real renderer rather than assume
it, since the whole justification of `--null_form deleted` rests on it.

Fourth, that without the new flags nothing moves. `LEGACY_BUCKETS_SHA256` was
taken from `build-buckets` on the fixture below with the code that had no
injection flag, and is compared against the output of the current code.

Fifth, the cell driver `scripts/70_rebuild/e3a_cells.py`, through --dry-run.

Everything runs on fake policies, so no model, no GPU and no torch are needed.
"""

from __future__ import annotations

import hashlib
import json
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

from c3.analysis import analysis as analysis_cli  # noqa: E402
from c3.analysis import replay as replay_mod  # noqa: E402
from c3.analysis import replay_batched as batched_mod  # noqa: E402
from c3.analysis.buckets import read_buckets_jsonl, write_buckets_jsonl  # noqa: E402
from c3.protocol.prompt_render import build_render_context  # noqa: E402


# Output of `build-buckets` on the fixture below, taken with the code that had
# no --inject_literal_candidate flag (2026-09-15). Adding the flag must
# not move these bytes when the flag is absent.
LEGACY_BUCKETS_SHA256 = "f2e77be137b4f5c65d91554f548f9bc942626e29ec4becbcb00ff59c8d50287e"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _digest(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=6).hexdigest()


class _Policy:
    """Deterministic policy whose text is a hash of the prompt.

    The text ignores the seed, so the sequential path and the batched path can
    be compared byte for byte; they derive seeds differently on purpose. One
    call returns n distinct texts, so nothing is lost to de-duplication.
    """

    def __init__(self) -> None:
        self.sample_calls = 0
        self.sample_many_calls = 0
        self.sequences = 0
        self.requests: List[Tuple[str, int, Any]] = []

    def texts(self, prompt: str, n: int, seed: Any) -> List[str]:
        head = _digest(str(prompt))
        return [f"[{head}|{k}]" for k in range(int(n))]

    def sample(self, prompt: str, n: int = 1, **decoding: Any) -> List[str]:
        self.sample_calls += 1
        self.sequences += int(n)
        self.requests.append((str(prompt), int(n), decoding.get("seed")))
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
            self.requests.append((str(prompt), int(n), seed))
            out.append(self.texts(prompt, n, seed))
        return out


class _EchoPolicy(_Policy):
    """Policy that answers with a fixed text per role, whatever the prompt.

    Used where the point is the shape of the bucket rather than the texts.
    """

    def texts(self, prompt: str, n: int, seed: Any) -> List[str]:
        return [f"<{k}>" for k in range(int(n))]


class _ClashingPolicy(_Policy):
    """Policy whose first sample of every call is the empty string.

    The realistic way an injected empty arm meets de-duplication: the model
    itself falls silent, and that sample must not become a second null arm.
    """

    def texts(self, prompt: str, n: int, seed: Any) -> List[str]:
        return [""] + super().texts(prompt, n, seed)[1:]


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


# ---------------------------------------------------------------------------
# Runner and config builders
# ---------------------------------------------------------------------------


#: Three roles, which is the shape of the E3a workflow: the measured role, the
#: role whose answer is recorded, and one more after it.
ROLES = ["r0", "r1", "r2"]


def _make_runner(
    policy: Any,
    *,
    n_questions: int = 4,
    tag: str = "e3a",
    roles: Sequence[str] = ROLES,
) -> replay_mod.ReplayRunner:
    dataset = [
        {"id": f"{tag}-q{i}", "input": f"question {i} of {tag}", "answer": "42"} for i in range(n_questions)
    ]
    return replay_mod.ReplayRunner(
        task_spec=f"wp-r10-{tag}",
        policy=policy,
        evaluator=_Evaluator(),
        prompt_renderer=_Renderer(),
        runner_meta={"task": "fake-task", "split": "FAKE", "method": "sft", "seed": 0},
        dataset=dataset,
        eval_suite_name="FAKE",
        roles_topo=list(roles),
    )


def _restart_states(runner: replay_mod.ReplayRunner, limit: int = 4) -> List[Any]:
    return list(
        runner.iter_restart_states(task="fake-task", split="FAKE", target_role="r0", limit=limit, seed=0)
    )


def _make_cfg(*, num_candidates: int, num_completions: int = 2, include_real: bool = False) -> Any:
    return replay_mod.ReplayConfig(
        target_role="r0",
        num_candidates=int(num_candidates),
        num_completions_per_candidate=int(num_completions),
        decoding={"temperature": 0.7, "top_p": 0.8, "top_k": 20, "max_new_tokens": 16, "seed": 0},
        record_next_teammate=True,
        next_role="r1",
        include_real_as_j0=bool(include_real),
        num_extra_v_samples=0,
        prefix_decoding={},
    )


def _as_rows(buckets: Sequence[Any], path: Path) -> List[Dict[str, Any]]:
    """Write buckets through the real writer and read them back as dicts."""
    write_buckets_jsonl(path, list(buckets), overwrite=True)
    return list(read_buckets_jsonl(path))


# ---------------------------------------------------------------------------
# 1. The forced alternative inside run_bucket
# ---------------------------------------------------------------------------


def test_the_injected_text_is_alternative_zero() -> None:
    runner = _make_runner(_Policy())
    state = _restart_states(runner)[0]
    bucket = runner.run_bucket(state, _make_cfg(num_candidates=4), forced_actions=[""])

    assert bucket.candidates[0].action_text == ""
    assert len(bucket.candidates) == 4
    assert all(c.action_text != "" for c in bucket.candidates[1:])


def test_a_non_empty_injection_also_lands_at_zero() -> None:
    runner = _make_runner(_Policy())
    state = _restart_states(runner)[0]
    bucket = runner.run_bucket(state, _make_cfg(num_candidates=3), forced_actions=["(no message)"])

    assert bucket.candidates[0].action_text == "(no message)"
    assert len(bucket.candidates) == 3


def test_a_silent_sample_does_not_become_a_second_null_arm() -> None:
    runner = _make_runner(_ClashingPolicy())
    state = _restart_states(runner)[0]
    bucket = runner.run_bucket(state, _make_cfg(num_candidates=4), forced_actions=[""])

    texts = [c.action_text for c in bucket.candidates]
    assert texts[0] == ""
    assert len(texts) == len(set(texts))
    assert texts.count("") == 1
    # Three of the four slots could be filled; the short bucket is recorded.
    assert bucket.meta["candidate_total_req"] == 4
    assert bucket.meta["candidate_total_got"] == len(texts) == 3
    assert bucket.meta["credit_n"] == 3


def test_the_batched_path_also_refuses_a_duplicate_null_arm() -> None:
    runner = _make_runner(_ClashingPolicy())
    buckets = batched_mod.run_buckets_batched(
        runner,
        _restart_states(runner),
        _make_cfg(num_candidates=4),
        target_role="r0",
        next_role="r1",
        forced_actions=[""],
    )
    for bucket in buckets:
        texts = [c.action_text for c in bucket.candidates]
        assert texts[0] == ""
        assert texts.count("") == 1
        assert len(texts) == len(set(texts)) == 3


def test_the_injected_arm_is_replayed_like_any_other_alternative() -> None:
    runner = _make_runner(_Policy())
    state = _restart_states(runner)[0]
    bucket = runner.run_bucket(state, _make_cfg(num_candidates=3, num_completions=2), forced_actions=[""])

    for cand in bucket.candidates:
        assert len(cand.returns) == 2
        assert len(cand.next_actions) == 2


# ---------------------------------------------------------------------------
# 2. The command line: what the flag does to the bucket
# ---------------------------------------------------------------------------


_FACTORY_MODULE = "c3_wp_r10_fake_runner_factory"


def _register_factory(builder: Callable[[], Any]) -> str:
    module = types.ModuleType(_FACTORY_MODULE)
    module.make_runner = lambda _args: builder()  # type: ignore[attr-defined]
    sys.modules[_FACTORY_MODULE] = module
    return f"{_FACTORY_MODULE}:make_runner"


def _cli_runner(policy: Any = None) -> replay_mod.ReplayRunner:
    return _make_runner(policy if policy is not None else _Policy())


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


def _rows(out_path: Path) -> List[Dict[str, Any]]:
    return list(read_buckets_jsonl(out_path))


def test_the_flag_adds_one_alternative_and_keeps_the_sampled_ones() -> None:
    """--num_candidates counts sampled alternatives, the null arm comes on top."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        plain = Path(tmp) / "plain.jsonl"
        injected = Path(tmp) / "injected.jsonl"
        _run_cli(plain)
        _run_cli(injected, "--inject_literal_candidate", "")

        plain_rows, injected_rows = _rows(plain), _rows(injected)
        assert len(plain_rows) == len(injected_rows) == 4

        for before, after in zip(plain_rows, injected_rows):
            assert len(before["candidates"]) == 3
            assert len(after["candidates"]) == 4
            assert after["candidates"][0]["action_text"] == ""
            # The three sampled alternatives are the ones the plain run drew:
            # same prompt, same seed, same count, so the injection cannot move
            # the measurement it is compared against.
            assert [c["action_text"] for c in after["candidates"][1:]] == [
                c["action_text"] for c in before["candidates"]
            ]


def test_the_credit_set_covers_every_alternative_including_the_null_arm() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "injected.jsonl"
        _run_cli(out, "--inject_literal_candidate", "")
        for row in _rows(out):
            meta = row["meta"]
            assert meta["credit_n"] == 4
            assert meta["candidate_total_req"] == 4
            assert meta["candidate_total_got"] == 4
            assert meta["v_extra_n"] == 0
            # The consumer takes range(credit_n) and then drops null_arm, so
            # this leaves exactly the three sampled alternatives.
            assert [j for j in range(meta["credit_n"]) if j != meta["null_arm"]] == [1, 2, 3]


def test_the_three_null_keys_are_written() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "injected.jsonl"
        _run_cli(out, "--inject_literal_candidate", "")
        for row in _rows(out):
            meta = row["meta"]
            assert meta["null_arm"] == 0
            assert meta["null_form"] == "empty"
            assert meta["null_text"] == ""


def test_the_deleted_form_is_recorded_with_its_note() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "deleted.jsonl"
        _run_cli(out, "--inject_literal_candidate", "", "--null_form", "deleted")
        for row in _rows(out):
            assert row["meta"]["null_form"] == "deleted"
            assert row["meta"]["null_form_note"] == analysis_cli.NULL_FORM_NOTE
            assert "alias deleted" in row["meta"]["null_form_note"]


def test_the_two_forms_differ_only_by_the_recorded_label() -> None:
    """Same seeds, same prompts: the two cells are one measurement."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        empty = Path(tmp) / "empty.jsonl"
        deleted = Path(tmp) / "deleted.jsonl"
        _run_cli(empty, "--inject_literal_candidate", "", "--null_form", "empty")
        _run_cli(deleted, "--inject_literal_candidate", "", "--null_form", "deleted")

        for before, after in zip(_rows(empty), _rows(deleted)):
            assert after["bucket_id"] == before["bucket_id"]
            assert after["candidates"] == before["candidates"]
            meta_before = dict(before["meta"])
            meta_after = dict(after["meta"])
            assert meta_before.pop("null_form") == "empty"
            assert meta_after.pop("null_form") == "deleted"
            assert meta_after == meta_before


def test_a_literal_text_is_recorded_verbatim() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "literal.jsonl"
        _run_cli(out, "--inject_literal_candidate", "  ")
        for row in _rows(out):
            assert row["meta"]["null_text"] == "  "
            assert row["candidates"][0]["action_text"] == "  "


def test_meta_json_cannot_contradict_the_recorded_injection() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "stamped.jsonl"
        _run_cli(
            out,
            "--inject_literal_candidate",
            "",
            "--null_form",
            "deleted",
            "--meta_json",
            json.dumps({"null_arm": 7, "null_form": "empty", "rule": "e3a_deleted"}, sort_keys=True),
        )
        for row in _rows(out):
            assert row["meta"]["null_arm"] == 0
            assert row["meta"]["null_form"] == "deleted"
            assert row["meta"]["rule"] == "e3a_deleted"


def test_the_injection_refuses_to_claim_j0_is_the_real_action() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "never.jsonl"
        with pytest.raises(SystemExit):
            _run_cli(out, "--inject_literal_candidate", "", "--include_real_as_j0", "true")
        assert not out.exists()


def test_include_real_as_j0_still_works_without_an_injection() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "real.jsonl"
        _run_cli(out, "--include_real_as_j0", "true")
        for row in _rows(out):
            assert row["meta"]["real_j"] == 0
            assert "null_arm" not in row["meta"]


# ---------------------------------------------------------------------------
# 3. Without the flags nothing moves
# ---------------------------------------------------------------------------


def test_without_the_flag_the_output_is_the_one_the_old_code_wrote(tmp_path: Path) -> None:
    out = tmp_path / "legacy.jsonl"
    _run_cli(out)
    rows = _rows(out)

    assert len(rows) == 4
    assert all("null_arm" not in row["meta"] for row in rows)
    assert all("null_form" not in row["meta"] for row in rows)
    assert hashlib.sha256(out.read_bytes()).hexdigest() == LEGACY_BUCKETS_SHA256


def test_null_form_alone_writes_nothing(tmp_path: Path) -> None:
    """The form is a label on an injection; with no injection it is inert."""
    out = tmp_path / "form_only.jsonl"
    _run_cli(out, "--null_form", "deleted")
    assert hashlib.sha256(out.read_bytes()).hexdigest() == LEGACY_BUCKETS_SHA256


def test_the_flags_are_off_by_default() -> None:
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
    assert args.inject_literal_candidate is None
    assert args.null_form == "empty"

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
            "--inject_literal_candidate",
            "",
            "--null_form",
            "deleted",
        ]
    )
    assert flagged.inject_literal_candidate == ""
    assert flagged.null_form == "deleted"


def test_an_unknown_form_is_refused() -> None:
    parser = analysis_cli._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
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
                "--null_form",
                "blank",
            ]
        )


# ---------------------------------------------------------------------------
# 4. The batched path carries the injection too
# ---------------------------------------------------------------------------


def test_the_batched_path_puts_the_injection_at_zero() -> None:
    runner = _make_runner(_Policy())
    states = _restart_states(runner)
    cfg = _make_cfg(num_candidates=4)
    buckets = batched_mod.run_buckets_batched(
        runner, states, cfg, target_role="r0", next_role="r1", forced_actions=[""]
    )

    assert len(buckets) == 4
    for bucket in buckets:
        assert bucket.candidates[0].action_text == ""
        assert len(bucket.candidates) == 4
        assert bucket.meta["credit_n"] == 4


def test_the_two_paths_agree_on_the_injected_bucket(tmp_path: Path) -> None:
    """Seed independent policy, so the agreement can be byte for byte."""
    cfg = _make_cfg(num_candidates=4)

    seq_runner = _make_runner(_Policy())
    seq = [
        seq_runner.run_bucket(state, cfg, forced_actions=[""]) for state in _restart_states(seq_runner)
    ]

    bat_runner = _make_runner(_Policy())
    bat = batched_mod.run_buckets_batched(
        bat_runner,
        _restart_states(bat_runner),
        cfg,
        target_role="r0",
        next_role="r1",
        forced_actions=[""],
    )

    seq_rows = _as_rows(seq, tmp_path / "sequential.jsonl")
    bat_rows = _as_rows(bat, tmp_path / "batched.jsonl")
    for before, after in zip(seq_rows, bat_rows):
        stripped = dict(after)
        meta = dict(after["meta"])
        assert meta.pop("seed_scheme") == batched_mod.SEED_SCHEME
        stripped["meta"] = meta
        assert stripped == before


def test_the_batched_path_is_unchanged_without_an_injection(tmp_path: Path) -> None:
    cfg = _make_cfg(num_candidates=3)
    runner_a = _make_runner(_Policy())
    plain = batched_mod.run_buckets_batched(
        runner_a, _restart_states(runner_a), cfg, target_role="r0", next_role="r1"
    )
    runner_b = _make_runner(_Policy())
    explicit_none = batched_mod.run_buckets_batched(
        runner_b,
        _restart_states(runner_b),
        cfg,
        target_role="r0",
        next_role="r1",
        forced_actions=None,
    )
    assert _as_rows(plain, tmp_path / "a.jsonl") == _as_rows(explicit_none, tmp_path / "b.jsonl")


def test_the_batched_path_does_not_sample_a_real_action_over_the_injection() -> None:
    """include_real_as_j0 asks for a sample that the injection already filled."""
    runner = _make_runner(_Policy())
    states = _restart_states(runner)
    cfg = _make_cfg(num_candidates=3, include_real=True)

    with_injection = batched_mod.run_buckets_batched(
        runner, states, cfg, target_role="r0", next_role="r1", forced_actions=["x"]
    )
    assert all(b.candidates[0].action_text == "x" for b in with_injection)
    assert all(len(b.candidates) == 3 for b in with_injection)


def test_the_batched_flag_and_the_injection_work_together(tmp_path: Path) -> None:
    out = tmp_path / "batched.jsonl"
    _run_cli(out, "--batched", "--inject_literal_candidate", "", "--null_form", "deleted")
    rows = _rows(out)
    assert len(rows) == 4
    for row in rows:
        assert row["meta"]["seed_scheme"] == batched_mod.SEED_SCHEME
        assert row["meta"]["null_arm"] == 0
        assert row["meta"]["null_form"] == "deleted"
        assert len(row["candidates"]) == 4
        assert row["candidates"][0]["action_text"] == ""


def test_the_batched_call_counts_stay_flat_under_the_injection() -> None:
    """One stage A call, then one call per (replay, downstream role)."""
    policy = _Policy()
    runner = _make_runner(policy)
    cfg = _make_cfg(num_candidates=4, num_completions=2)
    batched_mod.run_buckets_batched(
        runner, _restart_states(runner), cfg, target_role="r0", next_role="r1", forced_actions=[""]
    )
    # roles after r0 are r1 and r2, two replays: 1 + 2 * 2 = 5 calls.
    assert policy.sample_many_calls == 5
    assert policy.sample_calls == 0


# ---------------------------------------------------------------------------
# 5. Why the deleted form needs no separate assembly
# ---------------------------------------------------------------------------


def test_an_empty_output_leaves_no_trace_in_the_context() -> None:
    silent = build_render_context(question="q", role_outputs={"a": "", "b": "B"}, topo_so_far=["a", "b"])
    removed = build_render_context(question="q", role_outputs={"b": "B"}, topo_so_far=["a", "b"])
    assert silent["context"] == removed["context"] == "B"


def test_an_empty_first_output_leaves_no_separator() -> None:
    silent = build_render_context(
        question="q", role_outputs={"a": "", "b": "B", "c": "C"}, topo_so_far=["a", "b", "c"]
    )
    assert silent["context"] == "B\n\nC"
    assert not silent["context"].startswith("\n")


@dataclass
class _RoleSpec:
    name: str
    prompt: str
    depends_on: Tuple[str, ...] = ()


@dataclass
class _TaskSpec:
    roles: List[_RoleSpec] = field(default_factory=list)
    environment: Dict[str, Any] = field(default_factory=dict)
    env_name: str = "fake"


class _Tokenizer:
    """Minimal stand-in for the chat template branch of the composer."""

    def apply_chat_template(self, messages: Sequence[Mapping[str, str]], **_: Any) -> str:
        return "\n".join(f"<{m['role']}>{m['content']}" for m in messages)


def _openrlhf_renderer() -> Any:
    task_spec = _TaskSpec(
        roles=[
            _RoleSpec("reasoner", "you are the reasoner"),
            _RoleSpec("actor", "you are the actor", ("reasoner",)),
            _RoleSpec("verifier", "you are the verifier", ("actor",)),
        ]
    )
    return replay_mod._OpenRLHFPromptRenderer(task_spec=task_spec, tokenizer=_Tokenizer())


def test_the_real_renderer_cannot_tell_an_empty_message_from_a_deleted_one() -> None:
    renderer = _openrlhf_renderer()
    roles = ["reasoner", "actor", "verifier"]

    silent = renderer.render_role_prompt(
        question="q", roles_topo=roles, role_outputs={"reasoner": ""}, target_role="actor"
    )
    removed = renderer.render_role_prompt(
        question="q", roles_topo=roles, role_outputs={}, target_role="actor"
    )
    assert silent == removed
    assert "Context:" not in silent

    spoke = renderer.render_role_prompt(
        question="q", roles_topo=roles, role_outputs={"reasoner": "plan"}, target_role="actor"
    )
    assert spoke != silent
    assert "Context:" in spoke


def test_the_same_holds_on_the_configuration_e3a_will_run() -> None:
    """The check above, on the real three agent task rather than a stand-in."""
    from c3.task.config import load_task, topo_sort_roles  # noqa: PLC0415

    spec = load_task(str(REPO_ROOT / "configs" / "tasks" / "math_a3.yaml"))
    roles = [r.name for r in topo_sort_roles(spec.roles)]
    assert roles == ["reasoner", "actor", "verifier"]

    renderer = replay_mod._OpenRLHFPromptRenderer(task_spec=spec, tokenizer=_Tokenizer())
    question = "What is 2 + 2?"

    for target, silent_outputs, removed_outputs in (
        ("actor", {"reasoner": ""}, {}),
        ("verifier", {"reasoner": "", "actor": "so the answer is 4"}, {"actor": "so the answer is 4"}),
    ):
        silent = renderer.render_role_prompt(
            question=question, roles_topo=roles, role_outputs=silent_outputs, target_role=target
        )
        removed = renderer.render_role_prompt(
            question=question, roles_topo=roles, role_outputs=removed_outputs, target_role=target
        )
        assert silent == removed

    spoke = renderer.render_role_prompt(
        question=question, roles_topo=roles, role_outputs={"reasoner": "plan"}, target_role="actor"
    )
    assert "Context:" in spoke


def test_the_deletion_is_invisible_two_roles_downstream() -> None:
    renderer = _openrlhf_renderer()
    roles = ["reasoner", "actor", "verifier"]

    silent = renderer.render_role_prompt(
        question="q",
        roles_topo=roles,
        role_outputs={"reasoner": "", "actor": "draft"},
        target_role="verifier",
    )
    removed = renderer.render_role_prompt(
        question="q", roles_topo=roles, role_outputs={"actor": "draft"}, target_role="verifier"
    )
    assert silent == removed
    assert "draft" in silent


# ---------------------------------------------------------------------------
# 6. The cell driver
# ---------------------------------------------------------------------------


def _e3a_cells() -> Any:
    script_dir = REPO_ROOT / "scripts" / "70_rebuild"
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    import e3a_cells  # noqa: PLC0415

    return e3a_cells


def _e3a_commands(capsys: pytest.CaptureFixture, *extra: str) -> List[str]:
    assert _e3a_cells().main(["--model_root", "/models", "--dry-run", *extra]) == 0
    out = capsys.readouterr().out
    return [line for line in out.splitlines() if line and not line.startswith("#")]


def _argv_of(command: str) -> List[str]:
    import shlex

    return shlex.split(command)


def _flag(argv: Sequence[str], name: str) -> str:
    idx = argv.index(name)
    return argv[idx + 1]


def test_the_driver_emits_one_command_per_form(capsys: pytest.CaptureFixture) -> None:
    commands = _e3a_commands(capsys)
    assert len(commands) == 2

    forms = [_flag(_argv_of(line), "--null_form") for line in commands]
    assert forms == ["empty", "placeholder"]


def test_the_cell_driver_asks_for_the_measurement_the_analysis_plan_fixed(
    capsys: pytest.CaptureFixture,
) -> None:
    for line in _e3a_commands(capsys):
        argv = _argv_of(line)
        assert _flag(argv, "--task") == "configs/tasks/math_a3.yaml"
        assert _flag(argv, "--target_role") == "reasoner"
        assert _flag(argv, "--next_role") == "actor"
        assert _flag(argv, "--limit") == "150"
        assert _flag(argv, "--num_candidates") == "4"
        assert _flag(argv, "--num_completions") == "4"
        assert _flag(argv, "--seed") == "0"
        assert _flag(argv, "--split") == "MATHPOOL"
        assert _flag(argv, "--inject_literal_candidate") == _e3a_cells().NULL_TEXTS[_flag(argv, "--null_form")]


def test_the_cell_driver_writes_where_the_results_layout_says(capsys: pytest.CaptureFixture) -> None:
    outs = [_flag(_argv_of(line), "--out") for line in _e3a_commands(capsys, "--results_root", "/abs/results")]
    assert outs == [
        "/abs/results/E3a/a3/4b/empty/buckets.jsonl",
        "/abs/results/E3a/a3/4b/placeholder/buckets.jsonl",
    ]
    assert len(set(outs)) == 2


def test_the_cell_driver_stamps_the_meta_the_results_layout_requires(capsys: pytest.CaptureFixture) -> None:
    required = {
        "workflow",
        "model",
        "rule",
        "n_alternatives",
        "replays_per_alt",
        "temperature",
        "top_p",
        "top_k",
        "seed",
        "policy_tag",
        "dataset",
        "question_order_seed",
        "context_scope",
    }
    for line, form in zip(_e3a_commands(capsys), ["empty", "placeholder"]):
        meta = json.loads(_flag(_argv_of(line), "--meta_json"))
        assert required <= set(meta)
        assert meta["workflow"] == "a3"
        assert meta["model"] == "4b"
        assert meta["rule"] == f"e3a_{form}"
        assert meta["policy_tag"] == "sft"
        assert meta["context_scope"] == "ancestors"
        assert meta["dataset"] == "MATHPOOL"
        # The null arm is not one of the alternatives it is compared against.
        assert meta["n_alternatives"] == 4
        assert meta["replays_per_alt"] == 4
        assert meta["include_real_as_j0"] is False


def test_the_driver_reads_the_sampling_parameters_it_records(capsys: pytest.CaptureFixture) -> None:
    """The recorded decoding is the one the run will apply, not a copy by hand."""
    import yaml

    raw = yaml.safe_load((REPO_ROOT / "configs" / "analysis.yaml").read_text(encoding="utf-8"))
    decoding = raw["influence"]["decoding"]

    for line in _e3a_commands(capsys):
        meta = json.loads(_flag(_argv_of(line), "--meta_json"))
        assert meta["temperature"] == decoding["temperature"]
        assert meta["top_p"] == decoding["top_p"]
        assert meta["top_k"] == decoding["top_k"]


def test_one_form_can_be_selected(capsys: pytest.CaptureFixture) -> None:
    commands = _e3a_commands(capsys, "--only", "placeholder")
    assert len(commands) == 1
    assert _flag(_argv_of(commands[0]), "--null_form") == "placeholder"


def test_an_unknown_form_is_refused_by_the_driver() -> None:
    assert _e3a_cells().main(["--model_root", "/models", "--dry-run", "--only", "blank"]) == 2


def test_the_driver_reports_the_cell_count(capsys: pytest.CaptureFixture) -> None:
    assert _e3a_cells().main(["--model_root", "/models", "--dry-run"]) == 0
    assert "# cells: 2" in capsys.readouterr().out


def test_the_emitted_command_survives_a_shell_round_trip(capsys: pytest.CaptureFixture) -> None:
    """The injected text is empty, which a shell drops unless it is quoted."""
    for line in _e3a_commands(capsys):
        assert ("--inject_literal_candidate ''" in line) or ("--inject_literal_candidate 'No message.'" in line)
        argv = _argv_of(line)
        assert argv[argv.index("--inject_literal_candidate") + 1] in ("", "No message.")


def test_the_driver_batches_by_default_and_can_be_told_not_to(capsys: pytest.CaptureFixture) -> None:
    for line in _e3a_commands(capsys):
        assert " --batched" in line
    for line in _e3a_commands(capsys, "--no-batched"):
        assert "--batched" not in line


def test_the_driver_points_at_the_base_model(capsys: pytest.CaptureFixture) -> None:
    for line in _e3a_commands(capsys, "--model_root", "/models"):
        argv = _argv_of(line)
        assert _flag(argv, "--policy_ckpt") == "/models/Qwen3-4B-Instruct-2507"
        assert _flag(argv, "--method") == "sft"


def test_the_driver_refuses_a_measurement_size_that_makes_no_sense() -> None:
    assert _e3a_cells().main(["--model_root", "/models", "--dry-run", "--limit", "0"]) == 2
    assert _e3a_cells().main(["--model_root", "/models", "--dry-run", "--alternatives", "0"]) == 2
    assert _e3a_cells().main(["--model_root", "/models", "--dry-run", "--completions", "0"]) == 2


def test_the_emitted_command_is_the_analysis_cli(capsys: pytest.CaptureFixture) -> None:
    for line in _e3a_commands(capsys):
        argv = _argv_of(line)
        assert argv[1:4] == ["-m", "c3.analysis.analysis", "build-buckets"]


def test_the_emitted_command_builds_the_bucket_the_aggregator_expects(
    capsys: pytest.CaptureFixture, tmp_path: Path
) -> None:
    """End to end on the emitted argv, with a fake runner in place of the model.

    Everything except the policy is the real thing: the real defaults file, the
    real resolution of --defaults_section auto for the reasoner, the real
    writer. Only the size of the cell is cut down.
    """
    commands = _e3a_commands(capsys)
    spec = _register_factory(
        lambda: _make_runner(_Policy(), roles=["reasoner", "actor", "verifier"], tag="a3")
    )

    for line, form in zip(commands, ["empty", "placeholder"]):
        out = tmp_path / f"{form}.jsonl"
        argv = _argv_of(line)[3:]  # drop python -m <module>
        analysis_cli.main(
            argv
            + [
                "--analysis_yaml",
                str(REPO_ROOT / "configs" / "analysis.yaml"),
                "--runner_factory",
                spec,
                "--limit",
                "4",
                "--out",
                str(out),
            ]
        )

        rows = _rows(out)
        assert len(rows) == 4
        for row in rows:
            meta = row["meta"]
            assert meta["null_arm"] == 0
            assert meta["null_form"] == form
            assert meta["null_text"] == _e3a_cells().NULL_TEXTS[form]
            assert meta["rule"] == f"e3a_{form}"
            assert meta["context_scope"] == "ancestors"
            assert meta["record_next_teammate"] is True
            assert meta["include_real_as_j0"] is False
            assert meta["seed_scheme"] == batched_mod.SEED_SCHEME
            assert meta["credit_n"] == 5
            assert len(row["candidates"]) == 5
            assert row["candidates"][0]["action_text"] == _e3a_cells().NULL_TEXTS[form]
            assert all(c["next_actions"] for c in row["candidates"])


def test_the_emitted_commands_parse_as_the_cli_would(capsys: pytest.CaptureFixture) -> None:
    """The strongest check available here: the real parser accepts them."""
    parser = analysis_cli._build_parser()
    for line, form in zip(_e3a_commands(capsys), ["empty", "placeholder"]):
        argv = _argv_of(line)
        args = parser.parse_args(argv[3:])  # drop python -m <module>
        assert args.inject_literal_candidate == _e3a_cells().NULL_TEXTS[form]
        assert args.null_form == form
        assert args.num_candidates == 4
        assert args.num_completions == 4
        assert args.limit == 150
        assert args.target_role == "reasoner"
        assert args.next_role == "actor"
