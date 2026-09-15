"""Contract tests for the E1 cell driver and for the bucket meta injection.

Two things are pinned here.

First, `scripts/70_rebuild/e1_cells.py`: the sweep it emits is the whole of the
E1 measurement, so its cell count, its output paths, the measured position per
workflow and the meta block it attaches are all paper-facing facts. The tests
read the commands the script prints, which needs neither a GPU nor a model.

Second, the `--meta_json` flag of `c3.analysis.analysis build-buckets`: the
experiment level keys the results layout requires are not derivable from the
runner, so the caller passes them in, and they must reach every written bucket
without displacing anything the runner recorded. A fake runner supplies the
buckets, so this too runs without a model.
"""

from __future__ import annotations

import json
import shlex
import sys
import types
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "scripts" / "70_rebuild"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import e1_cells  # noqa: E402
import replay_tokens  # noqa: E402

from c3.analysis import analysis as analysis_cli  # noqa: E402
from c3.analysis import replay as replay_mod  # noqa: E402
from c3.analysis.buckets import read_buckets_jsonl  # noqa: E402


MODEL_ROOT = "/models"

# Every key the results layout demands on an E1 bucket.
REQUIRED_META_KEYS = {
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
    "include_real_as_j0",
}

# Revision 1 of the results layout retired these two as directories of their own.
RETIRED_RULE_NAMES = ("fixed_b8", "pd_n4")

EXPECTED_POSITIONS = {
    "a2": ("reasoner", "actor"),
    "a3": ("reasoner", "actor"),
    "mt4": ("reasoner_1", "actor_1"),
    "branch": ("planner", "solver_a"),
    "c5": ("planner", "solver"),
    "c10": ("reader", "planner"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dry_run(capsys: pytest.CaptureFixture, *extra: str) -> List[str]:
    """Run the driver in dry-run mode and return the printed lines."""
    code = e1_cells.main(["--model_root", MODEL_ROOT, "--dry-run", *extra])
    assert code == 0
    out = capsys.readouterr().out
    return out.splitlines()


def _commands(lines: Sequence[str]) -> List[str]:
    return [line for line in lines if line and not line.startswith("#")]


def _flag(argv: Sequence[str], name: str) -> str:
    """Return the single value of one flag, failing loudly when repeated."""
    hits = [i for i, tok in enumerate(argv) if tok == name]
    assert len(hits) == 1, f"{name} appears {len(hits)} times in {argv}"
    index = hits[0]
    assert index + 1 < len(argv), f"{name} has no value"
    return argv[index + 1]


# ---------------------------------------------------------------------------
# 1. The sweep
# ---------------------------------------------------------------------------


def test_dry_run_emits_six_workflows_times_two_models_times_five_branching_factors(
    capsys: pytest.CaptureFixture,
) -> None:
    lines = _dry_run(capsys)
    commands = _commands(lines)

    assert len(commands) == 60
    assert lines[-1] == "# cells: 60"


def test_every_emitted_command_is_a_build_buckets_call(capsys: pytest.CaptureFixture) -> None:
    for line in _commands(_dry_run(capsys)):
        argv = shlex.split(line)
        assert argv[:4] == ["python", "-m", "c3.analysis.analysis", "build-buckets"]


def test_only_selects_exactly_one_cell_with_the_right_arguments(capsys: pytest.CaptureFixture) -> None:
    commands = _commands(_dry_run(capsys, "--only", "a3/4b/3"))
    assert len(commands) == 1

    argv = shlex.split(commands[0])
    assert _flag(argv, "--task") == "configs/tasks/math_a3.yaml"
    assert _flag(argv, "--split") == "MATHPOOL"
    assert _flag(argv, "--policy_ckpt") == "/models/Qwen3-4B-Instruct-2507"
    assert _flag(argv, "--target_role") == "reasoner"
    assert _flag(argv, "--next_role") == "actor"
    assert _flag(argv, "--num_candidates") == "3"
    assert _flag(argv, "--num_completions") == "4"
    assert _flag(argv, "--limit") == "500"
    assert _flag(argv, "--seed") == "0"
    assert _flag(argv, "--analysis_yaml") == "configs/analysis.yaml"
    assert _flag(argv, "--engine") == "auto"
    assert _flag(argv, "--out") == "20_data/results/E1/a3/4b/sweep_n3/buckets.jsonl"

    meta = json.loads(_flag(argv, "--meta_json"))
    assert meta["workflow"] == "a3"
    assert meta["model"] == "4b"
    assert meta["rule"] == "sweep_n3"
    assert meta["n_alternatives"] == 3


def test_every_cell_of_the_sweep_is_drawn_from_the_screened_pool(capsys: pytest.CaptureFixture) -> None:
    """`--split` is one value for the whole sweep, so the default has to be a
    suite every workflow task file declares. `MATHPOOL` is that suite, and it is
    what the depth study measures on."""
    assert e1_cells.DEFAULT_SPLIT == "MATHPOOL"
    for line in _commands(_dry_run(capsys)):
        argv = shlex.split(line)
        assert _flag(argv, "--split") == "MATHPOOL"
        assert json.loads(_flag(argv, "--meta_json"))["dataset"] == "MATHPOOL"


def test_output_paths_follow_the_results_layout(capsys: pytest.CaptureFixture) -> None:
    seen: List[str] = []
    for line in _commands(_dry_run(capsys)):
        argv = shlex.split(line)
        out = _flag(argv, "--out")
        meta = json.loads(_flag(argv, "--meta_json"))
        expected = (
            f"20_data/results/E1/{meta['workflow']}/{meta['model']}/"
            f"sweep_n{meta['n_alternatives']}/buckets.jsonl"
        )
        assert out == expected
        seen.append(out)

    assert len(set(seen)) == 60, "two cells would write to the same bucket file"


def test_results_root_is_honoured(capsys: pytest.CaptureFixture) -> None:
    commands = _commands(_dry_run(capsys, "--results_root", "/abs/results", "--only", "c5/m2/8"))
    argv = shlex.split(commands[0])
    assert _flag(argv, "--out") == "/abs/results/E1/c5/m2/sweep_n8/buckets.jsonl"


def test_meta_json_parses_and_carries_every_required_key(capsys: pytest.CaptureFixture) -> None:
    for line in _commands(_dry_run(capsys)):
        argv = shlex.split(line)
        meta = json.loads(_flag(argv, "--meta_json"))

        assert isinstance(meta, dict)
        assert REQUIRED_META_KEYS <= set(meta), sorted(REQUIRED_META_KEYS - set(meta))

        n = int(_flag(argv, "--num_candidates"))
        assert meta["rule"] == f"sweep_n{n}"
        assert meta["n_alternatives"] == n
        assert meta["replays_per_alt"] == int(_flag(argv, "--num_completions"))
        assert meta["seed"] == int(_flag(argv, "--seed"))
        assert meta["question_order_seed"] == meta["seed"]
        assert meta["policy_tag"] == "sft"
        assert meta["dataset"] == _flag(argv, "--split")
        assert isinstance(meta["include_real_as_j0"], bool)


def test_meta_sampling_parameters_come_from_analysis_yaml(capsys: pytest.CaptureFixture) -> None:
    """The recorded decoding must be the decoding the run will apply.

    The target role of every E1 cell is a non-actor role, so `--defaults_section
    auto` resolves to the influence section; the meta is read from that same
    section so the two cannot drift.
    """
    import yaml

    raw = yaml.safe_load((REPO_ROOT / "configs" / "analysis.yaml").read_text(encoding="utf-8"))
    influence = raw["influence"]

    for line in _commands(_dry_run(capsys)):
        argv = shlex.split(line)
        meta = json.loads(_flag(argv, "--meta_json"))
        assert e1_cells.defaults_section_for(_flag(argv, "--target_role")) == "influence"
        assert meta["temperature"] == influence["decoding"]["temperature"]
        assert meta["top_p"] == influence["decoding"]["top_p"]
        assert meta["top_k"] == influence["decoding"]["top_k"]
        assert meta["include_real_as_j0"] == bool(influence["fidelity"]["include_real_as_j0"])


def test_no_retired_rule_name_appears_anywhere(capsys: pytest.CaptureFixture) -> None:
    """Revision 1 derives the fixed-budget readings, so no cell may carry them."""
    text = "\n".join(_dry_run(capsys))
    for name in RETIRED_RULE_NAMES:
        assert name not in text


def test_measured_position_is_the_first_role_and_its_first_successor() -> None:
    for workflow, expected in EXPECTED_POSITIONS.items():
        assert e1_cells.measured_positions(workflow) == expected


def test_every_workflow_reads_its_own_task_file() -> None:
    """The two-agent arm reads `math_a2.yaml`, not the paper's own task file.

    The depth study measures on `MATHPOOL`, and `configs/tasks/math.yaml` does
    not declare that suite, on purpose: the task file a reader of the paper runs
    should not name a file that exists only after the screening pass. So the
    two-agent arm reads a copy of it that does declare the pool. Pointing this
    back at `math.yaml` would leave the a2 cells unable to run the default
    split at all.
    """
    for workflow in ("a2", "a3", "mt4", "branch", "c5", "c10"):
        assert e1_cells.task_yaml_for(workflow) == f"configs/tasks/math_{workflow}.yaml"


def test_tensor_parallel_size_is_left_out_unless_asked_for(capsys: pytest.CaptureFixture) -> None:
    plain = _commands(_dry_run(capsys, "--only", "a2/4b/4"))[0]
    assert "--tensor_parallel_size" not in plain

    with_tp = _commands(_dry_run(capsys, "--only", "a2/4b/4", "--tp", "2"))[0]
    argv = shlex.split(with_tp)
    assert _flag(argv, "--tensor_parallel_size") == "2"


def test_overwrite_is_passed_through_only_when_requested(capsys: pytest.CaptureFixture) -> None:
    plain = _commands(_dry_run(capsys, "--only", "a2/4b/4"))[0]
    assert "--overwrite" not in shlex.split(plain)

    forced = _commands(_dry_run(capsys, "--only", "a2/4b/4", "--overwrite"))[0]
    assert "--overwrite" in shlex.split(forced)


# ---------------------------------------------------------------------------
# 2. Argument parsing and refusals
# ---------------------------------------------------------------------------


def test_models_parser_rejects_malformed_entries() -> None:
    assert e1_cells.parse_models("4b=Qwen3-4B,m2=Qwen3-8B") == {"4b": "Qwen3-4B", "m2": "Qwen3-8B"}
    for bad in ("4b", "=Qwen3-4B", "4b=", "4b=A,4b=B", ""):
        with pytest.raises(ValueError):
            e1_cells.parse_models(bad)


def test_only_parser_rejects_malformed_selectors() -> None:
    assert e1_cells.parse_only("branch/m2/6") == ("branch", "m2", 6)
    for bad in ("branch/m2", "branch/m2/six", "branch//6", "branch/m2/6/extra"):
        with pytest.raises(ValueError):
            e1_cells.parse_only(bad)


@pytest.mark.parametrize(
    "selector",
    ["nosuch/4b/3", "a3/nosuch/3", "a3/4b/5"],
)
def test_unknown_selector_exits_with_an_error(selector: str, capsys: pytest.CaptureFixture) -> None:
    code = e1_cells.main(["--model_root", MODEL_ROOT, "--dry-run", "--only", selector])
    assert code == 2
    assert "ERROR" in capsys.readouterr().err


def test_an_unknown_workflow_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown workflow"):
        e1_cells.task_yaml_for("a7")


def test_a_present_but_empty_bucket_file_does_not_count_as_done(tmp_path: Path) -> None:
    empty = tmp_path / "buckets.jsonl"
    empty.write_text("", encoding="utf-8")
    assert e1_cells._already_done(str(empty)) is False

    filled = tmp_path / "filled.jsonl"
    filled.write_text("{}\n", encoding="utf-8")
    assert e1_cells._already_done(str(filled)) is True
    assert e1_cells._already_done(str(tmp_path / "absent.jsonl")) is False


# ---------------------------------------------------------------------------
# 3. Bucket meta injection: the merge rule
# ---------------------------------------------------------------------------


def _sample_bucket_dict() -> Dict[str, Any]:
    return {
        "ctx_hash": 7,
        "target_role": "reasoner",
        "question_id": "q0",
        "restart": {"roles_topo": ["reasoner", "actor"], "role_outputs_prefix": {}},
        "candidates": [{"j": 0, "action_text": "a", "returns": [1.0], "next_actions": ["y"]}],
        "meta": {"seed": 11, "workflow": "runner-wins"},
    }


def test_merge_adds_new_keys_and_keeps_the_ones_the_runner_wrote() -> None:
    bucket = _sample_bucket_dict()
    merged = analysis_cli._merge_extra_meta(bucket, {"workflow": "caller-loses", "model": "4b"})

    assert merged["meta"]["workflow"] == "runner-wins"
    assert merged["meta"]["model"] == "4b"
    assert merged["meta"]["seed"] == 11


def test_merge_with_nothing_to_add_returns_the_very_same_object() -> None:
    """This is what keeps the default output byte identical to before the flag."""
    bucket = _sample_bucket_dict()
    assert analysis_cli._merge_extra_meta(bucket, {}) is bucket
    assert analysis_cli._merge_extra_meta(bucket, None) is bucket  # type: ignore[arg-type]


def test_merge_repairs_a_missing_or_non_mapping_meta() -> None:
    without = _sample_bucket_dict()
    del without["meta"]
    assert analysis_cli._merge_extra_meta(without, {"model": "4b"})["meta"] == {"model": "4b"}

    wrong = _sample_bucket_dict()
    wrong["meta"] = "not-a-mapping"
    repaired = analysis_cli._merge_extra_meta(wrong, {"model": "4b"})["meta"]
    assert repaired == {"meta_raw": "not-a-mapping", "model": "4b"}


def test_build_buckets_accepts_meta_json_and_refuses_junk(capsys: pytest.CaptureFixture) -> None:
    parser = analysis_cli._build_parser()
    args = parser.parse_args(
        [
            "build-buckets",
            "--task",
            "configs/tasks/math.yaml",
            "--split",
            "MATH500",
            "--policy_ckpt",
            "/unused",
            "--target_role",
            "reasoner",
            "--out",
            "/unused/buckets.jsonl",
            "--meta_json",
            '{"workflow":"a3"}',
        ]
    )
    assert args.meta_json == '{"workflow":"a3"}'


# ---------------------------------------------------------------------------
# 4. Bucket meta injection: end to end through build-buckets
# ---------------------------------------------------------------------------

_FAKE_RUNNER_MODULE = "c3_fake_replay_runner_for_rebuild_tests"

# Distinctive base so these context hashes never meet another test's collision guard.
_CTX_BASE = 0x5EED_0000


class _FakeRunner:
    """The smallest object `_cmd_build_buckets` accepts in place of a real runner."""

    roles_topo = ["reasoner", "actor"]

    def iter_restart_states(
        self,
        *,
        task: str,
        split: str,
        target_role: str,
        limit: int,
        seed: int,
        prefix_decoding: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Any]:
        for i in range(int(limit)):
            yield replay_mod.RestartState(
                question_id=f"fake-q{i}",
                question=f"fake question {i}",
                roles_topo=list(self.roles_topo),
                role_outputs_prefix={},
                meta={"label": "42"},
            )

    def run_bucket(self, restart_state: Any, cfg: Any, forced_actions: Any = None) -> Any:
        index = int(str(restart_state.question_id).rsplit("q", 1)[-1])
        candidates = [
            replay_mod.CandidateResult(
                action_text=f"action {index}.{j}",
                returns=[float(j % 2)] * int(cfg.num_completions_per_candidate),
                next_actions=["downstream"] * int(cfg.num_completions_per_candidate),
            )
            for j in range(int(cfg.num_candidates))
        ]
        # Mirrors the meta block the real ReplayRunner writes, with a fixed seed
        # so that an injected seed can be seen losing to the recorded one.
        meta = {
            "decoding": dict(cfg.decoding),
            "seed": 11,
            "record_next_teammate": bool(cfg.record_next_teammate),
            "next_role": cfg.next_role,
            "include_real_as_j0": bool(cfg.include_real_as_j0),
        }
        return replay_mod.Bucket(
            ctx_hash=_CTX_BASE + index,
            restart=restart_state,
            candidates=candidates,
            meta=meta,
            target_role=cfg.target_role,
        )

    def close(self) -> None:
        return None


def _register_fake_runner_module() -> None:
    if _FAKE_RUNNER_MODULE in sys.modules:
        return
    module = types.ModuleType(_FAKE_RUNNER_MODULE)
    module.make_runner = lambda _args: _FakeRunner()  # type: ignore[attr-defined]
    sys.modules[_FAKE_RUNNER_MODULE] = module


def _run_build_buckets(out_path: Path, *extra: str) -> None:
    _register_fake_runner_module()
    analysis_cli.main(
        [
            "build-buckets",
            "--task",
            str(REPO_ROOT / "configs" / "tasks" / "math.yaml"),
            "--split",
            "MATH500",
            "--policy_ckpt",
            "/unused",
            "--method",
            "sft",
            "--target_role",
            "reasoner",
            "--next_role",
            "actor",
            "--num_candidates",
            "2",
            "--num_completions",
            "1",
            "--limit",
            "3",
            "--seed",
            "0",
            "--analysis_yaml",
            str(REPO_ROOT / "configs" / "analysis.yaml"),
            "--runner_factory",
            f"{_FAKE_RUNNER_MODULE}:make_runner",
            "--out",
            str(out_path),
            *extra,
        ]
    )


def test_build_buckets_without_meta_json_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    _run_build_buckets(first)
    _run_build_buckets(second)

    assert first.read_bytes() == second.read_bytes()
    assert len(first.read_text(encoding="utf-8").strip().splitlines()) == 3


def test_meta_json_reaches_every_written_bucket(tmp_path: Path) -> None:
    plain = tmp_path / "plain.jsonl"
    stamped = tmp_path / "stamped.jsonl"

    injected = {
        "workflow": "a2",
        "model": "4b",
        "rule": "sweep_n2",
        "n_alternatives": 2,
        "replays_per_alt": 4,
        "policy_tag": "sft",
        "dataset": "MATH500",
        "question_order_seed": 0,
        "seed": 999,
    }

    _run_build_buckets(plain)
    _run_build_buckets(stamped, "--meta_json", json.dumps(injected, sort_keys=True))

    plain_rows = list(read_buckets_jsonl(plain))
    stamped_rows = list(read_buckets_jsonl(stamped))
    assert len(plain_rows) == len(stamped_rows) == 3

    for before, after in zip(plain_rows, stamped_rows):
        for key in ("workflow", "model", "rule", "n_alternatives", "replays_per_alt", "policy_tag", "dataset"):
            assert after["meta"][key] == injected[key]

        # target_role=reasoner, so `--defaults_section auto` must have resolved to
        # the influence section of configs/analysis.yaml.
        assert after["meta"]["include_real_as_j0"] is False
        assert after["meta"]["record_next_teammate"] is True

        # The runner already recorded a seed, so the injected one must not land.
        assert after["meta"]["seed"] == before["meta"]["seed"] != injected["seed"]

        # Nothing outside meta moves, and the identity of the bucket is unchanged.
        assert after["bucket_id"] == before["bucket_id"]
        for key in ("ctx_hash", "target_role", "question_id", "restart", "candidates"):
            assert after[key] == before[key]

        # Every key the runner wrote survives, byte for byte.
        for key, value in before["meta"].items():
            assert after["meta"][key] == value


def test_the_meta_the_driver_emits_agrees_with_what_the_run_records(tmp_path: Path) -> None:
    """The one key where the two could disagree is include_real_as_j0.

    The driver reads it from the section `--defaults_section auto` resolves to,
    and the runner writes the value it actually used. Injected keys never
    displace runner keys, so a mismatch would be silent; this test makes it loud.
    """
    target_role, _next_role = e1_cells.measured_positions("a2")
    sampling = e1_cells.read_sampling_defaults(
        str(REPO_ROOT / "configs" / "analysis.yaml"),
        e1_cells.defaults_section_for(target_role),
    )
    meta = e1_cells.build_meta(
        workflow="a2",
        model="4b",
        n=2,
        completions=4,
        seed=0,
        split="MATH500",
        sampling=sampling,
    )

    out = tmp_path / "agreement.jsonl"
    _run_build_buckets(out, "--meta_json", json.dumps(meta, sort_keys=True))

    for row in read_buckets_jsonl(out):
        assert row["meta"]["include_real_as_j0"] == meta["include_real_as_j0"]
        assert row["meta"]["decoding"]["temperature"] == meta["temperature"]
        assert row["meta"]["decoding"]["top_p"] == meta["top_p"]
        assert row["meta"]["decoding"]["top_k"] == meta["top_k"]


def test_build_buckets_refuses_a_meta_json_that_is_not_an_object(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        _run_build_buckets(tmp_path / "bad.jsonl", "--meta_json", "[1,2,3]")
    assert excinfo.value.code == 2

    with pytest.raises(SystemExit) as excinfo:
        _run_build_buckets(tmp_path / "bad2.jsonl", "--meta_json", "{not json")
    assert excinfo.value.code == 2


# ---------------------------------------------------------------------------
# scripts/70_rebuild/replay_tokens.py (2026-09-15)
#
# The one E1 key the aggregation cannot produce, `E1.c10.median_prefix_tokens`,
# needs a tokenizer and the model files. Its statistic is the length of what one
# replay regenerates downstream, which lives in `candidates[j].next_actions[k]`,
# one entry per replay of `candidates[j].returns`. What is testable without a
# tokenizer is the scan: which cells there are, how many replays they record and
# how many of those carry downstream text. That is `--dry-run`. The counting
# path is testable too, with a tokenizer object that splits on whitespace.
# ---------------------------------------------------------------------------


def _c10_bucket(qid: str, replays: Sequence[Sequence[Any]],
                returns_len: Optional[int] = None) -> Dict[str, Any]:
    """One bucket in the schema of c3/analysis/buckets.py, c10 shaped.

    `replays` is one list of downstream entries per alternative. `returns_len`
    overrides the number of returns recorded per alternative, which is how a
    bucket that ran replays without recording their text is built.
    """
    cands: List[Dict[str, Any]] = []
    for j, nexts in enumerate(replays):
        n_ret = len(nexts) if returns_len is None else returns_len
        cands.append({
            "j": j,
            "action_text": "a%d" % j,
            "returns": [1.0] * n_ret,
            "next_actions": list(nexts),
        })
    return {
        "bucket_id": "bkt_" + qid,
        "ctx_hash": 1,
        "target_role": "reader",
        "question_id": qid,
        "restart": {
            "roles_topo": ["reader", "planner", "solver"],
            "role_outputs_prefix": {},
            "question": "What is 2 + 2?",
        },
        "candidates": cands,
        "meta": {"workflow": "c10", "model": "4b", "rule": "sweep_n4"},
    }


def _write_c10_cell(root: Path, rows: Sequence[Dict[str, Any]], n: int = 4) -> Path:
    path = root / "c10" / "4b" / ("sweep_n%d" % n) / "buckets.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


class _WhitespaceTokenizer:
    """Stand-in for `transformers.AutoTokenizer`: one token per whitespace run.

    It is called the way `replay_tokens.count_tokens` calls the real one, so it
    pins the call shape as well as the arithmetic. The real tokenizer is a
    server-side dependency this machine does not have.
    """

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, text: str, add_special_tokens: bool = True) -> Dict[str, Any]:
        self.calls.append({"text": text, "add_special_tokens": add_special_tokens})
        ids = [0] * len(str(text).split())
        if add_special_tokens:
            ids = ids + [1]
        return {"input_ids": ids}


def test_replay_tokens_dry_run_needs_neither_transformers_nor_a_model(
        tmp_path: Path, capsys: Any) -> None:
    root = tmp_path / "20_data" / "results" / "E1"
    # three buckets, two alternatives each, two replays each: twelve replays
    _write_c10_cell(root, [_c10_bucket("q%d" % i, [["x y", "z"], ["p", "q r"]])
                           for i in range(3)])
    had_transformers = "transformers" in sys.modules

    rc = replay_tokens.main(["--results", str(root), "--dry-run"])
    assert rc == 0
    out = capsys.readouterr()
    assert "# cells: 1" in out.out
    assert "# buckets: 3" in out.out
    assert "# replays: 12" in out.out
    assert "# with downstream text: 12" in out.out
    assert "20_data/results/E1/c10/4b/sweep_n4/buckets.jsonl" in out.err
    assert "3 bucket(s), 12 replay(s), 12 with downstream text" in out.err
    if not had_transformers:
        assert "transformers" not in sys.modules


def test_replay_tokens_counts_one_replay_per_recorded_return(
        tmp_path: Path, capsys: Any) -> None:
    """A replay is an entry of `returns`; `next_actions[k]` is its downstream
    text. With `record_next_teammate` off the replays still happened and still
    cost what they cost, so they are counted and reported as carrying no text
    rather than silently leaving the denominator."""
    root = tmp_path / "E1"
    _write_c10_cell(root, [_c10_bucket("q0", [[]], returns_len=4),
                           _c10_bucket("q1", [["x y z"]])])

    assert replay_tokens.main(["--results", str(root), "--dry-run"]) == 0
    out = capsys.readouterr()
    assert "# replays: 5" in out.out
    assert "# with downstream text: 1" in out.out
    assert ("4 replay(s) contribute nothing: candidate.next_actions is empty, so no "
            "downstream text was recorded") in out.err


def test_replay_tokens_ignores_an_empty_downstream_text(tmp_path: Path, capsys: Any) -> None:
    root = tmp_path / "E1"
    _write_c10_cell(root, [_c10_bucket("q0", [["x", "   ", ""]])])

    assert replay_tokens.main(["--results", str(root), "--dry-run"]) == 0
    out = capsys.readouterr()
    assert "# replays: 3" in out.out
    assert "# with downstream text: 1" in out.out
    assert "2 replay(s) contribute nothing: replay has no downstream text" in out.err


def test_replay_tokens_joins_several_downstream_roles_of_one_replay() -> None:
    """Every bucket on disk carries `next_actions[k]` as a string, and the bucket
    schema guard refuses anything else. A writer that one day records the whole
    downstream tail of a replay would hand a mapping or a list instead, and the
    texts of that one replay are joined rather than dropped."""
    assert replay_tokens.downstream_text("x y") == "x y"
    assert replay_tokens.downstream_text({"planner": "P", "solver": "S"}) == "P\n\nS"
    assert replay_tokens.downstream_text(["P", "S"]) == "P\n\nS"
    assert replay_tokens.downstream_text({"planner": "", "solver": "S"}) == "S"
    assert replay_tokens.downstream_text("") is None
    assert replay_tokens.downstream_text("  \n ") is None
    assert replay_tokens.downstream_text({}) is None
    assert replay_tokens.downstream_text(17) is None
    assert replay_tokens.downstream_text(None) is None


def test_replay_tokens_refuses_the_appendix_tree_under_the_default_prefix(
        tmp_path: Path, capsys: Any) -> None:
    root = tmp_path / "E1_math500all"
    _write_c10_cell(root, [_c10_bucket("q0", [["x"]])])
    assert replay_tokens.main(["--results", str(root), "--dry-run"]) == 2
    assert "--key_prefix E1app" in capsys.readouterr().err


def test_replay_tokens_refuses_a_real_run_without_the_pieces_it_needs(
        tmp_path: Path, capsys: Any) -> None:
    root = tmp_path / "E1"
    _write_c10_cell(root, [_c10_bucket("q0", [["x"]])])
    assert replay_tokens.main(["--results", str(root)]) == 2
    err = capsys.readouterr().err
    for flag in ("--tokenizer", "--manifest", "--out"):
        assert flag in err


def test_replay_tokens_reports_both_medians() -> None:
    assert replay_tokens.medians([]) == (None, None)
    assert replay_tokens.medians([7]) == (7, 7.0)
    low, linear = replay_tokens.medians([10, 20, 30, 40])
    assert low == 20 and linear == pytest.approx(25.0)
    assert isinstance(low, int)


def test_replay_tokens_writes_the_key_through_the_shared_writer(
        tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
    """The whole counting path, with a tokenizer object standing in for the real
    one: the key value, its n, its source list and the auxiliary top-level block
    that does not belong in the manifest."""
    root = tmp_path / "20_data" / "results" / "E1"
    # replay lengths in whitespace-separated words: 1, 2, 3, 4
    _write_c10_cell(root, [_c10_bucket("q0", [["a", "a b"], ["a b c", "a b c d"]])])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "E1.c10.median_prefix_tokens": {"unit": "tokens", "fmt": "{:,d}", "value": None},
    }), encoding="utf-8")
    out_path = tmp_path / "replay_tokens_summary.json"

    fake = _WhitespaceTokenizer()
    monkeypatch.setattr(replay_tokens, "load_tokenizer", lambda path: fake)

    rc = replay_tokens.main(["--results", str(root), "--tokenizer", "/models/Qwen3-4B",
                             "--manifest", str(manifest), "--out", str(out_path)])
    assert rc == 0
    assert [call["add_special_tokens"] for call in fake.calls] == [False] * 4

    doc = json.loads(out_path.read_text(encoding="utf-8"))
    entry = doc["keys"]["E1.c10.median_prefix_tokens"]
    assert entry["value"] == 2 and entry["n"] == 4 and entry["unit"] == "tokens"
    assert entry["source"] == ["20_data/results/E1/c10/4b/sweep_n4/buckets.jsonl"]
    assert "low median 2, linear median 2.5, mean 2.5" in entry["note"]
    assert doc["results_root"] == "20_data/results/E1"
    assert doc["key_prefix"] == "E1"
    aux = doc["replay_tokens"]
    assert aux["median_low"] == 2 and aux["median_linear"] == pytest.approx(2.5)
    assert aux["mean_tokens"] == pytest.approx(2.5)
    assert aux["min_tokens"] == 1 and aux["max_tokens"] == 4
    assert aux["n_replays"] == 4 and aux["n_replays_counted"] == 4
    assert aux["n_cells"] == 1 and aux["n_buckets"] == 1
    assert "wrote 1 key(s)" in capsys.readouterr().out


def test_replay_tokens_linear_median_and_special_tokens_are_flags(
        tmp_path: Path, monkeypatch: Any) -> None:
    root = tmp_path / "20_data" / "results" / "E1"
    _write_c10_cell(root, [_c10_bucket("q0", [["a", "a b"], ["a b c", "a b c d"]])])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "E1.c10.median_prefix_tokens": {"unit": "tokens", "fmt": "{:,d}", "value": None},
    }), encoding="utf-8")
    out_path = tmp_path / "s.json"

    fake = _WhitespaceTokenizer()
    monkeypatch.setattr(replay_tokens, "load_tokenizer", lambda path: fake)
    assert replay_tokens.main(["--results", str(root), "--tokenizer", "/models/Qwen3-4B",
                               "--manifest", str(manifest), "--out", str(out_path),
                               "--median", "linear", "--add_special_tokens"]) == 0

    doc = json.loads(out_path.read_text(encoding="utf-8"))
    # one extra token per text, so the counts are 2, 3, 4, 5 and the linear
    # median is 3.5, which rounds to 4
    assert doc["keys"]["E1.c10.median_prefix_tokens"]["value"] == 4
    assert doc["replay_tokens"]["add_special_tokens"] is True
    assert [call["add_special_tokens"] for call in fake.calls] == [True] * 4


def test_replay_tokens_says_so_when_no_replay_carries_text(
        tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
    root = tmp_path / "20_data" / "results" / "E1"
    _write_c10_cell(root, [_c10_bucket("q0", [[]], returns_len=4)])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({}), encoding="utf-8")

    monkeypatch.setattr(replay_tokens, "load_tokenizer",
                        lambda path: pytest.fail("the tokenizer must not be loaded"))
    assert replay_tokens.main(["--results", str(root), "--tokenizer", "/models/Qwen3-4B",
                               "--manifest", str(manifest),
                               "--out", str(tmp_path / "s.json")]) == 2
    assert "no replay carries downstream text" in capsys.readouterr().err
    assert not (tmp_path / "s.json").exists()
