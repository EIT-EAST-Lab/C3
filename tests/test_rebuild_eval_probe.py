"""Contract tests for the start-accuracy probe and the candidate benchmark builders.

Three things are pinned here, and none of them needs a GPU, a model or the
`datasets` package.

First, `scripts/70_rebuild/eval_probe.py`: the two commands it emits are what
actually gets run on the platform, so the sample count, the temperature and the
artifact paths are checked against the layout `scripts/50_eval/paper_main_results.sh`
really writes. The probe overrides the profile defaults by repeating the flags,
so a test also pins the argparse rule that makes the override work.

Second, the summary arithmetic: the Wilson interval is checked against an
independent solution of the equation it comes from and against hand values, and
the per-suite aggregation is checked on both shapes the evaluation dump has.

Third, the preparation functions of the five candidate benchmarks: their field
mapping is checked on fixture rows shaped like the upstream ones, including the
two OlympiadBench shapes that need a ruling before the file can be built at all.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterable, List

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "scripts" / "70_rebuild"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import eval_probe  # noqa: E402


PROBE_TASK = REPO_ROOT / "configs" / "tasks" / "math_eval_probe.yaml"
MANIFEST = REPO_ROOT / "configs" / "data_manifest.yaml"

EXPECTED_SUITES = [
    "MATH500",
    "Minerva-Math",
    "OlympiadBench",
    "AMC23",
    "AIME24",
    "AIME25",
    "GSM8K-test",
    "CMATH-test",
]

PROBE_LIMIT = 100

Z95 = 1.959963984540054


def _load_prepare_math() -> ModuleType:
    """`scripts/10_data/prepare_math.py` as a module; its directory is not a package."""
    path = REPO_ROOT / "scripts" / "10_data" / "prepare_math.py"
    spec = importlib.util.spec_from_file_location("prepare_math_under_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare_math = _load_prepare_math()


def _args(*extra: str):
    argv = ["--policy", "/models/Qwen3-4B-Instruct-2507", *extra]
    return eval_probe.build_parser().parse_args(argv)


def _flag(argv: List[str], name: str) -> str:
    """The value that follows `name` in an argv list."""
    index = argv.index(name)
    return argv[index + 1]


# ---------------------------------------------------------------------------
# The emitted commands
# ---------------------------------------------------------------------------


def test_probe_emits_one_greedy_and_one_sampling_run() -> None:
    runs = eval_probe.build_runs(_args())

    assert [r.label for r in runs] == ["greedy", "sample"]

    greedy, sample = runs
    assert (greedy.n_samples_per_prompt, greedy.temperature) == (1, 0.0)
    assert (sample.n_samples_per_prompt, sample.temperature) == (4, 0.7)


def test_each_command_states_its_sample_count_and_temperature() -> None:
    """The profile is only a label here, so both numbers ride in extra_eval_args."""
    greedy, sample = eval_probe.build_runs(_args())

    assert _flag(greedy.argv, "--extra_eval_args") == "--eval_n_samples_per_prompt 1 --eval_temperature 0.0"
    assert _flag(sample.argv, "--extra_eval_args") == "--eval_n_samples_per_prompt 4 --eval_temperature 0.7"


def test_command_drives_the_existing_evaluation_script_in_direct_mode() -> None:
    greedy, _ = eval_probe.build_runs(_args())

    assert greedy.argv[2] == "one"
    assert greedy.argv[1].endswith("scripts/50_eval/paper_main_results.sh")
    assert _flag(greedy.argv, "--source_type") == "hf_base"
    assert _flag(greedy.argv, "--hf_base") == "/models/Qwen3-4B-Instruct-2507"
    assert _flag(greedy.argv, "--method") == "SFT"

    # The task path has to be absolute: the evaluation script only takes the base
    # name of an absolute path, and would otherwise turn `configs/tasks/x.yaml`
    # into a nested `configs/tasks/x` directory under the run root.
    task = _flag(greedy.argv, "--task")
    assert Path(task).is_absolute()
    assert task.endswith("configs/tasks/math_eval_probe.yaml")


def test_generation_cap_travels_as_the_environment_the_script_reads() -> None:
    greedy, _ = eval_probe.build_runs(_args())
    assert greedy.env == {"GENERATE_MAX_LEN": "512", "PROMPT_MAX_LEN": "2560"}
    assert "GENERATE_MAX_LEN=512" in greedy.command_line()


def test_artifact_paths_follow_the_layout_of_the_evaluation_script() -> None:
    greedy, sample = eval_probe.build_runs(_args())

    tail = "ckpt/_runs/_sft_main_results/eval_probe/eval_probe/math_eval_probe"
    assert greedy.eval_dir.endswith(f"{tail}/greedy")
    assert sample.eval_dir.endswith(f"{tail}/n10")
    assert greedy.eval_jsonl.endswith("/greedy/eval_only.jsonl")
    assert greedy.metrics_jsonl.endswith("/greedy/eval_only.jsonl.metrics.jsonl")


def test_an_absolute_posix_root_keeps_its_leading_slash() -> None:
    """
    The platform is Linux and its artifact root is an absolute path.

    A join that strips the leading slash turns `/data/ckpt/...` into
    `data/ckpt/...`, which resolves against whatever the working directory
    happens to be, so the probe would read an empty directory and report every
    suite as missing.
    """
    assert eval_probe.posix_join("/data/ckpt", "_runs", "eval_probe") == "/data/ckpt/_runs/eval_probe"
    assert eval_probe.posix_join("relative/root", "x") == "relative/root/x"
    assert eval_probe.posix_join("/data/ckpt/", "/x/") == "/data/ckpt/x"

    eval_dir = eval_probe.eval_dir_for(
        ckpt_root="/data/ckpt",
        run_id="eval_probe",
        out_subdir="eval_probe",
        task_path="/repo/configs/tasks/math_eval_probe.yaml",
        profile="greedy",
    )
    assert eval_dir == "/data/ckpt/_runs/_sft_main_results/eval_probe/eval_probe/math_eval_probe/greedy"


def test_only_selects_a_single_decoding() -> None:
    runs = eval_probe.build_runs(_args("--only", "sample"))
    assert [r.label for r in runs] == ["sample"]


def test_dry_run_prints_both_commands_and_runs_nothing(capsys: Any, monkeypatch: Any) -> None:
    def _explode(*_a: Any, **_k: Any) -> None:
        raise AssertionError("a dry run must not start a subprocess")

    monkeypatch.setattr(eval_probe.subprocess, "run", _explode)

    code = eval_probe.main(["--policy", "/models/m", "--dry-run"])
    out = capsys.readouterr().out

    assert code == 0
    assert out.count("paper_main_results.sh one") == 2
    assert "--eval_n_samples_per_prompt 4 --eval_temperature 0.7" in out
    assert "# runs: 2" in out


def test_a_repeated_argparse_flag_takes_the_last_value() -> None:
    """
    Why extra_eval_args can override the profile.

    The evaluation script appends extra_eval_args after the flags the profile
    fixed, so the override only works if the parser on the far end keeps the last
    occurrence. That parser is argparse with plain store actions, and this pins
    the rule rather than trusting it.
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_n_samples_per_prompt", type=int, default=1)
    parser.add_argument("--eval_temperature", type=float, default=0.7)

    parsed = parser.parse_args(
        [
            "--eval_n_samples_per_prompt",
            "10",
            "--eval_temperature",
            "0.7",
            "--eval_n_samples_per_prompt",
            "4",
            "--eval_temperature",
            "0.0",
        ]
    )
    assert parsed.eval_n_samples_per_prompt == 4
    assert parsed.eval_temperature == 0.0


# ---------------------------------------------------------------------------
# The probe task file
# ---------------------------------------------------------------------------


def test_probe_task_lists_the_eight_suites_each_capped_at_the_probe_budget() -> None:
    spec = yaml.safe_load(PROBE_TASK.read_text(encoding="utf-8"))
    suites = spec["environment"]["eval_suites"]

    assert [s["name"] for s in suites] == EXPECTED_SUITES
    assert {s["limit"] for s in suites} == {PROBE_LIMIT}
    assert eval_probe.expected_suites(PROBE_TASK) == EXPECTED_SUITES


def test_probe_task_reuses_the_training_block_and_scorer_of_the_math_task() -> None:
    probe = yaml.safe_load(PROBE_TASK.read_text(encoding="utf-8"))["environment"]
    math_task = yaml.safe_load((REPO_ROOT / "configs" / "tasks" / "math.yaml").read_text(encoding="utf-8"))
    main = math_task["environment"]

    assert probe["train_datasets"] == main["train_datasets"]
    for key in ("env_name", "sampling_mode", "episode_length", "math_backend", "reward_mode", "use_math_verify"):
        assert probe[key] == main[key], key


def test_every_probe_suite_points_at_the_file_its_manifest_entry_declares() -> None:
    suites = yaml.safe_load(PROBE_TASK.read_text(encoding="utf-8"))["environment"]["eval_suites"]
    outputs = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))["outputs"]
    by_name = {entry["name"]: entry for entry in outputs}

    for suite in suites:
        name = suite["name"]
        assert name in by_name, f"{name} has no manifest entry"
        assert suite["path"] == by_name[name]["output_path"], name


def test_candidate_manifest_entries_are_pinned_to_an_immutable_revision() -> None:
    outputs = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))["outputs"]
    by_name = {entry["name"]: entry for entry in outputs}

    for name in ("Minerva-Math", "OlympiadBench", "AMC23", "AIME24", "AIME25"):
        source = by_name[name]["source"]
        assert prepare_math._is_pinned_revision(source["revision"]), f"{name} revision is not a commit sha"
        assert source["split"] == "test", name


# ---------------------------------------------------------------------------
# Wilson interval
# ---------------------------------------------------------------------------


def _wilson_by_quadratic(k: int, n: int, z: float = Z95):
    """
    The same interval, solved rather than assembled.

    The Wilson bounds are the two roots of (phat - p)^2 = z^2 p (1 - p) / n. The
    implementation uses the centre-and-half-width form; this uses the quadratic
    formula, so agreement is a check on the algebra, not a restatement of it.
    """
    p = k / n
    a = 1.0 + z * z / n
    b = -(2.0 * p + z * z / n)
    c = p * p
    disc = b * b - 4.0 * a * c
    return ((-b - math.sqrt(disc)) / (2.0 * a), (-b + math.sqrt(disc)) / (2.0 * a))


@pytest.mark.parametrize("k,n", [(50, 100), (1, 40), (27, 30), (13, 100), (80, 100), (3, 400)])
def test_wilson_interval_solves_its_own_equation(k: int, n: int) -> None:
    got = eval_probe.wilson_interval(k, n)
    want = _wilson_by_quadratic(k, n)
    assert got[0] == pytest.approx(want[0], abs=1e-12)
    assert got[1] == pytest.approx(want[1], abs=1e-12)


def test_wilson_interval_matches_hand_values() -> None:
    low, high = eval_probe.wilson_interval(50, 100)
    assert (round(low, 6), round(high, 6)) == (0.403832, 0.596168)

    low, high = eval_probe.wilson_interval(13, 100)
    assert (round(low, 6), round(high, 6)) == (0.077572, 0.209804)


def test_wilson_interval_stays_inside_the_unit_interval_at_the_extremes() -> None:
    """The reason for Wilson: the normal interval leaves [0, 1] exactly here."""
    low, high = eval_probe.wilson_interval(0, 100)
    assert 0.0 <= low <= 1.0 and low == pytest.approx(0.0, abs=1e-12)
    assert high == pytest.approx(0.036993, abs=1e-6)

    low, high = eval_probe.wilson_interval(100, 100)
    assert low == pytest.approx(0.963007, abs=1e-6)
    assert 0.0 <= high <= 1.0 and high == pytest.approx(1.0, abs=1e-12)


def test_wilson_interval_is_undefined_without_trials() -> None:
    assert eval_probe.wilson_interval(0, 0) == (None, None)


# ---------------------------------------------------------------------------
# Summarizing a dump
# ---------------------------------------------------------------------------


def _marl_row(suite: str, qid: int, kid: int, reward: float) -> Dict[str, Any]:
    """The shape the multi-agent branch of the evaluation dump writes."""
    return {
        "global_step": 0,
        "question_id": qid,
        "k_id": kid,
        "datasource": suite,
        "prompt": f"q{qid}",
        "label": "42",
        "answer_role": "actor",
        "answer_text": "the answer is \\boxed{42}",
        "answer_reward": reward,
    }


def _single_agent_row(suite: str, qid: int, kid: int, reward: float, length: int, clipped: bool) -> Dict[str, Any]:
    """The shape the single-agent branch writes, which also carries token counts."""
    return {
        "global_step": 0,
        "question_id": qid,
        "k_id": kid,
        "datasource": suite,
        "text": "...",
        "reward": reward,
        "info": {
            "response_length": [length],
            "total_length": [length + 100],
            "response_clip_ratio": [clipped],
        },
    }


def test_accuracy_is_the_mean_over_problems_of_their_sample_means() -> None:
    rows = [
        _marl_row("AMC23", 0, 0, 1.0),
        _marl_row("AMC23", 0, 1, 0.0),
        _marl_row("AMC23", 1, 0, 1.0),
        _marl_row("AMC23", 1, 1, 1.0),
    ]
    suites = eval_probe.summarize_dump(rows, generate_max_len=512)
    amc = suites["AMC23"]

    assert amc["n_questions"] == 2
    assert amc["n_draws"] == 4
    assert amc["successes"] == 3
    # (0.5 + 1.0) / 2, which is what the trainer logs as pass1.
    assert amc["accuracy"] == pytest.approx(0.75)
    assert amc["wilson95"]["low"] == pytest.approx(_wilson_by_quadratic(3, 4)[0])


def test_each_suite_is_counted_on_its_own() -> None:
    rows = [
        _marl_row("AMC23", 0, 0, 1.0),
        _marl_row("AIME24", 1, 0, 0.0),
        _marl_row("AIME24", 2, 0, 0.0),
    ]
    suites = eval_probe.summarize_dump(rows, generate_max_len=512)

    assert sorted(suites) == ["AIME24", "AMC23"]
    assert suites["AMC23"]["accuracy"] == pytest.approx(1.0)
    assert suites["AIME24"]["accuracy"] == pytest.approx(0.0)
    assert suites["AIME24"]["n_questions"] == 2


def test_a_multi_agent_dump_cannot_report_generation_length() -> None:
    rows = [_marl_row("AMC23", 0, 0, 1.0)]
    length = eval_probe.summarize_dump(rows, generate_max_len=512)["AMC23"]["length"]

    assert length["source"] == "unavailable"
    assert length["mean_tokens"] is None
    assert length["truncation_rate"] is None
    assert "response_length" in length["note"]


def test_a_dump_with_token_counts_reports_length_and_truncation() -> None:
    rows = [
        _single_agent_row("MATH500", 0, 0, 1.0, length=100, clipped=False),
        _single_agent_row("MATH500", 1, 0, 0.0, length=512, clipped=True),
        _single_agent_row("MATH500", 2, 0, 0.0, length=300, clipped=False),
    ]
    length = eval_probe.summarize_dump(rows, generate_max_len=512)["MATH500"]["length"]

    assert length["source"] == "eval dump info.response_length"
    assert length["mean_tokens"] == pytest.approx((100 + 512 + 300) / 3)
    assert length["max_tokens"] == 512
    assert length["truncation_rate"] == pytest.approx(1 / 3)


def test_a_reward_that_is_neither_zero_nor_one_is_counted_and_not_hidden() -> None:
    rows = [_marl_row("AMC23", 0, 0, 0.6), _marl_row("AMC23", 1, 0, 1.0)]
    amc = eval_probe.summarize_dump(rows, generate_max_len=512)["AMC23"]

    assert amc["n_non_binary_reward"] == 1
    assert amc["successes"] == 2
    assert amc["accuracy"] == pytest.approx(0.8)


def test_a_row_without_a_reward_is_reported_and_left_out_of_the_accuracy() -> None:
    rows = [_marl_row("AMC23", 0, 0, 1.0), {"datasource": "AMC23", "question_id": 1, "k_id": 0}]
    amc = eval_probe.summarize_dump(rows, generate_max_len=512)["AMC23"]

    assert amc["n_draws"] == 2
    assert amc["n_scored_draws"] == 1
    assert amc["n_missing_reward"] == 1
    assert amc["accuracy"] == pytest.approx(1.0)


def test_rows_without_a_question_id_are_not_folded_into_one_problem() -> None:
    rows = [
        {"datasource": "AMC23", "answer_reward": 1.0},
        {"datasource": "AMC23", "answer_reward": 0.0},
    ]
    amc = eval_probe.summarize_dump(rows, generate_max_len=512)["AMC23"]

    assert amc["n_missing_question_id"] == 2
    assert amc["n_questions"] == 2
    assert amc["accuracy"] == pytest.approx(0.5)


def test_metrics_of_the_last_evaluation_record_win() -> None:
    rows = [
        {"kind": "eval_metrics", "global_step": 0, "metrics": {"eval_AMC23_pass1": 0.1}},
        {"kind": "train_metrics", "metrics": {"eval_AMC23_pass1": 0.9}},
        {"kind": "eval_metrics", "global_step": 1, "metrics": {"eval_AMC23_pass1": 0.5, "eval_AMC23_pass4": 0.8}},
    ]
    logged = eval_probe.metrics_by_suite(rows)

    assert logged == {"AMC23": {"pass1": 0.5, "pass4": 0.8}}


def test_a_suite_name_with_a_hyphen_survives_the_metric_key() -> None:
    rows = [{"kind": "eval_metrics", "metrics": {"eval_CMATH-test_pass1": 0.92, "eval_Minerva-Math_pass1": 0.3}}]
    assert eval_probe.metrics_by_suite(rows) == {"CMATH-test": {"pass1": 0.92}, "Minerva-Math": {"pass1": 0.3}}


# ---------------------------------------------------------------------------
# Suite coverage: the silent failure this probe has to catch
# ---------------------------------------------------------------------------


def _write_run(tmp_path: Path, rows: Iterable[Dict[str, Any]], metrics: Dict[str, float]) -> eval_probe.ProbeRun:
    eval_dir = tmp_path / "run"
    eval_dir.mkdir(parents=True, exist_ok=True)
    with (eval_dir / "eval_only.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (eval_dir / "eval_only.jsonl.metrics.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps({"kind": "eval_metrics", "global_step": 0, "metrics": metrics}) + "\n")
    return eval_probe.ProbeRun(
        label="greedy",
        profile="greedy",
        n_samples_per_prompt=1,
        temperature=0.0,
        eval_dir=eval_dir.as_posix(),
    )


def test_a_suite_missing_from_the_run_is_reported_as_a_problem(tmp_path: Path) -> None:
    run = _write_run(tmp_path, [_marl_row("MATH500", 0, 0, 1.0)], {"eval_MATH500_pass1": 1.0})
    block = eval_probe.summarize_run(run, task_path=PROBE_TASK, generate_max_len=512)

    assert block["observed_suites"] == ["MATH500"]
    assert block["missing_suites"] == [s for s in EXPECTED_SUITES if s != "MATH500"]
    assert block["problems"], "a run that covered one suite of eight must not look clean"


def test_a_complete_run_reports_no_problem(tmp_path: Path) -> None:
    rows = [_marl_row(name, index, 0, 1.0) for index, name in enumerate(EXPECTED_SUITES)]
    metrics = {f"eval_{name}_pass1": 1.0 for name in EXPECTED_SUITES}
    run = _write_run(tmp_path, rows, metrics)

    block = eval_probe.summarize_run(run, task_path=PROBE_TASK, generate_max_len=512)

    assert block["missing_suites"] == []
    assert block["problems"] == []
    assert all(values["accuracy_matches_trainer"] for values in block["suites"].values())


def test_an_accuracy_that_disagrees_with_the_trainer_is_flagged(tmp_path: Path) -> None:
    run = _write_run(tmp_path, [_marl_row("MATH500", 0, 0, 1.0)], {"eval_MATH500_pass1": 0.25})
    block = eval_probe.summarize_run(run, task_path=PROBE_TASK, generate_max_len=512)

    assert block["suites"]["MATH500"]["accuracy_logged_by_trainer"] == 0.25
    assert block["suites"]["MATH500"]["accuracy_matches_trainer"] is False


def test_a_missing_dump_is_a_problem_not_an_empty_summary(tmp_path: Path) -> None:
    run = eval_probe.ProbeRun(
        label="greedy",
        profile="greedy",
        n_samples_per_prompt=1,
        temperature=0.0,
        eval_dir=(tmp_path / "nothing_here").as_posix(),
    )
    block = eval_probe.summarize_run(run, task_path=PROBE_TASK, generate_max_len=512)

    assert block["suites"] == {}
    assert any("not found" in p for p in block["problems"])


# ---------------------------------------------------------------------------
# Preparation functions of the candidate benchmarks
# ---------------------------------------------------------------------------


def _spec(repo: str, **source: Any) -> Dict[str, Any]:
    src = {"kind": "huggingface", "id": repo, "config": "default", "split": "test", "revision": "abc123"}
    src.update(source)
    return {"name": repo, "source": src}


def _feed(monkeypatch: Any, rows: List[Dict[str, Any]]) -> None:
    """Replace the download with fixture rows shaped like the upstream ones."""

    def _fake(src: Dict[str, Any], *, logical_name: str) -> Iterable[Dict[str, Any]]:
        return iter(rows)

    monkeypatch.setattr(prepare_math, "_iter_hf_rows", _fake)


def test_minerva_maps_question_and_answer(monkeypatch: Any) -> None:
    _feed(monkeypatch, [{"question": " What is 1 + 1? ", "answer": " 2 "}])
    rows = list(prepare_math._prepare_minerva(_spec("math-ai/minervamath")))

    assert rows == [
        {
            "input": "What is 1 + 1?",
            "answer": "2",
            "source": "math-ai/minervamath:test@abc123",
        }
    ]


def test_amc23_keeps_the_upstream_id(monkeypatch: Any) -> None:
    _feed(monkeypatch, [{"id": 7, "question": "How many miles?", "answer": "27", "url": "https://example.org"}])
    row = list(prepare_math._prepare_amc23(_spec("math-ai/amc23")))[0]

    assert row["input"] == "How many miles?"
    assert row["answer"] == "27"
    assert row["unique_id"] == "7"
    assert "url" not in row


def test_aime24_takes_its_answer_out_of_the_boxed_solution(monkeypatch: Any) -> None:
    """Upstream has no answer column at all, only a solution of the form \\boxed{n}."""
    _feed(monkeypatch, [{"id": 60, "problem": "Every morning Aya walks.", "solution": "\\boxed{204}"}])
    row = list(prepare_math._prepare_aime24(_spec("math-ai/aime24")))[0]

    assert row["answer"] == "204"
    assert row["solution"] == "\\boxed{204}"
    assert row["unique_id"] == "60"


def test_aime24_refuses_a_row_whose_solution_has_no_box(monkeypatch: Any) -> None:
    _feed(monkeypatch, [{"id": 61, "problem": "A problem.", "solution": "no box here"}])
    with pytest.raises(SystemExit) as excinfo:
        list(prepare_math._prepare_aime24(_spec("math-ai/aime24")))
    assert "empty answer" in str(excinfo.value)


def test_aime25_maps_problem_and_answer(monkeypatch: Any) -> None:
    _feed(monkeypatch, [{"problem": "Find the sum.", "answer": "70", "id": "0"}])
    row = list(prepare_math._prepare_aime25(_spec("math-ai/aime25")))[0]

    assert (row["input"], row["answer"], row["unique_id"]) == ("Find the sum.", "70", "0")


def test_a_row_without_a_problem_statement_is_refused(monkeypatch: Any) -> None:
    _feed(monkeypatch, [{"question": "   ", "answer": "2"}])
    with pytest.raises(SystemExit) as excinfo:
        list(prepare_math._prepare_minerva(_spec("math-ai/minervamath")))
    assert "empty problem statement" in str(excinfo.value)


# ---------------------------------------------------------------------------
# OlympiadBench: the two shapes that need a ruling
# ---------------------------------------------------------------------------


OLYMPIAD_SINGLE = {
    "id": 1606,
    "question": "Xenia and Sergey play a game.",
    "solution": ["First we show two moves suffice.", "Now the lower bound."],
    "final_answer": ["2"],
    "context": None,
    "unit": None,
    "is_multiple_answer": False,
    "answer_type": "Numerical",
    "subfield": "Combinatorics",
    "difficulty": "Competition",
}

OLYMPIAD_MULTI = dict(OLYMPIAD_SINGLE, id=1607, final_answer=["2", "3"], is_multiple_answer=True)

OLYMPIAD_UNIT = dict(OLYMPIAD_SINGLE, id=1608, final_answer=["12"], unit="km/h")


def _olympiad_spec(multi: str, unit: str) -> Dict[str, Any]:
    return _spec("math-ai/olympiadbench", multi_answer_policy=multi, unit_policy=unit)


@pytest.mark.parametrize("field", ["multi_answer_policy", "unit_policy"])
def test_olympiadbench_refuses_to_build_while_a_policy_is_undecided(monkeypatch: Any, field: str) -> None:
    """
    The two policies change both the answers and the row count, so an unanswered
    one fails loudly instead of being guessed.
    """
    _feed(monkeypatch, [OLYMPIAD_SINGLE])
    spec = _olympiad_spec("drop", "drop")
    spec["source"][field] = prepare_math.POLICY_UNSET

    with pytest.raises(SystemExit) as excinfo:
        list(prepare_math._prepare_olympiadbench(spec))
    message = str(excinfo.value)
    assert field in message
    assert "DECIDE-ME" in message


def test_olympiadbench_refuses_an_unknown_policy(monkeypatch: Any) -> None:
    _feed(monkeypatch, [OLYMPIAD_SINGLE])
    with pytest.raises(SystemExit) as excinfo:
        list(prepare_math._prepare_olympiadbench(_olympiad_spec("median", "drop")))
    assert "not one of" in str(excinfo.value)


def test_olympiadbench_maps_a_single_answer_row(monkeypatch: Any) -> None:
    _feed(monkeypatch, [OLYMPIAD_SINGLE])
    row = list(prepare_math._prepare_olympiadbench(_olympiad_spec("drop", "drop")))[0]

    assert row["input"] == "Xenia and Sergey play a game."
    assert row["answer"] == "2"
    assert row["solution"] == "First we show two moves suffice.\n\nNow the lower bound."
    assert row["subject"] == "Combinatorics"
    assert row["level"] == "Competition"
    assert row["unique_id"] == "1606"


@pytest.mark.parametrize(
    "policy,expected",
    [("drop", []), ("first", ["2"]), ("join_comma", ["2, 3"])],
)
def test_multi_answer_policy_decides_what_a_two_answer_row_becomes(
    monkeypatch: Any, policy: str, expected: List[str]
) -> None:
    _feed(monkeypatch, [OLYMPIAD_MULTI])
    rows = list(prepare_math._prepare_olympiadbench(_olympiad_spec(policy, "drop")))
    assert [r["answer"] for r in rows] == expected


@pytest.mark.parametrize(
    "policy,expected",
    [("drop", []), ("ignore", ["12"]), ("append", ["12 km/h"])],
)
def test_unit_policy_decides_what_a_row_with_a_unit_becomes(
    monkeypatch: Any, policy: str, expected: List[str]
) -> None:
    _feed(monkeypatch, [OLYMPIAD_UNIT])
    rows = list(prepare_math._prepare_olympiadbench(_olympiad_spec("drop", policy)))
    assert [r["answer"] for r in rows] == expected


def test_olympiadbench_stops_on_a_row_that_carries_context(monkeypatch: Any) -> None:
    """All 674 rows of the pinned revision have an empty context; one that does not is unruled."""
    _feed(monkeypatch, [dict(OLYMPIAD_SINGLE, context="Let n be a positive integer.")])
    with pytest.raises(SystemExit) as excinfo:
        list(prepare_math._prepare_olympiadbench(_olympiad_spec("drop", "drop")))
    assert "context" in str(excinfo.value)


def test_olympiadbench_refuses_an_empty_answer_list(monkeypatch: Any) -> None:
    _feed(monkeypatch, [dict(OLYMPIAD_SINGLE, final_answer=[])])
    with pytest.raises(SystemExit) as excinfo:
        list(prepare_math._prepare_olympiadbench(_olympiad_spec("drop", "drop")))
    assert "empty final_answer" in str(excinfo.value)


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, []),
        ("", []),
        ("2", ["2"]),
        (["2", " 3 "], ["2", "3"]),
        (["2", None, ""], ["2"]),
        (("2",), ["2"]),
    ],
)
def test_answer_list_accepts_every_shape_the_column_arrives_in(raw: Any, expected: List[str]) -> None:
    assert prepare_math._answer_list(raw) == expected


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def test_the_five_candidates_are_registered_in_preparation_order() -> None:
    names = [name for name, _ in prepare_math.EVAL_PROBE_OUTPUTS]
    assert names == ["Minerva-Math", "OlympiadBench", "AMC23", "AIME24", "AIME25"]


def test_preparing_the_candidates_is_opt_in() -> None:
    """The default preparation run and the release checks must be unchanged."""
    source = (REPO_ROOT / "scripts" / "10_data" / "prepare_math.py").read_text(encoding="utf-8")
    assert '"--prepare_eval_probe_sets", type=int, default=0' in source
