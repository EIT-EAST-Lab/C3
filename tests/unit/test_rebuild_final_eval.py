"""Tests for the evaluation of record and the task file it reads.

Three things are pinned here, and none of them needs a GPU, a model or the
`datasets` package.

First, `scripts/70_rebuild/final_eval.py` splits the suites by their sample
count, because one evaluation run has exactly one `--eval_n_samples_per_prompt`
and the protocol asks for four different ones. The grouping, the greedy reading
of the single-sample group, and the generated task file of each group are what
actually gets run on the platform, so they are checked here.

Second, the arithmetic of the record: avg@k over problems rather than over
draws, the boxed rate and its denominator, and the merge of AIME24 and AIME25
into the 60-problem suite the main table reports. The merge is checked on parts
of unequal size, where the mean over problems and the mean of the two suite
accuracies differ, so the test can tell them apart.

Third, `configs/tasks/math_final_eval.yaml`: its suites, their manifest entries,
and that the sample counts the driver ships cover exactly those suites.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / "scripts" / "70_rebuild"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import final_eval  # noqa: E402


FINAL_TASK = REPO_ROOT / "configs" / "tasks" / "math_final_eval.yaml"
MANIFEST = REPO_ROOT / "configs" / "data_manifest.yaml"

# The suites of the final evaluation, in the order the task file declares them.
EXPECTED_SUITES = [
    "MATH500",
    "Minerva-Math",
    "AMC23",
    "AIME24",
    "AIME25",
    "GSM8K-test",
    "CMATH-test",
]

# The sample counts the noise arithmetic of the analysis plan fixed. The two saturation
# controls are read with the same estimator as everything else, at k = 4.
EXPECTED_K = {
    "MATH500": 4,
    "Minerva-Math": 4,
    "AMC23": 16,
    "AIME24": 16,
    "AIME25": 16,
    "GSM8K-test": 4,
    "CMATH-test": 4,
}


def _args(*extra: str):
    argv = ["--policy", "/models/Qwen3-4B-Instruct-2507", *extra]
    return final_eval.build_parser().parse_args(argv)


def _flag(argv: List[str], name: str) -> str:
    """The value that follows `name` in an argv list."""
    index = argv.index(name)
    return argv[index + 1]


def _by_label(groups: Iterable[final_eval.EvalGroup]) -> Dict[str, final_eval.EvalGroup]:
    return {group.label: group for group in groups}


# ---------------------------------------------------------------------------
# Grouping the suites by sample count
# ---------------------------------------------------------------------------


def test_suites_with_the_same_sample_count_share_one_run() -> None:
    """One run has one k, so the two distinct k of the protocol are two runs."""
    groups = final_eval.plan_groups(_args())

    assert [g.label for g in groups] == ["k4", "k16"]
    by_label = _by_label(groups)
    assert by_label["k4"].suites == ("MATH500", "Minerva-Math", "GSM8K-test", "CMATH-test")
    assert by_label["k16"].suites == ("AMC23", "AIME24", "AIME25")
    assert [g.k for g in groups] == [4, 16]


def test_every_suite_of_the_protocol_is_sampled_the_same_way() -> None:
    """One estimator across the protocol: the saturation controls are not greedy."""
    for group in final_eval.plan_groups(_args()):
        assert (group.profile, group.temperature) == ("n10", 0.7)
        assert (group.top_p, group.top_k) == (0.8, 20)


def test_a_group_at_one_sample_per_problem_would_be_decoded_greedily() -> None:
    """k=1 stays a value the driver accepts, and one sample at 0.7 is not it.

    No suite of the shipped protocol asks for it. This pins what it would mean
    if one did, so a later suite added at k=1 cannot quietly become a single
    temperature 0.7 draw.
    """
    groups = final_eval.plan_groups(
        _args("--k_by_suite", "MATH500=4,Minerva-Math=4,AMC23=16,AIME24=16,AIME25=16,GSM8K-test=1,CMATH-test=1")
    )
    by_label = _by_label(groups)

    assert [g.label for g in groups] == ["k1", "k4", "k16"]
    greedy = by_label["k1"]
    assert greedy.suites == ("GSM8K-test", "CMATH-test")
    assert (greedy.profile, greedy.temperature) == ("greedy", 0.0)
    assert (greedy.top_p, greedy.top_k) == (None, None)
    assert _flag(greedy.argv, "--extra_eval_args") == "--eval_n_samples_per_prompt 1 --eval_temperature 0.0"


def test_each_command_states_its_sample_count_and_its_decoding() -> None:
    """The profile is only a label here, so every number rides in extra_eval_args."""
    by_label = _by_label(final_eval.plan_groups(_args()))

    assert _flag(by_label["k4"].argv, "--extra_eval_args") == (
        "--eval_n_samples_per_prompt 4 --eval_temperature 0.7 --top_p 0.8 --top_k 20"
    )
    assert _flag(by_label["k16"].argv, "--extra_eval_args") == (
        "--eval_n_samples_per_prompt 16 --eval_temperature 0.7 --top_p 0.8 --top_k 20"
    )


def test_command_drives_the_existing_evaluation_script_in_direct_mode() -> None:
    group = final_eval.plan_groups(_args())[0]

    assert group.argv[2] == "one"
    assert group.argv[1].endswith("scripts/50_eval/paper_main_results.sh")
    assert _flag(group.argv, "--source_type") == "hf_base"
    assert _flag(group.argv, "--hf_base") == "/models/Qwen3-4B-Instruct-2507"
    assert _flag(group.argv, "--method") == "SFT"
    assert _flag(group.argv, "--seed") == "0"


def test_the_generation_cap_is_the_protocol_cap_and_travels_as_an_environment_variable() -> None:
    """2048 for training and evaluation alike; 512 truncates AIME before the answer."""
    group = final_eval.plan_groups(_args())[0]

    assert group.env == {"GENERATE_MAX_LEN": "2048", "PROMPT_MAX_LEN": "2560"}
    assert "GENERATE_MAX_LEN=2048" in group.command_line()


def test_each_group_runs_a_generated_task_file_of_its_own() -> None:
    """The artifact directory is named after the task file, so the names must differ."""
    groups = final_eval.plan_groups(_args())
    paths = [g.task_path for g in groups]

    assert len(set(paths)) == len(paths)
    for group in groups:
        assert group.task_path.endswith(f"_task_groups/math_final_eval_k{group.k}.yaml")
        assert _flag(group.argv, "--task") == group.task_path
        assert Path(group.task_path).is_absolute() or group.task_path.startswith("/")


def test_artifact_paths_follow_the_layout_of_the_evaluation_script() -> None:
    by_label = _by_label(final_eval.plan_groups(_args()))

    tail = "ckpt/_runs/_sft_main_results/final_eval/final_eval"
    assert by_label["k4"].eval_dir.endswith(f"{tail}/math_final_eval_k4/n10")
    assert by_label["k16"].eval_dir.endswith(f"{tail}/math_final_eval_k16/n10")
    assert by_label["k4"].eval_jsonl.endswith("/n10/eval_only.jsonl")
    assert by_label["k4"].metrics_jsonl.endswith("/n10/eval_only.jsonl.metrics.jsonl")


def test_only_selects_a_single_group() -> None:
    groups = final_eval.plan_groups(_args("--only", "k16"))
    assert [g.label for g in groups] == ["k16"]

    with pytest.raises(ValueError):
        final_eval.plan_groups(_args("--only", "k7"))


def test_dry_run_prints_every_command_and_runs_nothing(capsys: Any, monkeypatch: Any) -> None:
    def _explode(*_a: Any, **_k: Any) -> None:
        raise AssertionError("a dry run must not start a subprocess")

    def _no_write(*_a: Any, **_k: Any) -> None:
        raise AssertionError("a dry run must not write a task file")

    monkeypatch.setattr(final_eval.subprocess, "run", _explode)
    monkeypatch.setattr(final_eval, "write_group_task", _no_write)

    code = final_eval.main(["--policy", "/models/m", "--dry-run"])
    out = capsys.readouterr().out

    assert code == 0
    assert out.count("paper_main_results.sh one") == 2
    assert "--eval_n_samples_per_prompt 16 --eval_temperature 0.7 --top_p 0.8 --top_k 20" in out
    assert "# groups: 2" in out


# ---------------------------------------------------------------------------
# Reading the sample counts
# ---------------------------------------------------------------------------


def test_the_shipped_sample_counts_are_the_ones_the_noise_arithmetic_fixed() -> None:
    assert final_eval.parse_k_by_suite(final_eval.DEFAULT_K_BY_SUITE) == EXPECTED_K


@pytest.mark.parametrize(
    "text",
    [
        "",
        "MATH500",
        "MATH500=",
        "=4",
        "MATH500=four",
        "MATH500=0",
        "MATH500=-1",
        "MATH500=4,MATH500=16",
    ],
)
def test_an_unreadable_sample_count_is_refused_rather_than_guessed(text: str) -> None:
    with pytest.raises(ValueError):
        final_eval.parse_k_by_suite(text)


def test_a_suite_the_task_does_not_declare_is_refused() -> None:
    with pytest.raises(ValueError) as excinfo:
        final_eval.plan_groups(_args("--k_by_suite", "MATH500=4,NotASuite=4"))
    assert "NotASuite" in str(excinfo.value)


def test_a_suite_without_a_sample_count_is_refused() -> None:
    """A missing suite would be evaluated at nothing, or silently left out."""
    with pytest.raises(ValueError) as excinfo:
        final_eval.plan_groups(_args("--k_by_suite", "MATH500=4"))
    assert "CMATH-test" in str(excinfo.value)


# ---------------------------------------------------------------------------
# The generated task file of a group
# ---------------------------------------------------------------------------


def test_the_generated_task_narrows_the_suites_and_changes_nothing_else() -> None:
    main_spec = yaml.safe_load(FINAL_TASK.read_text(encoding="utf-8"))
    document = final_eval.group_task_document(main_spec, ["AMC23", "AIME24", "AIME25"])

    assert [s["name"] for s in document["environment"]["eval_suites"]] == ["AMC23", "AIME24", "AIME25"]

    stripped_source = {k: v for k, v in main_spec["environment"].items() if k != "eval_suites"}
    stripped_group = {k: v for k, v in document["environment"].items() if k != "eval_suites"}
    assert stripped_group == stripped_source
    assert document["experiment_name"] == main_spec["experiment_name"]
    assert document["mas"] == main_spec["mas"]
    # The source document is not mutated, so a second group sees every suite.
    assert len(main_spec["environment"]["eval_suites"]) == len(EXPECTED_SUITES)


def test_the_generated_task_pins_the_roles_file_of_the_source_task() -> None:
    """A relative roles path means something else under the run root.

    The loader resolves `mas.roles_path` against the directory of the task file
    it is reading, and the generated file does not sit beside the task it came
    from, so the path is pinned absolute when it writes.
    """
    main_spec = yaml.safe_load(FINAL_TASK.read_text(encoding="utf-8"))
    assert main_spec["mas"]["roles_path"] == "../roles/math/roles_duo.json"

    pinned = final_eval.absolute_roles_path(FINAL_TASK, main_spec["mas"]["roles_path"])
    assert pinned is not None
    assert Path(pinned) == (REPO_ROOT / "configs" / "roles" / "math" / "roles_duo.json").resolve()

    document = final_eval.group_task_document(main_spec, ["AMC23"], roles_path=pinned)
    assert document["mas"]["roles_path"] == pinned


@pytest.mark.parametrize("value", ["", None, "../roles/math/does_not_exist.json"])
def test_a_roles_path_that_resolves_to_nothing_is_left_alone_rather_than_guessed_at(value: Any) -> None:
    assert final_eval.absolute_roles_path(FINAL_TASK, value) is None


def test_a_group_naming_a_suite_the_task_lacks_is_refused() -> None:
    main_spec = yaml.safe_load(FINAL_TASK.read_text(encoding="utf-8"))
    with pytest.raises(ValueError):
        final_eval.group_task_document(main_spec, ["OlympiadBench"])


def test_the_written_group_task_parses_and_carries_its_provenance(tmp_path: Path) -> None:
    group = _by_label(final_eval.plan_groups(_args("--ckpt_root", tmp_path.as_posix())))["k16"]
    written = final_eval.write_group_task(group, main_task=FINAL_TASK)

    text = Path(written).read_text(encoding="utf-8")
    assert "Generated by scripts/70_rebuild/final_eval.py" in text
    assert "math_final_eval.yaml" in text

    document = yaml.safe_load(text)
    assert [s["name"] for s in document["environment"]["eval_suites"]] == ["AMC23", "AIME24", "AIME25"]
    assert document["environment"]["train_datasets"][0]["name"] == "MATH-train"
    assert Path(document["mas"]["roles_path"]).is_file()
    assert Path(document["mas"]["roles_path"]).name == "roles_duo.json"


# ---------------------------------------------------------------------------
# Reading the evaluation dump
# ---------------------------------------------------------------------------


def _reward_info(pred_method: str) -> str:
    """The `reward_info_json` string the multi-agent dump carries, abbreviated."""
    return json.dumps(
        {
            "env": "MathEnv",
            "reward": 1.0,
            "per_role": [{"role": "actor", "pred_method": pred_method, "pred_raw": "42"}],
        },
        ensure_ascii=False,
    )


def _row(
    suite: str,
    qid: int,
    kid: int,
    reward: float,
    pred_method: Optional[str] = "boxed",
    reward_info: Any = "use_pred_method",
) -> Dict[str, Any]:
    """The shape the multi-agent branch of the evaluation dump writes."""
    if reward_info == "use_pred_method":
        reward_info = None if pred_method is None else _reward_info(pred_method)
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
        "reward_info_json": reward_info,
    }


def test_accuracy_is_the_mean_over_problems_of_their_sample_means() -> None:
    """avg@k, not the pooled draw rate: the two differ as soon as k varies."""
    rows = [
        _row("AMC23", 0, 0, 1.0),
        _row("AMC23", 0, 1, 0.0),
        _row("AMC23", 0, 2, 0.0),
        _row("AMC23", 0, 3, 0.0),
        _row("AMC23", 1, 0, 1.0),
        _row("AMC23", 1, 1, 1.0),
        _row("AMC23", 1, 2, 1.0),
        _row("AMC23", 1, 3, 1.0),
    ]
    amc = final_eval.summarize_dump(rows, k=4)["AMC23"]

    assert amc["n_questions"] == 2
    assert amc["n_draws"] == 8
    assert amc["n_scored_draws"] == 8
    assert amc["successes"] == 5
    # (0.25 + 1.0) / 2
    assert amc["accuracy"] == pytest.approx(0.625)
    assert amc["k"] == 4
    assert amc["n_questions_off_k"] == 0


def test_per_question_carries_the_k_rewards_of_every_problem_in_dump_order() -> None:
    rows = [
        _row("AIME24", 0, 0, 1.0),
        _row("AIME24", 0, 1, 0.0),
        _row("AIME24", 1, 0, 0.0),
        _row("AIME24", 1, 1, 0.0),
    ]
    block = final_eval.summarize_dump(rows, k=2)["AIME24"]

    assert block["per_question"] == [
        {"suite": "AIME24", "question_id": 0, "rewards": [1.0, 0.0]},
        {"suite": "AIME24", "question_id": 1, "rewards": [0.0, 0.0]},
    ]


def test_a_problem_that_did_not_get_its_k_samples_is_counted() -> None:
    """A dropped draw changes the estimator, so it may not pass unremarked."""
    rows = [_row("AMC23", 0, 0, 1.0), _row("AMC23", 0, 1, 0.0), _row("AMC23", 1, 0, 1.0)]
    block = final_eval.summarize_dump(rows, k=2)["AMC23"]

    assert block["n_questions"] == 2
    assert block["n_questions_off_k"] == 1


def test_each_suite_of_one_dump_is_counted_on_its_own() -> None:
    rows = [_row("AMC23", 0, 0, 1.0), _row("AIME24", 1, 0, 0.0), _row("AIME25", 2, 0, 1.0)]
    suites = final_eval.summarize_dump(rows, k=1)

    assert sorted(suites) == ["AIME24", "AIME25", "AMC23"]
    assert suites["AIME24"]["accuracy"] == pytest.approx(0.0)
    assert suites["AIME25"]["accuracy"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# The boxed rate
# ---------------------------------------------------------------------------


def test_the_boxed_rate_counts_the_generations_whose_answer_came_out_of_a_box() -> None:
    rows = [
        _row("MATH500", 0, 0, 1.0, pred_method="boxed"),
        _row("MATH500", 0, 1, 0.0, pred_method="last_line"),
        _row("MATH500", 1, 0, 0.0, pred_method="anchor"),
        _row("MATH500", 1, 1, 1.0, pred_method="boxed"),
    ]
    block = final_eval.summarize_dump(rows, k=2)["MATH500"]

    assert block["boxed"]["n_boxed"] == 2
    assert block["boxed"]["n_with_pred_method"] == 4
    assert block["boxed_rate"] == pytest.approx(0.5)


def test_a_generation_that_carried_no_reward_detail_stays_out_of_the_denominator() -> None:
    """An unparsed detail is missing data, not a generation that failed to box."""
    rows = [
        _row("MATH500", 0, 0, 1.0, pred_method="boxed"),
        _row("MATH500", 0, 1, 0.0, pred_method=None),
        _row("MATH500", 1, 0, 0.0, reward_info="null"),
        _row("MATH500", 1, 1, 0.0, reward_info="{not json"),
    ]
    block = final_eval.summarize_dump(rows, k=2)["MATH500"]

    assert block["boxed"]["n_boxed"] == 1
    assert block["boxed"]["n_with_pred_method"] == 1
    assert block["boxed"]["n_without_pred_method"] == 3
    assert block["boxed_rate"] == pytest.approx(1.0)


def test_the_boxed_rate_is_undefined_rather_than_zero_when_nothing_carried_a_detail() -> None:
    rows = [_row("MATH500", 0, 0, 1.0, pred_method=None)]
    assert final_eval.summarize_dump(rows, k=1)["MATH500"]["boxed_rate"] is None


@pytest.mark.parametrize("wrapped", [False, True])
def test_the_reward_detail_is_read_whether_or_not_it_is_still_wrapped_in_a_list(wrapped: bool) -> None:
    """
    The rollout writes `reward_info_json` as a one-element list and the dump
    writer unwraps it. Both shapes are read, neither is assumed.
    """
    detail: Any = _reward_info("boxed")
    if wrapped:
        detail = [detail]
    assert final_eval.row_pred_method({"reward_info_json": detail}) == "boxed"


def test_a_row_with_no_reward_detail_at_all_has_no_extraction_method() -> None:
    assert final_eval.row_pred_method({}) is None
    assert final_eval.row_pred_method({"reward_info_json": None}) is None
    assert final_eval.row_pred_method({"reward_info_json": json.dumps({"per_role": []})}) is None


# ---------------------------------------------------------------------------
# Merging AIME24 and AIME25
# ---------------------------------------------------------------------------


def _suites_for_merge() -> Dict[str, Dict[str, Any]]:
    """Two parts of unequal size, so the two candidate merges disagree."""
    rows = (
        [_row("AIME24", qid, 0, 1.0) for qid in range(3)]
        + [_row("AIME24", qid, 0, 0.0) for qid in range(3, 4)]
        + [_row("AIME25", 10, 0, 0.0)]
    )
    return final_eval.summarize_dump(rows, k=1)


def test_the_two_aime_files_merge_problem_by_problem_not_accuracy_by_accuracy() -> None:
    suites = _suites_for_merge()
    assert suites["AIME24"]["accuracy"] == pytest.approx(0.75)
    assert suites["AIME25"]["accuracy"] == pytest.approx(0.0)

    merged = final_eval.merged_block("AIME", ("AIME24", "AIME25"), suites)

    assert merged["n_questions"] == 5
    # Three of five problems solved, not the 0.375 the mean of the two accuracies gives.
    assert merged["accuracy"] == pytest.approx(0.6)
    assert merged["merged_from"] == ["AIME24", "AIME25"]
    assert merged["k"] == 1
    assert [item["suite"] for item in merged["per_question"]] == ["AIME24"] * 4 + ["AIME25"]


def test_the_merged_entry_pools_the_boxed_counts() -> None:
    rows = [
        _row("AIME24", 0, 0, 1.0, pred_method="boxed"),
        _row("AIME24", 1, 0, 0.0, pred_method="last_line"),
        _row("AIME25", 2, 0, 0.0, pred_method="boxed"),
    ]
    merged = final_eval.merged_block("AIME", ("AIME24", "AIME25"), final_eval.summarize_dump(rows, k=1))

    assert merged["boxed"]["n_boxed"] == 2
    assert merged["boxed"]["n_with_pred_method"] == 3
    assert merged["boxed_rate"] == pytest.approx(2 / 3)


def test_the_merged_entry_refuses_parts_evaluated_at_different_sample_counts() -> None:
    suites = final_eval.summarize_dump([_row("AIME24", 0, 0, 1.0)], k=1)
    suites.update(final_eval.summarize_dump([_row("AIME25", 1, 0, 1.0)], k=16))

    with pytest.raises(ValueError):
        final_eval.merged_block("AIME", ("AIME24", "AIME25"), suites)


def test_a_merge_of_one_present_part_says_so() -> None:
    suites = final_eval.summarize_dump([_row("AIME24", 0, 0, 1.0)], k=1)
    merged = final_eval.merged_block("AIME", ("AIME24", "AIME25"), suites)

    assert merged["merged_from"] == ["AIME24"]
    assert "AIME25" in merged["note"]


def test_a_merge_with_no_present_part_is_not_written() -> None:
    suites = final_eval.summarize_dump([_row("MATH500", 0, 0, 1.0)], k=1)
    assert final_eval.merged_block("AIME", ("AIME24", "AIME25"), suites) is None


# ---------------------------------------------------------------------------
# The record as a whole
# ---------------------------------------------------------------------------


def _write_group(group: final_eval.EvalGroup, rows: List[Dict[str, Any]]) -> None:
    """The two artifacts one evaluation leaves behind, with the metrics the trainer logs."""
    eval_dir = Path(group.eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    with (eval_dir / "eval_only.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    questions: Dict[str, Dict[Any, List[float]]] = {}
    for row in rows:
        suite = str(row["datasource"])
        questions.setdefault(suite, {}).setdefault(row["question_id"], []).append(float(row["answer_reward"]))

    metrics = {
        f"eval_{suite}_pass1": sum(sum(v) / len(v) for v in per_question.values()) / len(per_question)
        for suite, per_question in questions.items()
    }
    with (eval_dir / "eval_only.jsonl.metrics.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps({"kind": "eval_metrics", "global_step": 0, "metrics": metrics}) + "\n")


def test_a_complete_record_reports_every_suite_plus_the_merged_aime(tmp_path: Path) -> None:
    args = _args("--ckpt_root", tmp_path.as_posix())
    groups = final_eval.plan_groups(args)
    by_label = _by_label(groups)

    _write_group(
        by_label["k4"],
        [_row("MATH500", 0, kid, 1.0) for kid in range(4)]
        + [_row("Minerva-Math", 1, kid, 0.0) for kid in range(4)]
        + [_row("GSM8K-test", 2, kid, 1.0) for kid in range(4)]
        + [_row("CMATH-test", 3, kid, 1.0) for kid in range(4)],
    )
    _write_group(
        by_label["k16"],
        [_row("AMC23", 0, kid, 1.0) for kid in range(16)]
        + [_row("AIME24", 1, kid, 1.0) for kid in range(16)]
        + [_row("AIME25", 2, kid, 0.0) for kid in range(16)],
    )

    record = final_eval.build_record(groups, args)

    assert record["problems"] == []
    assert record["ok"] is True
    assert sorted(record["suites"]) == sorted(EXPECTED_SUITES + ["AIME"])
    assert record["suites"]["MATH500"]["k"] == 4
    assert record["suites"]["MATH500"]["group"] == "k4"
    assert record["suites"]["AMC23"]["k"] == 16
    assert record["suites"]["AIME"]["n_questions"] == 2
    assert record["suites"]["AIME"]["accuracy"] == pytest.approx(0.5)
    assert record["k_by_suite"] == EXPECTED_K
    assert record["generate_max_len"] == 2048
    assert all(values["accuracy_matches_trainer"] for values in record["groups"]["k4"]["suites"].values())


def test_a_group_whose_dump_is_missing_is_a_problem_not_an_empty_record(tmp_path: Path) -> None:
    args = _args("--ckpt_root", tmp_path.as_posix())
    groups = final_eval.plan_groups(args)

    record = final_eval.build_record(groups, args)

    assert record["ok"] is False
    assert any("evaluation dump not found" in p for p in record["problems"])
    assert record["suites"] == {}


def test_a_suite_that_produced_no_rows_is_named(tmp_path: Path) -> None:
    """The silent failure this record has to catch: a suite that never ran."""
    args = _args("--ckpt_root", tmp_path.as_posix())
    group = _by_label(final_eval.plan_groups(args))["k16"]
    _write_group(group, [_row("AMC23", 0, kid, 1.0) for kid in range(16)])

    block = final_eval.summarize_group(group)

    assert block["observed_suites"] == ["AMC23"]
    assert block["missing_suites"] == ["AIME24", "AIME25"]
    assert block["problems"], "a group that covered one suite of three must not look clean"


# ---------------------------------------------------------------------------
# The task file
# ---------------------------------------------------------------------------


def test_the_final_task_declares_the_seven_suites_and_caps_none_of_them() -> None:
    spec = yaml.safe_load(FINAL_TASK.read_text(encoding="utf-8"))
    suites = spec["environment"]["eval_suites"]

    assert [s["name"] for s in suites] == EXPECTED_SUITES
    assert {s["limit"] for s in suites} == {None}


def test_the_final_task_reuses_the_training_block_and_scorer_of_the_math_task() -> None:
    final = yaml.safe_load(FINAL_TASK.read_text(encoding="utf-8"))["environment"]
    main = yaml.safe_load((REPO_ROOT / "configs" / "tasks" / "math.yaml").read_text(encoding="utf-8"))["environment"]

    assert final["train_datasets"] == main["train_datasets"]
    for key in ("env_name", "sampling_mode", "episode_length", "math_backend", "reward_mode", "use_math_verify"):
        assert final[key] == main[key], key


def test_the_final_task_covers_every_suite_the_training_time_evaluation_measures() -> None:
    """The main table comes from this file, so it may not be missing a monitored suite."""
    final_names = [s["name"] for s in yaml.safe_load(FINAL_TASK.read_text(encoding="utf-8"))["environment"]["eval_suites"]]
    main_names = [
        s["name"]
        for s in yaml.safe_load((REPO_ROOT / "configs" / "tasks" / "math.yaml").read_text(encoding="utf-8"))[
            "environment"
        ]["eval_suites"]
    ]

    assert set(main_names) <= set(final_names)


def test_every_final_task_suite_points_at_the_file_its_manifest_entry_declares() -> None:
    suites = yaml.safe_load(FINAL_TASK.read_text(encoding="utf-8"))["environment"]["eval_suites"]
    outputs = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))["outputs"]
    by_name = {entry["name"]: entry for entry in outputs}

    for suite in suites:
        name = suite["name"]
        assert name in by_name, f"{name} has no manifest entry"
        assert suite["path"] == by_name[name]["output_path"], name


def test_the_shipped_sample_counts_cover_exactly_the_suites_of_the_task() -> None:
    declared = [s["name"] for s in yaml.safe_load(FINAL_TASK.read_text(encoding="utf-8"))["environment"]["eval_suites"]]
    assert sorted(final_eval.parse_k_by_suite(final_eval.DEFAULT_K_BY_SUITE)) == sorted(declared)


def test_olympiadbench_is_not_evaluated() -> None:
    """It was dropped by the protocol: nine tenths of its answers never reach a box."""
    text = FINAL_TASK.read_text(encoding="utf-8")
    assert "OlympiadBench" not in text
    assert "OLYMPIADBENCH" not in text
