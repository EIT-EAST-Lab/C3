# tests/unit/test_e2_curves.py
"""The monitoring curve of one E2 run directory.

The fixtures are a few rows of the metrics file the trainer writes at each
evaluation point and a few rows of the per-problem dump beside it. Nothing here
reads real data. What is pinned is the shape of a point (avg@k and pass@k kept
apart, both in percent), the problem count that has to be recovered from the
other file because the metrics file drops it, and the two absences that are
reported rather than filled: no dump file, and no metrics file at all, which is
the state of an archived run until the platform copies that file too.

    python -m pytest tests/unit/test_e2_curves.py -q
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from c3.analysis.rebuild import e2_curves as ec

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def metrics_row(step, accuracies, *, k=4, pass_at_k=None, temperature=0.7, duration=120.0,
                kind=ec.METRICS_KIND):
    """One `eval_metrics` row in the shape `_maybe_dump_eval_metrics_jsonl` writes."""
    metrics = {}
    for suite, value in accuracies.items():
        metrics["eval_%s_pass1" % suite] = value
        best = (pass_at_k or {}).get(suite)
        if k > 1 and best is not None:
            metrics["eval_%s_pass%d" % (suite, k)] = best
    return {"kind": kind, "ts": 1_000_000.0 + step, "global_step": step,
            "duration_sec": duration, "temperature": temperature,
            "n_samples_per_prompt": k, "marl_algorithm": "c3", "metrics": metrics}


def write_metrics(run_dir, rows):
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, ec.METRICS_NAME)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return path


def log_line(step, accuracies, *, pass_at_k=None, k=4):
    """One trainer log line in the shape `evaluate` logs after an evaluation pass.

    The prefixes are the real ones: Ray stamps the stdout of the actor and the
    logger stamps a level, a timestamp and a source location before the message.
    The dictionary is printed the way `%s` on a dict prints it, single quotes
    and all, which is why the reader takes it as a Python literal.
    """
    metrics = {}
    for suite, value in accuracies.items():
        best = (pass_at_k or {}).get(suite)
        if k > 1 and best is not None:
            metrics["eval_%s_pass%d" % (suite, k)] = best
        metrics["eval_%s_pass1" % suite] = value
    return ("\x1b[36m(PPOTrainer pid=118945)\x1b[0m INFO 09-19 10:28:58 "
            "ppo_trainer_plugins.py:1358] Evaluation completed in 0:07:19, "
            "global_step %d, eval_metrics: %s\n" % (step, metrics))


def write_log(run_dir, lines):
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, ec.LOG_NAME)
    with open(path, "w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(line)
    return path


def write_questions(run_dir, per_step):
    """`{step: {suite: number of problems}}` as one dump row per problem and sample."""
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, ec.QUESTIONS_NAME)
    with open(path, "w", encoding="utf-8") as fh:
        for step, suites in sorted(per_step.items()):
            question_id = 0
            for suite, count in sorted(suites.items()):
                for _ in range(count):
                    for k_id in range(2):
                        fh.write(json.dumps({"global_step": step, "datasource": suite,
                                             "question_id": question_id, "k_id": k_id,
                                             "text": "x" * 8}) + "\n")
                    question_id += 1
    return path


def two_point_run(run_dir):
    """Two evaluation points over two suites, with the dump beside them."""
    write_metrics(run_dir, [
        metrics_row(78, {"MATH500": 0.415, "AIME24": 0.10},
                    pass_at_k={"MATH500": 0.62, "AIME24": 0.20}),
        metrics_row(156, {"MATH500": 0.4425, "AIME24": 0.1333},
                    pass_at_k={"MATH500": 0.65, "AIME24": 0.2667}),
    ])
    write_questions(run_dir, {78: {"MATH500": 5, "AIME24": 3},
                              156: {"MATH500": 5, "AIME24": 3}})
    return run_dir


# -----------------------------------------------------------------------------
# one metric name
# -----------------------------------------------------------------------------


def test_a_metric_name_splits_into_its_suite_and_its_sample_count():
    assert ec.split_metric_name("eval_MATH500_pass1") == {"suite": "MATH500", "k": 1}
    assert ec.split_metric_name("eval_Minerva-Math_pass4") == {"suite": "Minerva-Math", "k": 4}
    # A suite name holding the separator survives: the tail is anchored at the end.
    assert ec.split_metric_name("eval_GSM8K_test_pass4") == {"suite": "GSM8K_test", "k": 4}
    assert ec.split_metric_name("eval__pass4") is None
    assert ec.split_metric_name("eval_MATH500") is None
    assert ec.split_metric_name("reward/mean") is None


# -----------------------------------------------------------------------------
# the points of a run
# -----------------------------------------------------------------------------


def test_a_point_carries_both_readings_in_percent_and_keeps_them_apart(tmp_path):
    run = two_point_run(os.path.join(str(tmp_path), "train_s0"))
    document = ec.read_run(run)

    assert document["schema"] == ec.SCHEMA
    assert document["n_points"] == 2
    assert document["suites"] == ["AIME24", "MATH500"]
    assert [point["global_step"] for point in document["points"]] == [78, 156]

    first = document["points"][0]
    assert first["k"] == 4 and first["temperature"] == pytest.approx(0.7)
    assert first["duration_sec"] == pytest.approx(120.0)

    math500 = first["suites"]["MATH500"]
    assert math500["accuracy_pct"] == pytest.approx(41.5)
    assert math500["pass_at_k_pct"] == pytest.approx(62.0)
    assert math500["k_of_pass_at_k"] == 4
    assert math500["n_questions"] == 5
    assert first["suites"]["AIME24"]["n_questions"] == 3


def test_the_two_aime_years_stay_apart_because_the_task_evaluates_them_apart(tmp_path):
    """The final record merges them into one suite of 60; the curve does not."""
    run = os.path.join(str(tmp_path), "train_s0")
    write_metrics(run, [metrics_row(78, {"AIME24": 0.10, "AIME25": 0.20})])

    document = ec.read_run(run, with_counts=False)
    assert document["suites"] == ["AIME24", "AIME25"]
    named = {gap["quantity"] for gap in document["not_recorded"]}
    assert "the_merged_AIME_suite" in named


def test_a_point_without_a_pass_at_k_reading_leaves_it_undefined(tmp_path):
    """One sample per problem: `evaluate` writes pass1 and nothing else."""
    run = os.path.join(str(tmp_path), "train_s0")
    write_metrics(run, [metrics_row(78, {"MATH500": 0.40}, k=1)])

    entry = ec.read_run(run, with_counts=False)["points"][0]["suites"]["MATH500"]
    assert entry["accuracy_pct"] == pytest.approx(40.0)
    assert entry["pass_at_k_pct"] is None and entry["k_of_pass_at_k"] is None


def test_a_step_written_twice_keeps_its_last_row(tmp_path):
    run = os.path.join(str(tmp_path), "train_s0")
    write_metrics(run, [metrics_row(78, {"MATH500": 0.40}),
                        metrics_row(78, {"MATH500": 0.44})])

    document = ec.read_run(run, with_counts=False)
    assert document["metrics_file"]["n_duplicate_steps"] == 1
    assert document["n_points"] == 1
    assert document["points"][0]["suites"]["MATH500"]["accuracy_pct"] == pytest.approx(44.0)


def test_rows_of_another_kind_and_rows_that_do_not_parse_are_counted(tmp_path):
    run = os.path.join(str(tmp_path), "train_s0")
    path = write_metrics(run, [metrics_row(78, {"MATH500": 0.40}),
                               metrics_row(90, {"MATH500": 0.50}, kind="train_rollouts")])
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("{not json\n")
        fh.write(json.dumps({"kind": ec.METRICS_KIND, "global_step": "late"}) + "\n")

    document = ec.read_run(run, with_counts=False)
    assert document["metrics_file"]["rows"] == 1
    assert document["metrics_file"]["n_other_kind"] == 1
    assert document["metrics_file"]["n_unparsed"] == 2
    assert document["n_points"] == 1


def test_a_metric_that_is_not_a_number_is_named_rather_than_read_as_zero(tmp_path):
    run = os.path.join(str(tmp_path), "train_s0")
    row = metrics_row(78, {"MATH500": 0.40})
    row["metrics"]["eval_AMC23_pass1"] = None
    row["metrics"]["env/error_frac"] = 0.0
    write_metrics(run, [row])

    point = ec.read_run(run, with_counts=False)["points"][0]
    assert list(point["suites"]) == ["MATH500"]
    assert point["unreadable_metrics"] == ["env/error_frac", "eval_AMC23_pass1"]


# -----------------------------------------------------------------------------
# the counts and the two absences
# -----------------------------------------------------------------------------


def test_the_problem_count_comes_from_the_other_file_and_counts_problems_not_samples(tmp_path):
    run = two_point_run(os.path.join(str(tmp_path), "train_s0"))
    counts = ec.count_questions(os.path.join(run, ec.QUESTIONS_NAME))
    # two samples per problem were written, and the count is of distinct problems
    assert counts["rows"] == 2 * 2 * (5 + 3)
    assert counts["by_step"]["78"] == {"AIME24": 3, "MATH500": 5}


def test_without_the_dump_file_the_count_is_null_and_said_to_be(tmp_path):
    run = os.path.join(str(tmp_path), "train_s0")
    write_metrics(run, [metrics_row(78, {"MATH500": 0.40})])

    document = ec.read_run(run)
    assert document["points"][0]["suites"]["MATH500"]["n_questions"] is None
    reasons = [gap for gap in document["not_recorded"]
               if gap["quantity"] == "n_questions of every suite"]
    assert len(reasons) == 1 and ec.QUESTIONS_NAME in reasons[0]["why"]


def test_skipping_the_dump_file_is_reported_as_its_own_reason(tmp_path):
    run = two_point_run(os.path.join(str(tmp_path), "train_s0"))
    document = ec.read_run(run, with_counts=False)
    assert document["questions_file"]["read"] is False
    reasons = [gap["why"] for gap in document["not_recorded"]
               if gap["quantity"] == "n_questions of every suite"]
    assert len(reasons) == 1 and "--no_counts" in reasons[0]


def test_a_run_without_a_metrics_file_names_the_archive_step_that_left_it_behind(tmp_path):
    run = os.path.join(str(tmp_path), "train_s0")
    os.makedirs(run, exist_ok=True)

    document = ec.read_run(run)
    assert document["n_points"] == 0 and document["suites"] == []
    assert document["metrics_file"]["present"] is False
    reasons = [gap for gap in document["not_recorded"]
               if gap["quantity"] == "the whole monitoring curve"]
    assert len(reasons) == 1
    assert ec.METRICS_NAME in reasons[0]["why"]
    assert "archive step" in reasons[0]["would_need"]


def test_a_metrics_file_with_no_evaluation_point_is_its_own_reason(tmp_path):
    run = os.path.join(str(tmp_path), "train_s0")
    write_metrics(run, [])
    reasons = [gap["why"] for gap in ec.read_run(run)["not_recorded"]
               if gap["quantity"] == "the whole monitoring curve"]
    assert len(reasons) == 1 and ec.METRICS_KIND in reasons[0]


# -----------------------------------------------------------------------------
# the trainer-log fallback
# -----------------------------------------------------------------------------


def test_the_points_come_off_the_trainer_log_when_the_metrics_file_never_landed(tmp_path):
    """Three logged lines, one of them a step logged twice, and no metrics file.

    This is the archived run: the container is gone, the archive step copied
    train.log and not the metrics file, and the same numbers survive only as the
    line the evaluation pass logged.
    """
    run = os.path.join(str(tmp_path), "train_s0")
    write_log(run, [
        log_line(78, {"MATH500": 0.8875, "Minerva-Math": 0.59375},
                 pass_at_k={"MATH500": 0.94, "Minerva-Math": 0.6838235294117647}),
        log_line(156, {"MATH500": 0.8915}, pass_at_k={"MATH500": 0.936}),
        log_line(78, {"MATH500": 0.90}, pass_at_k={"MATH500": 0.95}),
    ])

    document = ec.read_run(run, with_counts=False)

    assert document["source"] == ec.LOG_NAME
    assert document["n_points"] == 2
    assert [point["global_step"] for point in document["points"]] == [78, 156]
    # The step logged twice keeps its last line, the rule the metrics file follows.
    assert document["log_fallback"]["n_duplicate_steps"] == 1
    assert document["log_fallback"]["rows"] == 3
    assert document["log_fallback"]["read"] is True
    assert document["metrics_file"]["present"] is False

    first = document["points"][0]["suites"]["MATH500"]
    assert first["accuracy_pct"] == pytest.approx(90.0)
    assert first["pass_at_k_pct"] == pytest.approx(95.0)
    assert first["k_of_pass_at_k"] == 4
    # The logged line carries no k, no temperature and no duration; none is invented.
    assert "k" not in document["points"][0]
    assert "temperature" not in document["points"][0]
    assert document["points"][1]["suites"]["MATH500"]["accuracy_pct"] == pytest.approx(89.15)
    # The later line replaces the earlier one whole, so the Minerva reading of the
    # first line of step 78 is gone with it rather than merged into the last.
    assert document["suites"] == ["MATH500"]
    # A run with points has no "whole monitoring curve" gap, whichever file they came from.
    assert not [gap for gap in document["not_recorded"]
                if gap["quantity"] == "the whole monitoring curve"]


def test_the_metrics_file_wins_and_the_log_is_not_read_when_it_has_points(tmp_path):
    run = two_point_run(os.path.join(str(tmp_path), "train_s0"))
    write_log(run, [log_line(999, {"MATH500": 0.99})])

    document = ec.read_run(run, with_counts=False)
    assert document["source"] == ec.METRICS_NAME
    assert document["log_fallback"]["read"] is False
    assert [point["global_step"] for point in document["points"]] == [78, 156]


def test_a_log_line_whose_dictionary_does_not_read_is_counted_not_guessed(tmp_path):
    run = os.path.join(str(tmp_path), "train_s0")
    write_log(run, [
        log_line(78, {"MATH500": 0.40}),
        "INFO ppo_trainer_plugins.py:1358] global_step 90, eval_metrics: {'eval_MATH500_pass1': }\n",
        "INFO ppo_trainer.py:2471] Global step 91 | epoch 1/2 | iter 91/312: {'reward/mean': 0.5}\n",
    ])

    block = ec.read_log_metrics(os.path.join(run, ec.LOG_NAME))
    assert block["rows"] == 1
    assert block["n_unparsed"] == 1
    assert block["n_other_kind"] == 0
    assert [point["global_step"] for point in block["points"]] == [78]


def test_with_neither_file_the_absence_names_both(tmp_path):
    run = os.path.join(str(tmp_path), "train_s0")
    os.makedirs(run, exist_ok=True)

    document = ec.read_run(run)
    assert document["source"] == ec.METRICS_NAME
    assert document["n_points"] == 0
    reasons = [gap for gap in document["not_recorded"]
               if gap["quantity"] == "the whole monitoring curve"]
    assert len(reasons) == 1
    assert ec.METRICS_NAME in reasons[0]["why"] and ec.LOG_NAME in reasons[0]["why"]


def test_a_log_without_an_evaluation_line_is_said_to_carry_none(tmp_path):
    run = os.path.join(str(tmp_path), "train_s0")
    write_log(run, ["INFO ppo_trainer.py:2471] Global step 1 | epoch 1/2 | iter 1/312: {}\n"])

    document = ec.read_run(run, with_counts=False)
    assert document["log_fallback"]["present"] is True
    reasons = [gap["why"] for gap in document["not_recorded"]
               if gap["quantity"] == "the whole monitoring curve"]
    assert len(reasons) == 1 and "carries no" in reasons[0]


def test_every_quantity_of_the_document_names_the_file_it_came_from(tmp_path):
    run = two_point_run(os.path.join(str(tmp_path), "train_s0"))
    provenance = ec.read_run(run, with_counts=False)["provenance"]
    assert provenance["points[].suites[].accuracy_pct"]["file"] == ec.METRICS_NAME
    assert provenance["points[].suites[].n_questions"]["file"] == ec.QUESTIONS_NAME
    for entry in provenance.values():
        assert set(entry) == {"file", "field", "how"}


# -----------------------------------------------------------------------------
# the command line
# -----------------------------------------------------------------------------


def test_the_command_line_writes_the_document_and_refuses_a_missing_directory(tmp_path, capsys):
    run = two_point_run(os.path.join(str(tmp_path), "train_s0"))
    out_path = os.path.join(str(tmp_path), "curve.json")

    assert ec.main(["--run", run, "--out", out_path]) == 0
    with open(out_path, "r", encoding="utf-8") as fh:
        document = json.load(fh)
    assert document["n_points"] == 2
    assert document["points"][0]["suites"]["MATH500"]["n_questions"] == 5

    assert ec.main(["--run", run, "--out", out_path, "--no_counts"]) == 0
    with open(out_path, "r", encoding="utf-8") as fh:
        assert json.load(fh)["points"][0]["suites"]["MATH500"]["n_questions"] is None

    assert ec.main(["--run", os.path.join(str(tmp_path), "nowhere"), "--out", out_path]) == 2
    assert "is not a directory" in capsys.readouterr().err


def test_it_runs_as_a_module(tmp_path):
    run = two_point_run(os.path.join(str(tmp_path), "train_s0"))
    out_path = os.path.join(str(tmp_path), "curve.json")
    done = subprocess.run(
        [sys.executable, "-m", "c3.analysis.rebuild.e2_curves", "--run", run, "--out", out_path],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stderr
    with open(out_path, "r", encoding="utf-8") as fh:
        assert json.load(fh)["schema"] == ec.SCHEMA
