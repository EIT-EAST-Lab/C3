# tests/unit/test_e2_tokens.py
"""The training token account of one E2 run directory.

Every fixture is a few synthetic lines of a trainer log and a few rows of a
budget ledger; nothing here reads real data or needs the training stack. What
is pinned is mostly the refusal to guess: the January script this one follows
fell back to 2048 samples and 512 tokens for a step line that did not carry the
field, and the whole point of the rewrite is that a step like that is counted
and left out instead.

    python -m pytest tests/unit/test_e2_tokens.py -q
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from c3.analysis.rebuild import e2_tokens as et

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: The prefix a Ray actor's stdout carries in front of the logger's own prefix.
PREFIX = "(PPOTrainerActor pid=1234) INFO 09-19 05:12:33 ppo_trainer.py:2471] "


def step_line(step, *, response=512.0, total=900.0, samples=320, epoch=1, iteration=None,
              extra=None, prefix=PREFIX):
    """One `Global step` line in the shape `ppo_trainer.py:fit` prints it."""
    status = {}
    if response is not None:
        status["response_length"] = response
    if total is not None:
        status["total_length"] = total
    if samples is not None:
        status["reward/count"] = samples
    status.update(extra or {})
    where = "%s | epoch %d/2 | iter %d/312" % ("Global step %d" % step, epoch,
                                               step if iteration is None else iteration)
    return "%s%s: %r\n" % (prefix, where, status)


def write_log(run_dir, lines, name=et.LOG_NAME):
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, name)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("INFO 09-19 05:00:00 train_ppo_ray.py:1] a line that is not a step\n")
        for line in lines:
            fh.write(line)
    return path


def write_ledger(run_dir, rows):
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, et.LEDGER_NAME)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return path


def write_config(run_dir, args):
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, et.CONFIG_NAME)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"ts": 0.0, "run_id": "x", "args": args}, fh)
    return path


def ledger_row(step, *, questions=128, samples=8, ts=1_000_000):
    return {"global_step": step, "epoch_idx": 0, "iter_in_epoch": step,
            "marl_algorithm": "c3", "n_questions_in_batch": questions,
            "n_samples_per_prompt": samples, "total_eval_calls": questions * samples,
            "ts_unix": ts}


def write_warmup_ledger(run_dir, rows):
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, et.WARMUP_LEDGER_NAME)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return path


def warmup_row(idx, *, questions=128, samples=8, ts=1_000_000, nodes=1024,
               generated=500_000, total=900_000):
    """One row in the shape `c3.utils.budget_ledger.make_warmup_record` writes."""
    row = {"stage": "critic_warmup", "warmup_rollout_idx": idx, "global_step": -1,
           "epoch_idx": -1, "iter_in_epoch": -1, "marl_algorithm": "mappo",
           "n_questions_in_batch": questions, "n_samples_per_prompt": samples,
           "total_eval_calls": questions * samples, "ts_unix": ts}
    for name, value in (("n_nodes", nodes), ("generated_tokens", generated),
                        ("total_tokens", total)):
        if value is not None:
            row[name] = value
    return row


def complete_run(run_dir):
    """Two steps, both complete, and a ledger spanning one hour."""
    write_log(run_dir, [step_line(1, response=500.0, total=900.0, samples=300),
                        step_line(2, response=600.0, total=1000.0, samples=320)])
    write_ledger(run_dir, [ledger_row(1, ts=1_000_000), ledger_row(2, ts=1_003_600)])
    write_config(run_dir, {"run_id": "a2_4b_c3_s0", "marl_algorithm": "c3",
                           "policy_sharing_mode": "shared", "n_samples_per_prompt": 8,
                           "rollout_batch_size": 128, "unrelated": 1})
    return run_dir


# -----------------------------------------------------------------------------
# one status line
# -----------------------------------------------------------------------------


def test_a_step_line_is_read_through_whatever_prefixes_it():
    parsed = et.parse_status_line(step_line(12, response=512.5, total=900.25, samples=320))
    assert parsed["global_step"] == 12
    assert parsed["status"]["response_length"] == 512.5
    assert parsed["status"]["reward/count"] == 320
    assert parsed["repaired"] is False
    assert (parsed["epoch"], parsed["train_epochs"]) == (1, 2)
    assert (parsed["iter"], parsed["iters_in_epoch"]) == (12, 312)


def test_a_line_without_a_step_or_without_a_dictionary_is_not_a_step_line():
    assert et.parse_status_line("INFO evaluation completed, global_step 4\n") is None
    assert et.parse_status_line("Global step 4 | epoch 1/2 | iter 4: no dictionary here\n") is None
    assert et.parse_status_line("Global step 4: {this is not a literal}\n") is None


def test_a_dictionary_holding_nan_is_repaired_rather_than_dropped():
    """`nan` is not a Python literal, so the whole line would otherwise be lost."""
    line = ("Global step 3 | epoch 1/2 | iter 3/312: "
            "{'response_length': 500.0, 'total_length': 900.0, 'reward/count': 320, 'kl': nan}\n")
    parsed = et.parse_status_line(line)
    assert parsed["repaired"] is True
    assert parsed["status"]["kl"] is None
    assert parsed["status"]["response_length"] == 500.0


def test_the_message_may_carry_no_iteration_total():
    """`total_batches` is optional in the message the trainer prints."""
    line = "Global step 7 | epoch 2/2 | iter 3: {'reward/count': 8}\n"
    parsed = et.parse_status_line(line)
    assert (parsed["iter"], parsed["iters_in_epoch"]) == (3, None)


#: What Ray flushes onto the end of a status line when a second actor writes at
#: the same moment: its own prefix, then a message of its own that carries
#: braces inside a string. Taken from the shape of the seven such lines in
#: 20_data/results/E2/a2_4b/c3/train_s0/train.log.
INTERLEAVED_TAIL = ("\x1b[36m(PPOTrainer pid=1)\x1b[0m We did not manage to extract a prediction "
                    "in the correct format. Gold: ['\\\\boxed{H, E}'], Pred: ['E, H']\n")


def test_another_actors_message_after_the_closing_brace_does_not_take_the_step_down():
    """The dictionary ends at the `}` that closes it, not at the last `}` of the line."""
    clean = step_line(97, response=512.5, total=900.25, samples=320)
    spliced = clean.rstrip("\n") + INTERLEAVED_TAIL

    parsed = et.parse_status_line(spliced)
    assert parsed is not None
    assert parsed["global_step"] == 97
    assert parsed["status"] == et.parse_status_line(clean)["status"]
    assert parsed["repaired"] is False
    assert (parsed["epoch"], parsed["iter"]) == (1, 97)


def test_a_line_nothing_was_spliced_onto_reads_exactly_as_before():
    clean = step_line(98, response=600.0, total=1000.0, samples=256)
    parsed = et.parse_status_line(clean)
    assert parsed["status"] == {"response_length": 600.0, "total_length": 1000.0,
                                "reward/count": 256}
    assert (parsed["global_step"], parsed["iters_in_epoch"]) == (98, 312)


def test_a_line_whose_dictionary_never_closes_is_still_refused(tmp_path):
    """A first `{` with no matching `}` is a damaged line, not a half-readable one."""
    damaged = ("Global step 2 | epoch 1/2 | iter 2/312: {'response_length': 500.0, 'total_"
               + INTERLEAVED_TAIL)
    assert et.parse_status_line(damaged) is None

    run = os.path.join(str(tmp_path), "train_s0")
    write_log(run, [step_line(1, response=500.0, total=900.0, samples=300), damaged])
    document = et.read_run(run)
    assert document["log"]["n_lines_with_a_step"] == 2
    assert document["log"]["n_unparsed"] == 1
    assert document["totals"]["n_steps"] == 1
    assert document["totals"]["n_complete_steps"] == 1


# -----------------------------------------------------------------------------
# the account of a whole run
# -----------------------------------------------------------------------------


def test_the_totals_are_the_products_of_the_two_means_and_the_sample_count(tmp_path):
    run = complete_run(os.path.join(str(tmp_path), "train_s0"))
    document = et.read_run(run)

    assert document["schema"] == et.SCHEMA
    assert [row["global_step"] for row in document["steps"]] == [1, 2]

    first, second = document["steps"]
    assert first["generated_tokens"] == pytest.approx(500.0 * 300)
    assert first["total_tokens"] == pytest.approx(900.0 * 300)
    assert first["prompt_tokens"] == pytest.approx(400.0 * 300)
    assert second["generated_tokens"] == pytest.approx(600.0 * 320)

    totals = document["totals"]
    assert totals["n_steps"] == 2 and totals["n_complete_steps"] == 2
    assert totals["samples"] == 620
    assert totals["generated_tokens"] == pytest.approx(500.0 * 300 + 600.0 * 320)
    assert totals["total_tokens"] == pytest.approx(900.0 * 300 + 1000.0 * 320)
    assert totals["prompt_tokens"] == pytest.approx(totals["total_tokens"] - totals["generated_tokens"])
    assert totals["total_tokens_M"] == pytest.approx(totals["total_tokens"] / 1e6)


def test_the_launch_arguments_are_narrowed_and_the_ledger_gives_the_wall_clock(tmp_path):
    run = complete_run(os.path.join(str(tmp_path), "train_s0"))
    document = et.read_run(run)

    assert document["config"]["marl_algorithm"] == "c3"
    assert document["config"]["n_samples_per_prompt"] == 8
    assert "unrelated" not in document["config"]

    ledger = document["ledger"]
    assert ledger["rows"] == 2 and ledger["global_steps"] == 2
    assert ledger["total_eval_calls"] == 2 * 128 * 8
    assert ledger["n_samples_per_prompt"] == [8]
    assert ledger["wall_clock_h"] == pytest.approx(1.0)
    # No hole, so the segmented sum and the first-to-last distance agree.
    assert ledger["n_segments"] == 1 and ledger["n_gaps_excluded"] == 0
    assert ledger["gap_hours_excluded"] == pytest.approx(0.0)
    assert ledger["first_last_span_h"] == pytest.approx(1.0)


# -----------------------------------------------------------------------------
# the wall clock of a run that was stopped and resumed
# -----------------------------------------------------------------------------


#: The shape of the MAGRPO arm of this round: a stretch of steps at a steady
#: pace, the container reclaimed, then the rest of the steps on another one.
STEADY_S = 600
HOLE_S = 24 * 3600 + 1_000


def resumed_ledger_rows(before=5, after=4, *, hole=HOLE_S, start=1_000_000, pace=STEADY_S):
    """Rows of a ledger whose run stopped after `before` steps and resumed."""
    rows = []
    stamp = start
    for step in range(1, before + 1):
        rows.append(ledger_row(step, ts=stamp))
        stamp += pace
    stamp += hole - pace
    for step in range(before + 1, before + after + 1):
        rows.append(ledger_row(step, ts=stamp))
        stamp += pace
    return rows


def test_a_ledger_with_a_day_long_hole_bills_only_its_two_training_segments(tmp_path):
    """The hole is the container swap, and the run was not training inside it."""
    run = os.path.join(str(tmp_path), "train_s0")
    write_ledger(run, resumed_ledger_rows())
    ledger = et.read_run(run)["ledger"]

    assert ledger["rows"] == 9
    assert ledger["median_row_interval_s"] == pytest.approx(float(STEADY_S))
    assert ledger["gap_threshold_s"] == pytest.approx(10.0 * STEADY_S)
    assert ledger["n_segments"] == 2 and ledger["n_gaps_excluded"] == 1
    assert ledger["gap_hours_excluded"] == pytest.approx(HOLE_S / 3600.0)
    assert [segment["n_rows"] for segment in ledger["segments"]] == [5, 4]

    # Four steps of 600 s, then three of 600 s: the hole is in neither span.
    assert ledger["wall_clock_h"] == pytest.approx((4 * STEADY_S + 3 * STEADY_S) / 3600.0)
    assert ledger["first_last_span_h"] == pytest.approx(
        (4 * STEADY_S + HOLE_S + 3 * STEADY_S) / 3600.0)
    assert ledger["wall_clock_h"] < ledger["first_last_span_h"]


def test_two_holes_leave_three_segments_and_both_are_named(tmp_path):
    run = os.path.join(str(tmp_path), "train_s0")
    rows = [ledger_row(1, ts=1_000_000), ledger_row(2, ts=1_000_600),
            ledger_row(3, ts=1_090_000), ledger_row(4, ts=1_090_600),
            ledger_row(5, ts=1_190_000), ledger_row(6, ts=1_190_600)]
    write_ledger(run, rows)
    ledger = et.read_run(run)["ledger"]

    assert ledger["n_segments"] == 3 and ledger["n_gaps_excluded"] == 2
    assert ledger["wall_clock_h"] == pytest.approx(3 * 600 / 3600.0)
    assert ledger["gap_hours_excluded"] == pytest.approx((89_400 + 99_400) / 3600.0)


def test_a_gap_of_exactly_ten_medians_is_still_training(tmp_path):
    """The rule cuts on longer than ten times, so the boundary stays inside."""
    run = os.path.join(str(tmp_path), "train_s0")
    write_ledger(run, [ledger_row(1, ts=1_000_000), ledger_row(2, ts=1_000_600),
                       ledger_row(3, ts=1_001_200), ledger_row(4, ts=1_007_200)])
    ledger = et.read_run(run)["ledger"]

    assert ledger["median_row_interval_s"] == pytest.approx(600.0)
    assert ledger["n_segments"] == 1 and ledger["n_gaps_excluded"] == 0
    assert ledger["wall_clock_h"] == pytest.approx(7_200 / 3600.0)


def test_rows_written_out_of_order_are_cut_in_time_order(tmp_path):
    """A resumed run's ledger is two files appended, so the order on disk is not the order in time."""
    run = os.path.join(str(tmp_path), "train_s0")
    rows = resumed_ledger_rows()
    write_ledger(run, list(reversed(rows)))
    ledger = et.read_run(run)["ledger"]

    assert ledger["n_segments"] == 2 and ledger["n_gaps_excluded"] == 1
    assert ledger["wall_clock_h"] == pytest.approx(7 * STEADY_S / 3600.0)
    assert all(segment["span_h"] > 0 for segment in ledger["segments"])


def test_a_ledger_whose_rows_share_one_second_is_left_in_one_piece(tmp_path):
    """Ten times a median of zero would make every interval a hole and the sum zero."""
    run = os.path.join(str(tmp_path), "train_s0")
    write_ledger(run, [ledger_row(step, ts=1_000_000) for step in range(1, 4)]
                 + [ledger_row(4, ts=1_003_600)])
    ledger = et.read_run(run)["ledger"]

    assert ledger["median_row_interval_s"] == pytest.approx(0.0)
    assert ledger["gap_threshold_s"] is None
    assert ledger["n_segments"] == 1 and ledger["n_gaps_excluded"] == 0
    assert ledger["wall_clock_h"] == pytest.approx(1.0)


def test_a_one_row_ledger_spans_nothing_and_is_one_segment(tmp_path):
    run = os.path.join(str(tmp_path), "train_s0")
    write_ledger(run, [ledger_row(1, ts=1_000_000)])
    ledger = et.read_run(run)["ledger"]

    assert ledger["rows"] == 1
    assert ledger["median_row_interval_s"] is None
    assert ledger["n_segments"] == 1 and ledger["n_gaps_excluded"] == 0
    assert ledger["wall_clock_h"] == pytest.approx(0.0)


def test_a_ledger_with_no_timestamp_at_all_reports_none_rather_than_zero(tmp_path):
    run = os.path.join(str(tmp_path), "train_s0")
    write_ledger(run, [{"global_step": 1, "total_eval_calls": 16},
                       {"global_step": 2, "total_eval_calls": 16}])
    ledger = et.read_run(run)["ledger"]

    assert ledger["rows"] == 2
    assert ledger["wall_clock_h"] is None and ledger["first_last_span_h"] is None
    assert ledger["n_segments"] == 0 and ledger["segments"] == []


def test_a_step_written_twice_keeps_its_last_line(tmp_path):
    """A resumed run writes the step it restarts from again."""
    run = os.path.join(str(tmp_path), "train_s0")
    write_log(run, [step_line(1, response=500.0, total=900.0, samples=300),
                    step_line(1, response=700.0, total=1100.0, samples=300)])

    document = et.read_run(run)
    assert document["log"]["n_duplicate_steps"] == 1
    assert len(document["steps"]) == 1
    assert document["steps"][0]["response_length"] == 700.0


def test_a_line_that_does_not_parse_is_counted_and_not_dropped_silently(tmp_path):
    run = os.path.join(str(tmp_path), "train_s0")
    write_log(run, [step_line(1), "Global step 2 | epoch 1/2 | iter 2/312: {broken\n"])

    document = et.read_run(run)
    assert document["log"]["n_lines_with_a_step"] == 2
    assert document["log"]["n_unparsed"] == 1
    assert document["totals"]["n_complete_steps"] == 1


# -----------------------------------------------------------------------------
# the refusal to guess
# -----------------------------------------------------------------------------


def test_a_step_missing_the_sample_count_is_left_out_of_the_totals(tmp_path):
    """The January script fell back to 2048 samples here. Nothing falls back."""
    run = os.path.join(str(tmp_path), "train_s0")
    write_log(run, [step_line(1, response=500.0, total=900.0, samples=300),
                    step_line(2, response=600.0, total=1000.0, samples=None)])

    document = et.read_run(run)
    second = document["steps"][1]
    assert second["samples"] is None
    assert second["generated_tokens"] is None
    assert second["missing"] == ["reward/count"]

    totals = document["totals"]
    assert totals["n_steps"] == 2 and totals["n_complete_steps"] == 1
    assert totals["incomplete_steps"] == [2]
    assert totals["generated_tokens"] == pytest.approx(500.0 * 300)
    assert [gap["why"] for gap in document["not_recorded"]
            if gap["quantity"] == "the token account of 1 of 2 steps"] == \
        ["those step lines carry no reward/count"]


def test_a_step_missing_a_length_is_left_out_the_same_way(tmp_path):
    """The January script fell back to 512 tokens here."""
    run = os.path.join(str(tmp_path), "train_s0")
    write_log(run, [step_line(1, response=None, total=900.0, samples=300)])

    document = et.read_run(run)
    assert document["steps"][0]["missing"] == ["response_length"]
    assert document["totals"]["generated_tokens"] is None
    assert document["totals"]["n_complete_steps"] == 0


def test_a_per_role_log_surfaces_its_role_lengths_instead_of_inventing_a_total(tmp_path):
    """`policy_sharing_mode per_role` prefixes every actor statistic with the role.

    There is no per-role sample count in the same line, so the step cannot be
    closed; what the line does carry is reported rather than dropped.
    """
    run = os.path.join(str(tmp_path), "train_s0")
    write_log(run, [step_line(1, response=None, total=None, samples=320,
                              extra={"reasoner/response_length": 300.0,
                                     "actor/response_length": 400.0,
                                     "reasoner/total_length": 700.0})])

    row = et.read_run(run)["steps"][0]
    assert row["response_length"] is None
    assert row["response_length_by_role"] == {"reasoner": 300.0, "actor": 400.0}
    assert row["total_length_by_role"] == {"reasoner": 700.0}
    assert sorted(row["missing"]) == ["response_length", "total_length"]


def test_the_quantities_no_file_records_are_named_on_every_run(tmp_path):
    run = complete_run(os.path.join(str(tmp_path), "train_s0"))
    named = {gap["quantity"] for gap in et.read_run(run)["not_recorded"]}
    assert "tokens_split_by_position_in_the_rollout_tree" in named
    assert "tokens_of_the_critic_warmup" in named
    assert "evaluation_tokens" in named
    assert "prompt_tokens_as_a_measurement" in named


def test_a_run_without_a_log_reports_that_and_computes_nothing(tmp_path):
    run = os.path.join(str(tmp_path), "train_s0")
    os.makedirs(run, exist_ok=True)

    document = et.read_run(run)
    assert document["log"]["present"] is False
    assert document["totals"]["total_tokens"] is None
    assert document["totals"]["n_steps"] == 0
    reasons = [gap for gap in document["not_recorded"]
               if gap["quantity"] == "every per-step length"]
    assert len(reasons) == 1 and et.LOG_NAME in reasons[0]["why"]

    ledger_gap = [gap for gap in document["not_recorded"]
                  if gap["quantity"] == "wall_clock_h and total_eval_calls"]
    assert len(ledger_gap) == 1


def test_a_log_with_no_step_line_at_all_is_its_own_reason(tmp_path):
    run = os.path.join(str(tmp_path), "train_s0")
    write_log(run, [])
    reasons = [gap["why"] for gap in et.read_run(run)["not_recorded"]
               if gap["quantity"] == "every per-step length"]
    assert len(reasons) == 1 and "no `Global step` status line" in reasons[0]


# -----------------------------------------------------------------------------
# the critic warmup ledger
# -----------------------------------------------------------------------------


def test_a_run_without_a_warmup_ledger_reports_nothing_rather_than_zero(tmp_path):
    run = complete_run(os.path.join(str(tmp_path), "train_s0"))
    warmup = et.read_run(run)["warmup"]

    assert warmup["present"] is False
    assert warmup["warmup_tokens_M"] is None
    assert warmup["generated_tokens_M"] is None
    assert warmup["note"].startswith("no warmup ledger")
    assert et.WARMUP_LEDGER_NAME in warmup["note"]


def test_a_missing_warmup_ledger_is_named_among_the_quantities_no_file_records(tmp_path):
    run = complete_run(os.path.join(str(tmp_path), "train_s0"))
    gaps = [gap for gap in et.read_run(run)["not_recorded"]
            if gap["quantity"] == "tokens_of_the_critic_warmup"]
    assert len(gaps) == 1
    assert et.WARMUP_LEDGER_NAME in gaps[0]["would_need"]


def test_the_warmup_rollouts_are_reported_apart_from_the_training_account(tmp_path):
    """The reason for the second file: a warmup rollout is not a step and not a training token."""
    run = complete_run(os.path.join(str(tmp_path), "train_s0"))
    before = et.read_run(run)

    write_warmup_ledger(run, [warmup_row(1, generated=500_000, total=900_000, ts=1_000_000),
                              warmup_row(2, generated=400_000, total=800_000, ts=1_001_800)])
    after = et.read_run(run)

    assert after["totals"] == before["totals"]
    assert after["ledger"]["rows"] == before["ledger"]["rows"] == 2
    assert after["ledger"]["global_steps"] == 2

    warmup = after["warmup"]
    assert warmup["present"] is True
    assert warmup["rows"] == 2
    assert warmup["warmup_tokens_M"] == pytest.approx(1.7)
    assert warmup["generated_tokens_M"] == pytest.approx(0.9)
    assert warmup["n_nodes"] == 2048
    assert warmup["wall_clock_h"] == pytest.approx(0.5)
    assert warmup["marl_algorithm"] == ["mappo"]


def test_a_recorded_warmup_is_no_longer_a_quantity_no_file_records(tmp_path):
    run = complete_run(os.path.join(str(tmp_path), "train_s0"))
    write_warmup_ledger(run, [warmup_row(1)])
    quantities = [gap["quantity"] for gap in et.read_run(run)["not_recorded"]]
    assert "tokens_of_the_critic_warmup" not in quantities


def test_a_warmup_row_without_a_token_field_is_counted_and_left_out_of_the_sum(tmp_path):
    run = complete_run(os.path.join(str(tmp_path), "train_s0"))
    write_warmup_ledger(run, [warmup_row(1, generated=500_000, total=900_000),
                              warmup_row(2, generated=None, total=None, nodes=None)])
    warmup = et.read_run(run)["warmup"]

    assert warmup["rows"] == 2
    assert warmup["rows_without_tokens"] == 1
    assert warmup["warmup_tokens_M"] == pytest.approx(0.9)
    assert "left out of the sums" in warmup["note"]


def test_a_warmup_ledger_that_carries_no_length_leaves_the_quantity_unrecorded(tmp_path):
    """A row of the shape the ledger wrote before the token sums were added to it."""
    run = complete_run(os.path.join(str(tmp_path), "train_s0"))
    write_warmup_ledger(run, [warmup_row(1, generated=None, total=None, nodes=None)])
    document = et.read_run(run)

    assert document["warmup"]["warmup_tokens_M"] is None
    assert document["warmup"]["rows"] == 1
    assert "none of which carries a token field" in document["warmup"]["note"]
    assert any(gap["quantity"] == "tokens_of_the_critic_warmup"
               for gap in document["not_recorded"])


def test_the_warmup_quantities_name_the_file_they_came_from(tmp_path):
    run = complete_run(os.path.join(str(tmp_path), "train_s0"))
    provenance = et.read_run(run)["provenance"]
    for name in ("warmup.warmup_tokens_M", "warmup.generated_tokens_M"):
        assert set(provenance[name]) == {"file", "field", "how"}
        assert provenance[name]["file"] == et.WARMUP_LEDGER_NAME


def test_whether_a_run_had_a_warmup_stage_at_all_is_carried_beside_the_account(tmp_path):
    run = os.path.join(str(tmp_path), "train_s0")
    complete_run(run)
    write_config(run, {"run_id": "a2_4b_mappo_s0", "marl_algorithm": "mappo",
                       "critic_warmup_steps": 64})
    assert et.read_run(run)["config"]["critic_warmup_steps"] == 64


# -----------------------------------------------------------------------------
# the document and the command line
# -----------------------------------------------------------------------------


def test_every_quantity_of_the_document_names_the_file_it_came_from(tmp_path):
    run = complete_run(os.path.join(str(tmp_path), "train_s0"))
    provenance = et.read_run(run)["provenance"]
    for name in ("steps[].response_length", "steps[].samples", "steps[].generated_tokens",
                 "ledger.total_eval_calls", "ledger.wall_clock_h"):
        assert set(provenance[name]) == {"file", "field", "how"}
        assert provenance[name]["file"] in (et.LOG_NAME, et.LEDGER_NAME, et.CONFIG_NAME)


def test_the_per_step_rows_can_be_left_out(tmp_path):
    run = complete_run(os.path.join(str(tmp_path), "train_s0"))
    document = et.read_run(run, include_steps=False)
    assert "steps" not in document
    assert "steps_note" in document
    assert document["totals"]["n_complete_steps"] == 2


def test_the_command_line_writes_the_document_and_refuses_a_missing_directory(tmp_path, capsys):
    run = complete_run(os.path.join(str(tmp_path), "train_s0"))
    out_path = os.path.join(str(tmp_path), "tokens.json")

    assert et.main(["--run", run, "--out", out_path]) == 0
    with open(out_path, "r", encoding="utf-8") as fh:
        document = json.load(fh)
    assert document["totals"]["n_complete_steps"] == 2
    assert len(document["steps"]) == 2

    assert et.main(["--run", run, "--out", out_path, "--no_steps"]) == 0
    with open(out_path, "r", encoding="utf-8") as fh:
        assert "steps" not in json.load(fh)

    assert et.main(["--run", os.path.join(str(tmp_path), "nowhere"), "--out", out_path]) == 2
    assert "is not a directory" in capsys.readouterr().err


def test_it_runs_as_a_module(tmp_path):
    run = complete_run(os.path.join(str(tmp_path), "train_s0"))
    out_path = os.path.join(str(tmp_path), "tokens.json")
    done = subprocess.run(
        [sys.executable, "-m", "c3.analysis.rebuild.e2_tokens", "--run", run, "--out", out_path],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stderr
    with open(out_path, "r", encoding="utf-8") as fh:
        assert json.load(fh)["schema"] == et.SCHEMA
