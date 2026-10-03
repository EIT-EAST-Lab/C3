# tests/unit/test_rebuild_aggregate_e2.py
"""The E2 aggregation and the two replication units it has to tell apart.

Every fixture is synthetic and hand computable; nothing here reads real data.
The interim tree is the one the platform is writing now (one training seed
evaluated five times, preregistration revision 20) and the full tree is the one
that replaces it (five training seeds, each evaluated five times). The two are
different measurements wearing the same shape, so what is pinned here is mostly
that the second is not silently reported as the first.

The last sections cover the two blocks the summary carries beside its keys: the
training token account of each run and its monitoring curve. The two readers
themselves are pinned in `test_e2_tokens.py` and `test_e2_curves.py`; what is
pinned here is that they are collected per run, that the curve reaches no
manifest key at all, and that the token account reaches exactly two of them
(`compute.<method>.train_tokens_M` and `.wall_clock_h`) and fills them from
the files of the run they belong to.

    python -m pytest tests/unit/test_rebuild_aggregate_e2.py -q
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys

import pytest
from scipy import stats

from c3.analysis.rebuild import aggregate_e2 as a2

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The keys of the two-agent Qwen3-4B arm, spelled as manifest_skeleton.py spells
# them. Only the keys the fixtures can reach are listed; anything else the
# aggregation tries would be refused, which is itself one of the tests.
# The main table of preregistration revision 18 and the two appendix controls.
# The appendix pair carries a mean, a standard deviation and a plain difference
# and no test at all, which is one of the things this file pins.
BENCHES = ("math500", "minerva", "amc23", "aime", "avg4")
APPX_BENCHES = ("gsm8k", "cmath")
METHODS = ("ppo1a", "mappo", "magrpo", "c3")

UNITS = {"mean": "%", "std": "%", "diff": "pp", "ci_lo": "pp", "ci_hi": "pp",
         "p": "p", "p_holm": "p", "name": "name", "arch_gain": "pp", "credit_gain": "pp"}


def manifest_keys(arm="a2_4b"):
    keys = {}

    def add(key, unit):
        keys[key] = {"value": None, "fmt": "{}", "unit": unit, "verdict": None,
                     "source": None, "n": None, "experiment": "E2", "note": ""}

    for bench in BENCHES:
        for method in METHODS:
            add("E2.%s.%s.%s.mean" % (arm, bench, method), "%")
            add("E2.%s.%s.%s.std" % (arm, bench, method), "%")
        for pair in ("c3_minus_best", "c3_minus_mappo"):
            add("E2.%s.%s.%s.diff" % (arm, bench, pair), "pp")
            add("E2.%s.%s.%s.ci_lo" % (arm, bench, pair), "pp")
            add("E2.%s.%s.%s.ci_hi" % (arm, bench, pair), "pp")
            add("E2.%s.%s.%s.p" % (arm, bench, pair), "p")
            add("E2.%s.%s.%s.p_holm" % (arm, bench, pair), "p")
    for bench in APPX_BENCHES:
        for method in METHODS:
            add("E2.%s.%s.%s.mean" % (arm, bench, method), "%")
            add("E2.%s.%s.%s.std" % (arm, bench, method), "%")
        add("E2.%s.%s.c3_minus_best.diff" % (arm, bench), "pp")
    add("E2.%s.strongest_baseline.name" % arm, "name")
    add("E2.%s.math500.best.mean" % arm, "%")
    add("E2.%s.math500.best.std" % arm, "%")
    add("E2.a2_4b.math500.arch_gain", "pp")
    add("E2.a2_4b.math500.credit_gain", "pp")
    # The verdict of the arm: a manifest key, and one no script may fill.
    keys["E2.%s.math500.paired" % arm] = {
        "value": None, "fmt": "{}", "unit": "verdict", "verdict": None,
        "source": None, "n": None, "experiment": "E2", "note": ""}
    return keys


def write_manifest(tmp_path, arm="a2_4b"):
    path = os.path.join(str(tmp_path), "manifest.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest_keys(arm), fh)
    return path


#: How many problems each suite of a record carries. `AIME` is the merged pair
#: of years, and the aggregation checks that count rather than trusting the name.
N_QUESTIONS = {"MATH500": 500, "Minerva-Math": 272, "AMC23": 40, "AIME": 60,
               "GSM8K-test": 1319, "CMATH-test": 1098}


def record(accuracies, *, schema="c3.final_eval.record/1", n_questions=None):
    """One final_eval_record.json in the shape scripts/70_rebuild/final_eval.py writes."""
    counts = dict(N_QUESTIONS)
    counts.update(n_questions or {})
    suites = {}
    for suite, value in accuracies.items():
        suites[suite] = {
            "k": 4,
            "n_questions": counts.get(suite, 500),
            "accuracy": value,
            "wilson95": {"low": 0.0, "high": 1.0},
            "boxed_rate": 1.0,
            "per_question": [],
        }
    return {"schema": schema, "policy": "ckpt", "method": "C3", "seed": 0,
            "suites": suites, "problems": [], "ok": True}


def write_run(root, arm, method, train_seed, eval_seed, accuracies, **kwargs):
    where = os.path.join(root, arm, method, "train_s%d" % train_seed, "eval_s%d" % eval_seed)
    os.makedirs(where, exist_ok=True)
    path = os.path.join(where, "final_eval_record.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(record(accuracies, **kwargs), fh)
    return path


def suite_values(math500, minerva=0.30, amc23=0.50, aime=0.10, gsm8k=0.92, cmath=0.90):
    """The six suites of one record, the four of the main table first."""
    return {"MATH500": math500, "Minerva-Math": minerva, "AMC23": amc23, "AIME": aime,
            "GSM8K-test": gsm8k, "CMATH-test": cmath}


# -----------------------------------------------------------------------------
# the two statistics helpers, on hand computed numbers
# -----------------------------------------------------------------------------


def test_holm_is_the_step_down_form_of_the_local_script():
    """Sorted ascending, multiplied by the tests still standing, carried forward."""
    assert a2.holm([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    # A later test may not come out below an earlier one: 0.04 x 1 is lifted to 0.06.
    assert a2.holm([0.2]) == pytest.approx([0.2])
    assert a2.holm([0.5, 0.6]) == pytest.approx([1.0, 1.0])


def test_paired_stats_matches_the_interval_the_local_script_computes():
    left = [80.0, 82.0, 81.0, 84.0, 83.0]
    right = [78.0, 79.0, 80.0, 79.0, 80.0]
    out = a2.paired_stats(left, right)

    # differences 2, 3, 1, 5, 3: mean 2.8, sample sd sqrt(2.2)
    assert out["n"] == 5
    assert out["diff"] == pytest.approx(2.8)
    half = 2.7764451051977987 * math.sqrt(2.2) / math.sqrt(5)
    assert half == pytest.approx(1.8417, abs=5e-4)
    assert out["ci_lo"] == pytest.approx(2.8 - half)
    assert out["ci_hi"] == pytest.approx(2.8 + half)
    assert out["p"] == pytest.approx(float(stats.ttest_rel(left, right).pvalue))


def test_paired_stats_leaves_the_interval_undefined_where_it_is_undefined():
    """One pair has no spread to speak of, and a constant difference has none either."""
    assert a2.paired_stats([1.0], [0.0])["ci_lo"] is None
    assert a2.paired_stats([1.0, 2.0], [0.0, 1.0])["p"] is None
    assert a2.paired_stats([], [])["diff"] is None


# -----------------------------------------------------------------------------
# interim: one training seed, five evaluation runs
# -----------------------------------------------------------------------------


def interim_tree(root):
    """c3 and magrpo, one training seed each, five evaluation runs each."""
    c3 = [0.80, 0.81, 0.82, 0.83, 0.84]
    magrpo = [0.70, 0.71, 0.72, 0.73, 0.74]
    for eval_seed in range(5):
        write_run(root, "a2_4b", "c3", 0, eval_seed, suite_values(c3[eval_seed]))
        write_run(root, "a2_4b", "magrpo", 0, eval_seed, suite_values(magrpo[eval_seed]))
    return c3, magrpo


def test_interim_reports_the_evaluation_runs_and_refuses_to_pair_them(tmp_path, capsys):
    root = os.path.join(str(tmp_path), "E2")
    interim_tree(root)
    manifest = json.load(open(write_manifest(tmp_path), encoding="utf-8"))

    builder = a2.build_e2_summary(root, manifest)
    doc = builder.payload()
    keys = doc["keys"]

    assert doc["replication"]["mode"] == "interim"
    assert doc["replication"]["by_arm"] == {"a2_4b": "interim"}

    # 80 to 84 in percent: mean 82, sample sd sqrt(2.5).
    assert keys["E2.a2_4b.math500.c3.mean"]["value"] == pytest.approx(82.0)
    assert keys["E2.a2_4b.math500.c3.std"]["value"] == pytest.approx(math.sqrt(2.5))
    assert keys["E2.a2_4b.math500.c3.mean"]["n"] == 5
    assert a2.INTERIM_NOTE in keys["E2.a2_4b.math500.c3.mean"]["note"]
    assert "evaluation runs of one training seed" in keys["E2.a2_4b.math500.c3.std"]["note"]

    # The difference is reported, the paired reading is not: there is one seed.
    assert keys["E2.a2_4b.math500.c3_minus_best.diff"]["value"] == pytest.approx(10.0)
    assert keys["E2.a2_4b.math500.c3_minus_best.diff"]["n"] == 1
    for tail in ("ci_lo", "ci_hi", "p", "p_holm"):
        assert "E2.a2_4b.math500.c3_minus_best.%s" % tail not in keys

    # The strongest baseline is the only baseline here, and it is named the way
    # the manifest names methods.
    assert keys["E2.a2_4b.strongest_baseline.name"]["value"] == "MAGRPO"
    assert keys["E2.a2_4b.math500.best.mean"]["value"] == pytest.approx(72.0)
    assert keys["E2.a2_4b.math500.best.std"]["value"] == pytest.approx(math.sqrt(2.5))

    # No verdict key is written, and the shared mechanism is what would refuse one:
    # nothing here even tries, so the guard is exercised directly.
    assert "E2.a2_4b.math500.paired" not in keys
    assert not builder.add("E2.a2_4b.math500.paired", 1.0, n=1, source=[])
    assert "E2.a2_4b.math500.paired: verdict key" in capsys.readouterr().err

    # Sources are the record files that were averaged, repository relative.
    sources = keys["E2.a2_4b.math500.c3.mean"]["source"]
    assert len(sources) == 5
    assert all(item.endswith("final_eval_record.json") for item in sources)

    # The two appendix controls are reported without a test, which is the
    # manifest's own key set rather than a second rule kept here.
    for bench in ("gsm8k", "cmath"):
        assert "E2.a2_4b.%s.c3.mean" % bench in keys
        assert "E2.a2_4b.%s.c3_minus_best.diff" % bench in keys
        assert "E2.a2_4b.%s.c3_minus_mappo.diff" % bench not in keys


def test_the_manifest_decides_which_keys_exist(tmp_path, capsys):
    """A key the manifest does not define is not attempted and is counted."""
    root = os.path.join(str(tmp_path), "E2")
    interim_tree(root)
    manifest = manifest_keys()
    del manifest["E2.a2_4b.math500.c3.std"]

    doc = a2.build_e2_summary(root, manifest).payload()
    assert "E2.a2_4b.math500.c3.mean" in doc["keys"]
    assert "E2.a2_4b.math500.c3.std" not in doc["keys"]
    assert "E2.a2_4b.math500.c3.std" in doc["extras"]["keys_the_manifest_does_not_define"]
    assert "key(s) the manifest does not define" in capsys.readouterr().err


def test_avg4_is_the_unweighted_mean_formed_inside_each_run(tmp_path):
    root = os.path.join(str(tmp_path), "E2")
    write_run(root, "a2_4b", "c3", 0, 0,
              {"MATH500": 0.60, "Minerva-Math": 0.90, "AMC23": 0.30, "AIME": 0.20})
    write_run(root, "a2_4b", "c3", 0, 1,
              {"MATH500": 0.30, "Minerva-Math": 0.60, "AMC23": 0.90, "AIME": 0.20})
    manifest = json.load(open(write_manifest(tmp_path), encoding="utf-8"))

    keys = a2.build_e2_summary(root, manifest).payload()["keys"]
    # Both runs average 50 over the four benchmarks, so the mean is 50 and the
    # spread is zero: the four are averaged inside a run, not across runs. The
    # mean is unweighted, so AIME's 60 problems count as much as MATH500's 500.
    assert keys["E2.a2_4b.avg4.c3.mean"]["value"] == pytest.approx(50.0)
    assert keys["E2.a2_4b.avg4.c3.std"]["value"] == pytest.approx(0.0)
    assert "unweighted" in keys["E2.a2_4b.avg4.c3.mean"]["note"]


def test_the_aime_key_is_the_merged_pair_and_a_short_merge_is_refused(tmp_path, capsys):
    """`final_eval.py` merges whichever years ran, so 30 problems can arrive as AIME.

    That is a different benchmark under the same name, so it is dropped with a
    line on stderr and recorded in extras rather than reported as AIME.
    """
    root = os.path.join(str(tmp_path), "E2")
    for eval_seed in range(2):
        write_run(root, "a2_4b", "c3", 0, eval_seed, suite_values(0.80, aime=0.10))
    # One run where only 2025 was evaluated, so the merge covers 30 problems.
    write_run(root, "a2_4b", "c3", 0, 2, suite_values(0.80, aime=0.99),
              n_questions={"AIME": 30})
    manifest = json.load(open(write_manifest(tmp_path), encoding="utf-8"))

    doc = a2.build_e2_summary(root, manifest).payload()
    keys = doc["keys"]
    # Three runs of MATH500, two of AIME: the short merge took nothing else with it.
    assert keys["E2.a2_4b.math500.c3.mean"]["n"] == 3
    assert keys["E2.a2_4b.aime.c3.mean"]["value"] == pytest.approx(10.0)
    assert keys["E2.a2_4b.aime.c3.mean"]["n"] == 2
    # avg4 needs all four, so the short run contributes no avg4 either.
    assert keys["E2.a2_4b.avg4.c3.mean"]["n"] == 2

    partial = doc["extras"]["aime_partial_merges"]
    assert len(partial) == 1 and partial[0]["n_questions"] == 30
    assert "not the 60 of the merged" in capsys.readouterr().err


# -----------------------------------------------------------------------------
# full: five training seeds, each evaluated five times
# -----------------------------------------------------------------------------


#: Percent accuracy of each training seed, per benchmark. Every benchmark moves
#: with the seed, so all four have a spread and the Holm family really has four
#: members; math500 is the one the interval below is hand computed for.
C3_BY_SEED = {"math500": [80.0, 82.0, 81.0, 84.0, 83.0],
              "minerva": [30.0, 32.0, 31.0, 33.0, 29.0],
              "amc23": [50.0, 52.0, 51.0, 55.0, 53.0],
              "aime": [10.0, 12.0, 11.0, 13.0, 9.0],
              "gsm8k": [92.0, 93.0, 91.0, 94.0, 92.0],
              "cmath": [90.0, 91.0, 92.0, 90.0, 93.0]}
MAGRPO_BY_SEED = {"math500": [78.0, 79.0, 80.0, 79.0, 80.0],
                  "minerva": [28.0, 30.0, 29.0, 28.0, 31.0],
                  "amc23": [48.0, 50.0, 49.0, 48.0, 51.0],
                  "aime": [8.0, 9.0, 10.0, 8.0, 9.0],
                  "gsm8k": [90.0, 90.0, 89.0, 91.0, 90.0],
                  "cmath": [88.0, 90.0, 89.0, 88.0, 91.0]}


def full_tree(root):
    """Five training seeds each evaluated five times.

    The five evaluation runs of a seed straddle the seed's value by the same
    amounts, so the seed's mean is exactly the number in the tables above and
    the arithmetic of the paired reading stays hand checkable.
    """
    offsets = [-0.02, -0.01, 0.0, 0.01, 0.02]
    for seed in range(5):
        for eval_seed, offset in enumerate(offsets):
            for method, table in (("c3", C3_BY_SEED), ("magrpo", MAGRPO_BY_SEED)):
                write_run(root, "a2_4b", method, seed, eval_seed,
                          suite_values(table["math500"][seed] / 100.0 + offset,
                                       minerva=table["minerva"][seed] / 100.0 + offset,
                                       amc23=table["amc23"][seed] / 100.0 + offset,
                                       aime=table["aime"][seed] / 100.0 + offset,
                                       gsm8k=table["gsm8k"][seed] / 100.0 + offset,
                                       cmath=table["cmath"][seed] / 100.0 + offset))


def test_full_pairs_over_training_seeds_and_corrects_across_the_four(tmp_path):
    root = os.path.join(str(tmp_path), "E2")
    full_tree(root)
    manifest = json.load(open(write_manifest(tmp_path), encoding="utf-8"))

    doc = a2.build_e2_summary(root, manifest).payload()
    keys = doc["keys"]

    assert doc["replication"]["mode"] == "full"
    assert doc["replication"]["by_arm"] == {"a2_4b": "full"}

    # Mean over the training seeds of the per-seed mean, std over the seeds.
    assert keys["E2.a2_4b.math500.c3.mean"]["value"] == pytest.approx(82.0)
    assert keys["E2.a2_4b.math500.c3.std"]["value"] == pytest.approx(
        float(stats.tstd(C3_BY_SEED["math500"])))
    assert keys["E2.a2_4b.math500.c3.mean"]["n"] == 5
    assert a2.INTERIM_NOTE not in keys["E2.a2_4b.math500.c3.mean"].get("note", "")
    assert "training seeds" in keys["E2.a2_4b.math500.c3.std"]["note"]

    # The paired reading, against the hand computed interval of the differences
    # 2, 3, 1, 5, 3: mean 2.8, half width t(0.975, 4) x sqrt(2.2 / 5).
    entry = keys["E2.a2_4b.math500.c3_minus_best.diff"]
    assert entry["value"] == pytest.approx(2.8)
    assert entry["n"] == 5
    half = 2.7764451051977987 * math.sqrt(2.2) / math.sqrt(5)
    assert keys["E2.a2_4b.math500.c3_minus_best.ci_lo"]["value"] == pytest.approx(2.8 - half)
    assert keys["E2.a2_4b.math500.c3_minus_best.ci_hi"]["value"] == pytest.approx(2.8 + half)

    raw = float(stats.ttest_rel(C3_BY_SEED["math500"], MAGRPO_BY_SEED["math500"]).pvalue)
    assert keys["E2.a2_4b.math500.c3_minus_best.p"]["value"] == a2.format_p(raw)
    # Holm runs over the four main benchmarks; avg4 is outside the family, and
    # the two appendix controls have no test at all.
    assert "Holm over 4 benchmarks" in keys["E2.a2_4b.math500.c3_minus_best.p_holm"]["note"]
    assert "outside the Holm family" in keys["E2.a2_4b.avg4.c3_minus_best.p_holm"]["note"]
    assert keys["E2.a2_4b.avg4.c3_minus_best.p_holm"]["value"] == \
        keys["E2.a2_4b.avg4.c3_minus_best.p"]["value"]
    for bench in ("gsm8k", "cmath"):
        assert "E2.a2_4b.%s.c3_minus_best.diff" % bench in keys
        for tail in ("ci_lo", "ci_hi", "p", "p_holm"):
            assert "E2.a2_4b.%s.c3_minus_best.%s" % (bench, tail) not in keys

    # A p value is a plain string, never a LaTeX fragment.
    for key, item in keys.items():
        if key.endswith(".p") or key.endswith(".p_holm"):
            assert not set(str(item["value"])) & set("\\{}$"), key


def test_a_seed_the_other_arm_does_not_have_is_left_out_of_the_pairing(tmp_path):
    root = os.path.join(str(tmp_path), "E2")
    full_tree(root)
    # One more C3 seed with no MAGRPO counterpart.
    for eval_seed in range(5):
        write_run(root, "a2_4b", "c3", 7, eval_seed, suite_values(0.99))
    manifest = json.load(open(write_manifest(tmp_path), encoding="utf-8"))

    keys = a2.build_e2_summary(root, manifest).payload()["keys"]
    # The mean of C3 now covers six seeds, while the difference still pairs five.
    assert keys["E2.a2_4b.math500.c3.mean"]["n"] == 6
    assert keys["E2.a2_4b.math500.c3_minus_best.diff"]["n"] == 5
    assert keys["E2.a2_4b.math500.c3_minus_best.diff"]["value"] == pytest.approx(2.8)


def test_the_two_decomposition_gains_of_the_two_agent_arm(tmp_path):
    root = os.path.join(str(tmp_path), "E2")
    for eval_seed in range(2):
        write_run(root, "a2_4b", "ppo1a", 0, eval_seed, suite_values(0.60))
        write_run(root, "a2_4b", "mappo", 0, eval_seed, suite_values(0.70))
        write_run(root, "a2_4b", "c3", 0, eval_seed, suite_values(0.75))
    manifest = json.load(open(write_manifest(tmp_path), encoding="utf-8"))

    keys = a2.build_e2_summary(root, manifest).payload()["keys"]
    assert keys["E2.a2_4b.math500.arch_gain"]["value"] == pytest.approx(10.0)
    assert keys["E2.a2_4b.math500.credit_gain"]["value"] == pytest.approx(5.0)
    # MAPPO is a candidate for strongest baseline; the single-agent reference is not.
    assert keys["E2.a2_4b.strongest_baseline.name"]["value"] == "MAPPO"


# -----------------------------------------------------------------------------
# a tree that is half written
# -----------------------------------------------------------------------------


def test_what_the_layout_does_not_define_is_skipped_rather_than_fatal(tmp_path, capsys):
    """A half written tree is the normal state while the platform evaluates.

    What is absent is skipped. What is present and damaged is a different thing
    and refuses the whole run; that is the test below this one.
    """
    root = os.path.join(str(tmp_path), "E2")
    interim_tree(root)

    # An arm that is not an arm, a method that is not a method, a seed directory
    # with no number, and an evaluation directory with no record in it yet.
    os.makedirs(os.path.join(root, "a9_nonsense", "c3", "train_s0", "eval_s0"))
    os.makedirs(os.path.join(root, "a2_4b", "not_a_method", "train_s0"))
    os.makedirs(os.path.join(root, "a2_4b", "c3", "train_sX"))
    os.makedirs(os.path.join(root, "a2_4b", "c3", "train_s0", "eval_s9"))

    manifest = json.load(open(write_manifest(tmp_path), encoding="utf-8"))
    keys = a2.build_e2_summary(root, manifest).payload()["keys"]

    # The five good runs are still the answer.
    assert keys["E2.a2_4b.math500.c3.mean"]["value"] == pytest.approx(82.0)
    assert keys["E2.a2_4b.math500.c3.mean"]["n"] == 5

    err = capsys.readouterr().err
    assert "is not an arm name" in err
    assert "is not a method name" in err
    assert "is not a training seed directory" in err
    assert "no final_eval_record.json" in err


def test_a_record_that_is_there_and_damaged_refuses_the_whole_run(tmp_path, capsys):
    """The 2026-09-22 failure: one bad file, four good ones, and a mean of four."""
    root = os.path.join(str(tmp_path), "E2")
    interim_tree(root)
    damaged = os.path.join(root, "a2_4b", "c3", "train_s0", "eval_s8")
    os.makedirs(damaged)
    with open(os.path.join(damaged, "final_eval_record.json"), "w", encoding="utf-8") as fh:
        fh.write("{not json")
    write_run(root, "a2_4b", "c3", 0, 7, suite_values(0.99), schema="c3.final_eval.record/9")

    manifest = json.load(open(write_manifest(tmp_path), encoding="utf-8"))
    with pytest.raises(a2.BrokenRecordError) as caught:
        a2.build_e2_summary(root, manifest)

    reasons = {os.path.basename(os.path.dirname(item["path"])): item["reason"]
               for item in caught.value.broken}
    assert set(reasons) == {"eval_s7", "eval_s8"}
    # Nine bytes beside six records of about a kilobyte: the size test reaches
    # this one before the parser does, which is the point of testing size first.
    assert "bytes" in reasons["eval_s8"] and "median" in reasons["eval_s8"]
    assert "has schema" in reasons["eval_s7"]
    for item in caught.value.broken:
        assert item["arm"] == "a2_4b" and item["method"] == "c3"
        assert item["train_seed"] == 0

    err = capsys.readouterr().err
    assert "median" in err and "has schema" in err


# The five MAPPO records of 2026-09-22, in bytes, as they sat on disk: four
# whole ones and the one a stopped transfer left at a quarter of its size. They
# are the fixture rather than round numbers because the rule has to catch THIS,
# and because the healthy four are within 78 bytes of each other, which is the
# spread the tolerance has to sit far above.
REAL_SIZES = {"eval_s0": 1445919, "eval_s1": 1445964, "eval_s2": 1445996,
              "eval_s3": 351232, "eval_s4": 1445997}


def write_run_of_size(root, arm, method, train_seed, eval_seed, accuracies, size,
                      pad_to=None, **kwargs):
    """One record grown to `pad_to` bytes with a filler field, then cut to `size`.

    A padded record is still a valid record of the right shape, which is what
    the whole ones have to be. Cutting a padded one is what a stopped transfer
    leaves: the 2026-09-22 file was a quarter of its siblings and ended in the
    middle of a string, so it did not parse either.
    """
    where = os.path.join(root, arm, method, "train_s%d" % train_seed, "eval_s%d" % eval_seed)
    os.makedirs(where, exist_ok=True)
    path = os.path.join(where, "final_eval_record.json")

    doc = record(accuracies, **kwargs)
    grown = max(size, int(pad_to or 0))
    text = json.dumps(doc)
    if len(text) < grown:
        # Converge on the exact length: adding the filler adds its own key too.
        doc["_filler"] = ""
        doc["_filler"] = "x" * (grown - len(json.dumps(doc)))
        text = json.dumps(doc)
    text = text[:size]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    assert os.path.getsize(path) == size, (os.path.getsize(path), size)
    return path


def test_the_size_test_flags_the_record_the_stopped_transfer_left():
    """Pure arithmetic on the real byte counts, with no tree and no parsing."""
    sizes = {name: value for name, value in REAL_SIZES.items()}
    flagged = a2._size_outliers(sizes)
    assert list(flagged) == ["eval_s3"]
    assert "351232 bytes" in flagged["eval_s3"]
    assert "76%" in flagged["eval_s3"]

    # The four whole ones on their own are unremarkable, and so is any one of
    # them put back beside the others.
    whole = {name: value for name, value in REAL_SIZES.items() if name != "eval_s3"}
    assert a2._size_outliers(whole) == {}
    assert a2._size_outliers(dict(whole, eval_s3=1445980)) == {}

    # Under three records beside each other the test does not run: a median of
    # two is the midpoint of the pair and cannot say which one is wrong.
    assert a2._size_outliers({"a": 1445919, "b": 351232}) == {}
    assert a2._size_outliers({"a": 351232}) == {}
    # A group of empty files is caught when they fail to parse, not here.
    assert a2._size_outliers({"a": 0, "b": 0, "c": 0}) == {}


def test_a_truncated_record_of_the_real_size_refuses_the_run(tmp_path, capsys):
    """End to end on the shape of 2026-09-22: five records, one cut short."""
    root = os.path.join(str(tmp_path), "E2")
    values = [0.80, 0.81, 0.82, 0.83, 0.84]
    for eval_seed in range(5):
        # The four whole ones are padded to their own size; eval_s3 is grown to
        # the size it should have been and then cut where the transfer stopped.
        write_run_of_size(root, "a2_4b", "c3", 0, eval_seed, suite_values(values[eval_seed]),
                          REAL_SIZES["eval_s%d" % eval_seed],
                          pad_to=1445980 if eval_seed == 3 else None)
    cut = os.path.join(root, "a2_4b", "c3", "train_s0", "eval_s3", "final_eval_record.json")
    assert os.path.getsize(cut) == 351232
    with pytest.raises(ValueError):
        json.load(open(cut, encoding="utf-8"))

    manifest = json.load(open(write_manifest(tmp_path), encoding="utf-8"))
    with pytest.raises(a2.BrokenRecordError) as caught:
        a2.build_e2_summary(root, manifest)

    assert len(caught.value.broken) == 1
    item = caught.value.broken[0]
    assert item["eval_seed"] == 3 and item["method"] == "c3"
    assert "351232 bytes" in item["reason"]


def test_a_record_missing_a_required_field_refuses_the_run(tmp_path):
    root = os.path.join(str(tmp_path), "E2")
    interim_tree(root)
    path = os.path.join(root, "a2_4b", "c3", "train_s0", "eval_s2", "final_eval_record.json")
    doc = json.load(open(path, encoding="utf-8"))
    del doc["ok"]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh)

    manifest = json.load(open(write_manifest(tmp_path), encoding="utf-8"))
    with pytest.raises(a2.BrokenRecordError) as caught:
        a2.build_e2_summary(root, manifest)
    assert [item["eval_seed"] for item in caught.value.broken] == [2]
    assert "does not carry ok" in caught.value.broken[0]["reason"]


def test_a_record_that_says_it_did_not_finish_clean_refuses_the_run(tmp_path):
    """`ok` is the evaluation's own verdict on itself and it is not advisory."""
    root = os.path.join(str(tmp_path), "E2")
    interim_tree(root)
    path = os.path.join(root, "a2_4b", "c3", "train_s0", "eval_s1", "final_eval_record.json")
    doc = json.load(open(path, encoding="utf-8"))
    doc["ok"] = False
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh)

    manifest = json.load(open(write_manifest(tmp_path), encoding="utf-8"))
    with pytest.raises(a2.BrokenRecordError) as caught:
        a2.build_e2_summary(root, manifest)
    assert "ok=False" in caught.value.broken[0]["reason"]


def test_five_whole_records_of_the_real_size_pass_without_a_word(tmp_path, capsys):
    """The control: the same sizes, none of them short, nothing refused."""
    root = os.path.join(str(tmp_path), "E2")
    values = [0.80, 0.81, 0.82, 0.83, 0.84]
    for eval_seed in range(5):
        size = REAL_SIZES["eval_s%d" % eval_seed]
        write_run_of_size(root, "a2_4b", "c3", 0, eval_seed, suite_values(values[eval_seed]),
                          1445980 if eval_seed == 3 else size)

    manifest = json.load(open(write_manifest(tmp_path), encoding="utf-8"))
    doc = a2.build_e2_summary(root, manifest).payload()

    assert doc["keys"]["E2.a2_4b.math500.c3.mean"]["value"] == pytest.approx(82.0)
    assert doc["keys"]["E2.a2_4b.math500.c3.mean"]["n"] == 5
    assert "broken_records" not in doc["extras"]
    assert "median" not in capsys.readouterr().err


def test_allow_broken_records_reports_what_is_readable_and_names_the_damage(tmp_path, capsys):
    """The permissive mode: the numbers that can be read, and n beside the damage."""
    root = os.path.join(str(tmp_path), "E2")
    interim_tree(root)
    path = os.path.join(root, "a2_4b", "c3", "train_s0", "eval_s3", "final_eval_record.json")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("{not json")

    manifest = json.load(open(write_manifest(tmp_path), encoding="utf-8"))
    doc = a2.build_e2_summary(root, manifest, allow_broken_records=True).payload()

    # 80, 81, 82 and 84 in percent, four of the five, and n says four.
    cell = doc["keys"]["E2.a2_4b.math500.c3.mean"]
    assert cell["value"] == pytest.approx((80.0 + 81.0 + 82.0 + 84.0) / 4)
    assert cell["n"] == 4
    assert "1 evaluation record(s) of this arm were damaged" in cell["note"]
    assert "c3/train_s0/eval_s3" in cell["note"]
    assert "the n of each key is how many were read" in cell["note"]

    # Every key of the arm carries it, the derived ones included.
    for key, entry in doc["keys"].items():
        if key.startswith("E2.a2_4b.") and ".compute." not in key:
            assert "were damaged" in entry["note"], key

    block = doc["extras"]["broken_records"]
    assert len(block) == 1 and block[0]["eval_seed"] == 3
    assert "--allow-broken-records was given" in capsys.readouterr().err


def test_the_command_line_refuses_and_writes_no_summary(tmp_path, capsys):
    root = os.path.join(str(tmp_path), "E2")
    interim_tree(root)
    path = os.path.join(root, "a2_4b", "c3", "train_s0", "eval_s3", "final_eval_record.json")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("{not json")
    out = os.path.join(str(tmp_path), "summary.json")

    code = a2.main(["--results", root, "--manifest", write_manifest(tmp_path), "--out", out])
    assert code == a2.EXIT_BROKEN_RECORDS
    assert not os.path.exists(out)
    err = capsys.readouterr().err
    assert "REFUSED" in err and "nothing was written" in err
    assert "--allow-broken-records" in err

    # The same tree with the flag writes the summary and exits zero.
    assert a2.main(["--results", root, "--manifest", write_manifest(tmp_path), "--out", out,
                    "--allow-broken-records"]) == 0
    assert os.path.isfile(out)


def test_a_refused_run_does_not_touch_a_summary_that_is_already_there(tmp_path):
    """The 2026-09-22 trap: a stale summary must not be mistaken for a fresh one."""
    root = os.path.join(str(tmp_path), "E2")
    interim_tree(root)
    out = os.path.join(str(tmp_path), "summary.json")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write('{"from": "an earlier run"}')
    before = open(out, encoding="utf-8").read()

    path = os.path.join(root, "a2_4b", "c3", "train_s0", "eval_s3", "final_eval_record.json")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("{not json")

    assert a2.main(["--results", root, "--manifest", write_manifest(tmp_path),
                    "--out", out]) == a2.EXIT_BROKEN_RECORDS
    assert open(out, encoding="utf-8").read() == before


def test_a_method_that_trained_but_was_not_evaluated_is_named_and_fills_nothing(tmp_path, capsys):
    """The state of the round the evening MAPPO finished training and had no cells yet."""
    root = os.path.join(str(tmp_path), "E2")
    interim_tree(root)
    trained = os.path.join(root, "a2_4b", "mappo", "train_s0")
    os.makedirs(trained)
    for name in ("run_config.json", "budget_ledger.jsonl", "train.log"):
        with open(os.path.join(trained, name), "w", encoding="utf-8") as fh:
            fh.write("{}\n")

    manifest = json.load(open(write_manifest(tmp_path), encoding="utf-8"))
    doc = a2.build_e2_summary(root, manifest).payload()

    # No MAPPO key, and no difference key that would need one.
    assert not [key for key in doc["keys"] if ".mappo." in key]
    assert not [key for key in doc["keys"] if "c3_minus_mappo" in key]

    block = doc["extras"]["methods_without_records"]["a2_4b"]
    assert block["mappo"]["why"] == "training run directory with no evaluation record"
    assert block["mappo"]["train_dirs"][0]["files"] == [
        "run_config.json", "budget_ledger.jsonl", "train.log"]
    assert block["mappo"]["train_dirs"][0]["n_eval_dirs"] == 0
    # A method with no directory at all is the other state, and is named too.
    assert block["ccpo"]["why"] == "no directory under the arm"
    assert block["ccpo"]["train_dirs"] == []
    # The two that did produce records are not in the block.
    assert "c3" not in block and "magrpo" not in block

    err = capsys.readouterr().err
    assert "a2_4b: no evaluation record for" in err
    assert "methods_without_records" in err


def test_an_arm_with_every_method_evaluated_carries_no_missing_block(tmp_path):
    root = os.path.join(str(tmp_path), "E2")
    for eval_seed in range(2):
        for method in a2.METHODS:
            write_run(root, "a2_4b", method, 0, eval_seed, suite_values(0.80))

    manifest = json.load(open(write_manifest(tmp_path), encoding="utf-8"))
    doc = a2.build_e2_summary(root, manifest).payload()
    assert "methods_without_records" not in doc["extras"]


def test_an_arm_directory_that_does_not_exist_is_not_an_error(tmp_path):
    """The arms of this round are two of five; the rest simply have no keys."""
    root = os.path.join(str(tmp_path), "E2")
    interim_tree(root)
    manifest = json.load(open(write_manifest(tmp_path), encoding="utf-8"))

    keys = a2.build_e2_summary(root, manifest).payload()["keys"]
    assert not [key for key in keys if key.startswith("E2.a3_4b.")]
    assert not [key for key in keys if key.startswith("E2.depth_cost.")]


def test_a_results_root_that_is_not_a_directory_is_an_error(tmp_path):
    manifest = json.load(open(write_manifest(tmp_path), encoding="utf-8"))
    with pytest.raises(ValueError):
        a2.build_e2_summary(os.path.join(str(tmp_path), "nowhere"), manifest)


# -----------------------------------------------------------------------------
# the run directory and the command line
# -----------------------------------------------------------------------------


def test_what_a_run_directory_carries_is_reported_and_not_written_to_a_key(tmp_path):
    root = os.path.join(str(tmp_path), "E2")
    interim_tree(root)
    train_dir = os.path.join(root, "a2_4b", "c3", "train_s0")
    with open(os.path.join(train_dir, "compute.json"), "w", encoding="utf-8") as fh:
        json.dump({"train_tokens_M": 12.5, "wall_clock_h": 17.5, "gpu_type": "A100", "n_gpus": 4}, fh)
    with open(os.path.join(train_dir, "budget_ledger.jsonl"), "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"global_step": 0, "total_eval_calls": 16}) + "\n")
        fh.write(json.dumps({"global_step": 1, "total_eval_calls": 16}) + "\n")
        fh.write("not json\n")

    manifest = json.load(open(write_manifest(tmp_path), encoding="utf-8"))
    doc = a2.build_e2_summary(root, manifest).payload()

    entry = doc["extras"]["runs"]["a2_4b/c3/train_s0"]
    assert entry["compute"]["train_tokens_M"] == 12.5
    assert entry["budget_ledger"] == {"rows": 2, "global_steps": 2, "total_eval_calls": 32}
    assert entry["files"] == ["compute.json", "budget_ledger.jsonl"]
    # The inventory itself fills nothing: this run has no log and its ledger rows
    # carry no timestamp, so neither compute key has a number to be written from.
    assert not [key for key in doc["keys"] if ".compute." in key]


# -----------------------------------------------------------------------------
# the token account and the monitoring curve
# -----------------------------------------------------------------------------


def write_token_and_curve_files(train_dir):
    """A trainer log, a budget ledger and one evaluation point, hand computable."""
    with open(os.path.join(train_dir, "train.log"), "w", encoding="utf-8") as fh:
        for step, (response, total, samples) in enumerate(
                ((500.0, 900.0, 300), (600.0, 1000.0, 320)), start=1):
            fh.write("INFO 09-19 05:0%d:00 ppo_trainer.py:2471] Global step %d | epoch 1/2 | "
                     "iter %d/312: %r\n"
                     % (step, step, step, {"response_length": response, "total_length": total,
                                           "reward/count": samples}))
    with open(os.path.join(train_dir, "budget_ledger.jsonl"), "w", encoding="utf-8") as fh:
        for step, stamp in ((1, 1_000_000), (2, 1_003_600)):
            fh.write(json.dumps({"global_step": step, "n_questions_in_batch": 128,
                                 "n_samples_per_prompt": 8, "total_eval_calls": 1024,
                                 "ts_unix": stamp}) + "\n")
    with open(os.path.join(train_dir, "eval_during_train.jsonl.metrics.jsonl"),
              "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"kind": "eval_metrics", "global_step": 1, "temperature": 0.7,
                             "n_samples_per_prompt": 4, "duration_sec": 90.0,
                             "metrics": {"eval_MATH500_pass1": 0.40,
                                         "eval_MATH500_pass4": 0.60,
                                         "eval_AIME24_pass1": 0.10}}) + "\n")


def test_the_token_account_of_every_run_is_carried_whole(tmp_path):
    root = os.path.join(str(tmp_path), "E2")
    interim_tree(root)
    write_token_and_curve_files(os.path.join(root, "a2_4b", "c3", "train_s0"))
    manifest = json.load(open(write_manifest(tmp_path), encoding="utf-8"))

    doc = a2.build_e2_summary(root, manifest).payload()
    assert sorted(doc["tokens"]) == ["a2_4b/c3/train_s0", "a2_4b/magrpo/train_s0"]

    account = doc["tokens"]["a2_4b/c3/train_s0"]
    assert account["totals"]["n_complete_steps"] == 2
    assert account["totals"]["total_tokens"] == 900.0 * 300 + 1000.0 * 320
    assert account["ledger"]["wall_clock_h"] == 1.0
    # The per-step rows stay in the standalone script's output.
    assert "steps" not in account and "steps_note" in account
    # This fixture manifest defines no compute key, so nothing here is written
    # anywhere: the two keys that do get filled have their own tests below.
    assert not [key for key in doc["keys"] if ".compute." in key]

    # A run whose log the platform has not copied yet says so instead of reporting zero.
    empty = doc["tokens"]["a2_4b/magrpo/train_s0"]
    assert empty["totals"]["total_tokens"] is None
    assert "every per-step length" in {gap["quantity"] for gap in empty["not_recorded"]}


def test_the_warmup_stage_is_carried_beside_the_training_account_and_never_inside_it(tmp_path):
    """A warmup rollout is not an optimizer step, so it enters no training total."""
    root = os.path.join(str(tmp_path), "E2")
    interim_tree(root)
    train_dir = os.path.join(root, "a2_4b", "c3", "train_s0")
    write_token_and_curve_files(train_dir)
    with open(os.path.join(train_dir, "budget_ledger_warmup.jsonl"), "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"stage": "critic_warmup", "warmup_rollout_idx": 1,
                             "global_step": -1, "marl_algorithm": "c3",
                             "n_questions_in_batch": 128, "n_samples_per_prompt": 8,
                             "total_eval_calls": 1024, "n_nodes": 1024,
                             "generated_tokens": 500_000, "total_tokens": 900_000,
                             "ts_unix": 999_000}) + "\n")
    manifest = json.load(open(write_manifest(tmp_path), encoding="utf-8"))

    doc = a2.build_e2_summary(root, manifest).payload()

    account = doc["tokens"]["a2_4b/c3/train_s0"]
    assert account["warmup"]["warmup_tokens_M"] == 0.9
    assert account["totals"]["total_tokens"] == 900.0 * 300 + 1000.0 * 320
    assert account["ledger"]["rows"] == 2

    # A run with no warmup ledger says so rather than reporting a zero.
    other = doc["tokens"]["a2_4b/magrpo/train_s0"]
    assert other["warmup"]["warmup_tokens_M"] is None
    assert other["warmup"]["note"].startswith("no warmup ledger")


def test_the_monitoring_curve_is_carried_with_the_bench_names_of_the_tables(tmp_path):
    root = os.path.join(str(tmp_path), "E2")
    interim_tree(root)
    write_token_and_curve_files(os.path.join(root, "a2_4b", "c3", "train_s0"))
    manifest = json.load(open(write_manifest(tmp_path), encoding="utf-8"))

    doc = a2.build_e2_summary(root, manifest).payload()
    curve = doc["curve"]["a2_4b/c3/train_s0"]
    assert curve["n_points"] == 1
    assert curve["suites"] == ["AIME24", "MATH500"]
    assert curve["points"][0]["suites"]["MATH500"]["accuracy_pct"] == 40.0
    # One year of AIME is not the merged benchmark of the manifest, so it is not labelled one.
    assert curve["bench_of_suite"] == {"MATH500": "math500"}
    # The curve feeds no key: preregistration revision 18 (d) keeps it out of the numbers.
    assert not [key for key in doc["keys"] if "curve" in key]

    missing = doc["curve"]["a2_4b/magrpo/train_s0"]
    assert missing["n_points"] == 0
    assert "the whole monitoring curve" in {gap["quantity"] for gap in missing["not_recorded"]}


# -----------------------------------------------------------------------------
# the two compute keys the token account fills
# -----------------------------------------------------------------------------


def manifest_with_compute(arm="a2_4b"):
    """The fixture manifest plus the compute keys the real one defines for this arm.

    `gen_tokens_M` is in it on purpose: the manifest defines it and nothing
    fills it, which is what a test of "these two and no more" needs.
    """
    keys = manifest_keys(arm)
    for method in ("c3", "mappo", "magrpo"):
        for tail, unit in (("train_tokens_M", "M tokens"), ("wall_clock_h", "hours"),
                           ("gen_tokens_M", "M tokens")):
            keys["E2.%s.compute.%s.%s" % (arm, method, tail)] = {
                "value": None, "fmt": "{:.1f}", "unit": unit, "verdict": None,
                "source": None, "n": None, "experiment": "E2",
                "note": "the skeleton's own sentence"}
    return keys


def test_the_two_compute_keys_of_a_run_are_written_from_that_runs_own_files(tmp_path):
    root = os.path.join(str(tmp_path), "E2")
    interim_tree(root)
    write_token_and_curve_files(os.path.join(root, "a2_4b", "c3", "train_s0"))

    doc = a2.build_e2_summary(root, manifest_with_compute()).payload()
    keys = doc["keys"]

    # 900 x 300 + 1000 x 320 tokens, in millions, to two decimals.
    tokens = keys["E2.a2_4b.compute.c3.train_tokens_M"]
    assert tokens["value"] == pytest.approx(0.59)
    assert tokens["n"] == 1
    assert tokens["unit"] == "M tokens"
    assert len(tokens["source"]) == 1 and tokens["source"][0].endswith("train.log")
    assert tokens["note"] == ("training stage only: prompt plus generated tokens summed over "
                              "the 2 training steps from train.log")

    # 1_003_600 minus 1_000_000 seconds, in hours: one segment, so the sum of
    # the segments and the first-to-last distance are the same number here.
    wall = keys["E2.a2_4b.compute.c3.wall_clock_h"]
    assert wall["value"] == pytest.approx(1.0)
    assert wall["n"] == 1
    assert len(wall["source"]) == 1 and wall["source"][0].endswith("budget_ledger.jsonl")
    assert wall["note"] == ("ledger span summed over contiguous training segments; "
                            "0 gap(s) of 0.00 h excluded")

    # Only those two: the split of the account by stage is in no file of a run.
    assert sorted(key for key in keys if ".compute." in key) == [
        "E2.a2_4b.compute.c3.train_tokens_M", "E2.a2_4b.compute.c3.wall_clock_h"]
    assert doc["extras"]["compute_keys_written"] == [
        "E2.a2_4b.compute.c3.train_tokens_M", "E2.a2_4b.compute.c3.wall_clock_h"]


def test_a_run_with_no_log_and_no_ledger_fills_neither_compute_key(tmp_path):
    """The state of a method between the end of its training and the end of its evaluation."""
    root = os.path.join(str(tmp_path), "E2")
    interim_tree(root)

    doc = a2.build_e2_summary(root, manifest_with_compute()).payload()
    assert not [key for key in doc["keys"] if ".compute." in key]
    assert doc["extras"]["compute_keys_written"] == []


def test_only_the_mappo_token_key_says_the_critic_warmup_is_outside_the_account(tmp_path):
    """The wall clock leaves it out too, but each manifest entry says that itself."""
    root = os.path.join(str(tmp_path), "E2")
    interim_tree(root)
    for eval_seed in range(5):
        write_run(root, "a2_4b", "mappo", 0, eval_seed, suite_values(0.75))
    write_token_and_curve_files(os.path.join(root, "a2_4b", "mappo", "train_s0"))
    write_token_and_curve_files(os.path.join(root, "a2_4b", "c3", "train_s0"))

    keys = a2.build_e2_summary(root, manifest_with_compute()).payload()["keys"]
    assert keys["E2.a2_4b.compute.mappo.train_tokens_M"]["note"].endswith(
        "; " + a2.COMPUTE_WARMUP_NOTE)
    assert a2.COMPUTE_WARMUP_NOTE not in keys["E2.a2_4b.compute.c3.train_tokens_M"]["note"]
    for method in ("mappo", "c3"):
        assert (keys["E2.a2_4b.compute.%s.wall_clock_h" % method]["note"]
                == a2.COMPUTE_WALL_NOTE % (0, 0.0))


def test_only_the_magrpo_token_key_says_the_run_was_stopped_and_resumed(tmp_path):
    """That arm trained on two containers; the sentence rides on its key alone."""
    root = os.path.join(str(tmp_path), "E2")
    interim_tree(root)
    write_token_and_curve_files(os.path.join(root, "a2_4b", "magrpo", "train_s0"))
    write_token_and_curve_files(os.path.join(root, "a2_4b", "c3", "train_s0"))

    keys = a2.build_e2_summary(root, manifest_with_compute()).payload()["keys"]
    assert keys["E2.a2_4b.compute.magrpo.train_tokens_M"]["note"].endswith(
        "; " + a2.COMPUTE_RESUMED_NOTE)
    assert a2.COMPUTE_RESUMED_NOTE not in keys["E2.a2_4b.compute.c3.train_tokens_M"]["note"]
    assert a2.COMPUTE_WARMUP_NOTE not in keys["E2.a2_4b.compute.magrpo.train_tokens_M"]["note"]


def test_the_wall_clock_of_a_resumed_run_drops_the_hole_and_the_note_counts_it(tmp_path):
    """The ruling of 2026-09-22: the stop between two containers is not training."""
    root = os.path.join(str(tmp_path), "E2")
    interim_tree(root)
    train_dir = os.path.join(root, "a2_4b", "magrpo", "train_s0")
    write_token_and_curve_files(train_dir)
    # Two steps 600 s apart, a day-long hole, then two more 600 s apart.
    with open(os.path.join(train_dir, "budget_ledger.jsonl"), "w", encoding="utf-8") as fh:
        for step, stamp in ((1, 1_000_000), (2, 1_000_600),
                            (3, 1_087_000), (4, 1_087_600)):
            fh.write(json.dumps({"global_step": step, "n_questions_in_batch": 128,
                                 "n_samples_per_prompt": 8, "total_eval_calls": 1024,
                                 "ts_unix": stamp}) + "\n")

    doc = a2.build_e2_summary(root, manifest_with_compute()).payload()
    wall = doc["keys"]["E2.a2_4b.compute.magrpo.wall_clock_h"]

    # 600 s plus 600 s, and not the 86_400 s of the hole between them.
    assert wall["value"] == pytest.approx(round(1200 / 3600.0, 2))
    assert wall["note"] == ("ledger span summed over contiguous training segments; "
                            "1 gap(s) of 24.00 h excluded")
    ledger = doc["tokens"]["a2_4b/magrpo/train_s0"]["ledger"]
    assert ledger["n_segments"] == 2
    assert ledger["first_last_span_h"] == pytest.approx(87_600 / 3600.0)


def test_a_log_whose_steps_are_not_all_readable_says_so_in_the_note(tmp_path):
    root = os.path.join(str(tmp_path), "E2")
    interim_tree(root)
    train_dir = os.path.join(root, "a2_4b", "c3", "train_s0")
    write_token_and_curve_files(train_dir)
    with open(os.path.join(train_dir, "train.log"), "a", encoding="utf-8") as fh:
        fh.write("INFO ppo_trainer.py:2471] Global step 3 | epoch 1/2 | iter 3/312: {broken\n")

    keys = a2.build_e2_summary(root, manifest_with_compute()).payload()["keys"]
    note = keys["E2.a2_4b.compute.c3.train_tokens_M"]["note"]
    assert "summed over the 2 training steps" in note
    assert "1 further step line(s) could not be read and are in no step of the account" in note
    # An unreadable line is not a repeat, and the note no longer says it might be.
    assert "repeat a step the log already carried" not in note


def test_a_resumed_log_says_its_repeated_steps_are_repeats_and_not_damage(tmp_path):
    """The MAGRPO arm of this round: the second container's log is appended to
    the first one's, so the steps it restarts from appear twice."""
    root = os.path.join(str(tmp_path), "E2")
    interim_tree(root)
    train_dir = os.path.join(root, "a2_4b", "c3", "train_s0")
    write_token_and_curve_files(train_dir)
    with open(os.path.join(train_dir, "train.log"), "a", encoding="utf-8") as fh:
        for step in (1, 2):
            fh.write("INFO 09-19 05:0%d:00 ppo_trainer.py:2471] Global step %d | epoch 1/2 | "
                     "iter %d/312: %r\n"
                     % (step, step, step, {"response_length": 500.0, "total_length": 900.0,
                                           "reward/count": 300}))

    keys = a2.build_e2_summary(root, manifest_with_compute()).payload()["keys"]
    note = keys["E2.a2_4b.compute.c3.train_tokens_M"]["note"]
    # Four lines, two unique steps, so two repeats and nothing damaged.
    assert "summed over the 2 training steps" in note
    assert ("2 further step line(s) repeat a step the log already carried, which is what a run "
            "resumed from a checkpoint writes, and the account keeps the last line of each") in note
    assert "could not be read" not in note
    assert "carry no length" not in note
    assert a2.COMPUTE_UNACCOUNTED_NOTE.split(" %d")[0] not in note


def test_a_log_whose_lines_all_count_says_nothing_beyond_the_step_total(tmp_path):
    root = os.path.join(str(tmp_path), "E2")
    interim_tree(root)
    write_token_and_curve_files(os.path.join(root, "a2_4b", "c3", "train_s0"))

    keys = a2.build_e2_summary(root, manifest_with_compute()).payload()["keys"]
    assert keys["E2.a2_4b.compute.c3.train_tokens_M"]["note"] == (
        "training stage only: prompt plus generated tokens summed over the 2 training steps "
        "from train.log")


def test_the_three_reasons_a_step_line_is_not_a_step_are_named_apart():
    """The split is arithmetic on the counters, so it is pinned without a tree."""
    log = {"n_lines_with_a_step": 10, "n_duplicate_steps": 3, "n_unparsed": 2}
    totals = {"n_steps": 5, "n_complete_steps": 4}
    clauses = a2._log_line_clauses(log, totals)
    assert clauses == [a2.COMPUTE_DUPLICATE_NOTE % 3,
                       a2.COMPUTE_UNPARSED_NOTE % 2,
                       a2.COMPUTE_INCOMPLETE_NOTE % 1]

    # Nothing to say when every line is a complete step.
    assert a2._log_line_clauses({"n_lines_with_a_step": 5, "n_duplicate_steps": 0,
                                 "n_unparsed": 0},
                                {"n_steps": 5, "n_complete_steps": 5}) == []
    # A run with no log at all gets no sentence rather than a zero.
    assert a2._log_line_clauses({}, {"n_steps": 0, "n_complete_steps": 0}) == []


def test_counters_that_do_not_add_up_are_reported_rather_than_papered_over():
    """The breakdown is arithmetic; if it fails, the note says so."""
    log = {"n_lines_with_a_step": 10, "n_duplicate_steps": 0, "n_unparsed": 0}
    totals = {"n_steps": 5, "n_complete_steps": 5}
    assert a2._log_line_clauses(log, totals) == [a2.COMPUTE_UNACCOUNTED_NOTE % 5]


def test_the_command_line_writes_a_summary_and_can_be_narrowed_to_one_arm(tmp_path):
    root = os.path.join(str(tmp_path), "E2")
    interim_tree(root)
    for eval_seed in range(5):
        write_run(root, "a3_4b", "c3", 0, eval_seed, suite_values(0.50))
    manifest_path = write_manifest(tmp_path)
    out_path = os.path.join(str(tmp_path), "summary.json")

    assert a2.main(["--results", root, "--manifest", manifest_path, "--out", out_path,
                    "--arm", "a2_4b"]) == 0
    with open(out_path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    assert doc["experiment"] == "E2"
    assert doc["generated_by"]["script"] == "c3.analysis.rebuild.aggregate_e2"
    assert doc["replication"]["by_arm"] == {"a2_4b": "interim"}
    assert not [key for key in doc["keys"] if key.startswith("E2.a3_4b.")]

    assert a2.main(["--results", root, "--manifest", manifest_path, "--out", out_path,
                    "--arm", "nonsense"]) == 2


def test_it_runs_as_a_module(tmp_path):
    root = os.path.join(str(tmp_path), "E2")
    interim_tree(root)
    manifest_path = write_manifest(tmp_path)
    out_path = os.path.join(str(tmp_path), "summary.json")

    done = subprocess.run(
        [sys.executable, "-m", "c3.analysis.rebuild.aggregate_e2",
         "--results", root, "--manifest", manifest_path, "--out", out_path],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stderr
    with open(out_path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    assert doc["keys"], done.stderr


# -----------------------------------------------------------------------------
# a supplementary checkpoint row
#
# `magrpo_s250` is MAGRPO's step-250 save, exported under its own method name so
# that it goes through this same aggregation (preregistration revision 25,
# 2026-09-22). What these tests hold down is the line between "reported" and
# "compared": the row gets its own accuracy cells and enters no test, and the
# preregistered numbers beside it come out exactly as they would without it.
# -----------------------------------------------------------------------------

SUPP = "magrpo_s250"


def test_a_checkpoint_name_is_recognised_and_a_typo_is_not():
    assert a2.supplementary_method(SUPP) == {"base": "magrpo", "checkpoint_step": 250}
    assert a2.supplementary_method("c3_s12") == {"base": "c3", "checkpoint_step": 12}
    for name in ("magrpo", "not_a_method", "magrpo_step250", "magrpos250", "magrpo_s",
                 "magrpo_sxyz", "nonsense_s250", "", "MAGRPO_S250"):
        assert a2.supplementary_method(name) is None, name
    assert a2.known_method("magrpo") and a2.known_method(SUPP)
    assert not a2.known_method("not_a_method")


def supplementary_manifest(tmp_path):
    """The fixture manifest plus the fourteen keys of the supplementary row."""
    keys = manifest_keys("a2_4b")
    for bench in BENCHES + APPX_BENCHES:
        for stat in ("mean", "std"):
            keys["E2.a2_4b.%s.%s.%s" % (bench, SUPP, stat)] = {
                "value": None, "fmt": "{:.1f}", "unit": "%", "verdict": None,
                "source": None, "n": None, "experiment": "E2", "note": ""}
    path = os.path.join(str(tmp_path), "manifest_supp.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(keys, fh)
    return json.load(open(path, encoding="utf-8"))


def supplementary_tree(root, *, run_config=True):
    """The interim tree plus the checkpoint row, five evaluation runs of its own."""
    c3, magrpo = interim_tree(root)
    supp = [0.86, 0.87, 0.88, 0.89, 0.90]
    for eval_seed in range(5):
        write_run(root, "a2_4b", SUPP, 0, eval_seed, suite_values(supp[eval_seed]))
    if run_config:
        where = os.path.join(root, "a2_4b", SUPP, "train_s0", "run_config.json")
        with open(where, "w", encoding="utf-8") as fh:
            json.dump({"method": "MAGRPO", "alg": SUPP, "checkpoint_tag": "global_step250",
                       "checkpoint_step": 250, "derived_from": "E2/a2_4b/magrpo/train_s0",
                       "note": "supplementary row; deviates from the preregistration"}, fh)
    return c3, magrpo, supp


def test_a_supplementary_row_gets_its_own_cells(tmp_path):
    root = os.path.join(str(tmp_path), "E2")
    _c3, _magrpo, supp = supplementary_tree(root)
    keys = a2.build_e2_summary(root, supplementary_manifest(tmp_path)).payload()["keys"]

    # 86 to 90 in percent: mean 88, sample sd sqrt(2.5), the same reduction as
    # any other interim cell.
    assert keys["E2.a2_4b.math500.%s.mean" % SUPP]["value"] == pytest.approx(88.0)
    assert keys["E2.a2_4b.math500.%s.std" % SUPP]["value"] == pytest.approx(math.sqrt(2.5))
    assert keys["E2.a2_4b.math500.%s.mean" % SUPP]["n"] == 5
    assert supp[0] == 0.86
    # Every bench the preregistered methods have, the supplementary row has too.
    for bench in BENCHES + APPX_BENCHES:
        assert "E2.a2_4b.%s.%s.mean" % (bench, SUPP) in keys, bench


def test_the_cells_of_a_supplementary_row_say_what_they_are(tmp_path):
    root = os.path.join(str(tmp_path), "E2")
    supplementary_tree(root)
    keys = a2.build_e2_summary(root, supplementary_manifest(tmp_path)).payload()["keys"]

    note = keys["E2.a2_4b.math500.%s.mean" % SUPP]["note"]
    assert "supplementary row: MAGRPO at step 250" in note
    assert "not the preregistered final model" in note
    assert "in no comparison" in note
    # And the preregistered rows do not pick the sentence up.
    assert "supplementary" not in keys["E2.a2_4b.math500.magrpo.mean"]["note"]
    assert "supplementary" not in keys["E2.a2_4b.math500.c3.mean"]["note"]


def test_a_supplementary_row_enters_no_comparison(tmp_path):
    """It scores highest of the three here, and the strongest baseline is still
    MAGRPO: a checkpoint chosen on the monitoring evaluation is not a
    preregistered baseline, and the preregistered test must not move because one
    was exported."""
    root = os.path.join(str(tmp_path), "E2")
    supplementary_tree(root)
    manifest = supplementary_manifest(tmp_path)
    doc = a2.build_e2_summary(root, manifest).payload()
    keys = doc["keys"]

    assert keys["E2.a2_4b.strongest_baseline.name"]["value"] == "MAGRPO"
    assert keys["E2.a2_4b.math500.best.mean"]["value"] == pytest.approx(72.0)
    assert not [key for key in keys if SUPP in key and ".mean" not in key
                and ".std" not in key]
    # The difference is C3 minus MAGRPO, unchanged: 82 - 72.
    assert keys["E2.a2_4b.math500.c3_minus_best.diff"]["value"] == pytest.approx(10.0)

    # And the same tree without the row gives the same preregistered numbers.
    plain = os.path.join(str(tmp_path), "E2plain")
    interim_tree(plain)
    plain_keys = a2.build_e2_summary(plain, manifest).payload()["keys"]
    for key in plain_keys:
        assert plain_keys[key]["value"] == keys[key]["value"], key


def test_the_supplementary_block_says_where_the_checkpoint_came_from(tmp_path, capsys):
    root = os.path.join(str(tmp_path), "E2")
    supplementary_tree(root)
    doc = a2.build_e2_summary(root, supplementary_manifest(tmp_path)).payload()

    block = doc["extras"]["supplementary"]["a2_4b"][SUPP]
    assert block["supplementary"] is True
    assert block["in_comparisons"] is False
    assert block["base_method"] == "magrpo"
    assert block["checkpoint_step"] == 250
    assert block["n_records"] == 5
    assert block["run_config"]["derived_from"] == "E2/a2_4b/magrpo/train_s0"
    assert block["run_config"]["checkpoint_tag"] == "global_step250"

    err = capsys.readouterr().err
    assert "supplementary checkpoint row(s) magrpo_s250" in err
    assert "in no comparison" in err


def test_a_tree_with_no_supplementary_row_carries_no_block(tmp_path):
    root = os.path.join(str(tmp_path), "E2")
    interim_tree(root)
    doc = a2.build_e2_summary(root, supplementary_manifest(tmp_path)).payload()
    assert "supplementary" not in doc["extras"]


def test_a_supplementary_row_without_a_run_config_is_still_reported(tmp_path):
    """The export writes run_config.json, but a row that reached the results tree
    without one is missing provenance, not missing data: the cells are still the
    cells and the block says the file is not there."""
    root = os.path.join(str(tmp_path), "E2")
    supplementary_tree(root, run_config=False)
    doc = a2.build_e2_summary(root, supplementary_manifest(tmp_path)).payload()

    block = doc["extras"]["supplementary"]["a2_4b"][SUPP]
    assert "run_config" not in block and "run_config_error" not in block
    assert doc["keys"]["E2.a2_4b.math500.%s.mean" % SUPP]["value"] == pytest.approx(88.0)


def test_a_supplementary_row_is_not_a_method_that_is_missing(tmp_path):
    """`methods_without_records` is the preregistered vocabulary, so the presence
    or absence of a checkpoint row never appears there."""
    root = os.path.join(str(tmp_path), "E2")
    supplementary_tree(root)
    doc = a2.build_e2_summary(root, supplementary_manifest(tmp_path)).payload()
    block = doc["extras"]["methods_without_records"]["a2_4b"]
    assert SUPP not in block
    assert "mappo" in block and "ccpo" in block


def test_a_supplementary_row_fills_no_compute_key(tmp_path):
    """The tokens and the wall clock belong to the training run the checkpoint is
    taken from, and are keyed there. A log left in the checkpoint row's own
    directory must not become a second account of the same training."""
    root = os.path.join(str(tmp_path), "E2")
    supplementary_tree(root)
    manifest = supplementary_manifest(tmp_path)
    doc = a2.build_e2_summary(root, manifest).payload()
    assert not [key for key in doc["keys"] if ".compute." in key and SUPP in key]
    assert not [key for key in manifest if ".compute." in key and SUPP in key]


def test_a_supplementary_row_does_not_turn_the_arm_into_a_multi_seed_one(tmp_path):
    """Its directory is `train_s0`, the same training seed the arm already has,
    so the replication unit of the round is unchanged."""
    root = os.path.join(str(tmp_path), "E2")
    supplementary_tree(root)
    doc = a2.build_e2_summary(root, supplementary_manifest(tmp_path)).payload()
    assert doc["replication"]["mode"] == "interim"
    assert doc["replication"]["by_arm"] == {"a2_4b": "interim"}


def test_the_summary_carries_the_fourteen_keys_a_manifest_writer_can_load(tmp_path):
    """The handover point of the chain: what `--from-summary` reads is the `keys`
    object of this summary, and a record it can load is one with a value, an n
    and at least one source. The other side of the handover (the writer, the
    printing formats of the real manifest) is pinned in the project tree, where
    that writer lives: 30_analysis/rebuild/test_manifest_writer.py."""
    root = os.path.join(str(tmp_path), "E2")
    supplementary_tree(root)
    manifest_path = os.path.join(str(tmp_path), "manifest_supp_on_disk.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(supplementary_manifest(tmp_path), fh)
    out_path = os.path.join(str(tmp_path), "summary.json")
    assert a2.main(["--results", root, "--manifest", manifest_path, "--out", out_path]) == 0

    with open(out_path, "r", encoding="utf-8") as fh:
        keys = json.load(fh)["keys"]
    mine = {key: rec for key, rec in keys.items() if SUPP in key}
    assert len(mine) == 14, sorted(mine)
    for key, rec in sorted(mine.items()):
        assert isinstance(rec["value"], float), key
        assert rec["n"] == 5, key
        assert rec["source"], key
        assert all("final_eval_record.json" in path for path in rec["source"]), key
        assert "supplementary row: MAGRPO at step 250" in rec["note"], key
