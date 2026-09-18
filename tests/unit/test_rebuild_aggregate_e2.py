# tests/unit/test_rebuild_aggregate_e2.py
"""The E2 aggregation and the two replication units it has to tell apart.

Every fixture is synthetic and hand computable; nothing here reads real data.
The interim tree is the one the platform is writing now (one training seed
evaluated five times, preregistration revision 20) and the full tree is the one
that replaces it (five training seeds, each evaluated five times). The two are
different measurements wearing the same shape, so what is pinned here is mostly
that the second is not silently reported as the first.

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


def test_what_is_missing_is_skipped_rather_than_fatal(tmp_path, capsys):
    root = os.path.join(str(tmp_path), "E2")
    interim_tree(root)

    # An arm that is not an arm, a method that is not a method, a seed directory
    # with no number, an evaluation directory with no record, and a record that
    # does not parse. None of these is the aggregation's business to fix.
    os.makedirs(os.path.join(root, "a9_nonsense", "c3", "train_s0", "eval_s0"))
    os.makedirs(os.path.join(root, "a2_4b", "not_a_method", "train_s0"))
    os.makedirs(os.path.join(root, "a2_4b", "c3", "train_sX"))
    os.makedirs(os.path.join(root, "a2_4b", "c3", "train_s0", "eval_s9"))
    broken = os.path.join(root, "a2_4b", "c3", "train_s0", "eval_s8")
    os.makedirs(broken)
    with open(os.path.join(broken, "final_eval_record.json"), "w", encoding="utf-8") as fh:
        fh.write("{not json")
    # A record from a schema this script does not know.
    write_run(root, "a2_4b", "c3", 0, 7, suite_values(0.99), schema="c3.final_eval.record/9")

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
    assert "does not parse" in err
    assert "has schema" in err


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
    # Two-agent compute never reaches the three-agent compute keys of the manifest.
    assert not [key for key in doc["keys"] if ".compute." in key]


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
