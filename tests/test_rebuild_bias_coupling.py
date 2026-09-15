# tests/test_rebuild_bias_coupling.py
"""The influence estimator, the E3a bias map and the E5 coupling.

Every fixture is synthetic and hand computable; nothing here reads real data.
Run with the light local environment:

    python -m pytest tests/test_rebuild_bias_coupling.py -q
"""

from __future__ import annotations

import io
import json
import math
import os

import pytest

from c3.analysis.rebuild import bias_map as bm
from c3.analysis.rebuild import coupling as cp
from c3.analysis.rebuild import influence as inf
from c3.analysis.rebuild import aggregate_e3a, aggregate_e5

LN3 = math.log(3.0)
LN4 = math.log(4.0)

# The E3a and E5 key names this work package writes. The list is the spelling
# snapshot the synthetic manifest is built from; test_key_names_are_the_spec_set
# checks that the modules emit exactly these.
E3A_KEYS = [
    "E3a.n_points",
    "E3a.bias.mean",
    "E3a.bias.p",
    "E3a.strata.high_infl_diff.n",
    "E3a.strata.high_infl_nodiff.n",
    "E3a.strata.high_infl_nodiff.zero_share_pct",
    "E3a.strata.high_infl_diff.zero_n",
    "E3a.strata.mw_p",
    "E3a.strata.low_infl.mean_abs_bias",
    "E3a.top_stratum.share_of_points_pct",
    "E3a.top_stratum.share_of_bias_pct",
    "E3a.top_stratum.mean_abs_bias",
    "E3a.rest.n",
    "E3a.rest.mean_abs_bias",
    "E3a.concentration_factor",
    "E3a.influence.mean_nats",
    "E3a.cutoffs.max_mw_p",
    "E3a.null_forms.paired_p",
]
E3A_EXTRA_KEYS = [
    "E3a.strata.high_infl_diff.mean_bias",
    "E3a.strata.high_infl_diff.median",
    "E3a.strata.high_infl_diff.iqr_lo",
    "E3a.strata.high_infl_diff.iqr_hi",
    "E3a.strata.high_infl_nodiff.mean_bias",
    "E3a.strata.high_infl_nodiff.median",
    "E3a.strata.high_infl_nodiff.iqr_lo",
    "E3a.strata.high_infl_nodiff.iqr_hi",
    "E3a.strata.high_infl_nodiff.zero_n",
]
E5_KEYS = [
    "E5.n_questions",
    "E5.rel.sft",
    "E5.rel.c3",
    "E5.rel.diff_p",
    "E5.influence.sft",
    "E5.influence.c3",
    "E5.influence.diff",
    "E5.influence.ci_lo",
    "E5.influence.ci_hi",
    "E5.influence.p",
    "E5.correction.sft",
    "E5.correction.c3",
    "E5.correction.diff",
    "E5.correction.ci_lo",
    "E5.correction.ci_hi",
    "E5.correction.p",
    "E5.correction.n_upstream_wrong.sft",
    "E5.correction.n_upstream_wrong.c3",
]
UNITS = {
    "E3a.n_points": "points", "E3a.bias.mean": "return", "E3a.bias.p": "p",
    "E3a.strata.mw_p": "p", "E3a.cutoffs.max_mw_p": "p", "E3a.null_forms.paired_p": "p",
    "E3a.concentration_factor": "ratio", "E3a.influence.mean_nats": "nats",
    "E5.rel.diff_p": "p", "E5.influence.p": "p", "E5.correction.p": "p",
}


# -----------------------------------------------------------------------------
# fixtures
# -----------------------------------------------------------------------------


def make_bucket(qid, returns, nexts, *, actions=None, meta=None):
    """One bucket in the schema of c3/analysis/buckets.py."""
    cands = []
    for j, (rets, nx) in enumerate(zip(returns, nexts)):
        cands.append({
            "j": j,
            "action_text": "" if actions is None else actions[j],
            "returns": [float(r) for r in rets],
            "next_actions": list(nx),
        })
    return {
        "bucket_id": "%s-b0" % qid,
        "ctx_hash": 0,
        "target_role": "Reasoner",
        "question_id": qid,
        "restart": {"roles_topo": ["Reasoner", "Actor"], "role_outputs_prefix": {}},
        "candidates": cands,
        "meta": dict(meta or {}),
    }


def boxed(text):
    return "\\boxed{%s}" % text


def e3a_buckets(per_stratum=4):
    """Decision points in three equal groups: zero-bias no-difference, top, low
    influence. `per_stratum` of each, four of each by default (twelve points).

    Alternative 0 is the injected null arm in every bucket. The three real
    alternatives carry a distinct downstream answer each in the first two groups
    (high influence) and a single shared answer in the last (influence exactly
    zero). `per_stratum` only repeats the same three patterns, so every number
    read off the map except the counts is the same at any size; the tests of the
    degeneracy warning use that.
    """
    out = []
    distinct = [[boxed("z")] * 4, [boxed("A")] * 4, [boxed("B")] * 4, [boxed("C")] * 4]
    shared = [[boxed("Z")] * 4] * 4
    meta = {"null_arm": 0, "null_form": "empty", "workflow": "a3", "model": "4b"}
    for i in range(per_stratum):  # high influence, no value difference, bias exactly zero
        out.append(make_bucket("A%d" % i,
                               [[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0]],
                               distinct, meta=meta))
    for i in range(per_stratum):  # high influence, values differ, large bias
        out.append(make_bucket("B%d" % i,
                               [[0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 0, 0]],
                               distinct, meta=meta))
    for i in range(per_stratum):  # low influence, small bias
        out.append(make_bucket("C%d" % i,
                               [[0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]],
                               shared, meta=meta))
    return out


#: Returns patterns whose even and odd halves agree (rho = 1) and disagree
#: (rho = 0.5); both are used unchanged in the two E5 arms.
REL_ONE = [[0, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1], [0, 0, 1, 1]]
REL_HALF = [[1, 0, 0, 0], [0, 1, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]]


def e5_arm(kind, n_questions=10):
    """One E5 arm. 'sft' has high influence and a low correction rate, 'c3' the
    reverse, with the returns (and therefore the reliability) identical."""
    out = []
    for i in range(n_questions):
        rets = REL_ONE if i % 2 == 0 else REL_HALF
        if kind == "sft":
            actions = [boxed("W"), boxed("W"), boxed("G"), boxed("G")]
            nexts = [[boxed("S%d" % j)] * 4 for j in range(4)]
            if i % 2 == 1:
                nexts[0] = [boxed("G")] + [boxed("S0")] * 3
        else:
            actions = [boxed("W"), boxed("G"), boxed("G"), boxed("G")]
            one = [boxed("G")] * 4 if i % 2 == 0 else [boxed("G")] * 3 + [boxed("W")]
            nexts = [list(one) for _ in range(4)]
        out.append(make_bucket("q%d" % i, rets, nexts, actions=actions,
                               meta={"workflow": "a2", "model": "4b"}))
    return out


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_manifest(path, keys):
    entries = {}
    for key in keys:
        entries[key] = {"value": None, "fmt": "{}", "unit": UNITS.get(key, "return"),
                        "verdict": None, "source": None, "n": None,
                        "experiment": key.split(".")[0], "note": ""}
    entries["E3a.bias_concentration"] = {"value": None, "fmt": "{}", "unit": "verdict",
                                         "verdict": None, "source": None, "n": None,
                                         "experiment": "E3a", "note": ""}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(entries, fh)
    return entries


# -----------------------------------------------------------------------------
# 1. plugin_mi_nats
# -----------------------------------------------------------------------------


def test_mi_independent_is_exactly_zero_uncorrected():
    y = [["a", "a", "b", "b"], ["a", "a", "b", "b"], ["a", "a", "b", "b"]]
    assert inf.plugin_mi_nats(y, miller_madow=False) == pytest.approx(0.0, abs=1e-15)


def test_mi_deterministic_is_log_n_alternatives():
    for n in (2, 3, 4):
        y = [[chr(ord("a") + j)] * 5 for j in range(n)]
        assert inf.plugin_mi_nats(y, miller_madow=False) == pytest.approx(math.log(n), abs=1e-12)


def test_miller_madow_direction_is_downward_when_every_group_spans_the_alphabet():
    """The correction subtracts here, which contradicts the wording of the
    specification's self-test 1 (see the report, objection 1).

    Mechanically: MM adds (m - 1) / (2N) to every entropy term, so it moves the
    mutual information by (m_Y - 1) / (2N) minus the weighted sum of the
    conditional terms. Two groups of four samples over the same two symbols give
    (2 - 1) / 16 for the marginal against 2 x (1/2) x (2 - 1) / 8 for the
    conditionals, that is 0.0625 - 0.125 = -0.0625.
    """
    y = [["a", "a", "b", "b"], ["a", "a", "b", "b"]]
    plain = inf.plugin_mi_nats(y, miller_madow=False)
    corrected = inf.plugin_mi_nats(y, miller_madow=True)
    assert plain == pytest.approx(0.0, abs=1e-15)
    assert corrected == pytest.approx(-0.0625, abs=1e-12)
    assert corrected < plain  # the correction is not clipped at zero


def test_miller_madow_direction_is_upward_when_each_group_is_a_point_mass():
    """Same correction, opposite sign: the conditional terms vanish."""
    y = [["a"] * 4, ["b"] * 4, ["c"] * 4]
    plain = inf.plugin_mi_nats(y, miller_madow=False)
    corrected = inf.plugin_mi_nats(y, miller_madow=True)
    assert plain == pytest.approx(LN3, abs=1e-12)
    assert corrected == pytest.approx(LN3 + 2.0 / 24.0, abs=1e-12)
    assert corrected > plain


def test_mi_hand_worked_two_by_four():
    """Two alternatives, four samples each, answer counts [3, 1] and [1, 3].

    H(Y) = ln 2 = 0.6931472, H(Y | j) = -(0.75 ln 0.75 + 0.25 ln 0.25) = 0.5623351,
    so the plug-in value is 0.1308121. The correction is 1/16 for the marginal
    minus 1/8 for the conditionals, that is -0.0625, giving 0.0683121.
    """
    y = [["a", "a", "a", "b"], ["a", "b", "b", "b"]]
    plain = inf.plugin_mi_nats(y, miller_madow=False)
    corrected = inf.plugin_mi_nats(y, miller_madow=True)
    assert plain == pytest.approx(0.13081203, abs=1e-8)
    assert corrected == pytest.approx(0.13081203 - 0.0625, abs=1e-8)


def test_mi_drops_empty_groups_and_returns_zero_below_two():
    assert inf.plugin_mi_nats([["a"], []], miller_madow=False) == 0.0
    assert inf.plugin_mi_nats([], miller_madow=True) == 0.0
    y = [["a"] * 4, ["b"] * 4, []]
    assert inf.plugin_mi_nats(y, miller_madow=False) == pytest.approx(math.log(2.0), abs=1e-12)


# -----------------------------------------------------------------------------
# 2. boxed_extractor and answer_symbol
# -----------------------------------------------------------------------------


def test_boxed_extractor_nested_braces():
    assert inf.boxed_extractor("so \\boxed{\\frac{1}{2}} done") == "\\frac{1}{2}"


def test_boxed_extractor_takes_the_last_one():
    assert inf.boxed_extractor("\\boxed{3} then \\boxed{ 41 }") == "41"


def test_boxed_extractor_missing_and_unbalanced():
    assert inf.boxed_extractor("no answer here") == inf.NO_ANSWER
    assert inf.boxed_extractor("") == inf.NO_ANSWER
    assert inf.boxed_extractor(None) == inf.NO_ANSWER
    assert inf.boxed_extractor("\\boxed{unclosed") == inf.NO_ANSWER
    assert inf.boxed_extractor("\\boxed{}") == inf.NO_ANSWER


def test_answer_symbol_maps_none_and_blank_to_no_answer():
    assert inf.answer_symbol("x", lambda t: None) == inf.NO_ANSWER
    assert inf.answer_symbol("x", lambda t: "   ") == inf.NO_ANSWER
    assert inf.answer_symbol("x", lambda t: " 7 ") == "7"
    assert inf.answer_symbol(None, inf.boxed_extractor) == inf.NO_ANSWER


# -----------------------------------------------------------------------------
# 2b. math_extractor, the extractor the aggregation runs (2026-09-15, narrowed
#     to explicit answers the same day)
#
# It is the repository's own parser (`c3.envs.math.parsing.parse_math_answer`),
# restricted to the methods that mean the output STATED an answer, followed by
# the repository's expression normaliser
# (`c3.envs.math.backends.marft.normalize.normalize_expr`). These tests pin the
# shapes the E3a downstream outputs actually take and the boundary of the
# explicit-answer rule, not the parser itself, which has its own tests.
# -----------------------------------------------------------------------------


def test_math_extractor_reads_a_boxed_answer():
    assert inf.math_extractor("work, then \\boxed{\\frac{1}{2}}") == "(1)/(2)"
    # The point of the normaliser: the same value written with and without the
    # display wrapper is one symbol, where the raw strings are two.
    assert inf.math_extractor("so \\boxed{$\\frac{1}{2}$}") == inf.math_extractor(
        "so \\boxed{\\frac{1}{2}}")


def test_math_extractor_does_not_fold_dfrac_into_frac():
    """A limit of the repository normaliser, pinned so it is not mistaken for a
    property of the estimator: `normalize_expr` rewrites \\frac and \\sqrt only,
    so a replay that writes \\dfrac keeps a symbol of its own."""
    assert inf.math_extractor("\\boxed{\\dfrac{1}{2}}") == "dfrac12"
    assert inf.math_extractor("\\boxed{\\dfrac{1}{2}}") != inf.math_extractor(
        "\\boxed{\\frac{1}{2}}")


def test_math_extractor_reads_a_hash_answer():
    assert inf.math_extractor("long working\n#### 42") == "42"
    # The last one wins, as in the parser.
    assert inf.math_extractor("#### 7\nmore\n#### 42") == "42"


def test_math_extractor_reads_an_anchored_answer():
    """`Final answer: 7` is a stated answer, method `anchor`, and it is the shape
    most real E3a downstream outputs end on."""
    from c3.envs.math.parsing import parse_math_answer

    assert parse_math_answer("long working\nFinal answer: 7")[1] == "anchor"
    assert inf.math_extractor("long working\nFinal answer: 7") == "7"
    assert inf.math_extractor("The answer is: 7") == "7"
    assert inf.math_extractor("Answer: 7") == "7"


def test_math_extractor_calls_the_prose_fallback_no_answer():
    """The maintainers' decision of 2026-09-15: only a STATED answer counts.

    The repository parser has no "unparseable" verdict; its last resort is the
    last non-empty line, method `last_line`, which is a sentence of the working.
    Such a sentence is all but unique to its replay, so accepting it as an answer
    symbol would manufacture influence. It is `<NONE>` instead.
    """
    from c3.envs.math.parsing import parse_math_answer

    prose = ("I checked the arithmetic once more.\n"
             "Now, the solution will be passed to the Verifier for final validation")
    token, method = parse_math_answer(prose)
    assert method == "last_line" and token  # the parser does return something
    assert inf.math_extractor(prose) == inf.NO_ANSWER

    # The cost of the rule, pinned so it is not discovered later: a bare number
    # on its own line is `last_line` too, so an output that states its answer
    # without a marker is read as no answer.
    assert parse_math_answer("42")[1] == "last_line"
    assert inf.math_extractor("42") == inf.NO_ANSWER


def test_math_extractor_accepts_exactly_the_three_methods_the_decision_names():
    """The accepted set is a whitelist of three, so the parser's two remaining
    anchor-ish outcomes fall outside it.

    `anchor_inline` is one of them: the parser reaches it when the anchor sits on
    the last non-empty line but not on the last line of the string, which is what
    a trailing blank line does. Whether that fourth method should join the three
    is the maintainers' call; this pins what the code does today so the choice is
    visible rather than silent.
    """
    from c3.envs.math.parsing import parse_math_answer

    assert inf.EXPLICIT_ANSWER_METHODS == ("boxed", "hash", "anchor")
    token, method = parse_math_answer("I think the answer is: 42\n\n")
    assert (token, method) == ("42", "anchor_inline")
    assert inf.math_extractor("I think the answer is: 42\n\n") == inf.NO_ANSWER
    # the same text without the trailing blank line is method `anchor`, accepted
    assert inf.math_extractor("I think the answer is: 42") == "42"


def test_math_extractor_has_no_answer_when_there_is_no_text():
    assert inf.math_extractor("") == inf.NO_ANSWER
    assert inf.math_extractor("   \n  ") == inf.NO_ANSWER
    assert inf.math_extractor(None) == inf.NO_ANSWER


def test_math_extractor_strips_the_dollar_signs_of_an_expression():
    assert inf.math_extractor("Final answer: $x + 1$") == "x + 1"
    assert inf.math_extractor("Final answer: $(3, \\frac{\\pi}{2})$") == "(3, (pi)/(2))"
    # This is the shape most of the real E3a outputs end on, and the one the
    # boxed placeholder reads as no answer at all.
    assert inf.boxed_extractor("Final answer: $x + 1$") == inf.NO_ANSWER


def test_math_extractor_keeps_the_parsed_token_when_normalising_empties_it():
    """The normaliser drops characters outside its allowed set, so it can empty a
    string the parser did find. That is not "no answer": no answer is reserved
    for the parser finding nothing, so the parsed token is kept instead."""
    assert inf.math_extractor("#### \\%\\%") == "\\%\\%"


def test_bucket_influence_excludes_the_null_arm_and_honours_credit_n():
    b = make_bucket("q", [[1]] * 4,
                    [[boxed("z")] * 4, [boxed("A")] * 4, [boxed("B")] * 4, [boxed("C")] * 4],
                    meta={"null_arm": 0})
    assert inf.bucket_influence(b, inf.boxed_extractor) == pytest.approx(LN3 + 2.0 / 24.0, abs=1e-12)
    b2 = make_bucket("q", [[1]] * 4,
                     [[boxed("A")] * 4, [boxed("B")] * 4, [boxed("C")] * 4, [boxed("D")] * 4],
                     meta={})
    assert inf.bucket_influence(b2, inf.boxed_extractor) == pytest.approx(LN4 + 3.0 / 32.0, abs=1e-12)
    b2["meta"]["credit_n"] = 3
    assert inf.bucket_influence(b2, inf.boxed_extractor) == pytest.approx(LN3 + 2.0 / 24.0, abs=1e-12)
    assert inf.bucket_influence(b2, inf.boxed_extractor, credit_only=False) == pytest.approx(
        LN4 + 3.0 / 32.0, abs=1e-12)


# -----------------------------------------------------------------------------
# 3. the E3a bias map
# -----------------------------------------------------------------------------


@pytest.fixture()
def e3a_report():
    points, skipped = bm.decision_points(e3a_buckets(), inf.boxed_extractor)
    assert sum(skipped.values()) == 0
    return points, bm.bias_map_report(points)


def test_decision_points_carry_the_hand_computed_quantities(e3a_report):
    points, _ = e3a_report
    assert len(points) == 12
    high_infl = LN3 + 2.0 / 24.0
    assert [p["bias"] for p in points[:4]] == [0.0] * 4
    assert [p["bias"] for p in points[4:8]] == [0.75] * 4
    assert [p["bias"] for p in points[8:]] == [0.25] * 4
    assert [p["delta_q"] for p in points[:4]] == [0.0] * 4
    assert [p["delta_q"] for p in points[4:8]] == [0.5] * 4
    assert [p["delta_q"] for p in points[8:]] == [0.0] * 4
    for p in points[:8]:
        assert p["influence"] == pytest.approx(high_infl, abs=1e-12)
    for p in points[8:]:
        assert p["influence"] == 0.0
    # red-team R1: the injected arm does not collapse the first four points
    assert [p["baseline_level"] for p in points[:4]] == [0.5] * 4


def test_decision_points_skip_buckets_without_a_landed_injection():
    good = e3a_buckets()[0]
    no_meta = make_bucket("x", [[1]] * 4, [["a"] * 4] * 4, meta={})
    out_of_range = make_bucket("y", [[1]] * 4, [["a"] * 4] * 4, meta={"null_arm": 9})
    empty_null = make_bucket("z", [[], [1], [1], [1]], [["a"] * 4] * 4, meta={"null_arm": 0})
    only_null = make_bucket("w", [[1], [], [], []], [["a"] * 4] * 4, meta={"null_arm": 0})
    points, skipped = bm.decision_points(
        [good, no_meta, out_of_range, empty_null, only_null], inf.boxed_extractor)
    assert len(points) == 1
    assert skipped == {"no_null_arm": 1, "null_arm_out_of_range": 1,
                       "no_null_returns": 1, "no_real_returns": 1}


def test_bias_map_keys_match_the_hand_computation(e3a_report):
    points, report = e3a_report
    keys = bm.e3a_key_values(report, null_forms=bm.null_forms_paired(points, points))
    value = {k: v["value"] for k, v in keys.items()}

    assert value["E3a.n_points"] == 12
    assert value["E3a.bias.mean"] == pytest.approx(1.0 / 3.0, abs=1e-12)
    # 4 x 0, 4 x 0.75, 4 x 0.25 against zero: t = 3.5456 on 11 degrees of freedom.
    # p is 0.0046, which the middle tier of the p-value convention prints as
    # "< 0.01" (the maintainers' decision; two decimals would read as "p equals zero").
    assert value["E3a.bias.p"] == "$<\\!0.01$"
    assert report["bias"]["p"] == pytest.approx(0.0045872026, abs=1e-9)

    assert value["E3a.strata.high_infl_diff.n"] == 4
    assert value["E3a.strata.high_infl_nodiff.n"] == 4
    assert value["E3a.strata.high_infl_nodiff.zero_share_pct"] == 100.0
    assert value["E3a.strata.high_infl_diff.zero_n"] == 0
    assert value["E3a.strata.low_infl.mean_abs_bias"] == pytest.approx(0.25, abs=1e-12)

    # Mann-Whitney on |bias|, [0.75] x 4 against [0] x 4. Complete separation
    # with ties inside both samples, so scipy takes the tie-corrected normal
    # approximation with continuity correction: z = (16 - 8 - 0.5) / 3.02372.
    assert report["strata"]["mw_p"] == pytest.approx(0.0131238068, abs=1e-9)
    assert value["E3a.strata.mw_p"] == "$= 0.01$"

    assert value["E3a.top_stratum.share_of_points_pct"] == pytest.approx(100.0 / 3.0, abs=1e-12)
    assert value["E3a.top_stratum.share_of_bias_pct"] == pytest.approx(75.0, abs=1e-12)
    assert value["E3a.top_stratum.mean_abs_bias"] == pytest.approx(0.75, abs=1e-12)
    assert value["E3a.rest.n"] == 8
    assert value["E3a.rest.mean_abs_bias"] == pytest.approx(0.125, abs=1e-12)
    assert value["E3a.concentration_factor"] == pytest.approx(6.0, abs=1e-12)
    assert value["E3a.influence.mean_nats"] == pytest.approx(
        (8.0 / 12.0) * (LN3 + 2.0 / 24.0), abs=1e-12)
    assert value["E3a.cutoffs.max_mw_p"] == "$= 0.01$"
    assert value["E3a.null_forms.paired_p"] == "$= 1.00$"
    assert set(value) == set(E3A_KEYS)
    # red-team R1 has no key of its own, so it must be readable in the note
    assert "R1 null arm return: mean 0.1667" in keys["E3a.bias.mean"]["note"]


def test_identical_null_forms_give_a_paired_p_of_one(e3a_report):
    points, _ = e3a_report
    paired = bm.null_forms_paired(points, points)
    assert paired["n_pairs"] == 12
    assert paired["mean_diff"] == 0.0
    # scipy returns nan on a vector of zeros; this work package reports 1.0,
    # which is the value the specification suggested.
    assert paired["p"] == 1.0


def test_null_forms_paired_detects_a_shifted_arm(e3a_report):
    points, _ = e3a_report
    shifted = [dict(p, bias=p["bias"] + 0.1 * (1 + i % 3)) for i, p in enumerate(points)]
    paired = bm.null_forms_paired(points, shifted)
    assert paired["n_pairs"] == 12
    assert paired["mean_diff"] == pytest.approx(0.2, abs=1e-12)
    assert paired["p"] < 0.001


def test_median_split_keeps_the_tied_points_on_the_high_side(e3a_report):
    """Eight of the twelve points share one influence value, which is the median.

    With the default rule (at or above the threshold) they all land in the high
    half, which is what makes the 4 / 4 / 4 design of this fixture work. The
    strict rule empties it; the parameter exists so the maintainers can see both.
    """
    points, _ = e3a_report
    strict = bm.bias_map_report(points, high_side="gt")
    assert strict["strata"]["high_infl_diff"]["n"] == 0
    assert strict["strata"]["high_infl_nodiff"]["n"] == 0
    assert strict["strata"]["low_infl"]["n"] == 12
    with pytest.raises(ValueError):
        bm.bias_map_report(points, high_side="above")


def test_cutoffs_take_the_largest_p_over_the_four_thresholds():
    """Twelve distinct influence values, so the four cutoffs really differ.

    Influence rises with the point index. The top 30% (4 points) holds two with
    a value difference and two without, while the wider cutoffs pull in points
    whose bias separation is weaker, so the reported number is the largest of
    the four p values, not the median split's.
    """
    points = []
    for i in range(12):
        points.append({"question_id": "q%d" % i, "influence": 0.1 * i,
                       "bias": 0.5 if i % 2 else 0.0,
                       "delta_q": 1.0 if i % 2 else 0.0,
                       "baseline_level": 0.0, "n_real": 3})
    report = bm.bias_map_report(points)
    per_cutoff = report["cutoffs"]["per_cutoff"]
    assert sorted(per_cutoff) == [30, 40, 50, 60]
    assert all(not math.isnan(v) for v in per_cutoff.values())
    assert report["cutoffs"]["max_mw_p"] == pytest.approx(max(per_cutoff.values()), abs=0)
    assert per_cutoff[30] > per_cutoff[60]  # fewer points, weaker separation


def test_concentration_factor_is_infinite_when_the_rest_is_flat():
    points = []
    for i in range(6):
        top = i >= 3
        points.append({"question_id": "q%d" % i, "influence": 1.0 if top else 0.0,
                       "bias": 0.4 if top else 0.0, "delta_q": 1.0 if top else 0.0,
                       "baseline_level": 0.0, "n_real": 3})
    report = bm.bias_map_report(points)
    assert math.isinf(report["concentration_factor"])
    keys = bm.e3a_key_values(report)
    assert "E3a.concentration_factor" not in keys  # not finite, so not written


def test_extra_strata_keys_are_opt_in(e3a_report):
    _, report = e3a_report
    default = bm.e3a_key_values(report)
    extra = bm.e3a_key_values(report, extra_strata_keys=True)
    assert set(extra) - set(default) == set(E3A_EXTRA_KEYS)
    assert extra["E3a.strata.high_infl_diff.mean_bias"]["value"] == pytest.approx(0.75, abs=1e-12)
    assert extra["E3a.strata.high_infl_nodiff.zero_n"]["value"] == 4


# -----------------------------------------------------------------------------
# 3b. the degenerate-split warning (2026-09-15)
#
# The first real E3a cell put 148 of 150 points on the high side of the median
# influence, leaving a low stratum of two. That is a reporting problem, not a
# rule problem: the split is fixed in the analysis plan and nothing here moves
# it. What is pinned below is that the summary says so out loud.
# -----------------------------------------------------------------------------


def _degeneracy_line(name, n):
    return ("stratum %s has n=%d points; the median split is degenerate on this data"
            % (name, n))


def test_bias_map_names_every_thin_stratum_and_warns_once_each():
    points, _ = bm.decision_points(e3a_buckets(), inf.boxed_extractor)
    stream = io.StringIO()
    report = bm.bias_map_report(points, stream=stream)

    degenerate = report["strata"]["degenerate"]
    assert set(degenerate) == set(bm.STRATUM_NAMES)          # four points in each
    for name in bm.STRATUM_NAMES:
        assert degenerate[name] == _degeneracy_line(name, 4)
        assert "[bias_map] " + _degeneracy_line(name, 4) in stream.getvalue()
    assert stream.getvalue().count("[bias_map]") == 3


def test_bias_map_is_quiet_when_no_stratum_is_thin():
    points, _ = bm.decision_points(e3a_buckets(per_stratum=bm.MIN_STRATUM_N),
                                   inf.boxed_extractor)
    stream = io.StringIO()
    report = bm.bias_map_report(points, stream=stream)

    assert report["strata"]["degenerate"] == {}
    assert stream.getvalue() == ""
    keys = bm.e3a_key_values(report)
    assert not any("degenerate" in entry["note"] for entry in keys.values())


def test_a_thin_stratum_reaches_the_note_of_the_keys_read_off_it():
    """The warning has to ride into the summary, since whoever reads the summary
    later is not the one who watched the stderr of the run."""
    points, _ = bm.decision_points(e3a_buckets(), inf.boxed_extractor)
    report = bm.bias_map_report(points, stream=io.StringIO())
    keys = bm.e3a_key_values(report, null_forms=bm.null_forms_paired(points, points))

    assert keys["E3a.strata.low_infl.mean_abs_bias"]["note"].endswith(
        _degeneracy_line("low_infl", 4))
    assert keys["E3a.strata.high_infl_diff.n"]["note"] == _degeneracy_line("high_infl_diff", 4)
    # the top stratum is the high-influence-and-differed stratum under another name
    assert _degeneracy_line("high_infl_diff", 4) in keys["E3a.top_stratum.mean_abs_bias"]["note"]
    assert _degeneracy_line("high_infl_diff", 4) in keys["E3a.concentration_factor"]["note"]
    # the Mann-Whitney key compares the two high strata, so it carries both
    for name in ("high_infl_diff", "high_infl_nodiff"):
        assert _degeneracy_line(name, 4) in keys["E3a.strata.mw_p"]["note"]
    # n_points stands for the whole point set and carries every warning there is
    for name in bm.STRATUM_NAMES:
        assert _degeneracy_line(name, 4) in keys["E3a.n_points"]["note"]
    # a key that reads no stratum is untouched
    assert "degenerate" not in keys["E3a.influence.mean_nats"]["note"]


def test_the_warning_does_not_move_the_split_or_drop_a_key():
    """Same points, same numbers: only the notes differ between a run that warns
    and the values themselves."""
    points, _ = bm.decision_points(e3a_buckets(), inf.boxed_extractor)
    report = bm.bias_map_report(points, stream=io.StringIO())
    keys = bm.e3a_key_values(report)

    assert set(keys) == set(E3A_KEYS) - {"E3a.null_forms.paired_p"}
    assert keys["E3a.strata.high_infl_diff.n"]["value"] == 4
    assert keys["E3a.strata.high_infl_nodiff.n"]["value"] == 4
    assert keys["E3a.strata.low_infl.mean_abs_bias"]["value"] == pytest.approx(0.25, abs=1e-12)


# -----------------------------------------------------------------------------
# 4. the E5 coupling contrast
# -----------------------------------------------------------------------------


def test_splithalf_evenodd_hand_values_and_exclusions():
    one = make_bucket("q", REL_ONE, [[""] * 4] * 4)
    half = make_bucket("q", REL_HALF, [[""] * 4] * 4)
    assert cp._splithalf_evenodd(one) == pytest.approx(1.0, abs=1e-12)
    assert cp._splithalf_evenodd(half) == pytest.approx(0.5, abs=1e-12)
    too_few = make_bucket("q", [[1, 0], [0, 1]], [[""] * 2] * 2)
    assert cp._splithalf_evenodd(too_few) is None
    too_short = make_bucket("q", [[1], [0], [1]], [[""]] * 3)
    assert cp._splithalf_evenodd(too_short) is None
    flat = make_bucket("q", [[1, 1], [1, 1], [1, 1]], [[""] * 2] * 3)
    assert cp._splithalf_evenodd(flat) is None


def test_coupling_directions_and_wrong_upstream_counts():
    gold = {"q%d" % i: "G" for i in range(10)}

    def is_correct(symbol, qid):
        return symbol == gold.get(qid)

    sft, _ = cp.arm_scan(e5_arm("sft"), inf.boxed_extractor, is_correct)
    c3, _ = cp.arm_scan(e5_arm("c3"), inf.boxed_extractor, is_correct)
    report = cp.coupling_report(sft, c3)

    assert report["n_questions"] == 10
    # reliability: identical returns in both arms, five buckets at 1.0 and five
    # at 0.5, so the means agree and the paired difference is exactly zero.
    assert report["rel"]["mean_a"] == pytest.approx(0.75, abs=1e-12)
    assert report["rel"]["mean_b"] == pytest.approx(0.75, abs=1e-12)
    assert report["rel"]["diff"] == 0.0
    assert report["rel"]["p"] == 1.0

    # influence falls: distinct downstream answers per alternative under sft,
    # one shared answer distribution under c3.
    assert report["influence"]["mean_a"] == pytest.approx(LN4 + 3.0 / 32.0, abs=1e-9)
    assert report["influence"]["mean_b"] == pytest.approx(0.5 * 0.0 + 0.5 * -0.09375, abs=1e-12)
    assert report["influence"]["diff"] < 0
    assert report["influence"]["ci_hi"] < 0
    assert report["influence"]["p"] < 0.001

    # correction rises: 0 and 0.125 under sft against 1.0 and 0.75 under c3.
    assert report["correction"]["mean_a"] == pytest.approx(0.0625, abs=1e-12)
    assert report["correction"]["mean_b"] == pytest.approx(0.875, abs=1e-12)
    assert report["correction"]["diff"] == pytest.approx(0.8125, abs=1e-12)
    assert report["correction"]["ci_lo"] == pytest.approx(0.8125 - 2.262157 * 0.0625, abs=1e-5)
    assert report["correction"]["ci_hi"] == pytest.approx(0.8125 + 2.262157 * 0.0625, abs=1e-5)
    assert report["correction"]["ci_lo"] > 0
    assert report["correction"]["p"] < 0.001

    # counted per alternative: two wrong upstream messages per bucket under sft,
    # one under c3, over ten buckets.
    assert report["correction"]["n_upstream_wrong_a"] == 20
    assert report["correction"]["n_upstream_wrong_b"] == 10
    assert report["correction"]["n_excluded"] == 0


def test_influence_is_reported_below_zero_when_the_correction_bites():
    gold = {"q1": "G"}

    def is_correct(symbol, qid):
        return symbol == gold.get(qid)

    arm, _ = cp.arm_scan([e5_arm("c3")[1]], inf.boxed_extractor, is_correct)
    assert arm["q1"]["influence"] == pytest.approx(-0.09375, abs=1e-12)


def test_identical_arms_show_no_significant_difference_anywhere():
    gold = {"q%d" % i: "G" for i in range(10)}

    def is_correct(symbol, qid):
        return symbol == gold.get(qid)

    arm, _ = cp.arm_scan(e5_arm("sft"), inf.boxed_extractor, is_correct)
    report = cp.coupling_report(arm, arm)
    for name in ("rel", "influence", "correction"):
        assert report[name]["diff"] == 0.0
        assert report[name]["p"] == 1.0
        assert report[name]["wilcoxon_p"] == 1.0
        assert report[name]["ci_lo"] == 0.0 and report[name]["ci_hi"] == 0.0
    keys = cp.e5_key_values(report)
    assert keys["E5.rel.diff_p"]["value"] == "$= 1.00$"
    assert keys["E5.influence.p"]["value"] == "$= 1.00$"
    assert keys["E5.correction.p"]["value"] == "$= 1.00$"
    assert set(keys) == set(E5_KEYS)


def test_correction_is_undefined_without_a_wrong_upstream_alternative():
    def always_right(symbol, qid):
        return True

    b = e5_arm("sft", n_questions=1)[0]
    rate, n_wrong, trials = cp.bucket_correction(b, inf.boxed_extractor, always_right)
    assert math.isnan(rate) and n_wrong == 0 and trials == 0

    def gold_only(symbol, qid):
        return symbol == "G"

    mixed, _ = cp.arm_scan(e5_arm("sft", n_questions=2), inf.boxed_extractor, gold_only)
    clean, _ = cp.arm_scan(e5_arm("sft", n_questions=2), inf.boxed_extractor, always_right)
    report = cp.coupling_report(clean, mixed)
    assert report["n_questions"] == 2
    assert report["correction"]["n"] == 0        # no question defined on both sides
    assert report["correction"]["n_excluded"] == 2
    keys = cp.e5_key_values(report)
    assert "E5.correction.sft" not in keys       # nan is dropped, never written as null
    # the wrong-alternative counts are integers, so only the zero pair count
    # keeps them out
    assert "E5.correction.n_upstream_wrong.sft" not in keys
    assert keys["E5.n_questions"]["value"] == 2


# -----------------------------------------------------------------------------
# 5. the two command line aggregations
# -----------------------------------------------------------------------------


def test_format_p_branches():
    """The three tiers of the maintainers' decision (analysis plan revision 13),
    with both boundaries: 0.001 belongs to the middle tier and 0.01 to the last."""
    assert inf.format_p(0.0) == "$<\\!0.001$"
    assert inf.format_p(0.0005) == "$<\\!0.001$"
    assert inf.format_p(0.0009999) == "$<\\!0.001$"
    assert inf.format_p(0.001) == "$<\\!0.01$"
    assert inf.format_p(0.0046) == "$<\\!0.01$"
    assert inf.format_p(0.009999) == "$<\\!0.01$"
    assert inf.format_p(0.01) == "$= 0.01$"
    assert inf.format_p(0.3) == "$= 0.30$"
    assert inf.format_p(1.0) == "$= 1.00$"
    assert inf.format_p(float("nan")) is None
    assert inf.format_p(None) is None


def test_build_summary_refuses_unknown_and_verdict_keys(tmp_path, capsys):
    """Contract revision 2: an unknown key and a verdict key are refused, one
    line each on stderr, and the document is written without them."""
    manifest_path = tmp_path / "manifest.json"
    manifest = write_manifest(str(manifest_path), E3A_KEYS)
    doc = inf.build_summary(
        "E3a", "s",
        {"E3a.not_a_key": {"value": 1},
         "E3a.bias_concentration": {"value": "yes"},
         "E3a.n_points": {"value": 12, "n": 12, "source": ["x"]}},
        manifest)
    assert set(doc["keys"]) == {"E3a.n_points"}
    err = capsys.readouterr().err
    assert "skip E3a.not_a_key: not a manifest key" in err
    assert "skip E3a.bias_concentration: verdict key" in err


def test_aggregate_e3a_cli(tmp_path, capsys):
    root = tmp_path / "20_data" / "results" / "E3a"
    write_jsonl(str(root / "a3" / "4b" / "empty" / "buckets.jsonl"), e3a_buckets())
    write_jsonl(str(root / "a3" / "4b" / "deleted" / "buckets.jsonl"), e3a_buckets())
    manifest_path = tmp_path / "manifest.json"
    write_manifest(str(manifest_path), E3A_KEYS + E5_KEYS)
    out = tmp_path / "E3a_summary.json"

    rc = aggregate_e3a.main(["--results", str(root), "--manifest", str(manifest_path),
                             "--out", str(out), "--extractor", "boxed"])
    assert rc == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["experiment"] == "E3a"
    assert doc["generated_by"]["script"] == "c3/analysis/rebuild/aggregate_e3a.py"
    assert set(doc["keys"]) <= set(E3A_KEYS + E5_KEYS)
    assert set(doc["keys"]) == set(E3A_KEYS)
    assert doc["keys"]["E3a.n_points"]["value"] == 12
    assert doc["keys"]["E3a.n_points"]["n"] == 12
    assert doc["keys"]["E3a.n_points"]["unit"] == "points"
    assert doc["keys"]["E3a.n_points"]["source"] == [
        "20_data/results/E3a/a3/4b/empty/buckets.jsonl"]
    assert doc["keys"]["E3a.null_forms.paired_p"]["source"] == [
        "20_data/results/E3a/a3/4b/empty/buckets.jsonl",
        "20_data/results/E3a/a3/4b/deleted/buckets.jsonl"]
    assert doc["keys"]["E3a.concentration_factor"]["value"] == pytest.approx(6.0, abs=1e-12)


def test_aggregate_e3a_keeps_going_when_the_manifest_lacks_a_key(tmp_path, capsys):
    """Results layout revision 2: a key the manifest does not define is
    refused with one line on stderr, the other keys are still written and the
    command still exits zero."""
    root = tmp_path / "E3a"
    write_jsonl(str(root / "a3" / "4b" / "empty" / "buckets.jsonl"), e3a_buckets())
    manifest_path = tmp_path / "manifest.json"
    write_manifest(str(manifest_path), [k for k in E3A_KEYS if k != "E3a.n_points"])
    out = tmp_path / "summary.json"

    rc = aggregate_e3a.main(["--results", str(root), "--manifest", str(manifest_path),
                             "--out", str(out)])
    assert rc == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert "E3a.n_points" not in doc["keys"]
    assert len(doc["keys"]) == len(E3A_KEYS) - 2      # n_points refused, paired_p has no arm
    assert "skip E3a.n_points: not a manifest key" in capsys.readouterr().err


def test_aggregate_e3a_without_the_placeholder_arm(tmp_path, capsys):
    """With neither the current directory name nor its legacy alias on
    disk, the message has to name both, or the operator looks for the wrong
    directory."""
    root = tmp_path / "E3a"
    write_jsonl(str(root / "a3" / "4b" / "empty" / "buckets.jsonl"), e3a_buckets())
    manifest_path = tmp_path / "manifest.json"
    write_manifest(str(manifest_path), E3A_KEYS)
    out = tmp_path / "summary.json"
    rc = aggregate_e3a.main(["--results", str(root), "--manifest", str(manifest_path),
                             "--out", str(out)])
    assert rc == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert "E3a.null_forms.paired_p" not in doc["keys"]
    err = capsys.readouterr().err
    assert "no placeholder arm (nor its legacy name deleted) under" in err


def test_aggregate_e3a_prefers_the_placeholder_arm(tmp_path, capsys):
    """`placeholder` is the arm the cell driver writes (results layout
    revision 3), so it is the one looked for, and finding it says
    nothing about a legacy name."""
    root = tmp_path / "20_data" / "results" / "E3a"
    write_jsonl(str(root / "a3" / "4b" / "empty" / "buckets.jsonl"), e3a_buckets())
    write_jsonl(str(root / "a3" / "4b" / "placeholder" / "buckets.jsonl"), e3a_buckets())
    manifest_path = tmp_path / "manifest.json"
    write_manifest(str(manifest_path), E3A_KEYS)
    out = tmp_path / "summary.json"

    rc = aggregate_e3a.main(["--results", str(root), "--manifest", str(manifest_path),
                             "--out", str(out), "--extractor", "boxed"])
    assert rc == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["keys"]["E3a.null_forms.paired_p"]["source"] == [
        "20_data/results/E3a/a3/4b/empty/buckets.jsonl",
        "20_data/results/E3a/a3/4b/placeholder/buckets.jsonl"]
    err = capsys.readouterr().err
    assert "legacy arm name" not in err


def test_aggregate_e3a_reads_the_legacy_arm_name_and_says_so(tmp_path, capsys):
    """A tree written before the rename still aggregates, with one
    line on stderr, because a summary built off a legacy tree is worth noticing."""
    root = tmp_path / "20_data" / "results" / "E3a"
    write_jsonl(str(root / "a3" / "4b" / "empty" / "buckets.jsonl"), e3a_buckets())
    write_jsonl(str(root / "a3" / "4b" / "deleted" / "buckets.jsonl"), e3a_buckets())
    manifest_path = tmp_path / "manifest.json"
    write_manifest(str(manifest_path), E3A_KEYS)
    out = tmp_path / "summary.json"

    rc = aggregate_e3a.main(["--results", str(root), "--manifest", str(manifest_path),
                             "--out", str(out), "--extractor", "boxed"])
    assert rc == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["keys"]["E3a.null_forms.paired_p"]["source"][1] == \
        "20_data/results/E3a/a3/4b/deleted/buckets.jsonl"
    assert "using the legacy arm name deleted" in capsys.readouterr().err


def test_aggregate_e3a_reports_a_missing_tree(tmp_path, capsys):
    manifest_path = tmp_path / "manifest.json"
    write_manifest(str(manifest_path), E3A_KEYS)
    rc = aggregate_e3a.main(["--results", str(tmp_path / "nothing"),
                             "--manifest", str(manifest_path),
                             "--out", str(tmp_path / "summary.json")])
    assert rc == 2
    assert "no empty arm" in capsys.readouterr().err


def test_aggregate_e5_cli(tmp_path):
    root = tmp_path / "20_data" / "results" / "E5"
    write_jsonl(str(root / "sft" / "buckets.jsonl"), e5_arm("sft"))
    write_jsonl(str(root / "c3_s0" / "buckets.jsonl"), e5_arm("c3"))
    gold = tmp_path / "gold.jsonl"
    with open(gold, "w", encoding="utf-8") as fh:
        for i in range(10):
            fh.write(json.dumps({"question_id": "q%d" % i, "gold_answer": "G"}) + "\n")
    manifest_path = tmp_path / "manifest.json"
    write_manifest(str(manifest_path), E3A_KEYS + E5_KEYS)
    out = tmp_path / "E5_summary.json"

    rc = aggregate_e5.main(["--results", str(root), "--manifest", str(manifest_path),
                            "--out", str(out), "--gold", str(gold)])
    assert rc == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["experiment"] == "E5"
    assert set(doc["keys"]) <= set(E3A_KEYS + E5_KEYS)
    assert set(doc["keys"]) == set(E5_KEYS)
    assert doc["keys"]["E5.n_questions"]["value"] == 10
    assert doc["keys"]["E5.correction.diff"]["value"] == pytest.approx(0.8125, abs=1e-12)
    assert doc["keys"]["E5.correction.n_upstream_wrong.sft"]["value"] == 20
    assert doc["keys"]["E5.correction.n_upstream_wrong.c3"]["value"] == 10
    assert doc["keys"]["E5.influence.diff"]["value"] < 0
    assert doc["keys"]["E5.influence.p"]["value"] == "$<\\!0.001$"
    assert doc["keys"]["E5.rel.sft"]["source"] == [
        "20_data/results/E5/sft/buckets.jsonl", "20_data/results/E5/c3_s0/buckets.jsonl"]


def test_aggregate_e5_gold_shapes_and_missing_gold(tmp_path, capsys):
    root = tmp_path / "E5"
    write_jsonl(str(root / "sft" / "buckets.jsonl"), e5_arm("sft"))
    write_jsonl(str(root / "c3_s0" / "buckets.jsonl"), e5_arm("c3"))
    gold = tmp_path / "gold.jsonl"
    with open(gold, "w", encoding="utf-8") as fh:
        for i in range(9):  # q9 has no gold answer
            row = {"question_id": "q%d" % i, "gold": "G"} if i % 2 else {"q%d" % i: "G"}
            fh.write(json.dumps(row) + "\n")
    parsed = aggregate_e5.read_gold(str(gold))
    assert len(parsed) == 9 and parsed["q0"] == "G" and parsed["q1"] == "G"

    manifest_path = tmp_path / "manifest.json"
    write_manifest(str(manifest_path), E5_KEYS)
    out = tmp_path / "summary.json"
    rc = aggregate_e5.main(["--results", str(root), "--manifest", str(manifest_path),
                            "--out", str(out), "--gold", str(gold)])
    assert rc == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["keys"]["E5.n_questions"]["value"] == 9
    assert "no gold answer" in doc["keys"]["E5.n_questions"]["note"]
    assert "dropped for having no gold answer" in capsys.readouterr().err


def test_aggregate_e5_rejects_an_unreadable_gold_row(tmp_path, capsys):
    root = tmp_path / "E5"
    write_jsonl(str(root / "sft" / "buckets.jsonl"), e5_arm("sft"))
    write_jsonl(str(root / "c3_s0" / "buckets.jsonl"), e5_arm("c3"))
    gold = tmp_path / "gold.jsonl"
    gold.write_text(json.dumps({"qid": "q0", "value": "G"}) + "\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    write_manifest(str(manifest_path), E5_KEYS)
    rc = aggregate_e5.main(["--results", str(root), "--manifest", str(manifest_path),
                            "--out", str(tmp_path / "summary.json"), "--gold", str(gold)])
    assert rc == 2
    assert "cannot tell which field" in capsys.readouterr().err


def test_key_names_are_the_spec_set(tmp_path):
    """The two key lists in this file are the spelling the modules emit."""
    points, _ = bm.decision_points(e3a_buckets(), inf.boxed_extractor)
    report = bm.bias_map_report(points)
    keys = bm.e3a_key_values(report, null_forms=bm.null_forms_paired(points, points))
    assert sorted(keys) == sorted(E3A_KEYS)
    assert not (set(E3A_KEYS) & set(E3A_EXTRA_KEYS))
