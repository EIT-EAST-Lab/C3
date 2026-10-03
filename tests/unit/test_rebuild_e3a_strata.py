# tests/unit/test_rebuild_e3a_strata.py
"""The stratified E3a ablation bias (Table 9) and its independent recheck.

The fixture is the hand computable bias map of `test_rebuild_bias_coupling.py`:
four decision points of high influence and no value difference (bias exactly
zero), four of high influence where the alternatives differ (bias 0.75 each)
and four of low influence. What is pinned: the Table 9 keys come out of the
aggregation's own code with the values the fixture implies, the numpy recheck
agrees with them, a disagreement stops the write, and the command line writes a
standalone summary.

    python -m pytest tests/unit/test_rebuild_e3a_strata.py -q
"""

from __future__ import annotations

import json

import pytest

from c3.analysis.rebuild import bias_map as bm
from c3.analysis.rebuild import e3a_strata as es
from c3.analysis.rebuild import influence as inf


def _boxed(text):
    return "\\boxed{%s}" % text


def _bucket(qid, returns, nexts):
    return {
        "bucket_id": "%s-b0" % qid,
        "ctx_hash": 0,
        "target_role": "Reasoner",
        "question_id": qid,
        "restart": {"roles_topo": ["Reasoner", "Actor"], "role_outputs_prefix": {}},
        "candidates": [
            {"j": j, "action_text": "", "returns": [float(r) for r in rets], "next_actions": list(nx)}
            for j, (rets, nx) in enumerate(zip(returns, nexts))
        ],
        "meta": {"null_arm": 0, "null_form": "empty", "workflow": "a3", "model": "4b"},
    }


def _buckets(per_stratum=4):
    distinct = [[_boxed("z")] * 4, [_boxed("A")] * 4, [_boxed("B")] * 4, [_boxed("C")] * 4]
    shared = [[_boxed("Z")] * 4] * 4
    out = []
    for i in range(per_stratum):
        out.append(_bucket("A%d" % i, [[1, 1, 0, 0]] * 4, distinct))
    for i in range(per_stratum):
        out.append(_bucket("B%d" % i, [[0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 0, 0]], distinct))
    for i in range(per_stratum):
        out.append(_bucket("C%d" % i, [[0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]], shared))
    return out


def _points():
    points, skipped = bm.decision_points(_buckets(), inf.boxed_extractor)
    assert sum(skipped.values()) == 0
    return points


def test_the_table9_keys_have_the_values_the_fixture_implies():
    keys = es.strata_keys(_points())
    assert set(keys) == set(es.TABLE9_KEYS)
    assert keys["E3a.strata.high_infl_diff.n"]["value"] == 4
    assert keys["E3a.strata.high_infl_nodiff.n"]["value"] == 4
    assert keys["E3a.strata.high_infl_diff.mean_bias"]["value"] == pytest.approx(0.75)
    assert keys["E3a.strata.high_infl_diff.median"]["value"] == pytest.approx(0.75)
    assert keys["E3a.strata.high_infl_diff.zero_n"]["value"] == 0
    assert keys["E3a.strata.high_infl_nodiff.mean_bias"]["value"] == pytest.approx(0.0)
    assert keys["E3a.strata.high_infl_nodiff.iqr_lo"]["value"] == pytest.approx(0.0)
    assert keys["E3a.strata.high_infl_nodiff.iqr_hi"]["value"] == pytest.approx(0.0)
    assert keys["E3a.strata.high_infl_nodiff.zero_n"]["value"] == 4


def test_the_numpy_recheck_agrees_and_a_disagreement_is_named():
    points = _points()
    keys = es.strata_keys(points)
    again = es.independent_strata(points)
    assert es.check(keys, again) == []
    again["E3a.strata.high_infl_diff.median"] += 0.01
    assert es.check(keys, again) == ["E3a.strata.high_infl_diff.median"]


def test_the_command_line_writes_a_standalone_summary(tmp_path, capsys):
    cell = tmp_path / "E3a" / "a3" / "4b"
    (cell / "empty").mkdir(parents=True)
    (cell / "empty" / "buckets.jsonl").write_text(
        "".join(json.dumps(b) + "\n" for b in _buckets()), encoding="utf-8")
    out = tmp_path / "out.json"
    assert es.main(["--results", str(tmp_path / "E3a"), "--out", str(out), "--extractor", "boxed"]) == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["experiment"] == "E3a"
    assert doc["n_points"] == 12
    assert list(doc["keys"]) == list(es.TABLE9_KEYS)
    for entry in doc["keys"].values():
        assert entry["source"] and entry["source"][0].endswith("empty/buckets.jsonl")


def test_a_tree_without_the_empty_arm_is_an_error(tmp_path):
    assert es.main(["--results", str(tmp_path), "--out", str(tmp_path / "o.json")]) == 2
