# tests/unit/test_rebuild_e1_signal_noise.py
"""The signal and noise decomposition of the E1 cells, and the reliability it implies.

Pinned here: the decomposition on a cell small enough to do by hand, the
skipping rule for a decision point with too few usable alternatives, that the
two terms come back on synthetic data whose true signal and noise are known,
the implied reliability S / (S + 2N), and the summary the command line writes,
including the deviation key that needs the E1 summary.

    python -m pytest tests/unit/test_rebuild_e1_signal_noise.py -q
"""

from __future__ import annotations

import json
import random
import statistics

import pytest

from c3.analysis.rebuild import e1_signal_noise as sn


def _row(qid, returns_by_alt):
    return {"question_id": qid, "candidates": [{"returns": list(r)} for r in returns_by_alt]}


def _write_cell(root, wf, rows, model="4b", n=4):
    path = root / wf / model / ("sweep_n%d" % n) / "buckets.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_the_decomposition_by_hand():
    rows = [_row("q0", [[1, 1, 0, 0], [1, 1, 1, 1]])]
    signal, noise, n = sn.decompose(rows)
    # means 0.5 and 1.0: variance across them (ddof 1) is 0.125;
    # within-alternative variances 1/3 and 0, each over 4 continuations: noise = (1/12 + 0) / 2.
    assert n == 1
    assert noise == pytest.approx((1.0 / 3.0 / 4.0) / 2.0)
    assert signal == pytest.approx(0.125 - noise)


def test_a_point_without_two_usable_alternatives_is_skipped():
    rows = [
        _row("q0", [[1, 0], [0]]),
        _row("q1", [[1, 0, 1], [0, 0, 1]]),
    ]
    _, _, n = sn.decompose(rows)
    assert n == 1
    with pytest.raises(ValueError):
        sn.decompose([_row("q0", [[1], [0]])])


def test_the_two_terms_come_back_on_synthetic_data():
    rng = random.Random(0)
    true_signal, per_draw_var, c = 0.04, 0.16, 4
    rows = []
    for q in range(4000):
        alts = []
        for _ in range(4):
            mu = rng.gauss(0.0, true_signal ** 0.5)
            alts.append([rng.gauss(mu, per_draw_var ** 0.5) for _ in range(c)])
        rows.append(_row("q%d" % q, alts))
    signal, noise, n = sn.decompose(rows)
    assert n == 4000
    assert noise == pytest.approx(per_draw_var / c, rel=0.05)
    assert signal == pytest.approx(true_signal, rel=0.1)


def test_the_implied_reliability_is_s_over_s_plus_2n():
    assert sn.implied_reliability(1.0, 1.0) == pytest.approx(1.0 / 3.0)
    assert sn.implied_reliability(0.3, 0.0) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        sn.implied_reliability(-1.0, 0.25)


def test_the_command_line_writes_every_workflow_and_the_deviation(tmp_path, capsys):
    results = tmp_path / "E1"
    rows = [_row("q%d" % i, [[1, 0, 1, 1], [0, 0, 1, 0], [1, 1, 1, 1], [0, 0, 0, 1]]) for i in range(3)]
    for wf in sn.WORKFLOWS:
        _write_cell(results, wf, rows)
    e1 = {"keys": {"E1.per_decision_n4.%s.4b.rel" % wf: {"value": 0.5} for wf in sn.WORKFLOWS}}
    e1_path = tmp_path / "e1_summary.json"
    e1_path.write_text(json.dumps(e1), encoding="utf-8")
    out = tmp_path / "out" / "summary.json"

    assert sn.main(["--results", str(results), "--out", str(out), "--e1-summary", str(e1_path)]) == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    keys = doc["keys"]
    assert doc["experiment"] == "E1"
    for wf in sn.WORKFLOWS:
        s = keys["E1.signal_noise.%s.4b.signal_var" % wf]["value"]
        nv = keys["E1.signal_noise.%s.4b.noise_var" % wf]["value"]
        assert keys["E1.predicted_rel.%s.4b" % wf]["value"] == pytest.approx(s / (s + 2 * nv))
        assert keys["E1.predicted_rel.%s.4b" % wf]["n"] == 3
    implied = keys["E1.predicted_rel.a2.4b"]["value"]
    assert keys["E1.predicted_rel.max_abs_dev"]["value"] == pytest.approx(abs(implied - 0.5))
    assert keys["E1.predicted_rel.max_abs_dev"]["n"] == len(sn.WORKFLOWS)


def test_a_missing_cell_is_skipped_and_no_deviation_is_claimed(tmp_path, capsys):
    results = tmp_path / "E1"
    rows = [_row("q0", [[1, 0], [0, 0], [1, 1], [0, 1]])]
    _write_cell(results, "a2", rows)
    keys = sn.build_keys(str(results), e1_summary={"keys": {}})
    assert "E1.signal_noise.a2.4b.signal_var" in keys
    assert "E1.signal_noise.a3.4b.signal_var" not in keys
    assert "E1.predicted_rel.max_abs_dev" not in keys
    err = capsys.readouterr().err
    assert "skip a3" in err


def test_an_empty_tree_is_an_error(tmp_path):
    assert sn.main(["--results", str(tmp_path), "--out", str(tmp_path / "s.json")]) == 2
