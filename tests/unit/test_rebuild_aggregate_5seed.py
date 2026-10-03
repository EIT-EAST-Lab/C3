"""CPU tests of c3.analysis.rebuild.aggregate_5seed on synthetic evaluation records.

The records are written in the shape scripts/70_rebuild/final_eval.py writes
(schema c3.final_eval.record/1), under the layout the aggregation reads, with
accuracies chosen so that every number of Tables 2, 8 and 10 can be checked by
hand or against scipy directly.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Dict

import pytest
from scipy import stats

from c3.analysis.rebuild import aggregate_5seed as a5
from c3.analysis.rebuild.aggregate_e2 import EXIT_BROKEN_RECORDS
from c3.analysis.rebuild.summary import RESULTS_KEYS_V3

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = REPO_ROOT / RESULTS_KEYS_V3

SUITES = {"math500": "MATH500", "aime25": "AIME25", "cmath": "CMATH-test", "gsm8k": "GSM8K-test"}

#: Percent accuracies per (method, bench), one value per training seed 0..4.
A2: Dict[str, Dict[str, list]] = {
    "mappo": {"math500": [69.0, 70.0, 68.4, 69.6, 69.5], "aime25": [3.33, 3.33, 3.33, 3.33, 3.33],
              "cmath": [95.1, 95.3, 95.5, 95.2, 95.4], "gsm8k": [92.8, 93.0, 92.9, 92.9, 92.8]},
    "magrpo": {"math500": [74.2, 74.8, 74.4, 74.9, 74.2], "aime25": [3.33, 6.67, 6.67, 3.33, 6.67],
               "cmath": [96.0, 96.2, 95.9, 96.3, 96.1], "gsm8k": [93.1, 93.6, 93.3, 93.7, 93.4]},
    "c3": {"math500": [82.6, 83.4, 82.0, 83.2, 82.8], "aime25": [6.67, 10.0, 6.67, 10.0, 6.67],
           "cmath": [96.2, 96.5, 96.1, 96.4, 96.3], "gsm8k": [93.3, 93.6, 93.2, 93.5, 93.4]},
}


def _record(values: Dict[str, float], *, k: int = 1, generate_max_len: int = 512, ok: bool = True) -> dict:
    suites = {}
    for bench, pct in values.items():
        suites[SUITES[bench]] = {"accuracy": pct / 100.0, "k": k, "n_questions": 30 if bench == "aime25" else 500}
    return {
        "schema": "c3.final_eval.record/1",
        "ok": ok,
        "method": "C3",
        "generate_max_len": generate_max_len,
        "suites": suites,
    }


def _write(root: Path, arm: str, method: str, seed: int, record: dict, eval_seed: int = 0) -> Path:
    path = root / arm / method / ("train_s%d" % seed) / ("eval_s%d" % eval_seed) / "final_eval_record.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def _tree(root: Path, arm: str = "a2_4b", data=A2) -> None:
    for method, benches in data.items():
        for seed in range(5):
            _write(root, arm, method, seed, _record({b: v[seed] for b, v in benches.items()}))


def _run(tmp_path: Path, *extra: str) -> dict:
    out = tmp_path / "summary.json"
    assert a5.main(["--results", str(tmp_path / "records"), "--out", str(out), *extra]) == 0
    return json.loads(out.read_text(encoding="utf-8"))


def test_means_stds_and_the_average_column(tmp_path: Path) -> None:
    _tree(tmp_path / "records")
    doc = _run(tmp_path)
    rows = doc["tables"]["a2_4b"]["rows"]
    for method, benches in A2.items():
        for bench, values in benches.items():
            cell = rows[method][bench]
            assert cell["mean"] == pytest.approx(statistics.mean(values))
            assert cell["std"] == pytest.approx(statistics.stdev(values))
            assert cell["n"] == 5
        avg = [statistics.mean([benches[b][s] for b in ("math500", "cmath", "gsm8k")]) for s in range(5)]
        assert rows[method]["avg3"]["mean"] == pytest.approx(statistics.mean(avg))
        assert rows[method]["avg3"]["std"] == pytest.approx(statistics.stdev(avg))
    keys = doc["keys"]
    assert keys["E2main.a2_4b.math500.c3.mean"]["value"] == pytest.approx(82.8)
    assert keys["E2main.a2_4b.math500.c3.mean"]["unit"] == "%"
    assert keys["E2main.a2_4b.avg3.mappo.std"]["n"] == 5
    assert doc["problems"] == []


def test_the_paired_test_and_holm_match_scipy(tmp_path: Path) -> None:
    _tree(tmp_path / "records")
    doc = _run(tmp_path)
    sig = doc["tables"]["significance"]
    assert sig["strongest_baseline"] == "magrpo"
    assert doc["keys"]["E2main.a2_4b.strongest_baseline.name"]["value"] == "MAGRPO"

    raw = {}
    for bench in ("math500", "aime25", "cmath", "gsm8k"):
        c3, best = A2["c3"][bench], A2["magrpo"][bench]
        diffs = [a - b for a, b in zip(c3, best)]
        pair = sig["benches"][bench]
        assert pair["diff"] == pytest.approx(statistics.mean(diffs))
        half = stats.t.ppf(0.975, 4) * statistics.stdev(diffs) / math.sqrt(5)
        assert pair["ci_lo"] == pytest.approx(statistics.mean(diffs) - half)
        assert pair["ci_hi"] == pytest.approx(statistics.mean(diffs) + half)
        raw[bench] = stats.ttest_rel(c3, best).pvalue
        assert pair["p"] == pytest.approx(raw[bench])
        assert pair["seeds"] == [0, 1, 2, 3, 4]

    # Holm by hand: sort ascending, scale the i-th smallest by m - i, running maximum.
    order = sorted(raw, key=raw.get)
    running = 0.0
    for rank, bench in enumerate(order):
        running = max(running, min(1.0, raw[bench] * (4 - rank)))
        assert sig["benches"][bench]["p_holm"] == pytest.approx(running)
        assert sig["benches"][bench]["holm_family_size"] == 4
    assert doc["keys"]["E2main.a2_4b.math500.c3_minus_best.p_holm"]["value"] == "< 0.001"


def test_every_key_written_is_a_key_of_the_paper(tmp_path: Path) -> None:
    _tree(tmp_path / "records")
    _tree(tmp_path / "records", arm="a3_4b")
    doc = _run(tmp_path)
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    assert doc["keys"], "nothing was written"
    assert set(doc["keys"]) <= set(schema)
    for key, entry in doc["keys"].items():
        assert entry["unit"] == schema[key]["unit"], key
    # Table 8 has means and stds; the significance test is Table 10, the 2-agent workflow only.
    assert "E2main.a3_4b.math500.c3.mean" in doc["keys"]
    assert not any(k.startswith("E2main.a3_4b") and "c3_minus_best" in k for k in doc["keys"])
    # Every cell of Tables 2 and 8 the paper prints is written.
    for arm in ("a2_4b", "a3_4b"):
        for method in ("mappo", "magrpo", "c3"):
            for bench in ("math500", "aime25", "cmath", "gsm8k", "avg3"):
                for stat in ("mean", "std"):
                    assert "E2main.%s.%s.%s.%s" % (arm, bench, method, stat) in doc["keys"]


def test_the_markdown_tables_print_at_the_papers_precision(tmp_path: Path) -> None:
    _tree(tmp_path / "records")
    table = tmp_path / "tables.md"
    _run(tmp_path, "--table", str(table))
    text = table.read_text(encoding="utf-8")
    assert "| Method | MATH500 | AIME 2025 | CMATH | GSM8K | Avg. |" in text
    c3 = A2["c3"]["math500"]
    assert "| C3 | %.1f \u00b1 %.1f |" % (statistics.mean(c3), statistics.stdev(c3)) in text
    assert "Table 10" in text and "C3 minus MAGRPO" in text
    diff = statistics.mean(a - b for a, b in zip(A2["c3"]["math500"], A2["magrpo"]["math500"]))
    assert "| MATH500 | %+.2f |" % diff in text
    assert "p < 0.001" in text
    assert "Table 8" not in text


def test_a_damaged_record_stops_the_run(tmp_path: Path) -> None:
    root = tmp_path / "records"
    _tree(root)
    _write(root, "a2_4b", "c3", 2, _record({"math500": 80.0}, ok=False))
    out = tmp_path / "summary.json"
    assert a5.main(["--results", str(root), "--out", str(out)]) == EXIT_BROKEN_RECORDS
    assert not out.exists()


def test_records_off_the_protocol_are_named(tmp_path: Path) -> None:
    root = tmp_path / "records"
    _tree(root)
    _write(root, "a2_4b", "mappo", 0, _record({b: v[0] for b, v in A2["mappo"].items()}, k=4,
                                              generate_max_len=2048))
    doc = _run(tmp_path)
    problems = " ".join(doc["problems"])
    assert "k != 1" in problems and "generation limit 2048" in problems


def test_a_seed_evaluated_twice_is_the_mean_of_its_evaluations(tmp_path: Path) -> None:
    root = tmp_path / "records"
    _tree(root)
    _write(root, "a2_4b", "c3", 0, _record({"math500": 90.0, "cmath": 96.0, "gsm8k": 93.0, "aime25": 10.0}),
           eval_seed=1)
    doc = _run(tmp_path)
    per_seed = doc["tables"]["a2_4b"]["rows"]["c3"]["math500"]["per_seed"]
    assert per_seed["0"] == pytest.approx((A2["c3"]["math500"][0] + 90.0) / 2)
    assert "more than one evaluation" in doc["keys"]["E2main.a2_4b.math500.c3.mean"]["note"]


def test_the_pairing_uses_only_the_seeds_both_arms_carry(tmp_path: Path) -> None:
    root = tmp_path / "records"
    data = {m: {b: list(v) for b, v in benches.items()} for m, benches in A2.items()}
    _tree(root, data=data)
    # C3 loses training seed 4.
    import shutil
    shutil.rmtree(root / "a2_4b" / "c3" / "train_s4")
    doc = _run(tmp_path)
    pair = doc["tables"]["significance"]["benches"]["math500"]
    assert pair["seeds"] == [0, 1, 2, 3]
    assert pair["n"] == 4
    assert doc["keys"]["E2main.a2_4b.math500.c3.mean"]["n"] == 4


def test_unknown_directories_are_passed_over(tmp_path: Path, capsys) -> None:
    root = tmp_path / "records"
    _tree(root)
    (root / "a2_code").mkdir()
    (root / "a2_4b" / "sft").mkdir()
    doc = _run(tmp_path)
    err = capsys.readouterr().err
    assert "a2_code is not an arm" in err
    assert "sft is not a method" in err
    assert "a2_4b" in doc["tables"]
