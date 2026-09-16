"""The data-prep write door refuses rows without an answer, and CMATH's `golden` column is read.

On 2026-09-15 the prepared CMATH train and test files carried an empty answer in every row
(upstream calls the column `golden`; the row builder knew only answer / output / label / target)
and the strict pass reported [OK] on both. These tests pin the gate that now stops that.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location("prepare_math_under_test", ROOT / "scripts" / "10_data" / "prepare_math.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_empty_answer_is_refused_at_the_write_door_and_nothing_is_left_behind(tmp_path):
    pm = _load()
    rows = [{"input": "q1", "answer": "1"}, {"input": "q2", "answer": ""}]
    out = tmp_path / "x.jsonl"
    with pytest.raises(SystemExit, match=r"row 1 has an empty answer"):
        pm._write_jsonl_atomic(out, pm._require_complete_rows("T", rows), overwrite=True)
    assert not out.exists()
    assert list(tmp_path.iterdir()) == []


def test_empty_input_is_refused():
    pm = _load()
    with pytest.raises(SystemExit, match=r"row 0 has an empty input"):
        list(pm._require_complete_rows("T", [{"input": "  ", "answer": "1"}]))


def test_complete_rows_pass_through_unchanged(tmp_path):
    pm = _load()
    rows = [{"input": "q1", "answer": "1", "source": "s"}, {"input": "q2", "answer": "0", "source": "s"}]
    out = tmp_path / "x.jsonl"
    n = pm._write_jsonl_atomic(out, pm._require_complete_rows("T", rows), overwrite=True)
    assert n == 2
    assert [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()] == rows


def test_an_existing_artifact_with_an_empty_answer_fails_the_strict_check(tmp_path):
    pm = _load()
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"input": "q", "answer": "1"}\n{"input": "q2", "answer": ""}\n', encoding="utf-8")
    with pytest.raises(SystemExit, match=r"row 1 has an empty input or answer"):
        pm._check_rows_complete_on_disk("T", bad)
    good = tmp_path / "good.jsonl"
    good.write_text('{"input": "q", "answer": "1"}\n\n', encoding="utf-8")
    pm._check_rows_complete_on_disk("T", good)


def test_boxed_extraction_covers_braces_bare_tokens_and_empty_boxes():
    pm = _load()
    assert pm._extract_boxed_answer(r"so $x = \boxed{\frac{1}{2}}$.") == r"\frac{1}{2}"
    assert pm._extract_boxed_answer(r"first \boxed{1}, our answer is $\boxed 2$.") == "2"
    assert pm._extract_boxed_answer(r"It follows that $x^2 + y^2 = \boxed 9$.") == "9"
    assert pm._extract_boxed_answer(r"there are $\boxed{}$ primes in that range.") is None
    assert pm._extract_boxed_answer("no box here") is None
    assert pm._extract_boxed_answer("") is None


def test_a_re_pinning_run_reports_a_changed_sha_instead_of_failing(tmp_path):
    pm = _load()
    missing = []
    kwargs = dict(name="X", out_path=tmp_path / "x.jsonl", expected_sha="a" * 64, computed_sha="b" * 64, strict=False, optional=False, missing_sha=missing)
    with pytest.raises(SystemExit, match="sha256 mismatch"):
        pm._verify_or_record_sha(**kwargs)
    pm._verify_or_record_sha(**kwargs, allow_mismatch=True)
    assert missing == []


def test_cmath_rows_read_the_golden_column(monkeypatch):
    pm = _load()
    upstream = [{"grade": "1", "question": "3 boxes of 6 bags: how many bags?", "golden": "18", "reasoning_step": "1", "num_digits": "2"}]
    monkeypatch.setattr(pm, "_iter_hf_rows", lambda src, logical_name: upstream)
    monkeypatch.setattr(pm, "_source_string", lambda *a, **k: "weitianwen/cmath:test@abc")
    rows = pm._cmath_rows({"split": "test"})
    assert rows == [{"input": "3 boxes of 6 bags: how many bags?", "answer": "18", "source": "weitianwen/cmath:test@abc", "grade": "1"}]
