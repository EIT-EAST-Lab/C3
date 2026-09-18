"""The write door of the data preparation scripts, and the two reductions MATH-train needs.

Issue 2 on the public repository reported three things at once, and all three are here.
The MBPP+ artifact was 378 rows in which only `task_id` and `source` carried a value,
because the row builder read MBPP column names out of an EvalPlus task, and its sha256
pin matched that empty file on every run. The EvalPlus task id `Mbpp/100` does not parse
as an integer, and the two places that tried fell back to the string and to 0. MATH-train
still held one problem that also occurs in MATH-test, and one that occurs twice in
MATH-train itself.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List

import pytest

from c3.task.datasets import _PROMPT_KEY_CANDIDATES

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        f"{name}_under_test", ROOT / "scripts" / "10_data" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _mbpp_row(**over: Any) -> Dict[str, Any]:
    row = {
        "task_id": 11,
        "text": "Write a function to add two numbers.",
        "code": "def add(a, b):\n    return a + b",
        "test_list": ["assert add(1, 2) == 3"],
        "test_setup_code": "",
        "source": "google-research-datasets/mbpp:test@" + "4" * 40,
    }
    row.update(over)
    return row


# ---------------------------------------------------------------------------
# The EvalPlus task id
# ---------------------------------------------------------------------------


def test_an_evalplus_task_id_becomes_its_integer() -> None:
    pc = _load("prepare_code")
    assert pc.mbpp_plus_task_id("Mbpp/100") == 100
    assert pc.mbpp_plus_task_id("mbpp/2") == 2
    assert pc.mbpp_plus_task_id(" Mbpp/809 ") == 809
    assert pc.mbpp_plus_task_id(100) == 100
    assert pc.mbpp_plus_task_id("100") == 100


@pytest.mark.parametrize("bad", ["Mbpp/abc", "", "HumanEval/0", None, "Mbpp/1/2"])
def test_a_task_id_that_does_not_parse_fails_instead_of_becoming_zero(bad: Any) -> None:
    """The old fallback gave every row of the file the same value, which is a silent bug."""
    pc = _load("prepare_code")
    with pytest.raises(SystemExit) as excinfo:
        pc.mbpp_plus_task_id(bad)
    assert "cannot read an integer task id" in str(excinfo.value)


# ---------------------------------------------------------------------------
# The write door of prepare_code.py
# ---------------------------------------------------------------------------


def test_the_shape_the_empty_mbpp_plus_file_had_is_refused(tmp_path: Path) -> None:
    pc = _load("prepare_code")
    as_written = [
        {
            "task_id": 100,
            "text": None,
            "code": None,
            "test_list": [],
            "test_setup_code": None,
            "source": "evalplus:mbpp_plus@0.3.1",
        }
    ]
    out = tmp_path / "test.jsonl"
    with pytest.raises(SystemExit) as excinfo:
        pc._write_checked("MBPP+", out, as_written)
    assert "row 0 has an empty problem statement" in str(excinfo.value)
    assert not out.exists()
    assert list(tmp_path.iterdir()) == []


def test_a_row_without_a_problem_statement_is_refused_and_leaves_nothing_behind(tmp_path: Path) -> None:
    pc = _load("prepare_code")
    rows = [_mbpp_row(), _mbpp_row(task_id=12, text="   ")]
    out = tmp_path / "test.jsonl"
    with pytest.raises(SystemExit) as excinfo:
        pc._write_checked("MBPP-test", out, rows)
    assert "row 1 has an empty problem statement" in str(excinfo.value)
    assert isinstance(excinfo.value.code, str)
    assert not out.exists()
    assert list(tmp_path.iterdir()) == []


def test_a_row_without_tests_is_refused(tmp_path: Path) -> None:
    pc = _load("prepare_code")
    out = tmp_path / "test.jsonl"
    with pytest.raises(SystemExit) as excinfo:
        pc._write_checked("MBPP-test", out, [_mbpp_row(test_list=[])])
    assert "row 0 has no tests" in str(excinfo.value)
    assert list(tmp_path.iterdir()) == []


def test_a_whole_script_counts_as_tests_and_an_apps_row_is_judged_on_its_own_key(tmp_path: Path) -> None:
    """HumanEval scores by one script, APPS by `input_output`; neither has a test_list."""
    pc = _load("prepare_code")
    human = {"task_id": "HumanEval/0", "prompt": "def f():\n", "test": "assert f() == 1", "entry_point": "f"}
    apps = {"problem_id": 1, "question": "add two numbers", "input_output": '{"inputs": [], "outputs": []}'}
    assert pc._write_checked("HumanEval", tmp_path / "he.jsonl", [human]) == 1
    assert pc._write_checked("APPS", tmp_path / "apps.jsonl", [apps]) == 1


def test_complete_rows_are_written_unchanged(tmp_path: Path) -> None:
    pc = _load("prepare_code")
    rows = [_mbpp_row(), _mbpp_row(task_id=12, text="Write a function to subtract.")]
    out = tmp_path / "test.jsonl"
    assert pc._write_checked("MBPP-test", out, rows) == 2
    written = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert written == rows


def test_an_artifact_already_on_disk_faces_the_same_gate(tmp_path: Path) -> None:
    """The empty MBPP+ file was skipped and verified against its pin on every rerun."""
    pc = _load("prepare_code")
    bad = tmp_path / "test.jsonl"
    bad.write_text(json.dumps({"task_id": 100, "test_list": []}) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        pc._check_rows_complete_on_disk("MBPP+", bad)
    assert "row 0 has an empty problem statement" in str(excinfo.value)

    good = tmp_path / "good.jsonl"
    good.write_text(json.dumps(_mbpp_row()) + "\n\n", encoding="utf-8")
    pc._check_rows_complete_on_disk("MBPP-test", good)


def test_both_preparation_scripts_ask_the_trainers_own_key_table() -> None:
    """The table is imported from c3/task/datasets.py, not copied into each script."""
    pc = _load("prepare_code")
    pm = _load("prepare_math")
    assert pc.PROMPT_KEY_CANDIDATES is _PROMPT_KEY_CANDIDATES
    assert pm.PROMPT_KEY_CANDIDATES is _PROMPT_KEY_CANDIDATES


# ---------------------------------------------------------------------------
# MATH-train: the repeat and the shared problem
# ---------------------------------------------------------------------------


def _math_spec() -> Dict[str, Any]:
    return {"source": {"id": "EleutherAI/hendrycks_math", "configs": ["algebra"], "split": "train", "revision": "0" * 40}}


def test_math_train_drops_a_repeated_statement_and_keeps_the_first_copy(monkeypatch, capsys) -> None:
    pm = _load("prepare_math")
    train: List[Dict[str, Any]] = [
        {"problem": "p1", "level": 1, "type": "Algebra", "solution": "first, so \\boxed{1}"},
        # The same problem again. Upstream ships it with another solution text, so the
        # deduplication by the input, answer and solution triple does not see it.
        {"problem": "  p1  ", "level": 1, "type": "Algebra", "solution": "again, so \\boxed{1}"},
        {"problem": "p2", "level": 2, "type": "Algebra", "solution": "so \\boxed{2}"},
    ]

    def _rows(src: Dict[str, Any], *, logical_name: str):
        return iter(train if src["split"] == "train" else [])

    monkeypatch.setattr(pm, "_iter_hf_rows", _rows)
    rows = list(pm._prepare_math_train(_math_spec()))

    assert [r["input"] for r in rows] == ["p1", "p2"]
    assert rows[0]["solution"] == "first, so \\boxed{1}"
    out = capsys.readouterr().out
    assert "3 rows minus 1 repeated statement(s) -> 2 rows" in out
    assert "repeated statement dropped: p1 | answer: 1" in out


def test_math_train_drops_the_problem_that_also_occurs_in_the_test_split(monkeypatch, capsys) -> None:
    pm = _load("prepare_math")
    shared = "The points $(0,0)$, $(a,11)$ and $(b,37)$ are the vertices of an equilateral triangle."
    train = [
        {"problem": "p1", "level": 1, "type": "Algebra", "solution": "so \\boxed{1}"},
        {"problem": shared, "level": 3, "type": "Precalculus", "solution": "so \\boxed{315}"},
    ]
    # The same statement with different whitespace, to pin that the comparison normalizes.
    test = [{"problem": shared.replace(" ", "  "), "level": 3, "type": "Precalculus", "solution": "so \\boxed{315}"}]

    seen_splits = []

    def _rows(src: Dict[str, Any], *, logical_name: str):
        seen_splits.append(src["split"])
        return iter(train if src["split"] == "train" else test)

    monkeypatch.setattr(pm, "_iter_hf_rows", _rows)
    rows = list(pm._prepare_math_train(_math_spec()))

    assert [r["input"] for r in rows] == ["p1"]
    assert seen_splits == ["train", "test"], "the builder reads the test split of the same revision"
    out = capsys.readouterr().out
    assert "also in MATH-test, dropped:" in out
    assert "answer: 315" in out
    assert "2 rows minus 1 shared with the 1 row test split -> 1 rows" in out


def test_math_test_is_not_subtracted_from_itself(monkeypatch) -> None:
    """Only the train builder subtracts; the test artifact is prepared as the plain union."""
    pm = _load("prepare_math")
    rows = [{"problem": "p1", "level": 1, "type": "Algebra", "solution": "so \\boxed{1}"}]
    monkeypatch.setattr(pm, "_iter_hf_rows", lambda src, *, logical_name: iter(rows))
    spec = {"source": {"id": "x/y", "configs": ["algebra"], "split": "test", "revision": "0" * 40}}
    assert [r["input"] for r in pm._prepare_math_test(spec)] == ["p1"]
