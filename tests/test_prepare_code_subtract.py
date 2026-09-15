"""MBPP-train is written minus the problems that also occur in MBPP-test.

Upstream MBPP (full config, revision 4bb6404) ships three problems in both splits; the
overlap gate of 2026-09-15 caught them. The subtraction keys on the same normalized problem
text the gate compares, so the two cannot disagree.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", ROOT / "scripts" / "10_data" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_train_rows_that_occur_in_test_are_dropped_and_counted(capsys):
    pc = _load("prepare_code")
    test_rows = [{"task_id": 11, "text": "Write a function to  check if a nested list is a subset."}]
    test_keys = {pc._problem_key(r) for r in test_rows}
    train_rows = [
        {"task_id": 601, "text": "Write a function to check if a nested list is a subset."},
        {"task_id": 602, "text": "Write a python function to find the first repeated character."},
    ]
    kept = list(pc._subtract_test_problems("MBPP-train", train_rows, test_keys))
    assert [r["task_id"] for r in kept] == [602]
    assert "dropped 1 row(s)" in capsys.readouterr().out


def test_problem_key_matches_the_overlap_gate_normalization():
    pc = _load("prepare_code")
    gate = _load("check_overlap")
    row = {"text": "  Write a\tfunction   to reverse\n a list. "}
    assert pc._problem_key(row) == gate.question_key(row) == "Write a function to reverse a list."


def test_problem_keys_are_read_back_from_an_existing_test_file(tmp_path):
    pc = _load("prepare_code")
    p = tmp_path / "test.jsonl"
    p.write_text(json.dumps({"text": "a  b"}) + "\n" + json.dumps({"text": ""}) + "\n\n", encoding="utf-8")
    assert pc._problem_keys_from_file(p) == {"a b"}
