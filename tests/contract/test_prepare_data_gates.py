"""The write door of the data preparation scripts, the MBPP+ row, and MATH-train's reductions.

Issue 2 on the public repository reported three things at once, and all three are here.
The MBPP+ artifact was 378 rows in which only `task_id` and `source` carried a value,
because the row builder read MBPP column names out of an EvalPlus task, and its sha256
pin matched that empty file on every run. The EvalPlus task id `Mbpp/100` does not parse
as an integer, and the two places that tried fell back to the string and to 0. MATH-train
still held one problem that also occurs in MATH-test, and one that occurs twice in
MATH-train itself.

The MBPP+ row that replaced the empty one is a differential test: it carries the inputs
and the reference solution, and the expected value is what the reference returns. What
keeps a candidate from writing its own verdict is the order the evaluator runs things in,
so the order is pinned here against the evaluator itself rather than described.
"""

from __future__ import annotations

import importlib.util
import builtins
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional

import pytest

from c3.envs.code import executor as code_executor
from c3.envs.code.executor import run_mbpp_tests
from c3.envs.code.reward import _coerce_timeout
from c3.task.datasets import _PROMPT_KEY_CANDIDATES

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        f"{name}_under_test", ROOT / "scripts" / "10_data" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
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
# The MBPP+ row: a differential test against the reference solution
# ---------------------------------------------------------------------------

# One EvalPlus task, in the shape evalplus 0.3.1 hands over: `prompt` is the statement
# wrapped in a docstring with the first MBPP assertion appended, `base_input` and
# `plus_input` are argument lists, and no expected value is stored anywhere.
def _evalplus_task(**over: Any) -> Dict[str, Any]:
    task = {
        "task_id": "Mbpp/100",
        "prompt": '"""\nWrite a function to add two numbers.\nassert add(1, 2) == 3\n"""\n',
        "canonical_solution": "\ndef add(a, b):\n    return a + b\n",
        "assertion": "\nassert add(1, 2) == 3\n",
        "contract": "\n    assert isinstance(a, int)\n",
        "entry_point": "add",
        "atol": 0,
        "base_input": [[1, 2], [2, 3]],
        "plus_input": [[10, 20], [0, 0], [-1, 1]],
    }
    task.update(over)
    return task


def _score(row: Dict[str, Any], candidate: str) -> Dict[str, Any]:
    """
    Run one prepared row against one candidate answer, in a child process.

    Going through `run_mbpp_tests` rather than calling `_exec_all` here is not a
    preference. `_exec_all` ends in `_apply_rlimits`, which sets RLIMIT_CPU and RLIMIT_AS
    on whatever process calls it, and it is written for the worker child, where that is
    the point. Called from a test it gave the pytest process a CPU budget of
    `timeout + 1` seconds and a 4 GB address space. On Windows `import resource` fails
    and the whole thing is a silent no-op, which is why it looked fine here; on Linux it
    took the run down with an INTERNALERROR in pathlib right after the first test that
    called it. Evaluation code runs in a child, and that includes the tests of it.
    """
    _score, info = run_mbpp_tests(
        candidate_code=candidate,
        ref_code=None,
        sample_meta=row,
        timeout=int(row.get("timeout_s", 15)),
    )
    return info


def _process_globals() -> Dict[str, Any]:
    """The objects a test that executes code in this process has been seen to replace."""
    return {
        "os.fspath": os.fspath,
        "os.path.abspath": os.path.abspath,
        "sys.modules['os']": sys.modules["os"],
        "builtins.str": builtins.str,
        "builtins.isinstance": builtins.isinstance,
        "sys.stdout": sys.stdout,
        "sys.stderr": sys.stderr,
    }


def _resource_limits() -> Optional[Dict[str, Any]]:
    """The limits `_apply_rlimits` sets, or None where the platform has no `resource`."""
    try:
        import resource  # type: ignore
    except Exception:
        return None
    return {
        "RLIMIT_CPU": resource.getrlimit(resource.RLIMIT_CPU),
        "RLIMIT_AS": resource.getrlimit(resource.RLIMIT_AS),
    }


def test_the_statement_comes_out_of_the_docstring_without_the_assertion() -> None:
    pc = _load("prepare_code")
    assert pc.mbpp_plus_statement(_evalplus_task()["prompt"]) == "Write a function to add two numbers."
    assert pc.mbpp_plus_statement('"""\nLine one.\nLine two.\nassert f(1) == 2\n"""\n') == "Line one. Line two."
    assert pc.mbpp_plus_statement("") == ""


def test_every_key_a_consumer_reads_is_filled_and_the_task_id_is_an_integer() -> None:
    pc = _load("prepare_code")
    row = pc.canon_mbpp_plus_row(_evalplus_task(), "evalplus:mbpp_plus@0.3.1")

    assert row["task_id"] == 100 and isinstance(row["task_id"], int)
    assert row["text"] == "Write a function to add two numbers."
    assert "def add(a, b):" in row["code"]
    assert row["entry_point"] == "add"
    assert row["source"] == "evalplus:mbpp_plus@0.3.1"
    assert row["timeout_s"] == pc.MBPP_PLUS_TIMEOUT_S
    # One test per input, and the count is on the row so the file can be checked alone.
    assert row["n_base_inputs"] == 2 and row["n_plus_inputs"] == 3
    assert row["test_list"] == [f"assert _c3_case({i})" for i in range(5)]
    assert "def _c3_case(" in row["test_setup_code"] and "def _c3_judge(" in row["test_setup_code"]
    assert "_C3_REF = add" in row["test_setup_code"]


def test_the_reference_solution_passes_its_own_tests_and_a_wrong_one_does_not() -> None:
    pc = _load("prepare_code")
    row = pc.canon_mbpp_plus_row(_evalplus_task(), "s")

    good = _score(row, row["code"])
    assert (good["passed"], good["total"]) == (5, 5)

    bad = _score(row, "def add(a, b):\n    return a * b\n")
    assert bad["passed"] < bad["total"]


def test_a_candidate_cannot_write_its_own_verdict() -> None:
    """The setup runs after the candidate, so a forged `_c3_case` is simply overwritten.

    This is the whole reason the harness is built the way it is, and it holds because
    the executor defers a setup block that raises NameError until after the candidate.
    """
    pc = _load("prepare_code")
    row = pc.canon_mbpp_plus_row(_evalplus_task(), "s")
    forged = "def add(a, b):\n    return a * b\n\n\ndef _c3_case(index):\n    return True\n"
    result = _score(row, forged)
    assert result["passed"] < result["total"]


def test_an_entry_point_that_shadows_a_builtin_still_defers() -> None:
    """`Mbpp/126` is called `sum`, so a bare reference to it would find the builtin.

    The `del` after the capture is what raises NameError in that case, which is what
    moves the block after the candidate.
    """
    pc = _load("prepare_code")
    task = _evalplus_task(
        task_id="Mbpp/126",
        entry_point="sum",
        canonical_solution="\ndef sum(a, b):\n    return a + b\n",
        prompt='"""\nWrite a function to add two numbers.\nassert sum(1, 2) == 3\n"""\n',
    )
    row = pc.canon_mbpp_plus_row(task, "s")
    assert row["test_setup_code"].splitlines()[3:5] == ["_C3_FN = sum", "del sum"]

    good = _score(row, row["code"])
    assert (good["passed"], good["total"]) == (5, 5)

    # The forgery buys nothing: the same wrong answer scores the same with and without it.
    wrong = "def sum(a, b):\n    return 0\n"
    forged = wrong + "\n\ndef _c3_case(index):\n    return True\n"
    plain_score = _score(row, wrong)
    assert plain_score["passed"] < plain_score["total"]
    assert _score(row, forged)["passed"] == plain_score["passed"]


def test_a_candidate_that_never_defines_the_entry_point_scores_nothing() -> None:
    pc = _load("prepare_code")
    row = pc.canon_mbpp_plus_row(_evalplus_task(), "s")
    result = _score(row, "def something_else():\n    return 1\n")
    assert result["passed"] == 0


def test_the_set_comparison_and_the_output_is_not_none_rules_are_applied() -> None:
    """Two of EvalPlus's special oracles, on the entry points it names."""
    pc = _load("prepare_code")

    set_task = _evalplus_task(
        entry_point="similar_elements",
        prompt='"""\nShared elements.\nassert similar_elements((1,2),(2,3)) == (2,)\n"""\n',
        canonical_solution="\ndef similar_elements(a, b):\n    return [x for x in a if x in b]\n",
        base_input=[[[1, 2, 3], [3, 2, 9]]],
        plus_input=[],
    )
    row = pc.canon_mbpp_plus_row(set_task, "s")
    assert "_C3_MODE = 'set_eq'" in row["test_setup_code"]
    # A different order is the same set, so it passes where plain equality would not.
    reordered = "def similar_elements(a, b):\n    return [x for x in a if x in b][::-1]\n"
    assert _score(row, reordered)["passed"] == 1

    none_task = _evalplus_task(
        entry_point="check_str",
        prompt='"""\nStarts with a vowel.\nassert check_str("annie")\n"""\n',
        canonical_solution="\ndef check_str(text):\n    return text[:1] in 'aeiou' or None\n",
        base_input=[["annie"], ["dog"]],
        plus_input=[],
    )
    row = pc.canon_mbpp_plus_row(none_task, "s")
    assert "_C3_MODE = 'not_none'" in row["test_setup_code"]
    # Any non-None answer counts where the reference returns one, which is the rule.
    assert _score(row, "def check_str(text):\n    return 'yes' if text[:1] in 'aeiou' else None\n")["passed"] == 2


def test_the_float_tolerance_follows_evalplus() -> None:
    pc = _load("prepare_code")
    task = _evalplus_task(
        entry_point="volume_sphere",
        prompt='"""\nVolume of a sphere.\nassert volume_sphere(10)\n"""\n',
        canonical_solution="\ndef volume_sphere(r):\n    return 4.0 / 3.0 * 3.141592653589793 * r ** 3\n",
        base_input=[[10]],
        plus_input=[],
        atol=1e-4,
    )
    row = pc.canon_mbpp_plus_row(task, "s")
    assert "_C3_ATOL = 0.0001" in row["test_setup_code"]
    close = "def volume_sphere(r):\n    return 4.0 / 3.0 * 3.14159265 * r ** 3\n"
    assert _score(row, close)["passed"] == 1
    far = "def volume_sphere(r):\n    return 4.0 / 3.0 * 3.14 * r ** 3\n"
    assert _score(row, far)["passed"] == 0


def test_the_inputs_are_written_as_literals_the_sandbox_can_read() -> None:
    pc = _load("prepare_code")
    assert pc.python_literal([1, (2,), {"a": 3}, set()]) == "[1, (2,), {'a': 3}, set()]"
    assert pc.python_literal(float("inf")) == "float('inf')"
    assert pc.python_literal(float("-inf")) == "float('-inf')"
    assert pc.python_literal(float("nan")) == "float('nan')"
    assert pc.python_literal(complex(1, 2)) == "(1+2j)"
    with pytest.raises(SystemExit, match="no literal form"):
        pc.python_literal(object())


def test_the_imports_of_the_reference_are_restored_after_the_candidate_runs() -> None:
    """The executor hoists import lines out of the setup, so they run before the candidate."""
    pc = _load("prepare_code")
    assert pc._rebind_import_lines("import re\n") == ['re = __import__("re")']
    assert pc._rebind_import_lines("import heapq as hq\n") == ['hq = __import__("heapq")']
    assert pc._rebind_import_lines("from collections import defaultdict, Counter\n") == [
        'defaultdict = __import__("collections").defaultdict',
        'Counter = __import__("collections").Counter',
    ]
    with pytest.raises(SystemExit, match="not one of the forms"):
        pc._rebind_import_lines("import os.path\n")

    task = _evalplus_task(
        canonical_solution="\nimport re\n\n\ndef add(a, b):\n    return len(re.findall('x', 'x' * (a + b)))\n",
    )
    row = pc.canon_mbpp_plus_row(task, "s")
    assert 're = __import__("re")' in row["test_setup_code"]
    # A candidate that replaces the module the reference uses cannot change its answer.
    sabotage = "import re\nre = None\n\n\ndef add(a, b):\n    return a + b\n"
    assert _score(row, sabotage)["passed"] == 5


def test_a_task_needing_an_oracle_this_builder_does_not_have_stops_the_run() -> None:
    pc = _load("prepare_code")
    with pytest.raises(SystemExit, match="special oracle"):
        pc.canon_mbpp_plus_row(_evalplus_task(entry_point="are_equivalent"), "s")


def test_evalplus_is_the_only_provenance_form_mbpp_plus_accepts() -> None:
    """`fallback_mbpp@<sha>` used to mean "write MBPP under the MBPP+ name instead"."""
    pc = _load("prepare_code")
    assert pc._is_pinned_evalplus_revision("evalplus@0.3.1")
    assert not pc._is_pinned_evalplus_revision("fallback_mbpp@" + "4" * 40)
    assert not pc._is_pinned_evalplus_revision("evalplus")


def test_the_generated_tests_stay_inside_the_evaluators_own_budget() -> None:
    """The budget is the default of C3_CODE_MAX_ASSERT_CHARS, not a number of our own."""
    pc = _load("prepare_code")
    if not os.environ.get("C3_CODE_MAX_ASSERT_CHARS", "").strip():
        assert pc.MBPP_PLUS_ASSERT_BUDGET == code_executor._ASSERT_MAX_CHARS

    task = _evalplus_task(base_input=[[1, 2]] * 3, plus_input=[[1, 2]] * 147)
    row = pc.canon_mbpp_plus_row(task, "s")
    assert len(row["test_list"]) == 150
    assert sum(len(test) for test in row["test_list"]) < pc.MBPP_PLUS_ASSERT_BUDGET


def test_an_mbpp_plus_row_declares_the_wall_clock_its_benchmark_needs() -> None:
    """Twice what EvalPlus allows one task, because every input runs twice here.

    EvalPlus spends its 60 seconds on the candidate alone; these tests run the reference
    over the same input as well. The task file cannot lower what the row asks for, and
    the row cannot ask for more than the ceiling.
    """
    assert _coerce_timeout({}, {"timeout_s": 120}) == 120
    assert _coerce_timeout({"task_env_cfg": {"code_timeout": 15}}, {"timeout_s": 120}) == 120
    assert _coerce_timeout({"task_env_cfg": {"code_timeout": 240}}, {"timeout_s": 120}) == 240
    assert _coerce_timeout({}, {}) == 15
    assert _coerce_timeout({}, None) == 15
    # A data file cannot ask for an hour.
    assert _coerce_timeout({}, {"timeout_s": 3600}) == 120


def test_the_sandbox_lets_the_complex_number_problems_run() -> None:
    """Three MBPP+ problems need cmath, and no correct answer to them can run without it.

    The import hook is called directly rather than through `exec`, so that nothing in
    this file executes evaluation code in the pytest process.
    """
    safe_import = code_executor._mk_safe_env()["__builtins__"]["__import__"]
    assert round(safe_import("cmath").polar(complex(1, 1))[0], 6) == round(2 ** 0.5, 6)
    with pytest.raises(ImportError):
        safe_import("sys")


def test_running_an_evaluation_leaves_this_process_alone() -> None:
    """An evaluation happens in a child, and this is what says so out loud.

    `_exec_all` applies RLIMIT_CPU and RLIMIT_AS to its own process, because it is
    written to run in the worker child. A test that called it directly therefore gave
    the pytest process a CPU budget of `timeout + 1` seconds and a 4 GB address space.
    On Windows `import resource` fails and it is a silent no-op; on Linux it ended a CI
    run with an INTERNALERROR, a TypeError raised inside pathlib while pytest was
    working out where the next test lived, one test after the first one that called it.

    So this pins the property rather than the symptom: after an evaluation, the objects
    the interpreter depends on are the same objects, and the resource limits are the
    same limits.
    """
    pc = _load("prepare_code")
    row = pc.canon_mbpp_plus_row(_evalplus_task(), "s")

    before = _process_globals()
    limits_before = _resource_limits()

    result = _score(row, row["code"])
    assert (result["passed"], result["total"]) == (5, 5)

    after = _process_globals()
    replaced = [name for name in before if before[name] is not after[name]]
    assert not replaced, f"an evaluation replaced these in the test process: {replaced}"
    assert _resource_limits() == limits_before


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
