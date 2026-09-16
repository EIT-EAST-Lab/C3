"""Schema alignment before concatenating evaluation suites or training sources.

The start-accuracy probe of 2026-09-15 declared eight suites and the trainer evaluated one:
HF `concatenate_datasets` refused the mix (MATH500 carries an integer `level`, OlympiadBench a
string `level`, Minerva has no `level` at all) and the trainer fell back to the first suite
without a word. The aligner keeps every column, fills what a source lacks, casts a column whose
type differs to string, and refuses a source that lacks a required column.
"""

from __future__ import annotations

import pytest

datasets = pytest.importorskip("datasets")

from c3.task.datasets import (  # noqa: E402
    REQUIRED_COLUMNS,
    _interleave_train_datasets,
    align_dataset_schemas,
    concatenate_datasets_aligned,
)


def _suite(name, rows):
    ds = datasets.Dataset.from_list(rows)
    return ds.add_column("datasource", [name] * len(ds))


def _eight_like_suites():
    """Three suites shaped like the real prepared files: different columns, one type conflict."""
    math500 = _suite(
        "MATH500",
        [
            {"input": "q1", "answer": "1", "level": 5, "solution": "s1", "unique_id": "m/1"},
            {"input": "q2", "answer": "2", "level": 3, "solution": "s2", "unique_id": "m/2"},
        ],
    )
    minerva = _suite("Minerva-Math", [{"input": "q3", "answer": "3", "source": "minerva"}])
    olympiad = _suite(
        "OlympiadBench",
        [{"input": "q4", "answer": "4", "level": "hard", "solution": "s4", "unique_id": "o/4"}],
    )
    return {"MATH500": math500, "Minerva-Math": minerva, "OlympiadBench": olympiad}


def test_hf_refuses_the_raw_mix_which_is_the_bug_being_fixed():
    suites = _eight_like_suites()
    with pytest.raises(Exception):
        datasets.concatenate_datasets(list(suites.values()))


def test_every_suite_reaches_the_concatenation():
    suites = _eight_like_suites()
    merged, alignment = concatenate_datasets_aligned(suites)
    assert len(merged) == 4
    assert merged["datasource"] == ["MATH500", "MATH500", "Minerva-Math", "OlympiadBench"]
    assert merged["input"] == ["q1", "q2", "q3", "q4"]
    assert merged["answer"] == ["1", "2", "3", "4"]


def test_conflicting_column_is_cast_to_string_and_missing_columns_are_filled():
    suites = _eight_like_suites()
    merged, alignment = concatenate_datasets_aligned(suites)
    assert alignment.cast_to_string == ("level",)
    assert merged["level"] == ["5", "3", None, "hard"]
    assert merged["source"] == [None, None, "minerva", None]
    assert merged["unique_id"] == ["m/1", "m/2", None, "o/4"]
    assert set(alignment.filled) == {"MATH500", "Minerva-Math", "OlympiadBench"}
    assert "level" in alignment.filled["Minerva-Math"]
    assert alignment.filled["MATH500"] == ("source",)
    assert "columns=" in alignment.describe() and "cast_to_string=['level']" in alignment.describe()


def test_identical_schemas_pass_through_untouched():
    a = _suite("A", [{"input": "q1", "answer": "1"}])
    b = _suite("B", [{"input": "q2", "answer": "2"}])
    aligned, alignment = align_dataset_schemas({"A": a, "B": b})
    assert alignment.cast_to_string == () and alignment.filled == {}
    assert aligned["A"].features == a.features
    merged, _ = concatenate_datasets_aligned({"A": a, "B": b})
    assert merged["input"] == ["q1", "q2"]


def test_a_source_without_a_required_column_is_an_error_not_a_dropped_source():
    good = _suite("A", [{"input": "q1", "answer": "1"}])
    bad = datasets.Dataset.from_list([{"input": "q2", "datasource": "B"}])  # no answer
    with pytest.raises(ValueError, match="'B' lacks required column"):
        concatenate_datasets_aligned({"A": good, "B": bad})
    assert REQUIRED_COLUMNS == ("input", "answer", "datasource")


def test_single_source_is_returned_as_is():
    a = _suite("A", [{"input": "q1", "answer": "1"}])
    merged, alignment = concatenate_datasets_aligned({"A": a})
    assert len(merged) == 1 and alignment.filled == {}


def test_train_concat_mode_uses_the_same_alignment():
    math_train = _suite("MATH-train", [{"input": "q1", "answer": "1", "level": 2}])
    cmath_train = _suite("CMATH-train", [{"input": "q2", "answer": "2", "grade": 4}])
    merged = _interleave_train_datasets([math_train, cmath_train], [1.0, 1.0], seed=0, sampling_mode="concat")
    assert len(merged) == 2
    assert merged["datasource"] == ["MATH-train", "CMATH-train"]
    assert merged["grade"] == [None, 4]
