"""Tests for the data contamination gate and the split rules it depends on.

Two bugs reached the prepared data and both were silent. A training artifact was
pinned to a mirror whose only split was named `train` but held the whole
benchmark, so every MATH500 problem sat in the training file. An evaluation
artifact was pinned to a revision with no test split, and the repository-file
fallback quietly prepared the validation split under the test name, so the
evaluation file was a copy of the training file.

These tests pin the three mechanisms that turn both into loud failures: the
overlap gate, the subtraction rule that keeps the CMATH training artifact
disjoint from its test split, and the loader refusing to substitute a split it
was not asked for.
"""

from __future__ import annotations

import importlib.util
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / 'tests' / 'fixtures' / 'data'
FIXTURE_MANIFEST = FIXTURES / 'overlap_manifest.yaml'
CHECK_OVERLAP = REPO_ROOT / 'scripts' / '10_data' / 'check_overlap.py'
MANIFEST = REPO_ROOT / 'configs' / 'data_manifest.yaml'

MATH_TASK_YAMLS = (
    'configs/tasks/math.yaml',
    'configs/tasks/math_a3.yaml',
    'configs/tasks/math_branch.yaml',
    'configs/tasks/math_c5.yaml',
    'configs/tasks/math_c10.yaml',
    'configs/tasks/math_mt4.yaml',
)


def _load_module(rel_path: str, module_name: str) -> ModuleType:
    path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Could not load module from {path}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# prepare_math.py must import on a machine without the download stack, because
# the split and subtraction rules live in it and are unit tested here.
gate = _load_module('scripts/10_data/check_overlap.py', 'c3_check_overlap')
preparer = _load_module('scripts/10_data/prepare_math.py', 'c3_prepare_math')


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------


def _manifest_outputs() -> List[Dict[str, Any]]:
    return yaml.safe_load(MANIFEST.read_text(encoding='utf-8'))['outputs']


def _entry(name: str) -> Dict[str, Any]:
    for item in _manifest_outputs():
        if item['name'] == name:
            return item
    raise AssertionError(f'{name} is not in the manifest')


def _run_gate(out_dir: Path, manifest: Path = FIXTURE_MANIFEST) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            '-B',
            str(CHECK_OVERLAP),
            '--manifest',
            str(manifest),
            '--out_dir',
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        encoding='utf-8',
    )


def _lay_out_fixture(tmp_path: Path, eval_fixture: str) -> Path:
    out_dir = tmp_path / 'prepared'
    (out_dir / 'overlap').mkdir(parents=True)
    shutil.copyfile(FIXTURES / 'overlap_train.jsonl', out_dir / 'overlap' / 'train.jsonl')
    shutil.copyfile(FIXTURES / eval_fixture, out_dir / 'overlap' / 'eval.jsonl')
    return out_dir


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='\n') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


# -----------------------------------------------------------------------------
# 1. the gate
# -----------------------------------------------------------------------------


def test_gate_exits_zero_and_prints_a_zero_for_every_pair(tmp_path: Path) -> None:
    out_dir = _lay_out_fixture(tmp_path, 'overlap_eval_clean.jsonl')
    proc = _run_gate(out_dir)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert 'FIXTURE-train x FIXTURE-eval : 0' in proc.stdout
    assert '1 pairs checked' in proc.stdout


def test_gate_exits_one_when_one_evaluation_problem_is_in_the_training_file(tmp_path: Path) -> None:
    out_dir = _lay_out_fixture(tmp_path, 'overlap_eval_leak.jsonl')
    proc = _run_gate(out_dir)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert 'FIXTURE-train x FIXTURE-eval : 1' in proc.stdout
    assert 'Compute the sum of the first three primes.' in proc.stdout


def test_the_leaking_fixture_differs_from_the_training_row_only_in_whitespace() -> None:
    """The gate must catch a reformatted copy, which is what the leak fixture is."""
    train = [json.loads(line) for line in FIXTURES.joinpath('overlap_train.jsonl').read_text(encoding='utf-8').splitlines()]
    leak = [json.loads(line) for line in FIXTURES.joinpath('overlap_eval_leak.jsonl').read_text(encoding='utf-8').splitlines()]
    assert leak[0]['input'] != train[1]['input']
    assert gate.normalize_question(leak[0]['input']) == gate.normalize_question(train[1]['input'])


@pytest.mark.parametrize(
    'raw, expected',
    [
        ('  a  b  ', 'a b'),
        ('a\tb', 'a b'),
        ('a\nb', 'a b'),
        ('a \u3000 b', 'a b'),  # ideographic space, escaped so this file stays ASCII
        ('a\xa0b', 'a b'),  # non breaking space
        ('', ''),
        (None, ''),
    ],
)
def test_normalization_trims_and_collapses_whitespace(raw: Any, expected: str) -> None:
    assert gate.normalize_question(raw) == expected


def test_the_gate_and_the_preparer_normalize_identically() -> None:
    samples = [
        'x  y',
        ' leading and trailing ',
        'tab\tseparated',
        'line\nbreak',
        'ideographic\u3000space',
        'non\xa0breaking',
        '',
    ]
    assert [gate.normalize_question(s) for s in samples] == [preparer.normalize_question(s) for s in samples]


def test_gate_reads_the_question_under_every_column_name_the_loader_accepts() -> None:
    assert gate.QUESTION_KEY_CANDIDATES == preparer.QUESTION_KEY_CANDIDATES
    for field in gate.QUESTION_KEY_CANDIDATES:
        assert gate.question_key({field: ' q '}) == 'q'
    assert gate.question_key({'answer': '2'}) == ''
    # `input` wins over the later names, as it does in the trainer.
    assert gate.question_key({'text': 'b', 'input': 'a'}) == 'a'


def test_gate_exits_two_when_no_file_was_prepared(tmp_path: Path) -> None:
    proc = _run_gate(tmp_path / 'prepared')
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert 'nothing to compare' in proc.stdout


def test_gate_exits_two_when_a_file_carries_no_problem_statement(tmp_path: Path) -> None:
    out_dir = _lay_out_fixture(tmp_path, 'overlap_eval_clean.jsonl')
    _write_jsonl(out_dir / 'overlap' / 'eval.jsonl', [{'golden': '2'}, {'golden': '3'}])
    proc = _run_gate(out_dir)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert 'no problem statement found' in proc.stdout


def test_gate_exits_two_on_a_malformed_line(tmp_path: Path) -> None:
    out_dir = _lay_out_fixture(tmp_path, 'overlap_eval_clean.jsonl')
    (out_dir / 'overlap' / 'eval.jsonl').write_text('{"input": "ok"}\nnot json\n', encoding='utf-8')
    proc = _run_gate(out_dir)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert 'line 2 is not valid JSON' in proc.stdout


def test_gate_skips_artifacts_that_were_not_prepared(tmp_path: Path) -> None:
    """The default preparation leaves HumanEval and APPS out, so missing is normal."""
    out_dir = tmp_path / 'prepared'
    _write_jsonl(out_dir / 'MATH' / 'train.jsonl', [{'input': 'train only'}])
    _write_jsonl(out_dir / 'MATH500' / 'test.jsonl', [{'input': 'eval only'}])
    proc = _run_gate(out_dir, manifest=MANIFEST)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert 'MATH-train x MATH500 : 0' in proc.stdout
    assert '[SKIP] not prepared: HumanEval' in proc.stdout
    assert '[SKIP] not prepared: CMATH-train' in proc.stdout


def test_gate_catches_a_leak_between_the_real_manifest_names(tmp_path: Path) -> None:
    out_dir = tmp_path / 'prepared'
    _write_jsonl(out_dir / 'MATH' / 'train.jsonl', [{'input': 'shared problem'}, {'input': 'other'}])
    _write_jsonl(out_dir / 'MATH500' / 'test.jsonl', [{'input': '  shared   problem '}])
    proc = _run_gate(out_dir, manifest=MANIFEST)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert 'MATH-train x MATH500 : 1' in proc.stdout


# -----------------------------------------------------------------------------
# 2. reading the manifest without a YAML dependency
# -----------------------------------------------------------------------------


def test_manifest_reader_agrees_with_pyyaml_on_the_real_manifest() -> None:
    parsed = gate.parse_manifest_outputs(MANIFEST.read_text(encoding='utf-8'))
    reference = _manifest_outputs()
    assert [e['name'] for e in parsed] == [e['name'] for e in reference]
    assert [e['output_path'] for e in parsed] == [e['output_path'] for e in reference]
    for mine, theirs in zip(parsed, reference):
        pinned = theirs.get('sha256')
        assert mine['sha256'] == (pinned if pinned else None)


def test_manifest_reader_survives_the_sha256_rewrite() -> None:
    """`--update_manifest_sha256 1` rewrites the file through PyYAML; it must still parse."""
    obj = yaml.safe_load(MANIFEST.read_text(encoding='utf-8'))
    for item in obj['outputs']:
        item['sha256'] = 'f' * 64
    buf = io.StringIO()
    yaml.safe_dump(obj, buf, sort_keys=False, allow_unicode=True)
    parsed = gate.parse_manifest_outputs(buf.getvalue())
    assert [e['name'] for e in parsed] == [e['name'] for e in obj['outputs']]
    assert all(e['sha256'] == 'f' * 64 for e in parsed)


def test_manifest_reader_ignores_nested_list_items() -> None:
    parsed = gate.parse_manifest_outputs(FIXTURE_MANIFEST.read_text(encoding='utf-8'))
    assert [e['name'] for e in parsed] == ['FIXTURE-train', 'FIXTURE-eval']


def test_manifest_reader_stops_at_the_next_top_level_key() -> None:
    text = 'outputs:\n- name: A\n  output_path: data/a.jsonl\nextras:\n- name: B\n  output_path: data/b.jsonl\n'
    assert [e['name'] for e in gate.parse_manifest_outputs(text)] == ['A']


def test_only_the_train_artifacts_are_treated_as_training_files() -> None:
    names = [e['name'] for e in _manifest_outputs()]
    trains = [n for n in names if gate.is_train_artifact(n)]
    # The classification is the naming contract and nothing else, so it stays
    # right as benchmarks are added to the manifest.
    assert trains == [n for n in names if n.endswith('-train')]
    assert set(trains) >= {'MATH-train', 'CMATH-train', 'MBPP-train'}
    assert not gate.is_train_artifact('MATH500')
    assert not gate.is_train_artifact('CMATH-test')


def test_output_paths_resolve_under_the_data_directory(tmp_path: Path) -> None:
    assert gate.resolve_output_path('data/MATH/train.jsonl', tmp_path) == tmp_path / 'MATH' / 'train.jsonl'
    assert gate.resolve_output_path('MATH/train.jsonl', tmp_path) == tmp_path / 'MATH' / 'train.jsonl'


# -----------------------------------------------------------------------------
# 3. subtract_by_question
# -----------------------------------------------------------------------------


def test_subtract_by_question_drops_shared_problems_and_keeps_order() -> None:
    rows = [{'input': 'a'}, {'input': 'b'}, {'input': 'c'}, {'input': 'd'}]
    other = [{'input': 'c'}, {'input': 'a'}]
    assert preparer.subtract_by_question(rows, other, ('input',)) == [{'input': 'b'}, {'input': 'd'}]


def test_subtract_by_question_matches_after_whitespace_normalization() -> None:
    rows = [{'input': 'one  two'}, {'input': 'three'}]
    other = [{'input': ' one\tTWO '}, {'input': '  one   two\n'}]
    assert preparer.subtract_by_question(rows, other, ('input',)) == [{'input': 'three'}]


def test_subtract_by_question_ignores_answers_and_ids() -> None:
    rows = [{'input': 'a', 'answer': '1', 'id': 7}]
    other = [{'input': 'a', 'answer': '2', 'id': 9}]
    assert preparer.subtract_by_question(rows, other, ('input',)) == []


def test_subtract_by_question_keeps_rows_with_no_usable_statement() -> None:
    rows = [{'answer': '1'}, {'input': '   '}, {'input': 'a'}]
    other = [{'answer': '1'}, {'input': 'a'}]
    assert preparer.subtract_by_question(rows, other, ('input',)) == [{'answer': '1'}, {'input': '   '}]


def test_subtract_by_question_respects_the_key_fields_given() -> None:
    rows = [{'question': 'a'}, {'question': 'b'}]
    other = [{'input': 'a'}]
    assert preparer.subtract_by_question(rows, other, ('question',)) == rows
    assert preparer.subtract_by_question(rows, other) == [{'question': 'b'}]


def test_subtract_by_question_returns_a_list_even_for_generators() -> None:
    rows = ({'input': str(i)} for i in range(3))
    other = ({'input': '1'} for _ in range(1))
    assert preparer.subtract_by_question(rows, other, ('input',)) == [{'input': '0'}, {'input': '2'}]


# -----------------------------------------------------------------------------
# 4. the loader never substitutes a split it was not asked for
# -----------------------------------------------------------------------------


def _fake_api(files: List[str]):
    class _Api:
        def __init__(self, endpoint: Any = None) -> None:
            self.endpoint = endpoint

        def list_repo_files(self, repo_id: str, repo_type: str, revision: str) -> List[str]:
            return list(files)

    return _Api


def _install_stub_hub(monkeypatch, tmp_path: Path, files: List[str], rows_by_file: Dict[str, List[Dict[str, Any]]]):
    downloaded: List[str] = []

    def _download(repo_id: str, repo_type: str, filename: str, revision: str) -> str:
        downloaded.append(filename)
        path = tmp_path / filename
        _write_jsonl(path, rows_by_file[filename])
        return str(path)

    def _load_dataset(*args: Any, **kwargs: Any) -> Any:
        raise ValueError('Unknown split "test". Should be one of ["validation"].')

    monkeypatch.setattr(preparer, 'HfApi', _fake_api(files))
    monkeypatch.setattr(preparer, 'hf_hub_download', _download)
    monkeypatch.setattr(preparer, 'load_dataset', _load_dataset)
    return downloaded


def test_pick_repo_files_returns_nothing_when_no_file_holds_the_split() -> None:
    """The CMATH revision that caused the bug: one validation file, no test file."""
    files = ['README.md', 'data/validation-00000-of-00001.parquet']
    assert preparer._pick_repo_files(files, split='validation', cfg=None) == [
        'data/validation-00000-of-00001.parquet'
    ]
    assert preparer._pick_repo_files(files, split='test', cfg=None) == []


def test_missing_split_fails_instead_of_preparing_another_split(monkeypatch, tmp_path: Path) -> None:
    rows = [{'question': 'q1', 'answer': 'a1'}, {'question': 'q2', 'answer': 'a2'}]
    downloaded = _install_stub_hub(
        monkeypatch,
        tmp_path,
        files=['validation.jsonl'],
        rows_by_file={'validation.jsonl': rows},
    )
    src = {'id': 'weitianwen/cmath', 'config': 'default', 'split': 'test', 'revision': '0' * 40}

    with pytest.raises(SystemExit) as excinfo:
        list(preparer._iter_hf_rows(src, logical_name='CMATH-test'))

    message = str(excinfo.value)
    assert "no data file matches split='test'" in message
    assert 'never substitutes another split' in message
    assert downloaded == []


def test_a_split_that_really_exists_is_still_read_through_the_fallback(monkeypatch, tmp_path: Path) -> None:
    rows = [{'question': 'q1', 'answer': 'a1'}, {'question': 'q2', 'answer': 'a2'}]
    downloaded = _install_stub_hub(
        monkeypatch,
        tmp_path,
        files=['validation.jsonl'],
        rows_by_file={'validation.jsonl': rows},
    )
    src = {'id': 'weitianwen/cmath', 'config': 'default', 'split': 'validation', 'revision': '0' * 40}

    assert list(preparer._iter_hf_rows(src, logical_name='CMATH-validation')) == rows
    assert downloaded == ['validation.jsonl']


def test_cmath_builder_subtracts_the_split_named_in_the_manifest(monkeypatch, tmp_path: Path) -> None:
    validation = [{'question': 'shared  problem'}, {'question': 'validation only'}]
    test = [{'question': ' shared problem '}, {'question': 'test only'}]
    _install_stub_hub(
        monkeypatch,
        tmp_path,
        files=['validation.jsonl', 'test.jsonl'],
        rows_by_file={'validation.jsonl': validation, 'test.jsonl': test},
    )
    spec = {
        'source': {
            'id': 'weitianwen/cmath',
            'config': 'default',
            'split': 'validation',
            'subtract_split': 'test',
            'revision': '0' * 40,
        }
    }

    kept = list(preparer._prepare_cmath(spec))
    assert [r['input'] for r in kept] == ['validation only']
    assert kept[0]['source'] == 'weitianwen/cmath:validation@' + '0' * 40


def test_cmath_builder_without_subtract_split_keeps_every_row(monkeypatch, tmp_path: Path) -> None:
    test = [{'question': 'q1'}, {'question': 'q2'}]
    _install_stub_hub(monkeypatch, tmp_path, files=['test.jsonl'], rows_by_file={'test.jsonl': test})
    spec = {'source': {'id': 'weitianwen/cmath', 'config': 'default', 'split': 'test', 'revision': '0' * 40}}
    assert [r['input'] for r in preparer._prepare_cmath(spec)] == ['q1', 'q2']


def test_cmath_builder_refuses_to_subtract_a_split_from_itself(monkeypatch, tmp_path: Path) -> None:
    _install_stub_hub(monkeypatch, tmp_path, files=['test.jsonl'], rows_by_file={'test.jsonl': [{'question': 'q'}]})
    spec = {
        'source': {
            'id': 'weitianwen/cmath',
            'config': 'default',
            'split': 'test',
            'subtract_split': 'test',
            'revision': '0' * 40,
        }
    }
    with pytest.raises(SystemExit) as excinfo:
        preparer._prepare_cmath(spec)
    assert 'subtract_split equals source.split' in str(excinfo.value)


def test_a_missing_download_stack_fails_loudly_instead_of_falling_back(monkeypatch, tmp_path: Path) -> None:
    """Deferring the imports must not turn a missing dependency into a fallback."""
    monkeypatch.setattr(preparer, 'load_dataset', None)
    monkeypatch.setattr(preparer, 'HfApi', _fake_api(['train.jsonl']))
    src = {'id': 'x/y', 'config': None, 'split': 'train', 'revision': '0' * 40}
    with pytest.raises(SystemExit) as excinfo:
        list(preparer._iter_hf_rows(src, logical_name='X'))
    assert '`datasets` is required' in str(excinfo.value)


# -----------------------------------------------------------------------------
# 5. multi-config MATH train
# -----------------------------------------------------------------------------


def test_source_configs_reads_a_list_and_falls_back_to_one_config() -> None:
    assert preparer._source_configs({'configs': ['algebra', 'geometry']}) == ['algebra', 'geometry']
    assert preparer._source_configs({'config': 'main'}) == ['main']
    assert preparer._source_configs({'config': 'default'}) == [None]
    assert preparer._source_configs({}) == [None]


@pytest.mark.parametrize('bad', [{'configs': []}, {'configs': 'algebra'}, {'configs': ['algebra', 'algebra']}, {'configs': ['algebra', 'default']}])
def test_source_configs_rejects_an_unusable_list(bad: Dict[str, Any]) -> None:
    with pytest.raises(SystemExit):
        preparer._source_configs(bad)


def test_math_train_builder_concatenates_every_config_and_deduplicates(monkeypatch) -> None:
    per_config = {
        'algebra': [
            {'problem': 'p1', 'level': 'Level 1', 'type': 'Algebra', 'solution': 'so \\boxed{1}'},
            {'problem': 'dup', 'level': 'Level 2', 'type': 'Algebra', 'solution': 'so \\boxed{9}'},
        ],
        'geometry': [
            {'problem': 'p2', 'level': 'Level 3', 'type': 'Geometry', 'solution': 'so \\boxed{2}'},
            {'problem': 'dup', 'level': 'Level 2', 'type': 'Algebra', 'solution': 'so \\boxed{9}'},
        ],
    }

    def _rows(src: Dict[str, Any], *, logical_name: str):
        return iter(per_config[src['config']])

    monkeypatch.setattr(preparer, '_iter_hf_rows', _rows)
    spec = {
        'source': {
            'id': 'EleutherAI/hendrycks_math',
            'configs': ['algebra', 'geometry'],
            'split': 'train',
            'revision': '0' * 40,
        }
    }

    rows = list(preparer._prepare_math_train(spec))
    assert [r['input'] for r in rows] == ['p1', 'dup', 'p2']
    assert rows[0]['answer'] == '1'
    # `level` is passed through: an integer stays an integer, anything else
    # keeps its upstream spelling. Which of the two this dataset ships is a
    # property of the upstream revision, not of this builder.
    assert rows[0]['level'] == 'Level 1'
    assert rows[0]['subject'] == 'Algebra'
    assert rows[0]['solution'] == 'so \\boxed{1}'
    assert rows[0]['source'] == 'EleutherAI/hendrycks_math:algebra:train@' + '0' * 40
    assert rows[2]['source'] == 'EleutherAI/hendrycks_math:geometry:train@' + '0' * 40


def test_math_train_builder_prefers_an_explicit_subject_column(monkeypatch) -> None:
    def _rows(src: Dict[str, Any], *, logical_name: str):
        return iter([{'problem': 'p', 'subject': 'Given', 'type': 'Ignored', 'solution': '\\boxed{3}'}])

    monkeypatch.setattr(preparer, '_iter_hf_rows', _rows)
    spec = {'source': {'id': 'x/y', 'configs': ['a'], 'split': 'train', 'revision': '0' * 40}}
    assert list(preparer._prepare_math_train(spec))[0]['subject'] == 'Given'


def test_math_train_builder_keeps_an_integer_level_an_integer(monkeypatch) -> None:
    def _rows(src: Dict[str, Any], *, logical_name: str):
        return iter([{'problem': 'p', 'level': 3, 'type': 'Algebra', 'solution': '\\boxed{3}'}])

    monkeypatch.setattr(preparer, '_iter_hf_rows', _rows)
    spec = {'source': {'id': 'x/y', 'configs': ['a'], 'split': 'train', 'revision': '0' * 40}}
    assert list(preparer._prepare_math_train(spec))[0]['level'] == 3


# -----------------------------------------------------------------------------
# 6. expected_rows
# -----------------------------------------------------------------------------


def _expected_rows_case(tmp_path: Path, *, rows: int, expected: int, strict: bool, sha: Any) -> None:
    path = tmp_path / 'artifact.jsonl'
    _write_jsonl(path, [{'input': str(i)} for i in range(rows)])
    preparer._check_expected_rows(
        name='FIXTURE',
        spec={'expected_rows': expected},
        out_path=path,
        strict=strict,
        expected_sha=sha,
    )


def test_expected_rows_fails_a_wrong_count_while_the_pin_is_empty(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        _expected_rows_case(tmp_path, rows=600, expected=597, strict=True, sha=None)
    assert 'row count mismatch' in str(excinfo.value)


def test_expected_rows_passes_the_right_count(tmp_path: Path) -> None:
    _expected_rows_case(tmp_path, rows=597, expected=597, strict=True, sha=None)


def test_expected_rows_is_skipped_once_the_sha256_pin_is_filled_in(tmp_path: Path) -> None:
    _expected_rows_case(tmp_path, rows=600, expected=597, strict=True, sha='a' * 64)


def test_expected_rows_is_skipped_outside_strict_mode(tmp_path: Path) -> None:
    _expected_rows_case(tmp_path, rows=600, expected=597, strict=False, sha=None)


def test_an_empty_sha256_pin_counts_as_unpinned(tmp_path: Path) -> None:
    assert not preparer._has_sha_pin(None)
    assert not preparer._has_sha_pin('')
    assert not preparer._has_sha_pin('   ')
    assert preparer._has_sha_pin('a' * 64)
    with pytest.raises(SystemExit):
        _expected_rows_case(tmp_path, rows=600, expected=597, strict=True, sha='  ')


# -----------------------------------------------------------------------------
# 7. what the manifest and the task configs now say
# -----------------------------------------------------------------------------


def test_math_train_is_pinned_to_the_canonical_split_not_the_full_mirror() -> None:
    source = _entry('MATH-train')['source']
    assert source['id'] == 'EleutherAI/hendrycks_math'
    assert source['split'] == 'train'
    assert source['configs'] == [
        'algebra',
        'counting_and_probability',
        'geometry',
        'intermediate_algebra',
        'number_theory',
        'prealgebra',
        'precalculus',
    ]
    assert 'competition_math' not in yaml.safe_dump({'id': source['id']})


def test_cmath_train_subtracts_the_test_split_of_the_same_revision() -> None:
    train = _entry('CMATH-train')
    test = _entry('CMATH-test')
    assert train['source']['split'] == 'validation'
    assert train['source']['subtract_split'] == 'test'
    assert test['source']['split'] == 'test'
    assert train['source']['revision'] == test['source']['revision']
    assert train['expected_rows'] == 597
    assert test['expected_rows'] == 1098


def test_the_three_repinned_entries_carry_an_immutable_revision() -> None:
    """Their upstream moved: the revision is a commit hash (pinned 2026-09-15) and the
    sha256 is either still unfilled or a real digest written on the platform."""
    for name in ('MATH-train', 'CMATH-train', 'CMATH-test'):
        item = _entry(name)
        rev = item['source']['revision']
        assert isinstance(rev, str) and len(rev) == 40 and all(c in '0123456789abcdef' for c in rev), name
        sha = item['sha256']
        assert sha is None or (isinstance(sha, str) and len(sha) == 64), name
    assert _entry('CMATH-train')['source']['revision'] == _entry('CMATH-test')['source']['revision']


def test_the_artifacts_that_did_not_change_keep_their_pins() -> None:
    """Nothing outside the repinned three may lose a revision or a hash."""
    unchanged = {
        'GSM8K-test': 'e53f048856ff4f594e959d75785d2c2d37b678ee',
        'MATH500': '33b8b6415e2c3765cb6b0ac1a63b550167e7eb87',
        'HumanEval': 'd009b64cde5bb5a7d0915975916644532214c91a',
        'APPS': 'ac6966973a0bd7bb274836fc34782df80e56dd93',
        'MBPP-train': '4bb6404fdc6cacfda99d4ac4205087b89d32030c',
        'MBPP-test': '4bb6404fdc6cacfda99d4ac4205087b89d32030c',
        'MBPP+': 'evalplus@0.3.1',
    }
    for name, revision in unchanged.items():
        item = _entry(name)
        assert item['source']['revision'] == revision, name
        assert isinstance(item['sha256'], str) and len(item['sha256']) == 64, name


def test_every_manifest_entry_carries_notes() -> None:
    for item in _manifest_outputs():
        notes = item['source'].get('notes')
        assert isinstance(notes, str) and len(notes) > 40, item['name']


@pytest.mark.parametrize('rel_path', MATH_TASK_YAMLS)
def test_math_tasks_sample_each_problem_once_per_epoch(rel_path: str) -> None:
    env = yaml.safe_load((REPO_ROOT / rel_path).read_text(encoding='utf-8'))['environment']
    assert env['sampling_mode'] == 'concat'
    assert env['reshuffle_each_epoch'] is True


def test_prepare_all_runs_the_gate_in_strict_mode() -> None:
    text = (REPO_ROOT / 'scripts' / '10_data' / 'prepare_all.sh').read_text(encoding='utf-8')
    assert 'scripts/10_data/check_overlap.py' in text
    assert '[[ "$STRICT" == "1" ]]' in text
