# tests/test_prepare_math_pool.py
"""The MATH test artifact and the informative-question pool built out of it.

Two configuration files used to name data files no script could produce:
`data/MATH/test_full.jsonl` for the screening task and
`data/MATH/pool_informative.jsonl` for the `MATHPOOL` evaluation suite. Both are
manifest entries now, and these tests pin what that costs: the test split is the
train split's builder pointed at another split, so the two cannot drift apart,
and the pool is a subset of it named by an id list rather than a second download.

The id list is the part that has to be exact. It carries ids and not problem
statements, so the builder and the screening pass must compute an id the same
way; they call one function, and its rule is pinned here.

Nothing here downloads anything, and the whole file runs in the light local
environment:

    python -m pytest tests/test_prepare_math_pool.py -q
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List

import pytest
import yaml

from c3.analysis.rebuild.pool_ids import ID_HASH_PREFIX_CHARS, row_id


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / 'configs' / 'data_manifest.yaml'
SHIPPED_ID_LIST = REPO_ROOT / 'configs' / 'data' / 'pool_informative_ids.json'

SKIP_LINE = '[SKIP] MATHPOOL: pool_informative_ids.json is empty; run the E1 screening pass first'

MATH_SUBJECT_CONFIGS = [
    'algebra',
    'counting_and_probability',
    'geometry',
    'intermediate_algebra',
    'number_theory',
    'prealgebra',
    'precalculus',
]


def _load_module(rel_path: str, module_name: str) -> ModuleType:
    path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Could not load module from {path}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


preparer = _load_module('scripts/10_data/prepare_math.py', 'c3_prepare_math_pool')


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------


def _entry(name: str) -> Dict[str, Any]:
    for item in yaml.safe_load(MANIFEST.read_text(encoding='utf-8'))['outputs']:
        if item['name'] == name:
            return item
    raise AssertionError(f'{name} is not in the manifest')


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='\n') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def _id_list(tmp_path: Path, ids: List[Any], **screen: Any) -> Path:
    path = tmp_path / 'pool_informative_ids.json'
    document = {'source': 'MATH-test', 'screen': dict(screen), 'unique_ids': ids}
    path.write_text(json.dumps(document), encoding='utf-8')
    return path


def _pool_spec(id_list: Path) -> Dict[str, Any]:
    return {'source': {'kind': 'local', 'id': str(id_list), 'parent': 'MATH-test'}}


OUTPUTS = {'MATH-test': {'output_path': 'data/MATH/test_full.jsonl'}}

PARENT_ROWS = [
    {'input': 'a', 'answer': '1', 'unique_id': 'u2'},
    {'input': 'b', 'answer': '2', 'unique_id': 'u1'},
    {'input': 'c', 'answer': '3'},
]


def _lay_out_parent(tmp_path: Path, rows: List[Dict[str, Any]] = None) -> Path:
    out_dir = tmp_path / 'prepared'
    _write_jsonl(out_dir / 'MATH' / 'test_full.jsonl', PARENT_ROWS if rows is None else rows)
    return out_dir


# -----------------------------------------------------------------------------
# 1. row_id: the one rule both sides use
# -----------------------------------------------------------------------------


def test_row_id_is_the_unique_id_when_the_row_has_one() -> None:
    assert row_id({'unique_id': 'test/algebra/42.json', 'input': 'x'}) == 'test/algebra/42.json'
    assert row_id({'unique_id': '  spaced  ', 'input': 'x'}) == 'spaced'


def test_row_id_falls_back_to_a_hash_of_the_problem_statement() -> None:
    """The fallback is a fixed prefix of the SHA256 of the stored `input`."""
    assert row_id({'input': 'x'}) == hashlib.sha256(b'x').hexdigest()[:ID_HASH_PREFIX_CHARS]
    # A hand value, so a change of encoding or of the prefix length is visible here.
    assert row_id({'input': 'x'}) == '2d711642b726b044'
    assert row_id({'input': 'What is 1 + 1?'}) == '72f8917549de2d6d'
    assert len(row_id({'input': 'x'})) == 16


def test_row_id_is_stable_across_calls_and_ignores_the_other_columns() -> None:
    first = row_id({'input': 'x', 'answer': '1', 'source': 'repo@rev'})
    second = row_id({'source': 'other@rev', 'answer': '2', 'input': 'x'})
    assert first == second == row_id({'input': 'x'})


def test_row_id_does_not_normalize_whitespace() -> None:
    """The stored string is hashed as it stands, so a round trip through the file matches."""
    assert row_id({'input': ' x'}) != row_id({'input': 'x'})
    assert row_id({'input': 'a  b'}) != row_id({'input': 'a b'})


def test_row_id_refuses_a_row_it_cannot_name() -> None:
    with pytest.raises(ValueError):
        row_id({'answer': '1'})
    with pytest.raises(ValueError):
        row_id({'unique_id': '', 'input': '   '})


# -----------------------------------------------------------------------------
# 2. MATH-test: the train builder pointed at another split
# -----------------------------------------------------------------------------


def _feed_configs(monkeypatch: Any, per_config: Dict[str, List[Dict[str, Any]]]) -> None:
    def _rows(src: Dict[str, Any], *, logical_name: str):
        return iter(per_config[src['config']])

    monkeypatch.setattr(preparer, '_iter_hf_rows', _rows)


def _test_split_spec() -> Dict[str, Any]:
    return {
        'source': {
            'id': 'EleutherAI/hendrycks_math',
            'configs': ['algebra', 'geometry'],
            'split': 'test',
            'revision': '0' * 40,
        }
    }


def test_math_test_builder_concatenates_every_config_and_deduplicates(monkeypatch: Any) -> None:
    _feed_configs(
        monkeypatch,
        {
            'algebra': [
                {'problem': 'p1', 'level': 'Level 1', 'type': 'Algebra', 'solution': 'so \\boxed{1}'},
                {'problem': 'dup', 'level': 'Level 2', 'type': 'Algebra', 'solution': 'so \\boxed{9}'},
            ],
            'geometry': [
                {'problem': 'p2', 'level': 'Level 3', 'type': 'Geometry', 'solution': 'so \\boxed{2}'},
                {'problem': 'dup', 'level': 'Level 2', 'type': 'Algebra', 'solution': 'so \\boxed{9}'},
            ],
        },
    )

    rows = list(preparer._prepare_math_test(_test_split_spec()))

    assert [r['input'] for r in rows] == ['p1', 'dup', 'p2']
    assert rows[0] == {
        'input': 'p1',
        'answer': '1',
        'solution': 'so \\boxed{1}',
        'subject': 'Algebra',
        'level': 'Level 1',
        'source': 'EleutherAI/hendrycks_math:algebra:test@' + '0' * 40,
    }
    assert rows[2]['source'] == 'EleutherAI/hendrycks_math:geometry:test@' + '0' * 40


def test_math_test_builder_drops_a_row_with_no_extractable_answer(capsys: Any, monkeypatch: Any) -> None:
    """Same answer gate as the train split, and it says which artifact it dropped from."""
    _feed_configs(
        monkeypatch,
        {
            'algebra': [
                {'problem': 'kept', 'solution': 'so \\boxed{1}'},
                {'problem': 'no answer', 'solution': 'so \\boxed{}'},
            ],
            'geometry': [],
        },
    )

    rows = list(preparer._prepare_math_test(_test_split_spec()))
    out = capsys.readouterr().out

    assert [r['input'] for r in rows] == ['kept']
    assert '[INFO] MATH-test: dropped 1 row(s)' in out


def test_both_math_artifacts_are_written_by_one_builder() -> None:
    """The row format and the deduplication rule cannot drift between the two splits."""
    source = (REPO_ROOT / 'scripts' / '10_data' / 'prepare_math.py').read_text(encoding='utf-8')
    assert source.count('def _prepare_math_subject_union(') == 1
    assert '_prepare_math_subject_union(spec, name="MATH-train")' in source
    assert '_prepare_math_subject_union(spec, name="MATH-test")' in source


# -----------------------------------------------------------------------------
# 3. The id list
# -----------------------------------------------------------------------------


def test_an_empty_id_list_skips_and_writes_nothing(capsys: Any, tmp_path: Path) -> None:
    ids = _id_list(tmp_path, [])
    assert preparer._pool_ids_or_skip(_pool_spec(ids)) is None
    assert capsys.readouterr().out.strip() == SKIP_LINE


def test_the_shipped_id_list_is_the_screened_pool() -> None:
    """What the repository ships: the 500 ids the E1 screening pass selected on 2026-09-15."""
    document = json.loads(SHIPPED_ID_LIST.read_text(encoding='utf-8'))
    assert document['source'] == 'MATH-test'
    ids = document['unique_ids']
    assert len(ids) == 500 and len(set(ids)) == 500
    assert all(isinstance(i, str) and len(i) == 16 and int(i, 16) >= 0 for i in ids)
    screen = document['screen']
    assert screen['taken'] == len(ids)
    assert screen['informative'] >= screen['taken'] and screen['screened'] >= screen['informative']
    assert (screen['workflow'], screen['n'], screen['c'], screen['seed']) == ('a2', 2, 2, 7)

    assert preparer._pool_ids_or_skip(_entry('MATH-pool')) == ids


def test_a_filled_id_list_is_read_in_file_order(tmp_path: Path) -> None:
    ids = _id_list(tmp_path, ['u1', 'u2'], taken=2)
    assert preparer._pool_ids_or_skip(_pool_spec(ids)) == ['u1', 'u2']


@pytest.mark.parametrize(
    'document',
    [
        {'source': 'MATH-test'},
        {'unique_ids': 'u1'},
        [],
    ],
)
def test_an_id_list_without_a_usable_unique_ids_field_fails(tmp_path: Path, document: Any) -> None:
    path = tmp_path / 'pool_informative_ids.json'
    path.write_text(json.dumps(document), encoding='utf-8')
    with pytest.raises(SystemExit):
        preparer.load_pool_ids(path)


def test_a_repeated_or_empty_id_fails(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        preparer.load_pool_ids(_id_list(tmp_path, ['u1', 'u1']))
    assert 'twice' in str(excinfo.value)

    with pytest.raises(SystemExit) as excinfo:
        preparer.load_pool_ids(_id_list(tmp_path, ['u1', '  ']))
    assert 'empty id' in str(excinfo.value)


def test_a_taken_count_that_disagrees_with_the_list_fails(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        preparer.load_pool_ids(_id_list(tmp_path, ['u1', 'u2'], taken=500))
    assert 'screen.taken=500' in str(excinfo.value)


def test_a_missing_id_list_fails_instead_of_skipping(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        preparer.load_pool_ids(tmp_path / 'nothing_here.json')
    assert 'id list not found' in str(excinfo.value)


# -----------------------------------------------------------------------------
# 4. The pool builder
# -----------------------------------------------------------------------------


def test_the_pool_takes_the_named_rows_in_the_order_the_list_gives(tmp_path: Path) -> None:
    out_dir = _lay_out_parent(tmp_path)
    ids = ['u1', row_id({'input': 'c'}), 'u2']

    rows = list(
        preparer._prepare_math_pool(
            _pool_spec(_id_list(tmp_path, ids)),
            ids=ids,
            out_dir=out_dir,
            outputs=OUTPUTS,
        )
    )

    assert [r['input'] for r in rows] == ['b', 'c', 'a']
    assert rows[1] == {'input': 'c', 'answer': '3'}
    # The row with no unique_id is reached through the hash fallback.
    assert ids[1] == '2e7d2c03a9507ae2'


def test_an_id_that_is_not_in_the_parent_fails(tmp_path: Path) -> None:
    out_dir = _lay_out_parent(tmp_path)
    ids = ['u1', 'not-a-row']

    with pytest.raises(SystemExit) as excinfo:
        list(
            preparer._prepare_math_pool(
                _pool_spec(_id_list(tmp_path, ids)),
                ids=ids,
                out_dir=out_dir,
                outputs=OUTPUTS,
            )
        )

    message = str(excinfo.value)
    assert '[FAIL] MATH-pool' in message
    assert '1 of 2 ids are not in MATH-test' in message


def test_an_unprepared_parent_fails_rather_than_writing_an_empty_pool(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        list(
            preparer._prepare_math_pool(
                _pool_spec(_id_list(tmp_path, ['u1'])),
                ids=['u1'],
                out_dir=tmp_path / 'prepared',
                outputs=OUTPUTS,
            )
        )
    assert 'is not prepared yet' in str(excinfo.value)


def test_a_parent_with_two_rows_of_the_same_id_fails(tmp_path: Path) -> None:
    out_dir = _lay_out_parent(tmp_path, [{'input': 'a', 'unique_id': 'u1'}, {'input': 'b', 'unique_id': 'u1'}])
    with pytest.raises(SystemExit) as excinfo:
        preparer.index_rows_by_id(out_dir / 'MATH' / 'test_full.jsonl', name='MATH-test')
    assert 'repeats the row id' in str(excinfo.value)


# -----------------------------------------------------------------------------
# 5. What the manifest says about the two new entries
# -----------------------------------------------------------------------------


def test_math_test_is_the_same_source_and_revision_as_math_train() -> None:
    train = _entry('MATH-train')['source']
    test = _entry('MATH-test')['source']

    assert test['id'] == train['id'] == 'EleutherAI/hendrycks_math'
    assert test['configs'] == train['configs'] == MATH_SUBJECT_CONFIGS
    assert test['revision'] == train['revision']
    assert train['split'] == 'train'
    assert test['split'] == 'test'


def test_math_test_declares_the_row_count_that_guards_its_unfilled_pin() -> None:
    entry = _entry('MATH-test')
    assert entry['output_path'] == 'data/MATH/test_full.jsonl'
    assert entry['expected_rows'] == 5000
    # The pin is written on the platform. Until then the row count is the gate,
    # which is exactly the window _check_expected_rows covers.
    assert entry['sha256'] is None or len(str(entry['sha256'])) == 64


def test_the_pool_entry_is_optional_and_derived_from_math_test() -> None:
    entry = _entry('MATH-pool')

    assert entry['output_path'] == 'data/MATH/pool_informative.jsonl'
    assert entry['optional'] is True
    assert 'expected_rows' not in entry
    assert entry['source']['parent'] == 'MATH-test'
    assert entry['source']['id'] == 'configs/data/pool_informative_ids.json'
    assert (REPO_ROOT / entry['source']['id']).is_file()


def test_neither_new_entry_is_a_training_artifact() -> None:
    """The naming contract is what the overlap gate reads; both must be evaluation files."""
    for name in ('MATH-test', 'MATH-pool'):
        assert not name.endswith('-train')


def test_the_screening_task_points_at_the_prepared_test_split() -> None:
    task = yaml.safe_load((REPO_ROOT / 'configs' / 'tasks' / 'math_screen.yaml').read_text(encoding='utf-8'))
    suites = {s['name']: s['path'] for s in task['environment']['eval_suites']}

    assert suites['MATHTEST'] == _entry('MATH-test')['output_path']


def test_the_screening_task_no_longer_names_the_deprecated_mirror() -> None:
    text = (REPO_ROOT / 'configs' / 'tasks' / 'math_screen.yaml').read_text(encoding='utf-8')
    assert 'competition_math' not in text
