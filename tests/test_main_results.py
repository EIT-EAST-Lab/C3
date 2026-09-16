from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from c3.reporting import main_results
from c3.reporting.main_results import main as main_results_main


def _write_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj) + '\n', encoding='utf-8')


def test_main_results_aggregate_from_metrics(tmp_path) -> None:
    ckpt_root = tmp_path / 'ckpt'
    run_root = ckpt_root / '_runs' / 'fixture_math_seed0'
    registry_path = tmp_path / 'registry.yaml'
    out_dir = tmp_path / 'out'

    registry = {
        'version': 1,
        'defaults': {'out_subdir': 'main_results'},
        'runs': [
            {
                'id': 'fixture_math_seed0',
                'method': 'SFT',
                'task': 'math',
                'seed': 0,
                'source': {
                    'type': 'train_run_dir',
                    'train_run_dir': str(run_root),
                },
            }
        ],
    }
    registry_path.write_text(yaml.safe_dump(registry, sort_keys=False), encoding='utf-8')

    greedy_metrics = {
        'metrics': {
            'eval_MATH500_pass1': 0.50,
            'eval_Minerva-Math_pass1': 0.60,
            'eval_AMC23_pass1': 0.70,
            'eval_AIME24_pass1': 0.10,
            'eval_AIME25_pass1': 0.20,
        },
        'global_step': 1,
    }
    n10_metrics = {
        'metrics': {
            'eval_MATH500_pass1': 0.51,
            'eval_MATH500_pass10': 0.52,
            'eval_Minerva-Math_pass1': 0.61,
            'eval_Minerva-Math_pass10': 0.62,
            'eval_AMC23_pass1': 0.71,
            'eval_AMC23_pass10': 0.72,
            'eval_AIME24_pass1': 0.11,
            'eval_AIME24_pass10': 0.12,
            'eval_AIME25_pass1': 0.21,
            'eval_AIME25_pass10': 0.22,
        },
        'global_step': 1,
    }

    _write_jsonl(run_root / 'main_results' / 'math' / 'greedy' / 'eval_only.jsonl.metrics.jsonl', greedy_metrics)
    _write_jsonl(run_root / 'main_results' / 'math' / 'n10' / 'eval_only.jsonl.metrics.jsonl', n10_metrics)

    rc = main_results_main([
        'aggregate',
        '--registry', str(registry_path),
        '--ckpt_root', str(ckpt_root),
        '--out_dir', str(out_dir),
        '--strict', '1',
        '--prefer_metrics', '1',
        '--validate_samples_when_strict', '0',
        '--expected_runs_per_method_task', '1',
    ])
    assert rc == 0

    summary = json.loads((out_dir / 'main_results.summary.json').read_text(encoding='utf-8'))
    assert summary['methods']['SFT']['MATH500']['Greedy']['mean'] == 0.50
    assert summary['methods']['SFT']['Minerva-Math']['P@10']['mean'] == 0.62
    assert (out_dir / 'main_results.table.tex').exists()


def test_the_expected_math_suites_are_the_main_table_suites() -> None:
    """The evaluation protocol of 2026-09-15 made these five the main table, and
    moved the two saturation controls to an appendix list of their own."""
    assert main_results.EXPECTED_DATASOURCES['math'] == [
        'MATH500', 'Minerva-Math', 'AMC23', 'AIME24', 'AIME25'
    ]
    assert main_results.APPENDIX_DATASOURCES['math'] == ['GSM8K-test', 'CMATH-test']
    assert main_results.TABLE_COLUMNS == [
        'MATH500', 'Minerva-Math', 'AMC23', 'AIME', 'MBPP+', 'MBPP-test'
    ]
    assert main_results.APPENDIX_TABLE_COLUMNS == ['GSM8K-test', 'CMATH-test']


def test_the_two_aime_suites_are_reported_as_one_merged_column(tmp_path) -> None:
    """`AIME` is a reported column and not a prepared file, the same merge
    `scripts/70_rebuild/final_eval.py` writes. Both parts have 30 problems, so
    the mean of the two accuracies is the mean over the 60 problems; a run that
    is missing one part gets no column rather than half of one.
    """
    ckpt_root = tmp_path / 'ckpt'
    run_root = ckpt_root / '_runs' / 'fixture_math_seed0'
    registry_path = tmp_path / 'registry.yaml'
    out_dir = tmp_path / 'out'

    registry = {
        'version': 1,
        'defaults': {'out_subdir': 'main_results'},
        'runs': [
            {
                'id': 'fixture_math_seed0',
                'method': 'SFT',
                'task': 'math',
                'seed': 0,
                'source': {'type': 'train_run_dir', 'train_run_dir': str(run_root)},
            }
        ],
    }
    registry_path.write_text(yaml.safe_dump(registry, sort_keys=False), encoding='utf-8')

    greedy = {'metrics': {}, 'global_step': 1}
    n10 = {'metrics': {}, 'global_step': 1}
    for suite, g, p1, p10 in [
        ('MATH500', 0.50, 0.51, 0.52),
        ('Minerva-Math', 0.60, 0.61, 0.62),
        ('AMC23', 0.70, 0.71, 0.72),
        ('AIME24', 0.10, 0.11, 0.12),
        ('AIME25', 0.20, 0.21, 0.22),
    ]:
        greedy['metrics']['eval_%s_pass1' % suite] = g
        n10['metrics']['eval_%s_pass1' % suite] = p1
        n10['metrics']['eval_%s_pass10' % suite] = p10

    _write_jsonl(run_root / 'main_results' / 'math' / 'greedy' / 'eval_only.jsonl.metrics.jsonl', greedy)
    _write_jsonl(run_root / 'main_results' / 'math' / 'n10' / 'eval_only.jsonl.metrics.jsonl', n10)

    rc = main_results_main([
        'aggregate',
        '--registry', str(registry_path),
        '--ckpt_root', str(ckpt_root),
        '--out_dir', str(out_dir),
        '--strict', '1',
        '--prefer_metrics', '1',
        '--validate_samples_when_strict', '0',
        '--expected_runs_per_method_task', '1',
    ])
    assert rc == 0

    summary = json.loads((out_dir / 'main_results.summary.json').read_text(encoding='utf-8'))
    aime = summary['methods']['SFT']['AIME']
    assert aime['Greedy']['mean'] == pytest.approx((0.10 + 0.20) / 2)
    assert aime['P@1']['mean'] == pytest.approx((0.11 + 0.21) / 2)
    assert aime['P@10']['mean'] == pytest.approx((0.12 + 0.22) / 2)

    # The parts stay in the summary; the merge adds a column, it replaces nothing.
    assert summary['methods']['SFT']['AIME24']['Greedy']['mean'] == 0.10
    assert summary['methods']['SFT']['AIME25']['Greedy']['mean'] == 0.20

    tex = (out_dir / 'main_results.table.tex').read_text(encoding='utf-8')
    assert '% Columns: MATH500 | Minerva-Math | AMC23 | AIME | MBPP+ | MBPP-test' in tex
