from __future__ import annotations

import json
from pathlib import Path

import yaml

from c3.tools.main_results import main as main_results_main


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
            'eval_CMATH-test_pass1': 0.60,
            'eval_GSM8K-test_pass1': 0.70,
        },
        'global_step': 1,
    }
    n10_metrics = {
        'metrics': {
            'eval_MATH500_pass1': 0.51,
            'eval_MATH500_pass10': 0.52,
            'eval_CMATH-test_pass1': 0.61,
            'eval_CMATH-test_pass10': 0.62,
            'eval_GSM8K-test_pass1': 0.71,
            'eval_GSM8K-test_pass10': 0.72,
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
    assert summary['methods']['SFT']['CMATH-test']['P@10']['mean'] == 0.62
    assert (out_dir / 'main_results.table.tex').exists()
