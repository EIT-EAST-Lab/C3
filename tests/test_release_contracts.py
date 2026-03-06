from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from c3.utils.budget_ledger import append_ledger, make_budget_record
from c3.utils.context_key import fingerprint, hash63


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(rel_path: str, module_name: str):
    path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Could not load module from {path}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_hash63_is_stable_and_non_negative() -> None:
    text = 'reasoner -> actor'
    assert hash63(text) == hash63(text)
    assert 0 <= hash63(text) < (1 << 63)
    assert len(fingerprint(text)) >= 8


def test_budget_ledger_appends_jsonl_and_stringifies_unknowns(tmp_path) -> None:
    run_dir = tmp_path / 'run'
    record = make_budget_record(
        global_step=3,
        epoch_idx=1,
        iter_in_epoch=2,
        marl_algorithm='c3',
        n_questions_in_batch=2,
        n_samples_per_prompt=8,
        roles_topo=['reasoner', 'actor'],
        fanout=[2, 4],
    )
    record['opaque'] = object()
    append_ledger(str(run_dir), record)
    payload = json.loads((run_dir / 'budget_ledger.jsonl').read_text(encoding='utf-8').splitlines()[0])
    assert payload['total_eval_calls'] == 16
    assert payload['roles_topo'] == ['reasoner', 'actor']
    assert isinstance(payload['opaque'], str)


def test_no_data_check_flags_non_empty_generated_dirs(tmp_path) -> None:
    mod = _load_module('scripts/audit/no_data_check.py', 'no_data_check')
    (tmp_path / 'data').mkdir()
    (tmp_path / 'data' / 'mini.jsonl').write_text('{}\n', encoding='utf-8')
    (tmp_path / 'artifacts').mkdir()
    (tmp_path / 'artifacts' / 'report.json').write_text('{}\n', encoding='utf-8')
    hits = mod._scan_files(tmp_path, max_bytes=1024 * 1024)
    reasons = [reason for _, reason in hits]
    assert 'Generated datasets must not be committed' in reasons
    assert 'Generated reports/plots must not be committed' in reasons
