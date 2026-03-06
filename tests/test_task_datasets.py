from __future__ import annotations

from pathlib import Path

from c3.integration.marl_specs import load_task
from c3.integration.task_datasets import load_task_datasets


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_load_task_exposes_dataset_specs_from_environment() -> None:
    spec = load_task(str(REPO_ROOT / 'configs' / 'tasks' / 'math.yaml'))
    assert spec.train_datasets[0]['name'] == 'MATH-train'
    assert spec.eval_suites[0]['name'] == 'MATH500'
    assert spec.repo_root == str(REPO_ROOT)


def test_load_task_datasets_uses_repo_relative_paths_and_eval_names(monkeypatch, tmp_path) -> None:
    spec = load_task(str(REPO_ROOT / 'tests' / 'fixtures' / 'tasks' / 'mini_math.yaml'))
    monkeypatch.chdir(tmp_path)
    td = load_task_datasets(spec)
    assert list(td.evals.keys()) == ['MATH500']
    row = td.train[0]
    assert row['datasource'] == 'fixture-math-train'
    assert row['input'] == 'What is 1 + 1?'
    assert row['answer'] == '2'


def test_load_task_datasets_supports_code_fixture(monkeypatch, tmp_path) -> None:
    spec = load_task(str(REPO_ROOT / 'tests' / 'fixtures' / 'tasks' / 'mini_code.yaml'))
    monkeypatch.chdir(tmp_path)
    td = load_task_datasets(spec)
    assert list(td.evals.keys()) == ['MBPP-test', 'MBPP+']
    row = td.train[0]
    assert row['datasource'] == 'fixture-code-train'
    assert row['input'].startswith('Write a function add')
