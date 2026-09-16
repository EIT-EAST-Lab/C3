from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

from c3.utils.budget_ledger import append_ledger, make_budget_record
from c3.utils.context_key import fingerprint, hash63
from c3.utils.paper_train_contract import (
    PAPER_TRAIN_BUDGET_B,
    PAPER_TRAIN_DEPARTURES,
    PAPER_TRAIN_METHOD_N_SAMPLES,
    PAPER_TRAIN_RECIPE,
    PAPER_TRAIN_STORE_TRUE_FLAGS,
    get_paper_train_n_samples,
    render_paper_train_args,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module(rel_path: str, module_name: str):
    path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Could not load module from {path}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _paper_train_script_text() -> str:
    return (REPO_ROOT / 'scripts' / '40_train' / 'paper_train.sh').read_text(encoding='utf-8')


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


def test_paper_train_contract_explicitly_keeps_b8_for_all_paper_methods() -> None:
    assert PAPER_TRAIN_BUDGET_B == 8
    assert dict(PAPER_TRAIN_METHOD_N_SAMPLES) == {
        'MAPPO': 8,
        'MAGRPO': 8,
        'C3': 8,
    }
    assert [get_paper_train_n_samples(m) for m in ('MAPPO', 'MAGRPO', 'C3')] == [8, 8, 8]


def test_paper_train_script_uses_contract_helper() -> None:
    text = (REPO_ROOT / 'scripts' / '40_train' / 'paper_train.sh').read_text(encoding='utf-8')
    assert 'from c3.utils.paper_train_contract import get_paper_train_n_samples' in text
    assert '--n_samples_per_prompt "$n_samples_per_prompt"' in text


def test_paper_train_script_renders_the_recipe_contract() -> None:
    # Half one: the script asks this module for the recipe and expands the answer.
    text = _paper_train_script_text()
    assert 'from c3.utils.paper_train_contract import render_paper_train_args' in text
    assert 'print(render_paper_train_args())' in text
    assert 'recipe_args_line="$(_paper_train_recipe_args)"' in text
    assert 'read -r -a recipe_args <<< "$recipe_args_line"' in text
    assert '"${recipe_args[@]}"' in text

    # Half two: the rendered string splits on whitespace into exactly the recipe,
    # which is what the shell does with it.
    tokens = render_paper_train_args().split()
    n_valued = 2 * len(PAPER_TRAIN_RECIPE)
    assert tokens[:n_valued:2] == list(PAPER_TRAIN_RECIPE)
    assert tokens[1:n_valued:2] == list(PAPER_TRAIN_RECIPE.values())
    assert tokens[n_valued:] == list(PAPER_TRAIN_STORE_TRUE_FLAGS)


def test_every_contract_flag_exists_in_the_trainer_parser() -> None:
    # Read the parser as text on purpose: importing it would pull in torch, which
    # the CPU tier does not install.
    text = (REPO_ROOT / 'openrlhf' / 'cli' / 'train_ppo_ray_tooling.py').read_text(encoding='utf-8')
    flags = list(PAPER_TRAIN_RECIPE) + list(PAPER_TRAIN_STORE_TRUE_FLAGS) + list(PAPER_TRAIN_DEPARTURES)
    missing = [
        flag
        for flag in flags
        if not re.search(r'add_argument\(\s*[\'"]' + re.escape(flag) + r'[\'"]', text)
    ]
    assert missing == [], f'flags the trainer parser does not define: {missing}'


def test_paper_train_departures_are_disjoint_and_explained() -> None:
    recipe = set(PAPER_TRAIN_RECIPE)
    store_true = set(PAPER_TRAIN_STORE_TRUE_FLAGS)
    departures = set(PAPER_TRAIN_DEPARTURES)
    assert recipe & store_true == set()
    assert recipe & departures == set()
    assert store_true & departures == set()

    for flag, departure in PAPER_TRAIN_DEPARTURES.items():
        assert departure.value.strip(), flag
        assert departure.paper_value.strip(), flag
        assert departure.value.strip() != departure.paper_value.strip(), flag
        # A departure has to say what it departs from and point at the document
        # that argues for it, otherwise it is just an undocumented difference.
        assert 'docs/' in departure.reason, flag


def test_paper_train_script_does_not_repeat_recipe_flags() -> None:
    text = _paper_train_script_text()
    repeated = [flag for flag in list(PAPER_TRAIN_RECIPE) + list(PAPER_TRAIN_STORE_TRUE_FLAGS) if flag in text]
    assert repeated == [], f'flags written in the script as well as in the recipe: {repeated}'


def test_paper_train_script_passes_the_declared_departures() -> None:
    text = _paper_train_script_text()
    missing = [f'{flag} {d.value}' for flag, d in PAPER_TRAIN_DEPARTURES.items() if f'{flag} {d.value}' not in text]
    assert missing == [], f'departures declared but not passed by the script: {missing}'


def test_experience_maker_no_longer_forces_mappo_to_single_sample() -> None:
    text = (REPO_ROOT / 'openrlhf' / 'trainer' / 'ppo_utils' / 'experience_maker.py').read_text(encoding='utf-8')
    assert 'n_samples_per_prompt must be 1 for MAS/C3 tasks' not in text


def test_no_data_check_flags_non_empty_generated_dirs(tmp_path) -> None:
    mod = _load_module('scripts/90_audit/no_data_check.py', 'no_data_check')
    (tmp_path / 'data').mkdir()
    (tmp_path / 'data' / 'mini.jsonl').write_text('{}\n', encoding='utf-8')
    (tmp_path / 'artifacts').mkdir()
    (tmp_path / 'artifacts' / 'report.json').write_text('{}\n', encoding='utf-8')
    hits = mod._scan_files(tmp_path, max_bytes=1024 * 1024)
    reasons = [reason for _, reason in hits]
    assert 'Generated datasets must not be committed' in reasons
    assert 'Generated reports/plots must not be committed' in reasons
