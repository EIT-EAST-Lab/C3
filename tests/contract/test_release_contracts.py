from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

from c3.utils.budget_ledger import append_ledger, make_budget_record
from c3.utils.context_key import fingerprint, hash63
from c3.utils.paper_train_contract import (
    PAPER_TRAIN_BUDGET_B,
    PAPER_TRAIN_DEFAULT_RECIPE,
    PAPER_TRAIN_METHOD_N_SAMPLES,
    PAPER_TRAIN_METHODS,
    PAPER_TRAIN_RECIPES,
    get_paper_train_n_samples,
    get_paper_train_recipe,
    paper_train_settings,
    render_paper_train_args,
)
from c3.utils.paper_train_contract import main as contract_main


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
    for recipe in PAPER_TRAIN_RECIPES:
        for method in PAPER_TRAIN_METHODS:
            assert paper_train_settings(recipe, method)['--n_samples_per_prompt'].value == '8'


def test_paper_train_recipes_are_the_three_trained_tables() -> None:
    assert list(PAPER_TRAIN_RECIPES) == ['main', 'long', 'a3']
    assert PAPER_TRAIN_DEFAULT_RECIPE == 'main'
    assert get_paper_train_recipe(None).name == 'main'
    assert get_paper_train_recipe('main').table == 'Table 2'
    assert get_paper_train_recipe('long').table == 'Table 7'
    assert get_paper_train_recipe('a3').table == 'Table 8'
    assert get_paper_train_recipe('main').task_yaml == 'configs/tasks/math_5seed.yaml'
    assert get_paper_train_recipe('long').task_yaml == 'configs/tasks/math.yaml'
    assert get_paper_train_recipe('a3').task_yaml == 'configs/tasks/math_5seed_a3.yaml'
    assert get_paper_train_recipe('main').train_seeds == (0, 1, 2, 3, 4)
    assert get_paper_train_recipe('long').train_seeds == (0,)
    assert get_paper_train_recipe('a3').train_seeds == (0, 1, 2, 3, 4)
    for recipe in PAPER_TRAIN_RECIPES.values():
        assert (REPO_ROOT / recipe.task_yaml).is_file(), recipe.task_yaml
    with pytest.raises(KeyError):
        get_paper_train_recipe('code')


def _value(recipe: str, method: str, flag: str):
    return paper_train_settings(recipe, method)[flag].value


def test_paper_train_recipes_carry_the_tables_values() -> None:
    # Table 5, the rows every run shares.
    for recipe in PAPER_TRAIN_RECIPES:
        for method in PAPER_TRAIN_METHODS:
            assert _value(recipe, method, '--actor_learning_rate') == '1e-6'
            assert _value(recipe, method, '--lr_warmup_ratio') == '0.03'
            assert _value(recipe, method, '--train_batch_size') == '256'
            assert _value(recipe, method, '--rollout_batch_size') == '128'
            assert _value(recipe, method, '--max_epochs') == '1'
            assert _value(recipe, method, '--eps_clip') == '0.2'
            assert _value(recipe, method, '--init_kl_coef') == '0.01'
            assert _value(recipe, method, '--kl_target') == '0.1'
            assert _value(recipe, method, '--kl_estimator') == 'k3'
            assert _value(recipe, method, '--prompt_max_len') == '2560'
            assert _value(recipe, method, '--l2') == '0.0'
            assert _value(recipe, method, '--max_norm') == '1.0'
            assert paper_train_settings(recipe, method)['--bf16'].value is None
    # Table 5, the rows that differ between the 5-seed and the long-generation run.
    for recipe in ('main', 'a3'):
        assert _value(recipe, 'C3', '--generate_max_len') == '512'
        assert _value(recipe, 'C3', '--train_epochs') == '5'
        assert _value(recipe, 'C3', '--micro_train_batch_size') == '64'
        assert _value(recipe, 'C3', '--micro_rollout_batch_size') == '32'
        assert _value(recipe, 'C3', '--eval_every_ratio') == '0.05'
        assert '--max_samples' not in paper_train_settings(recipe, 'C3')
    assert _value('long', 'C3', '--generate_max_len') == '2048'
    assert _value('long', 'C3', '--train_epochs') == '2'
    assert _value('long', 'C3', '--max_samples') == '5000'
    assert _value('long', 'C3', '--micro_train_batch_size') == '4'
    assert _value('long', 'C3', '--micro_rollout_batch_size') == '8'
    assert _value('long', 'C3', '--eval_every_ratio') == '0.25'
    # Table 6 and Appendix B.2, per method.
    for recipe in PAPER_TRAIN_RECIPES:
        assert _value(recipe, 'C3', '--c3_credit_variant') == 'reward_only'
        assert _value(recipe, 'C3', '--c3_baseline_mode') == 'loo'
        assert _value(recipe, 'MAPPO', '--critic_pretrain') == 'jhu-clsp/mmBERT-base'
        assert _value(recipe, 'MAPPO', '--critic_learning_rate') == '5e-5'
        assert _value(recipe, 'MAPPO', '--critic_warmup_steps') == '64'
        assert _value(recipe, 'MAPPO', '--mappo_normalize_scope') == 'global'
        assert _value(recipe, 'MAPPO', '--mappo_state_max_len') == '2560'
        assert _value(recipe, 'MAGRPO', '--magrpo_baseline') == 'group_mean'
        assert _value(recipe, 'MAGRPO', '--magrpo_adv_unit') == 'joint_action'
        assert '--critic_pretrain' not in paper_train_settings(recipe, 'C3')
        assert '--critic_pretrain' not in paper_train_settings(recipe, 'MAGRPO')
    assert _value('main', 'C3', '--c3_fanout') == '2,4'
    assert _value('long', 'C3', '--c3_fanout') == '2,4'
    assert _value('a3', 'C3', '--c3_fanout') == '2,2,2'


def test_the_5seed_recipes_carry_the_run_memory_layout_and_c3_estimator() -> None:
    for recipe in ('main', 'a3'):
        for method in PAPER_TRAIN_METHODS:
            settings = paper_train_settings(recipe, method)
            assert settings['--packing_samples'].value is None
            assert settings['--use_dynamic_batch'].value is None
            assert settings['--train_max_tokens_per_gpu'].value == '32768'
            assert settings['--rollout_max_tokens_per_gpu'].value == '32768'
            for flag in ('--packing_samples', '--use_dynamic_batch',
                         '--train_max_tokens_per_gpu', '--rollout_max_tokens_per_gpu'):
                assert settings[flag].source == 'run configuration'
        assert _value(recipe, 'C3', '--advantage_estimator') == 'reinforce'
        assert paper_train_settings(recipe, 'C3')['--advantage_estimator'].source == 'run configuration'
        assert '--advantage_estimator' not in paper_train_settings(recipe, 'MAPPO')
        assert '--advantage_estimator' not in paper_train_settings(recipe, 'MAGRPO')
    for method in PAPER_TRAIN_METHODS:
        settings = paper_train_settings('long', method)
        for flag in ('--packing_samples', '--use_dynamic_batch', '--advantage_estimator'):
            assert flag not in settings


def test_the_long_recipe_keeps_the_step_250_checkpoint_to_the_end() -> None:
    """Replays the trainer's save and rotation rules on the long recipe's schedule.

    Saves: the evaluation at the start, one after every evaluation (every
    round(0.25 x max_steps) steps), one every --save_steps steps, and one after a
    final evaluation when max_steps is not a multiple of the interval. Before each
    save the oldest checkpoints are deleted while --max_ckpt_num or more are on
    disk (openrlhf/utils/deepspeed/deepspeed.py, save_ckpt).
    """
    for method in PAPER_TRAIN_METHODS:
        settings = paper_train_settings('long', method)
        save_steps = int(settings['--save_steps'].value)
        keep = int(settings['--max_ckpt_num'].value)
        ratio = float(settings['--eval_every_ratio'].value)
        per_pass = int(settings['--max_samples'].value) // 32  # --vllm_generate_batch_size default, drop_last
        max_steps = per_pass * int(settings['--train_epochs'].value)
        assert max_steps == 312
        eval_steps = max(1, min(max_steps, int(round(max_steps * ratio))))
        saves = ['eval_start_step0']
        for step in range(1, max_steps + 1):
            if step % eval_steps == 0:
                saves.append('eval_step%d' % step)
            if step % save_steps == 0:
                saves.append('global_step%d' % step)
        if max_steps % eval_steps:
            saves.append('eval_step%d_final' % max_steps)
        on_disk = []
        for tag in saves:
            while len(on_disk) >= keep:
                on_disk.pop(0)
            on_disk.append(tag)
        assert 'global_step250' in on_disk, (method, on_disk)
        assert settings['--save_steps'].source.startswith('Table 7')


def test_paper_train_script_passes_extra_args_through() -> None:
    text = _paper_train_script_text()
    assert '#   EXTRA_ARGS' in text
    assert 'EXTRA_ARGS="${EXTRA_ARGS:-}"' in text
    assert 'read -r -a EXTRA <<< "$EXTRA_ARGS"' in text
    assert '${EXTRA[@]+"${EXTRA[@]}"}' in text


def _dry_run_commands(recipe: str) -> list[list[str]]:
    import os
    import shlex
    import shutil
    import subprocess
    import sys

    bash = shutil.which('bash')
    if not bash:
        pytest.skip('no bash on this machine')
    env = dict(os.environ, DRY_RUN='1', RECIPE=recipe, PRETRAIN='Qwen/Qwen3-4B-Instruct-2507',
               PYTHON_BIN=Path(sys.executable).as_posix(), EXTRA_ARGS='')
    out = subprocess.run([bash, 'scripts/40_train/paper_train.sh'], cwd=REPO_ROOT, env=env,
                         capture_output=True, text=True, check=True).stdout
    return [shlex.split(line) for line in out.splitlines() if line.strip() and not line.startswith('#')]


def _flag(cmd: list[str], flag: str) -> str:
    assert cmd.count(flag) == 1, (flag, cmd)
    return cmd[cmd.index(flag) + 1]


def test_paper_train_script_gives_every_run_its_own_checkpoint_directory() -> None:
    text = _paper_train_script_text()
    assert '--ckpt_path "$run_dir"' in text
    assert '--ckpt_path "$CKPT_ROOT"' not in text
    seen: dict[str, str] = {}
    for recipe in PAPER_TRAIN_RECIPES:
        cmds = _dry_run_commands(recipe)
        assert len(cmds) == len(PAPER_TRAIN_METHODS) * len(get_paper_train_recipe(recipe).train_seeds), recipe
        for cmd in cmds:
            ckpt = _flag(cmd, '--ckpt_path')
            assert ckpt == _flag(cmd, '--run_dir'), cmd
            assert ckpt == 'ckpt/_runs/' + _flag(cmd, '--run_id'), cmd
            assert ckpt not in seen, (ckpt, seen.get(ckpt), _flag(cmd, '--marl_algorithm'))
            seen[ckpt] = '%s/%s' % (recipe, _flag(cmd, '--marl_algorithm'))
    assert len(seen) == 15 + 3 + 15


def test_every_c3_fanout_spends_the_budget() -> None:
    for recipe in PAPER_TRAIN_RECIPES:
        fanout = [int(x) for x in _value(recipe, 'C3', '--c3_fanout').split(',')]
        product = 1
        for x in fanout:
            product *= x
        assert product == PAPER_TRAIN_BUDGET_B, recipe


def test_every_setting_names_its_source() -> None:
    for recipe in PAPER_TRAIN_RECIPES:
        for method in PAPER_TRAIN_METHODS:
            for flag, setting in paper_train_settings(recipe, method).items():
                assert setting.source.strip(), (recipe, method, flag)
                assert setting.source.startswith(('Table ', 'Appendix ', 'run configuration')), (
                    recipe, method, flag, setting.source)


def test_paper_train_script_uses_the_contract_command_line() -> None:
    text = _paper_train_script_text()
    assert '"$PYTHON_BIN" -m c3.utils.paper_train_contract "$@"' in text
    assert 'TASK_YAML="$(_contract task --recipe "$RECIPE")"' in text
    assert 'SEEDS_LINE="$(_contract seeds --recipe "$RECIPE")"' in text
    assert 'recipe_args_line="$(_contract args --recipe "$RECIPE" --method "$alg")"' in text
    assert 'read -r -a recipe_args <<< "$recipe_args_line"' in text
    assert '"${recipe_args[@]}"' in text
    assert 'RECIPE="${RECIPE:-main}"' in text
    assert 'METHODS=(MAPPO MAGRPO C3)' in text


def test_the_rendered_string_splits_into_exactly_the_settings() -> None:
    for recipe in PAPER_TRAIN_RECIPES:
        for method in PAPER_TRAIN_METHODS:
            expected = []
            for flag, setting in paper_train_settings(recipe, method).items():
                expected.append(flag)
                if setting.value is not None:
                    expected.append(setting.value)
            assert render_paper_train_args(recipe, method).split() == expected


def test_the_contract_command_line_prints_each_piece(capsys) -> None:
    assert contract_main(['task', '--recipe', 'a3']) == 0
    assert capsys.readouterr().out.strip() == 'configs/tasks/math_5seed_a3.yaml'
    assert contract_main(['seeds', '--recipe', 'long']) == 0
    assert capsys.readouterr().out.strip() == '0'
    assert contract_main(['args', '--recipe', 'main', '--method', 'c3']) == 0
    assert capsys.readouterr().out.strip() == render_paper_train_args('main', 'C3')
    assert contract_main(['args', '--recipe', 'nope', '--method', 'C3']) == 2
    assert contract_main(['args', '--recipe', 'main', '--method', 'GRPO']) == 2


def test_every_contract_flag_exists_in_the_trainer_parser() -> None:
    # Read the parser as text on purpose: importing it would pull in torch, which
    # the CPU tier does not install.
    text = (REPO_ROOT / 'openrlhf' / 'cli' / 'train_ppo_ray_tooling.py').read_text(encoding='utf-8')
    flags = set()
    for recipe in PAPER_TRAIN_RECIPES:
        for method in PAPER_TRAIN_METHODS:
            flags.update(paper_train_settings(recipe, method))
    missing = sorted(
        flag
        for flag in flags
        if not re.search(r'add_argument\(\s*[\'"]' + re.escape(flag) + r'[\'"]', text)
    )
    assert missing == [], f'flags the trainer parser does not define: {missing}'


def test_paper_train_script_does_not_repeat_recipe_flags() -> None:
    text = _paper_train_script_text()
    flags = set()
    for recipe in PAPER_TRAIN_RECIPES:
        for method in PAPER_TRAIN_METHODS:
            flags.update(paper_train_settings(recipe, method))
    repeated = sorted(flag for flag in flags if re.search(re.escape(flag) + r'(?![a-z_])', text))
    assert repeated == [], f'flags written in the script as well as in the recipe: {repeated}'


def test_paper_train_script_trains_math_only() -> None:
    text = _paper_train_script_text()
    assert 'code.yaml' not in text
    assert 'TASKS=' not in text


def test_the_trainer_default_credit_variant_is_reward_only() -> None:
    text = (REPO_ROOT / 'openrlhf' / 'cli' / 'train_ppo_ray_tooling.py').read_text(encoding='utf-8')
    block = text[text.index('"--c3_credit_variant",'):]
    block = block[:block.index(')')]
    assert 'default="reward_only"' in block
    assert '"value_assisted"' in block


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
