# tests/unit/test_rebuild_e4_reference_cells.py
"""The E4 reference cell driver: the build-buckets commands it emits.

The reference is what the critic comparison is scored against, so the replay
count, the replay seeds, the refusal of the evaluation cell's own seed, the
reuse of the recorded alternatives and the output layout are pinned here. The
prep directory is the one `e4_critic.py prepare-reference` writes for a small
synthetic cell, so the two scripts are checked against each other.

    python -m pytest tests/unit/test_rebuild_e4_reference_cells.py -q
"""

from __future__ import annotations

import importlib.util
import json
import shlex
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / "scripts" / "70_rebuild"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import e4_critic as e4  # noqa: E402
import e4_reference_cells as ref  # noqa: E402


def _critic_tests():
    path = Path(__file__).with_name("test_rebuild_e4_critic.py")
    spec = importlib.util.spec_from_file_location("_e4_critic_fixtures", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def prep(tmp_path):
    fx = _critic_tests()
    _cell, split_path, doc, _ = fx._split(tmp_path, frac=0.3, seed=0)
    out = tmp_path / "prep"
    e4.run_prepare(split_path, str(out))
    return out, doc


def _commands(capsys, prep_dir, *extra):
    argv = ["--prep", str(prep_dir), "--model_root", "/models", "--results_root", "/r", "--dry-run", *extra]
    assert ref.main(argv) == 0
    lines = [l for l in capsys.readouterr().out.splitlines() if l and not l.startswith("#")]
    return [shlex.split(l) for l in lines]


def _flag(argv, name):
    return argv[argv.index(name) + 1]


def test_two_reference_cells_at_sixteen_continuations(prep, capsys):
    prep_dir, doc = prep
    cmds = _commands(capsys, prep_dir)
    assert [int(_flag(a, "--seed")) for a in cmds] == [3003, 4004]
    for argv in cmds:
        assert argv[1:4] == ["-m", "c3.analysis.analysis", "build-buckets"]
        assert _flag(argv, "--num_completions") == "16"
        assert _flag(argv, "--num_candidates") == "4"
        assert _flag(argv, "--limit") == str(doc["n_heldout"])
        assert _flag(argv, "--split") == e4.HELDOUT_SUITE
        assert _flag(argv, "--target_role") == "reasoner"
        assert _flag(argv, "--next_role") == "actor"
        assert _flag(argv, "--policy_ckpt") == "/models/Qwen3-4B-Instruct-2507"
        assert _flag(argv, "--include_real_as_j0") == "false"
        assert "--batched" in argv


def test_the_alternatives_are_the_recorded_ones(prep, capsys):
    prep_dir, _ = prep
    for argv in _commands(capsys, prep_dir):
        reuse = _flag(argv, "--reuse_candidates_from")
        assert reuse.endswith("/" + e4.HELDOUT_REUSE_FILE)
        assert Path(reuse).is_file()
        assert Path(_flag(argv, "--task")).name == e4.HELDOUT_TASK_FILE


def test_the_cells_land_where_evaluate_reads_them(prep, capsys):
    prep_dir, _ = prep
    outs = [_flag(a, "--out") for a in _commands(capsys, prep_dir)]
    assert outs == [
        "/r/E4/reference_heldout/a2/4b/n4_c16_seed3003/buckets.jsonl",
        "/r/E4/reference_heldout/a2/4b/n4_c16_seed4004/buckets.jsonl",
    ]


def test_the_meta_records_the_reference_seed_and_the_cells_seed(prep, capsys):
    prep_dir, _ = prep
    for argv in _commands(capsys, prep_dir):
        meta = json.loads(_flag(argv, "--meta_json"))
        assert meta["seed"] == int(_flag(argv, "--seed"))
        assert meta["source_seed"] == 0
        assert meta["replays_per_alt"] == 16
        assert meta["temperature"] == 0.7 and meta["top_p"] == 0.8 and meta["top_k"] == 20
        decoding = json.loads(_flag(argv, "--decoding_json"))
        assert "seed" not in decoding


def test_the_cells_own_seed_is_refused(prep, capsys):
    prep_dir, _ = prep
    assert ref.main(["--prep", str(prep_dir), "--model_root", "/m", "--seeds", "0", "--dry-run"]) == 2
    assert "own seed" in capsys.readouterr().err


def test_a_missing_prep_directory_is_refused(tmp_path, capsys):
    assert ref.main(["--prep", str(tmp_path), "--model_root", "/m", "--dry-run"]) == 2


def test_all_points_replays_the_whole_cell(prep, tmp_path, capsys):
    prep_dir, _ = prep
    source = tmp_path / "cell.jsonl"
    cmds = _commands(capsys, prep_dir, "--all-points", "--source", str(source), "--seeds", "3003")
    assert len(cmds) == 1
    argv = cmds[0]
    assert _flag(argv, "--task") == "configs/tasks/math_a2.yaml"
    assert _flag(argv, "--split") == "MATHPOOL"
    assert _flag(argv, "--limit") == "10"
    assert _flag(argv, "--reuse_candidates_from") == str(source)
    assert "/E4/reference_all/" in _flag(argv, "--out")
    assert ref.main(["--prep", str(prep_dir), "--model_root", "/m", "--all-points", "--dry-run"]) == 2
