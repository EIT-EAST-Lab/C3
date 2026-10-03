# tests/unit/test_rebuild_e1b_cells.py
"""The E1b grid driver: the commands it emits for the noise-law grid.

The grid is the upper block of Table 3, so its cell count, its seeds, its
output layout and the reuse of seed A's alternatives by seed B are
paper-facing facts. The tests read the printed commands, which needs neither a
GPU nor a model, and drive the run loop with a fake process runner.

    python -m pytest tests/unit/test_rebuild_e1b_cells.py -q
"""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path
from typing import List

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / "scripts" / "70_rebuild"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import e1_cells  # noqa: E402
import e1b_cells  # noqa: E402

from c3.analysis.rebuild import aggregate_e1b  # noqa: E402


def _commands(capsys, *extra: str) -> List[List[str]]:
    assert e1b_cells.main(["--model_root", "/models", "--dry-run", *extra]) == 0
    lines = [l for l in capsys.readouterr().out.splitlines() if l and not l.startswith("#")]
    return [shlex.split(l) for l in lines]


def _flag(argv: List[str], name: str) -> str:
    return argv[argv.index(name) + 1]


def test_the_grid_is_three_by_three_by_two_seeds(capsys):
    cmds = _commands(capsys)
    assert len(cmds) == 18
    pairs = {(int(_flag(a, "--num_candidates")), int(_flag(a, "--num_completions"))) for a in cmds}
    assert pairs == {(n, c) for n in (2, 4, 8) for c in (2, 4, 8)}


def test_every_cell_is_the_paper_cell(capsys):
    for argv in _commands(capsys):
        assert argv[1:4] == ["-m", "c3.analysis.analysis", "build-buckets"]
        assert _flag(argv, "--task") == "configs/tasks/math_a2.yaml"
        assert _flag(argv, "--split") == "MATHPOOL"
        assert _flag(argv, "--limit") == "500"
        assert _flag(argv, "--target_role") == "reasoner"
        assert _flag(argv, "--next_role") == "actor"
        assert _flag(argv, "--policy_ckpt") == "/models/Qwen3-4B-Instruct-2507"
        assert "--batched" in argv


def test_seed_b_reuses_the_alternatives_seed_a_recorded(capsys):
    cmds = _commands(capsys)
    by_out = {_flag(a, "--out"): a for a in cmds}
    for out, argv in by_out.items():
        seed = int(_flag(argv, "--seed"))
        if out.endswith("/seedA/buckets.jsonl"):
            assert seed == 1001
            assert "--reuse_candidates_from" not in argv
        else:
            assert out.endswith("/seedB/buckets.jsonl")
            assert seed == 2002
            source = _flag(argv, "--reuse_candidates_from")
            assert source == out.replace("/seedB/", "/seedA/")
            assert source in by_out


def test_seed_a_comes_before_its_seed_b(capsys):
    outs = [_flag(a, "--out") for a in _commands(capsys)]
    for i, out in enumerate(outs):
        if "/seedB/" in out:
            assert outs.index(out.replace("/seedB/", "/seedA/")) < i


def test_the_layout_is_the_one_the_aggregation_reads(capsys):
    outs = [_flag(a, "--out") for a in _commands(capsys, "--results_root", "/r")]
    for out in outs:
        parts = out.split("/")
        assert parts[:3] == ["", "r", "E1b"]
        assert parts[3] in aggregate_e1b.WORKFLOWS
        assert parts[4] == "4b"
        assert aggregate_e1b.CELL_RE.match(parts[5])
        assert parts[6] in aggregate_e1b.SEED_DIRS
        assert parts[7] == "buckets.jsonl"


def test_the_meta_records_the_grid_point_and_the_seed_tag(capsys):
    for argv in _commands(capsys):
        meta = json.loads(_flag(argv, "--meta_json"))
        n, c = int(_flag(argv, "--num_candidates")), int(_flag(argv, "--num_completions"))
        assert meta["rule"] == "e1b_n%d_c%d" % (n, c)
        assert meta["n_alternatives"] == n and meta["replays_per_alt"] == c
        assert meta["seed"] == int(_flag(argv, "--seed"))
        assert meta["seed_tag"] == ("A" if meta["seed"] == 1001 else "B")
        assert meta["question_order_seed"] == 0
        assert meta["limit"] == 500
        assert meta["dataset"] == "MATHPOOL"


def test_only_selects_one_point_with_both_seeds(capsys):
    cmds = _commands(capsys, "--only", "4/8")
    assert len(cmds) == 2
    assert {_flag(a, "--seed") for a in cmds} == {"1001", "2002"}


@pytest.mark.parametrize("selector", ["4", "3/4", "a/b"])
def test_a_bad_selector_is_refused(selector, capsys):
    assert e1b_cells.main(["--model_root", "/m", "--dry-run", "--only", selector]) == 2


def test_two_models_in_one_tree_are_refused(capsys):
    assert e1b_cells.main(["--model_root", "/m", "--dry-run", "--models", "4b=A,m2=B"]) == 2


def test_seed_b_is_refused_while_seed_a_is_missing(tmp_path, monkeypatch, capsys):
    args = e1b_cells.build_parser().parse_args(
        ["--model_root", "/m", "--results_root", str(tmp_path), "--only", "2/2"])
    cells = e1b_cells.build_cells(args)
    ran = []

    def fake_run(argv, cwd, timeout, check):
        ran.append(argv)

        class Done:
            returncode = 1

        return Done()

    monkeypatch.setattr(e1b_cells.subprocess, "run", fake_run)
    failures = e1b_cells.run_cells(cells, timeout_s=10.0, overwrite=False)
    assert len(ran) == 1 and "--reuse_candidates_from" not in ran[0]
    names = [name for name, _ in failures]
    assert any(name.endswith("seedA") for name in names)
    assert any(name.endswith("seedB") and "seed A buckets missing" in why for name, why in failures)


def test_the_shared_helpers_are_the_e1_drivers():
    assert e1b_cells.measured_positions is e1_cells.measured_positions
    assert e1b_cells.build_argv is e1_cells.build_argv
