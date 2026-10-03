"""CPU tests of the 5-seed task files and of evaluating a 3-agent model with final_eval.py.

Covers configs/tasks/math_5seed.yaml and math_5seed_a3.yaml (the training tasks
of RECIPE=main and RECIPE=a3) and configs/tasks/math_final_eval_a3.yaml (the
evaluation of the 3-agent 5-seed run, Table 8): the training data is the MATH
training split alone, the roles are the team the recipe trains, and
scripts/70_rebuild/final_eval.py plans, writes and summarizes a greedy
evaluation of a 3-agent model on the four benchmarks of Table 8, whose record
c3.analysis.rebuild.aggregate_5seed then reads.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, List

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / "scripts" / "70_rebuild"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import final_eval  # noqa: E402

from c3.analysis.rebuild import aggregate_5seed  # noqa: E402
from c3.utils.paper_train_contract import get_paper_train_recipe  # noqa: E402

TASKS = REPO_ROOT / "configs" / "tasks"
EVAL_A3 = "configs/tasks/math_final_eval_a3.yaml"
TABLE8_SUITES = ["MATH500", "AIME25", "CMATH-test", "GSM8K-test"]
K1 = "MATH500=1,AIME25=1,CMATH-test=1,GSM8K-test=1"


def _load(name: str) -> dict:
    return yaml.safe_load((TASKS / name).read_text(encoding="utf-8"))


def _roles(task_name: str) -> List[str]:
    spec = _load(task_name)
    path = (TASKS / spec["mas"]["roles_path"]).resolve()
    return [entry["role"] for entry in json.loads(path.read_text(encoding="utf-8"))]


def _flag(argv: List[str], name: str) -> str:
    return argv[argv.index(name) + 1]


@pytest.mark.parametrize("name", ["math_5seed.yaml", "math_5seed_a3.yaml", "math_final_eval_a3.yaml"])
def test_the_5seed_tasks_train_on_the_math_training_split_alone(name: str) -> None:
    env = _load(name)["environment"]
    assert [(d["name"], d["path"]) for d in env["train_datasets"]] == [("MATH-train", "data/MATH/train.jsonl")]


def test_the_recipes_train_the_5seed_tasks_with_their_teams() -> None:
    assert get_paper_train_recipe("main").task_yaml == "configs/tasks/math_5seed.yaml"
    assert get_paper_train_recipe("a3").task_yaml == "configs/tasks/math_5seed_a3.yaml"
    assert _roles("math_5seed.yaml") == ["reasoner", "actor"]
    assert _roles("math_5seed_a3.yaml") == ["reasoner", "actor", "verifier"]
    # The long-generation run keeps MATH-train with CMATH.
    assert get_paper_train_recipe("long").task_yaml == "configs/tasks/math.yaml"


def test_the_3agent_evaluation_task_is_the_training_task_with_the_table8_suites() -> None:
    train, evaluation = _load("math_5seed_a3.yaml"), _load("math_final_eval_a3.yaml")
    assert evaluation["mas"] == train["mas"]
    for field in ("train_datasets", "math_backend", "reward_mode", "use_math_verify", "episode_length"):
        assert evaluation["environment"][field] == train["environment"][field], field
    assert [s["name"] for s in evaluation["environment"]["eval_suites"]] == TABLE8_SUITES
    assert all(s["limit"] is None for s in evaluation["environment"]["eval_suites"])


def _args(tmp_path: Path, *extra: str):
    argv = ["--policy", "ckpt/_runs/paper_a3_C3_math_seed0/final_hf", "--task", EVAL_A3,
            "--k_by_suite", K1, "--generate_max_len", "512", "--method", "C3",
            "--run_id", "paper_a3_C3_math_seed0", "--ckpt_root", str(tmp_path / "ckpt"), *extra]
    return final_eval.build_parser().parse_args(argv)


def test_final_eval_plans_one_greedy_group_at_512_tokens(tmp_path: Path) -> None:
    groups = final_eval.plan_groups(_args(tmp_path))
    assert [g.label for g in groups] == ["k1"]
    group = groups[0]
    assert list(group.suites) == TABLE8_SUITES
    assert (group.profile, group.temperature, group.top_p, group.top_k) == ("greedy", 0.0, None, None)
    assert group.env["GENERATE_MAX_LEN"] == "512"
    assert _flag(group.argv, "--method") == "C3"
    assert _flag(group.argv, "--extra_eval_args") == "--eval_n_samples_per_prompt 1 --eval_temperature 0.0"


def test_the_generated_group_task_keeps_the_three_agent_roles(tmp_path: Path) -> None:
    group = final_eval.plan_groups(_args(tmp_path))[0]
    written = final_eval.write_group_task(group, main_task=REPO_ROOT / EVAL_A3)
    spec = yaml.safe_load(Path(written).read_text(encoding="utf-8"))
    roles_path = Path(spec["mas"]["roles_path"])
    assert roles_path.is_absolute()
    assert roles_path.name == "roles_trio.json"
    assert [s["name"] for s in spec["environment"]["eval_suites"]] == TABLE8_SUITES


def test_dry_run_with_the_3agent_task(tmp_path: Path, capsys: Any, monkeypatch: Any) -> None:
    def _explode(*_a: Any, **_k: Any) -> None:
        raise AssertionError("a dry run must not start a subprocess")

    monkeypatch.setattr(final_eval.subprocess, "run", _explode)
    code = final_eval.main(["--policy", "/models/m", "--task", EVAL_A3, "--k_by_suite", K1,
                            "--generate_max_len", "512", "--method", "C3", "--dry-run",
                            "--ckpt_root", str(tmp_path / "ckpt")])
    out = capsys.readouterr().out
    assert code == 0
    assert "# groups: 1" in out
    assert "suites=MATH500,AIME25,CMATH-test,GSM8K-test" in out
    assert "GENERATE_MAX_LEN=512" in out


def test_the_default_sample_counts_do_not_fit_the_3agent_task(tmp_path: Path) -> None:
    """The defaults name suites this task does not declare, so --k_by_suite has to be given."""
    args = final_eval.build_parser().parse_args(["--policy", "/models/m", "--task", EVAL_A3])
    with pytest.raises(ValueError):
        final_eval.plan_groups(args)


def _fake_dump(group: final_eval.EvalGroup, solved: dict, sizes: dict) -> None:
    eval_dir = Path(group.eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for suite, n in sizes.items():
        for q in range(n):
            rows.append({"datasource": suite, "question_id": "%s-%d" % (suite, q),
                         "answer_reward": 1.0 if q < solved[suite] else 0.0})
    Path(group.eval_jsonl).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    metrics = {"eval_%s_pass1" % s: solved[s] / sizes[s] for s in sizes}
    Path(group.metrics_jsonl).write_text(json.dumps({"kind": "eval_metrics", "metrics": metrics}) + "\n",
                                         encoding="utf-8")


def test_a_3agent_record_reaches_table8(tmp_path: Path) -> None:
    """final_eval.py's record of a 3-agent model is what aggregate_5seed reads into Table 8."""
    sizes = {"MATH500": 20, "AIME25": 10, "CMATH-test": 10, "GSM8K-test": 10}
    results = tmp_path / "records"
    for seed, solved_math in enumerate((15, 16, 17)):
        args = _args(tmp_path, "--run_id", "a3_c3_seed%d" % seed)
        group = final_eval.plan_groups(args)[0]
        _fake_dump(group, {"MATH500": solved_math, "AIME25": 1, "CMATH-test": 9, "GSM8K-test": 9}, sizes)
        record = final_eval.build_record([group], args)
        assert record["ok"], record["problems"]
        assert record["suites"]["AIME25"]["k"] == 1
        out = results / "a3_4b" / "c3" / ("train_s%d" % seed) / "eval_s0" / "final_eval_record.json"
        out.parent.mkdir(parents=True)
        out.write_text(json.dumps(record), encoding="utf-8")

    summary = tmp_path / "summary.json"
    assert aggregate_5seed.main(["--results", str(results), "--out", str(summary)]) == 0
    doc = json.loads(summary.read_text(encoding="utf-8"))
    cell = doc["tables"]["a3_4b"]["rows"]["c3"]
    assert cell["math500"]["mean"] == pytest.approx(80.0)
    assert cell["math500"]["std"] == pytest.approx(5.0)
    assert cell["aime25"]["mean"] == pytest.approx(10.0)
    assert cell["avg3"]["mean"] == pytest.approx((80.0 + 90.0 + 90.0) / 3)
    assert doc["keys"]["E2main.a3_4b.math500.c3.mean"]["value"] == pytest.approx(80.0)
    assert doc["problems"] == []
