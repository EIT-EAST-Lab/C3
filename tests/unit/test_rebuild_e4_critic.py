# tests/unit/test_rebuild_e4_critic.py
"""Tests for scripts/70_rebuild/e4_critic.py on small synthetic cells.

Nothing here reads a results tree or needs torch: split, prepare-reference and
evaluate run for real; train and score run their orchestration (sweep,
selection, refit, data accounting, alignment) with the torch parts replaced by
small fakes, so the suite runs on the CPU tier. The model code itself needs a
GPU machine with torch and transformers.

    python -m pytest tests/unit/test_rebuild_e4_critic.py -q
"""
from __future__ import annotations

import json
import math
import os
import sys
import types

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT_DIR = os.path.join(REPO_ROOT, "scripts", "70_rebuild")
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import e4_critic as e4  # noqa: E402

ENV = {
    "env_name": "MathEnv", "episode_length": 1,
    "eval_suites": [{"limit": None, "name": "MATHPOOL", "path": "data/MATH/pool_informative.jsonl"}],
    "math_backend": "marft", "mix_seed": 42, "reshuffle_each_epoch": True,
    "reward_mode": "vote_binary", "sampling_mode": "concat",
    "train_datasets": [{"limit": None, "name": "MATH-train", "path": "data/MATH/train.jsonl", "weight": 1.0}],
    "use_math_verify": True,
}
DEC = {"max_new_tokens": 512, "temperature": 0.7, "top_k": 20, "top_p": 0.8}
PRETRAIN = "/models/Qwen3-4B-Instruct-2507/"

# Four replays per alternative for ten questions. Question 3 has a constant C3
# estimate (every alternative 2/4), so its C3 rank correlation is undefined.
RETURNS = [
    [[1, 1, 1, 0], [0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0]],
    [[0, 0, 0, 1], [1, 1, 1, 1], [0, 1, 1, 0], [0, 0, 0, 0]],
    [[1, 0, 1, 0], [1, 1, 1, 0], [0, 0, 0, 0], [1, 0, 0, 0]],
    [[1, 1, 0, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 0, 1, 1]],
    [[0, 1, 0, 0], [1, 1, 0, 1], [0, 0, 0, 0], [1, 1, 1, 1]],
    [[1, 1, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 1]],
    [[0, 0, 1, 0], [1, 0, 1, 1], [1, 1, 1, 0], [0, 0, 0, 0]],
    [[1, 0, 0, 1], [0, 0, 0, 0], [1, 1, 1, 1], [0, 1, 0, 0]],
    [[0, 1, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0], [1, 1, 1, 0]],
    [[1, 1, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 1, 1]],
]


def _bucket(qid, texts, returns, *, seed=0, question=None, ctx_hash=None, label=None,
            reuse=False, env=None, nexts_tag="c3", decoding=None, pretrain=PRETRAIN):
    question = question if question is not None else "Question number %d?" % qid
    return {
        "bucket_id": "bkt_%d_%s" % (qid, seed),
        "ctx_hash": ctx_hash if ctx_hash is not None else 7000 + qid,
        "target_role": "reasoner",
        "question_id": qid,
        "restart": {
            "question": question, "question_id": qid, "role_outputs_prefix": {},
            "roles_topo": ["reasoner", "actor"],
            "meta": {"label": label if label is not None else str(qid), "dataset_meta": None,
                     "env_name": "MathEnv", "eval_suite": "MATHPOOL",
                     "task_env_cfg": env if env is not None else ENV},
        },
        "candidates": [
            {"j": j, "action_text": t, "returns": [float(x) for x in r],
             "next_actions": ["%s reply q%d j%d r%d \\boxed{%d}" % (nexts_tag, qid, j, k, k)
                              for k in range(len(r))]}
            for j, (t, r) in enumerate(zip(texts, returns))
        ],
        "meta": {
            "seed": seed, "decoding": dict(decoding or DEC, seed=seed), "next_role": "actor",
            "candidate_total_req": len(texts), "candidate_total_got": len(texts), "v_extra_n": 0,
            "pretrain": pretrain, "seed_scheme": "batched_v1", "workflow": "a2", "model": "4b",
            "rule": "sweep_n%d" % len(texts), "n_alternatives": len(texts),
            "replays_per_alt": len(returns[0]), "dataset": "MATHPOOL", "limit": 10,
            "candidate_reuse": True if reuse else None,
        },
    }


def _texts(qid, n=4, tag="alt"):
    return ["Q%d %s %d: some reasoning text" % (qid, tag, j) for j in range(n)]


def _write(path, buckets):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for b in buckets:
            fh.write(json.dumps(b, ensure_ascii=False) + "\n")
    return str(path)


def _cell(tmp_path, name="cell.jsonl"):
    buckets = [_bucket(q, _texts(q), RETURNS[q]) for q in range(10)]
    return _write(tmp_path / name, buckets), buckets


def _split(tmp_path, frac=0.3, seed=0):
    cell_path, buckets = _cell(tmp_path)
    split_path = str(tmp_path / "split.json")
    doc = e4.run_split(cell_path, split_path, heldout_frac=frac, seed=seed)
    return cell_path, split_path, doc, buckets


def _reference(tmp_path, split_doc, *, seed=3003, replays=16, drop=None, swap_texts_at=None,
               name="ref.jsonl", extra_points=()):
    """Reference buckets for the held-out points: 16 returns, reference mean = pattern / 16."""
    out = []
    for idx, p in enumerate(split_doc["heldout"]):
        if drop is not None and idx == drop:
            continue
        q = p["question_id"]
        texts = [c["action_text"] for c in p["candidates"]]
        if swap_texts_at is not None and idx == swap_texts_at:
            texts = list(reversed(texts))
        rets = []
        for j in range(len(texts)):
            k = (q * 3 + j * 5) % 17  # 0..16 correct replays, varies by point and alternative
            k = min(k, replays)
            rets.append([1.0] * k + [0.0] * (replays - k))
        out.append(_bucket(q, texts, rets, seed=seed, question=p["question"],
                           ctx_hash=p["ctx_hash"], label=p["label"], reuse=True, nexts_tag="ref"))
    for q in extra_points:
        out.append(_bucket(q, _texts(q), [[1.0] * replays] * 4, seed=seed, reuse=True, nexts_tag="ref"))
    return _write(tmp_path / name, out)


def _scores(tmp_path, split_path, split_doc, mult=1, name=None, fn=None, debug=None):
    split_sha = e4.sha256_file(split_path)
    fn = fn or (lambda p, j: float((p["question_id"] * 7 + j * 3) % 5) + 0.1 * j)
    points = [{"question_key": p["question_key"], "question_id": p["question_id"],
               "ctx_hash": p["ctx_hash"], "scores": [fn(p, j) for j in range(len(p["candidates"]))]}
              for p in split_doc["heldout"]]
    doc = {"schema": e4.SCORES_SCHEMA, "split": {"path": split_path, "sha256": split_sha},
           "model": {"data_multiplier": mult,
                     "selected": {"lr": 3e-5, "head": "small", "widths": [], "val_mse": 0.1},
                     "hyperparameters": {"epochs": 3},
                     "data": {"base": {"n_rows": 28, "n_rollouts": 112, "n_points": 7},
                              "all": {"n_rows": 28 * mult, "n_rollouts": 112 * mult, "n_points": 7},
                              "ratio_rows": float(mult), "ratio_rollouts": float(mult),
                              "debug_max_rows": debug},
                     "files": ["20_data/results/E1/a2/4b/sweep_n4/buckets.jsonl"]},
           "truncation": {"max_len": 512, "n": 12, "n_truncated": 0},
           "points": points}
    path = tmp_path / (name or "scores_%dx.json" % mult)
    path.write_text(json.dumps(doc), encoding="utf-8")
    return str(path)


def _eval_args(split_path, ref_path, scores, out_dir, undefined="exclude", manifest=None, **kw):
    argv = ["evaluate", "--split", split_path, "--reference", ref_path, "--scores", scores[1],
            "--undefined-rho", undefined, "--out-dir", str(out_dir),
            "--manifest", manifest if manifest is not None else str(out_dir) + "_no_manifest.json"]
    if 2 in scores:
        argv += ["--scores-2x", scores[2]]
    if 4 in scores:
        argv += ["--scores-4x", scores[4]]
    for k, v in kw.items():
        argv += ["--" + k.replace("_", "-"), str(v)]
    return argv


# -----------------------------------------------------------------------------
# rank statistics against hand-worked examples
# -----------------------------------------------------------------------------


def test_spearman_hand_examples():
    # one swapped pair among four: 1 - 6 * 2 / (4 * 15) = 0.8
    assert e4.spearman([1, 2, 3, 4], [1, 3, 2, 4]) == pytest.approx(0.8)
    assert e4.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    # ties take average ranks: x ranks (1.5, 1.5, 3, 4), y ranks (1, 2, 3, 4);
    # centred products sum to 4.5, squares to 4.5 and 5, so r = 4.5 / sqrt(22.5)
    assert e4.spearman([0.25, 0.25, 0.5, 1.0], [0.1, 0.2, 0.3, 0.4]) == pytest.approx(4.5 / math.sqrt(22.5))
    # a constant side has no ordering: undefined, not zero
    assert e4.spearman([0.5, 0.5, 0.5, 0.5], [1, 2, 3, 4]) is None
    assert e4.spearman([1, 2], [3, 3]) is None
    assert e4.spearman([1], [1]) is None


def test_loo_advantage_hand_example_and_ranking():
    means = [0.25, 0.5, 0.75, 1.0]
    # each mean minus the mean of the other three: 0.25 - 0.75, 0.5 - 2/3, 0.75 - 7/12, 1.0 - 0.5
    assert e4.loo_advantages(means) == pytest.approx([-0.5, 0.5 - 2.0 / 3.0, 0.75 - 1.75 / 3.0, 0.5])
    ref = [0.1, 0.9, 0.4, 0.6]
    assert e4.spearman(e4.loo_advantages(means), ref) == pytest.approx(e4.spearman(means, ref))
    with pytest.raises(e4.E4Error):
        e4.loo_advantages([1.0])


def test_paired_stats_hand_example():
    # differences 0.8, 0.4, 0.6: mean 0.6, sample sd 0.2, t = 0.6 / (0.2 / sqrt 3);
    # with two degrees of freedom the two-sided p is 1 - t / sqrt(2 + t^2)
    st = e4.paired_stats([1.0, 0.8, 0.6], [0.2, 0.4, 0.0], n_boot=200, seed=0)
    t = 0.6 / (0.2 / math.sqrt(3))
    assert st["n"] == 3
    assert st["mean_diff"] == pytest.approx(0.6)
    assert st["t"] == pytest.approx(t)
    assert st["p_t"] == pytest.approx(1 - t / math.sqrt(2 + t * t))
    lo, hi = st["boot_ci95"]
    assert 0.4 - 1e-12 <= lo <= 0.6 <= hi <= 0.8 + 1e-12
    empty = e4.paired_stats([], [])
    assert empty["n"] == 0 and empty["mean_diff"] is None


def test_format_p_tiers():
    assert e4.format_p(2.4e-13) == "< 0.001"
    assert e4.format_p(0.0009999) == "< 0.001"
    assert e4.format_p(0.001) == "< 0.01"
    assert e4.format_p(0.0099) == "< 0.01"
    assert e4.format_p(0.01) == "= 0.01"
    assert e4.format_p(0.304) == "= 0.30"
    assert e4.format_p(None) is None
    assert e4.format_p(float("nan")) is None


def _repo_module(name):
    return __import__(name, fromlist=["_"])


def test_format_p_and_source_path_agree_with_the_repository():
    summary = _repo_module("c3.analysis.rebuild.summary")
    for p in (1e-9, 0.0005, 0.001, 0.004, 0.0099, 0.01, 0.049, 0.5, 1.0, None, float("inf")):
        assert e4.format_p(p) == summary.format_p(p)
    for path in ("20_data/results/E4/x/buckets.jsonl", "/abs/runs/E4/b.jsonl",
                 "a/20_data/results/E1/c/buckets.jsonl"):
        assert e4.source_path(path) == summary.source_path(path)


# -----------------------------------------------------------------------------
# split: held-out points never reach training
# -----------------------------------------------------------------------------


def test_split_is_deterministic_and_records_the_heldout_points(tmp_path):
    cell_path, split_path, doc, buckets = _split(tmp_path, frac=0.3, seed=0)
    assert doc["n_heldout"] == 3 and doc["n_train_points"] == 7
    again = e4.run_split(cell_path, str(tmp_path / "again.json"), heldout_frac=0.3, seed=0)
    assert [p["question_id"] for p in again["heldout"]] == [p["question_id"] for p in doc["heldout"]]
    other = e4.run_split(cell_path, str(tmp_path / "other.json"), heldout_frac=0.3, seed=5)
    assert len(other["heldout"]) == 3
    p = doc["heldout"][0]
    src = buckets[p["question_id"]]
    assert p["ctx_hash"] == src["ctx_hash"] and p["context"] == src["restart"]["question"]
    assert [c["returns"] for c in p["candidates"]] == [c["returns"] for c in src["candidates"]]
    assert [c["action_text"] for c in p["candidates"]] == [c["action_text"] for c in src["candidates"]]
    assert doc["eval_cell"]["sha256"] == e4.sha256_file(cell_path)
    assert doc["eval_cell"]["facts"]["seed"] == 0
    held = {q["question_key"] for q in doc["heldout"]}
    assert held.isdisjoint(doc["train_question_keys"])
    assert len(held) + len(doc["train_question_keys"]) == 10


def test_split_leaves_no_heldout_question_or_alternative_in_training(tmp_path):
    cell_path, split_path, doc, _ = _split(tmp_path, frac=0.3, seed=0)
    held_q, held_c = e4.heldout_sets(doc)
    held_ids = [p["question_id"] for p in doc["heldout"]]
    train_ids = [q for q in range(10) if q not in held_ids]
    # a second file over the same questions with other alternatives (another seed),
    # and one training question that carries a held-out alternative's text verbatim,
    # the way cells built under one seed share alternatives across branching factors
    shared_text = doc["heldout"][0]["candidates"][1]["action_text"]
    other = []
    for q in range(10):
        texts = _texts(q, tag="other")
        if q == train_ids[0]:
            texts[2] = shared_text
        other.append(_bucket(q, texts, RETURNS[q], seed=1001))
    other_path = _write(tmp_path / "other.jsonl", other)
    rows, acct = e4.collect_rows([cell_path, other_path], held_q, held_c)
    assert all(r["question_key"] not in {p["question_key"] for p in doc["heldout"]} for r in rows)
    assert all(r["action"] not in held_c for r in rows)
    assert all(r["context"] not in held_q for r in rows)
    assert acct[0]["dropped_heldout_question"] == 3 and acct[1]["dropped_heldout_question"] == 3
    assert acct[1]["dropped_heldout_alternative"] == 1
    assert acct[0]["n_rows"] == 28 and acct[1]["n_rows"] == 27
    assert acct[0]["n_rollouts"] == 112
    tot = e4.totals(acct)
    assert tot["n_rows"] == 55


def test_split_refuses_a_repeated_question(tmp_path):
    buckets = [_bucket(q, _texts(q), RETURNS[q]) for q in range(4)]
    buckets.append(_bucket(9, _texts(9), RETURNS[9], question=buckets[0]["restart"]["question"]))
    path = _write(tmp_path / "dup.jsonl", buckets)
    with pytest.raises(e4.E4Error):
        e4.run_split(path, str(tmp_path / "s.json"), 0.25, 0)


def test_split_refuses_a_cell_with_two_seeds(tmp_path):
    buckets = [_bucket(q, _texts(q), RETURNS[q]) for q in range(4)]
    buckets.append(_bucket(5, _texts(5), RETURNS[5], seed=7))
    path = _write(tmp_path / "mixed.jsonl", buckets)
    with pytest.raises(e4.E4Error):
        e4.run_split(path, str(tmp_path / "s.json"), 0.25, 0)


def test_inner_split_keeps_each_question_on_one_side():
    rows = [{"question_key": "q%d" % (i // 4), "target": 0.5} for i in range(40)]
    tr, va = e4.inner_split(rows, 0.2, 0)
    assert len(va) == 8 and len(tr) == 32
    assert {rows[i]["question_key"] for i in tr}.isdisjoint({rows[i]["question_key"] for i in va})


def test_parse_heads():
    assert e4.parse_heads(["small=linear", "large=768"]) == [("small", []), ("large", [768])]
    assert e4.parse_heads(["deep=512,256"]) == [("deep", [512, 256])]
    for bad in (["small"], ["a=linear", "a=768"], ["x=0"], ["x="], ["x=abc"]):
        with pytest.raises(e4.E4Error):
            e4.parse_heads(bad)


# -----------------------------------------------------------------------------
# prepare-reference: what build-buckets reads
# -----------------------------------------------------------------------------


def test_prepare_reference_writes_the_heldout_points(tmp_path):
    cell_path, split_path, doc, buckets = _split(tmp_path, frac=0.3, seed=0)
    out = tmp_path / "prep"
    ids = [e4.question_key(b["restart"]["question"]) for b in buckets]
    ids_path = tmp_path / "ids.json"
    ids_path.write_text(json.dumps({"unique_ids": ids}), encoding="utf-8")
    rep = e4.run_prepare(split_path, str(out), pool_ids=str(ids_path))
    assert rep["n_points"] == 3 and rep["pool_ids_checked"] is True
    assert rep["source_seed"] == 0
    bb = rep["build_buckets"]
    assert bb["--split"] == e4.HELDOUT_SUITE and bb["--limit"] == 3 and bb["--num_candidates"] == 4
    assert bb["--target_role"] == "reasoner" and bb["--next_role"] == "actor"
    assert bb["--decoding_json"] == DEC
    # the reuse file is the held-out lines of the cell, byte for byte
    src_lines = [l.rstrip("\n") for l in open(cell_path, encoding="utf-8")]
    reuse_lines = [l.rstrip("\n") for l in open(out / e4.HELDOUT_REUSE_FILE, encoding="utf-8")]
    assert reuse_lines == [src_lines[p["question_id"]] for p in doc["heldout"]]
    pool = [json.loads(l) for l in open(out / e4.HELDOUT_POOL_FILE, encoding="utf-8")]
    assert pool == [{"id": p["question_id"], "input": p["question"], "answer": p["label"]} for p in doc["heldout"]]
    text = (out / e4.HELDOUT_TASK_FILE).read_text(encoding="utf-8")
    task = json.loads(text[text.index("{"):])
    assert task["environment"]["eval_suites"] == [{"name": e4.HELDOUT_SUITE, "path": e4.HELDOUT_POOL_FILE, "limit": None}]
    assert {k: v for k, v in task["environment"].items() if k != "eval_suites"} == \
        {k: v for k, v in ENV.items() if k != "eval_suites"}
    assert task["mas"]["roles_path"] == "math/roles_duo.json"
    yaml = pytest.importorskip("yaml")
    assert yaml.safe_load(text) == task


def test_prepare_reference_reuse_file_is_read_by_the_repository_loader(tmp_path):
    replay = _repo_module("c3.analysis.replay")
    _cell_path, split_path, doc, _ = _split(tmp_path, frac=0.3, seed=0)
    out = tmp_path / "prep"
    e4.run_prepare(split_path, str(out))
    reuse = replay.load_candidate_reuse(str(out / e4.HELDOUT_REUSE_FILE))
    assert len(reuse) == 3 and reuse.requested_total() == 4
    for p in doc["heldout"]:
        assert reuse.texts_for(p["question_id"], p["ctx_hash"], total_req=4) == \
            [c["action_text"] for c in p["candidates"]]


def test_prepare_reference_refuses_a_changed_cell_or_a_wrong_pool_id(tmp_path):
    cell_path, split_path, doc, buckets = _split(tmp_path, frac=0.3, seed=0)
    ids = [e4.question_key(b["restart"]["question"]) for b in buckets]
    ids[doc["heldout"][0]["question_id"]] = "0" * 16
    ids_path = tmp_path / "ids.json"
    ids_path.write_text(json.dumps({"unique_ids": ids}), encoding="utf-8")
    with pytest.raises(e4.E4Error):
        e4.run_prepare(split_path, str(tmp_path / "p1"), pool_ids=str(ids_path))
    with open(cell_path, "a", encoding="utf-8") as fh:
        fh.write("\n")
    with pytest.raises(e4.E4Error):
        e4.run_prepare(split_path, str(tmp_path / "p2"))


# -----------------------------------------------------------------------------
# evaluate
# -----------------------------------------------------------------------------


def _full_setup(tmp_path, frac=0.3):
    cell_path, split_path, doc, buckets = _split(tmp_path, frac=frac, seed=0)
    ref = _reference(tmp_path, doc)
    scores = {m: _scores(tmp_path, split_path, doc, mult=m) for m in (1, 2, 4)}
    return cell_path, split_path, doc, ref, scores


def test_evaluate_writes_every_key_the_manifest_expects(tmp_path):
    _c, split_path, doc, ref, scores = _full_setup(tmp_path)
    out = tmp_path / "eval"
    assert e4.main(_eval_args(split_path, ref, scores, out)) == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    ev = json.loads((out / "e4_eval.json").read_text(encoding="utf-8"))
    want = set(e4.PRIMARY_KEYS) | {e4.KEY_MULT[2], e4.KEY_MULT[4]}
    assert set(summary["keys"]) == want
    assert e4.VERDICT_KEY not in summary["keys"]
    assert summary["experiment"] == "E4" and summary["undefined_rho"] == "exclude"
    n = summary["keys"][e4.KEY_N]["value"]
    assert isinstance(n, int) and n > 0
    for key, rec in summary["keys"].items():
        assert set(rec) >= {"value", "n", "unit", "source", "note"}
        assert isinstance(rec["source"], list) and rec["source"]
        assert rec["unit"] == e4.KEY_UNITS[key]
    assert summary["keys"][e4.KEY_P]["value"] in ("< 0.001", "< 0.01") or \
        summary["keys"][e4.KEY_P]["value"].startswith("= ")
    for key in (e4.KEY_CRITIC, e4.KEY_C3):
        assert -1.0 <= summary["keys"][key]["value"] <= 1.0
    assert summary["keys"][e4.KEY_DIFF]["value"] == pytest.approx(
        summary["keys"][e4.KEY_C3]["value"] - summary["keys"][e4.KEY_CRITIC]["value"])
    assert ev["keys"] == summary["keys"]
    assert len(ev["points"]) == doc["n_heldout"]
    assert ev["criterion"]["decided_by"].endswith("not written by this script")
    assert ev["results"]["zero"]["1x"]["stats"]["n"] >= ev["results"]["exclude"]["1x"]["stats"]["n"]
    assert ev["diagnostics"]["reference_seed"] == 3003 and ev["diagnostics"]["cell_seed"] == 0
    assert ev["diagnostics"]["alternatives_whose_first_c_reference_replays_are_the_c3_replays"] == 0


def test_evaluate_keys_follow_a_given_manifest(tmp_path):
    assert e4.DEFAULT_MANIFEST is None
    _c, split_path, _doc, ref, scores = _full_setup(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({k: {"unit": u} for k, u in e4.KEY_UNITS.items()}), encoding="utf-8")
    out = tmp_path / "eval"
    assert e4.main(_eval_args(split_path, ref, scores, out, manifest=str(manifest_path))) == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    manifest = e4.read_json(str(manifest_path))
    assert set(summary["keys"]) == set(e4.PRIMARY_KEYS) | {e4.KEY_MULT[2], e4.KEY_MULT[4]}
    for key, rec in summary["keys"].items():
        assert key in manifest and rec["unit"] == manifest[key]["unit"]


def test_evaluate_undefined_conventions(tmp_path):
    # one held-out point with a constant C3 estimate (question 3); a split that holds it out
    cell_path, _b = _cell(tmp_path)
    for seed in range(50):
        doc = e4.run_split(cell_path, str(tmp_path / "s.json"), 0.3, seed)
        if 3 in [p["question_id"] for p in doc["heldout"]]:
            break
    else:
        pytest.fail("no split holds question 3 out")
    split_path = str(tmp_path / "s.json")
    ref = _reference(tmp_path, doc)
    scores = {1: _scores(tmp_path, split_path, doc)}
    e4.main(_eval_args(split_path, ref, scores, tmp_path / "ex", undefined="exclude"))
    e4.main(_eval_args(split_path, ref, scores, tmp_path / "ze", undefined="zero"))
    ex = json.loads((tmp_path / "ex" / "e4_eval.json").read_text(encoding="utf-8"))
    ze = json.loads((tmp_path / "ze" / "e4_eval.json").read_text(encoding="utf-8"))
    pts = ex["points"]
    q3 = [p for p in pts if p["question_id"] == 3][0]
    assert q3["rho_c3"] is None and q3["reference_defined"]
    defined = [p for p in pts if p["reference_defined"]]
    both = [p for p in defined if p["rho_c3"] is not None and p["critic"]["1x"]["rho"] is not None]
    assert ex["keys"][e4.KEY_N]["value"] == len(both)
    assert ze["keys"][e4.KEY_N]["value"] == len(defined)
    assert ex["keys"][e4.KEY_C3]["value"] == pytest.approx(sum(p["rho_c3"] for p in both) / len(both))
    assert ze["keys"][e4.KEY_C3]["value"] == pytest.approx(
        sum((p["rho_c3"] or 0.0) for p in defined) / len(defined))
    assert ex["results"] == ze["results"]  # both conventions are always written
    assert ex["results"]["exclude"]["1x"]["counts"]["c3_undefined"] >= 1


def test_evaluate_refuses_without_a_reference_file(tmp_path):
    _c, split_path, _doc, _ref, scores = _full_setup(tmp_path)
    out = tmp_path / "eval"
    rc = e4.main(_eval_args(split_path, str(tmp_path / "missing.jsonl"), scores, out))
    assert rc == 2
    assert not os.path.exists(out / "summary.json") and not os.path.exists(out / "e4_eval.json")


def test_evaluate_refuses_a_reference_that_misses_a_point(tmp_path):
    _c, split_path, doc, _ref, scores = _full_setup(tmp_path)
    ref = _reference(tmp_path, doc, drop=1, name="short.jsonl")
    with pytest.raises(e4.E4Error, match="missing question_id"):
        e4.run_evaluate(e4.build_parser().parse_args(_eval_args(split_path, ref, scores, tmp_path / "e")))
    assert not os.path.exists(tmp_path / "e" / "summary.json")


def test_evaluate_refuses_a_reference_under_the_cells_seed(tmp_path):
    _c, split_path, doc, _ref, scores = _full_setup(tmp_path)
    ref = _reference(tmp_path, doc, seed=0, name="same_seed.jsonl")
    with pytest.raises(e4.E4Error, match="own seed"):
        e4.run_evaluate(e4.build_parser().parse_args(_eval_args(split_path, ref, scores, tmp_path / "e")))


def test_evaluate_refuses_other_alternatives_or_replay_counts(tmp_path):
    _c, split_path, doc, _ref, scores = _full_setup(tmp_path)
    swapped = _reference(tmp_path, doc, swap_texts_at=0, name="swapped.jsonl")
    with pytest.raises(e4.E4Error, match="other alternatives"):
        e4.run_evaluate(e4.build_parser().parse_args(_eval_args(split_path, swapped, scores, tmp_path / "e1")))
    eight = _reference(tmp_path, doc, replays=8, name="eight.jsonl")
    with pytest.raises(e4.E4Error, match="expected 16"):
        e4.run_evaluate(e4.build_parser().parse_args(_eval_args(split_path, eight, scores, tmp_path / "e2")))


def test_evaluate_says_so_when_no_point_can_be_paired(tmp_path):
    # a critic that scores every alternative of a point alike has no ranking anywhere
    _c, split_path, doc, ref, _scores_all = _full_setup(tmp_path)
    flat = {1: _scores(tmp_path, split_path, doc, name="flat.json", fn=lambda p, j: 0.5)}
    out = tmp_path / "eval"
    assert e4.main(_eval_args(split_path, ref, flat, out, undefined="exclude")) == 3
    ev = json.loads((out / "e4_eval.json").read_text(encoding="utf-8"))
    assert set(ev["missing_keys"]) == set(e4.PRIMARY_KEYS)
    assert ev["results"]["exclude"]["1x"]["counts"]["critic_undefined"] == len(ev["points"])
    # under "zero" the same points count as 0 and the keys are there
    assert e4.main(_eval_args(split_path, ref, flat, tmp_path / "eval0", undefined="zero")) == 0
    ev0 = json.loads((tmp_path / "eval0" / "e4_eval.json").read_text(encoding="utf-8"))
    assert ev0["keys"][e4.KEY_CRITIC]["value"] == 0.0 and ev0["missing_keys"] == []


def test_evaluate_ignores_reference_points_that_are_not_heldout(tmp_path):
    _c, split_path, doc, _ref, scores = _full_setup(tmp_path)
    ref = _reference(tmp_path, doc, name="all.jsonl", extra_points=(20, 21))
    ev = e4.run_evaluate(e4.build_parser().parse_args(_eval_args(split_path, ref, scores, tmp_path / "e")))
    assert ev["diagnostics"]["reference_points_not_heldout"] == 2


def test_evaluate_refuses_debug_or_foreign_scores(tmp_path):
    _c, split_path, doc, ref, scores = _full_setup(tmp_path)
    debug = {1: _scores(tmp_path, split_path, doc, name="debug.json", debug=64)}
    with pytest.raises(e4.E4Error, match="debug"):
        e4.run_evaluate(e4.build_parser().parse_args(_eval_args(split_path, ref, debug, tmp_path / "e1")))
    wrong = {1: scores[1], 2: scores[4]}
    with pytest.raises(e4.E4Error, match="expected 2x"):
        e4.run_evaluate(e4.build_parser().parse_args(_eval_args(split_path, ref, wrong, tmp_path / "e2")))


def test_evaluate_refuses_other_decoding_or_reward(tmp_path):
    _c, split_path, doc, _ref, scores = _full_setup(tmp_path)
    rows = [json.loads(l) for l in open(_reference(tmp_path, doc, name="r.jsonl"), encoding="utf-8")]
    for r in rows:
        r["meta"]["decoding"]["max_new_tokens"] = 2048
    bad = _write(tmp_path / "dec.jsonl", rows)
    with pytest.raises(e4.E4Error, match="decoding"):
        e4.run_evaluate(e4.build_parser().parse_args(_eval_args(split_path, bad, scores, tmp_path / "e1")))
    for r in rows:
        r["meta"]["decoding"]["max_new_tokens"] = 512
        r["restart"]["meta"]["task_env_cfg"] = dict(ENV, reward_mode="avg_per_answer")
    bad = _write(tmp_path / "rew.jsonl", rows)
    with pytest.raises(e4.E4Error, match="reward"):
        e4.run_evaluate(e4.build_parser().parse_args(_eval_args(split_path, bad, scores, tmp_path / "e2")))


# -----------------------------------------------------------------------------
# train and score: orchestration, with the torch parts faked
# -----------------------------------------------------------------------------


class _FakeTok:
    pad_token_id = 0

    @classmethod
    def from_pretrained(cls, _path):
        return cls()

    def __call__(self, contexts, actions, truncation=False, max_length=None):
        ids = []
        for c, a in zip(contexts, actions):
            x = [1] * (len(c.split()) + len(a.split()) + 3)
            if truncation and max_length is not None:
                x = x[:max_length]
            ids.append(x)
        return {"input_ids": ids}


class _FakeModel:
    def __init__(self, widths):
        self.widths = list(widths)
        self.pooling = "cls"

    def to(self, _device):
        return self

    def state_dict(self):
        return {"widths": self.widths}

    def load_state_dict(self, state):
        self.loaded = state


def _fake_torch(saved):
    def save(obj, path):
        saved.append(path)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(obj, fh)

    def load(path, map_location=None):
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    return types.SimpleNamespace(
        manual_seed=lambda s: None, save=save, load=load, __version__="fake",
        cuda=types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None))


def _install_fakes(monkeypatch, saved, fit_calls):
    ft = _fake_torch(saved)
    monkeypatch.setattr(e4, "_torch", lambda: (ft, None, _FakeTok))
    monkeypatch.setattr(e4, "build_critic", lambda backbone, widths, dropout, pooling, seed: _FakeModel(widths))

    def fake_fit(model, ids, targets, hp, lr, pad_id, device, val=None):
        fit_calls.append({"lr": lr, "widths": list(model.widths), "n": len(ids), "val": val is not None})
        # validation MSE is lowest at lr 3e-5 with the linear head
        score = abs(math.log10(lr) - math.log10(3e-5)) + 0.01 * len(model.widths) + 0.05
        hist = {"train_mse": [0.3, 0.2, 0.1][:hp["epochs"]], "val_mse": [], "val_rank_corr": [],
                "diverged": False}
        if val is not None:
            hist["val_mse"] = [score + 0.1, score]
            hist["val_rank_corr"] = [0.1, 0.2]
        return hist

    monkeypatch.setattr(e4, "fit", fake_fit)
    # distinct scores per row, so every held-out point has a critic ranking
    monkeypatch.setattr(e4, "predict", lambda model, ids, pad, device, bs, bf16:
                        [float(len(x)) + 0.37 * ((7 * i) % 5) for i, x in enumerate(ids)])


def _train_args(split_path, out_dir, *extra):
    return e4.build_parser().parse_args(["train", "--split", split_path, "--out-dir", str(out_dir)] + list(extra))


def test_train_sweeps_selects_and_refits(tmp_path, monkeypatch):
    saved, calls = [], []
    _install_fakes(monkeypatch, saved, calls)
    _c, split_path, doc, _b = _split(tmp_path, frac=0.3, seed=0)
    out = tmp_path / "t1"
    rep = e4.run_train(_train_args(split_path, out, "--backbone", "/models/mmbert",
                                   "--heads", "small=linear", "large=768"))
    assert len(rep["grid"]) == 6
    assert [(g["lr"], g["head"]) for g in rep["grid"]] == [(lr, h) for lr in e4.PREREG_LRS for h in ("small", "large")]
    assert rep["selected"]["lr"] == 3e-5 and rep["selected"]["head"] == "small"
    assert rep["fixed"]["lrs_are_preregistered"] is True
    assert rep["hyperparameters"]["epochs"] == 3 and rep["hyperparameters"]["max_len"] == 512
    # the sweep trains on the inner-train rows, the refit on every training row
    assert all(c["val"] for c in calls[:6]) and not calls[6]["val"]
    assert calls[6]["n"] == rep["data"]["all"]["n_rows"] == 28
    assert calls[0]["n"] == rep["inner_split"]["n_train_rows"]
    assert rep["inner_split"]["n_train_rows"] + rep["inner_split"]["n_val_rows"] == 28
    assert rep["data"]["base_files"][0]["dropped_heldout_question"] == 3
    assert os.path.isfile(out / "train_report.json") and os.path.isfile(out / "final" / "critic.pt")
    cfg = json.loads((out / "final" / "critic_config.json").read_text(encoding="utf-8"))
    assert cfg["widths"] == [] and cfg["max_len"] == 512
    assert rep["truncation"]["final"]["n"] == 28


def test_train_requires_the_head_convention(tmp_path, monkeypatch):
    _install_fakes(monkeypatch, [], [])
    _c, split_path, _doc, _b = _split(tmp_path)
    with pytest.raises(e4.E4Error, match="--heads"):
        e4.run_train(_train_args(split_path, tmp_path / "t", "--backbone", "/m"))


def test_train_multiplier_retrains_the_selected_configuration(tmp_path, monkeypatch):
    saved, calls = [], []
    _install_fakes(monkeypatch, saved, calls)
    _c, split_path, doc, _b = _split(tmp_path, frac=0.3, seed=0)
    e4.run_train(_train_args(split_path, tmp_path / "t1", "--backbone", "/m",
                             "--heads", "small=linear", "large=768"))
    extra = _write(tmp_path / "extra.jsonl",
                   [_bucket(q, _texts(q, tag="x"), RETURNS[q], seed=1001) for q in range(10)])
    report_1x = str(tmp_path / "t1" / "train_report.json")
    calls.clear()
    rep = e4.run_train(_train_args(split_path, tmp_path / "t2", "--data-multiplier", "2",
                                   "--config-from", report_1x, "--extra-buckets", extra))
    assert len(calls) == 1 and calls[0]["n"] == 56 and calls[0]["lr"] == 3e-5 and calls[0]["widths"] == []
    assert rep["data"]["ratio_rows"] == pytest.approx(2.0)
    assert rep["data"]["ratio_rollouts"] == pytest.approx(2.0)
    assert rep["grid"] == [] and rep["data_multiplier"] == 2
    with pytest.raises(e4.E4Error, match="--seed"):
        e4.run_train(_train_args(split_path, tmp_path / "t3", "--data-multiplier", "4",
                                 "--config-from", report_1x, "--extra-buckets", extra, "--seed", "0"))


def test_score_lines_up_with_the_split(tmp_path, monkeypatch):
    saved, calls = [], []
    _install_fakes(monkeypatch, saved, calls)
    _c, split_path, doc, _b = _split(tmp_path, frac=0.3, seed=0)
    e4.run_train(_train_args(split_path, tmp_path / "t1", "--backbone", "/m", "--heads", "small=linear"))
    out = tmp_path / "scores.json"
    sdoc = e4.run_score(e4.build_parser().parse_args(
        ["score", "--model-dir", str(tmp_path / "t1"), "--split", split_path, "--out", str(out)]))
    assert [p["question_key"] for p in sdoc["points"]] == [p["question_key"] for p in doc["heldout"]]
    for sp, p in zip(sdoc["points"], doc["heldout"]):
        assert len(sp["scores"]) == len(p["candidates"])
    assert sdoc["model"]["data_multiplier"] == 1 and sdoc["model"]["data"]["debug_max_rows"] is None
    # a score file from this path is accepted by evaluate
    ref = _reference(tmp_path, doc)
    ev = e4.run_evaluate(e4.build_parser().parse_args(
        _eval_args(split_path, ref, {1: str(out)}, tmp_path / "eval")))
    assert e4.KEY_N in ev["keys"]
