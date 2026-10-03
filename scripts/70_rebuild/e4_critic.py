#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""e4_critic.py: E4, the learned-critic comparison run fairly.

A decision-value critic Qhat(H_u, A_u) (mmBERT-base with a regression head) is
trained on the rollout data C3 already has. The critic and C3's own low-budget
measurement are then scored on held-out decision points against ONE
independent reference: sixteen fresh replays of every held-out alternative.
In the paper (arXiv:2603.06859v3) this is the critic comparison of Section 4
and Appendix B.7. The primary criterion fixed before the run: best critic
< 0.5 and C3 >= 0.8.

The paper's run, in order (paths are examples; `20_data/results` is the results
root, the evaluation cell is the E1 sweep cell of the 2-agent workflow at n = 4):

    E1=20_data/results/E1/a2/4b/sweep_n4/buckets.jsonl
    H=20_data/results/E4/heldout_a2_4b_sweep_n4
    python scripts/70_rebuild/e4_critic.py split --eval-buckets $E1 --out $H/split.json
    python scripts/70_rebuild/e4_critic.py prepare-reference --split $H/split.json --out-dir $H/prep
    python scripts/70_rebuild/e4_reference_cells.py --prep $H/prep --model_root /models  # GPU
    python scripts/70_rebuild/e4_critic.py train --split $H/split.json --eval-buckets $E1 \\
        --backbone jhu-clsp/mmBERT-base --heads small=linear large=768 \\
        --lrs 1e-5 3e-5 1e-4 --epochs 3 --batch-size 16 --max-len 1024 \\
        --weight-decay 0.01 --head-dropout 0.0 --pooling config --val-frac 0.2 --seed 0 \\
        --out-dir runs/critic_1x                                                   # GPU
    python scripts/70_rebuild/e4_critic.py score --model-dir runs/critic_1x \\
        --split $H/split.json --out runs/scores_1x.json                              # GPU
    python scripts/70_rebuild/e4_critic.py evaluate --split $H/split.json \\
        --reference 20_data/results/E4/reference_heldout/a2/4b/n4_c16_seed3003/buckets.jsonl \\
        --scores runs/scores_1x.json --undefined-rho exclude --out-dir $H/eval

The 2x and 4x critics retrain the selected configuration with
`--data-multiplier 2|4 --config-from runs/critic_1x/train_report.json
--extra-buckets <more E1b seed A cells>`; the second reference is the same
reference cell under replay seed 4004.

Subcommands, in the order they run:

  split              draw the held-out decision points from the evaluation
                     cell (one E1 bucket file) and write split.json
  prepare-reference  write the held-out points in the form `build-buckets
                     --reuse_candidates_from` reads, their questions as a
                     one-suite question file, and a task file naming that file
  train              lr x head sweep on an inner validation split of the
                     training points, selection on validation MSE, then the
                     selected configuration refit on every training point;
                     with --data-multiplier 2|4, the selected configuration
                     retrained on the same base data plus more bucket files
  score              the trained critic scores every held-out alternative
  evaluate           per-point Spearman correlations against the reference,
                     for the critic and for C3's leave-one-out advantages,
                     their paired difference and paired tests; writes
                     e4_eval.json (every point) and summary.json (the shape
                     of c3/analysis/rebuild/summary.py)

What "held out" means here. The unit is the decision point, and in the E1
cells (first position of each workflow) a decision point is a question. A
held-out point is excluded from training in EVERY bucket file training reads:
any record with the same question text is dropped, and any training
alternative whose text equals a held-out alternative is dropped as well. The
second rule matters because cells built under one seed share alternatives
across branching factors (E1 a2/4b sweep_n2 and sweep_n4 share 271 of 1,000
alternatives at the same question).

Why the reference seed must differ from the cell's. The batched replay path
seeds replay r of alternative j from (run seed, question_id, j, r, role), not
from the replay count, so a sixteen-replay run under the cell's own seed would
reproduce the cell's four replays as its first four (measured: E1b_g500 n4_c4
seedA are the first four replays of n4_c8 seedA for 2,000 of 2,000
alternatives). evaluate refuses a reference whose seed equals the cell's.

split, prepare-reference and evaluate import numpy and scipy only; train and
score import torch and transformers when they run.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import datetime
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
#: No results manifest ships with the repository; pass --manifest to check the
#: written keys against one.
DEFAULT_MANIFEST: Optional[str] = None
SCRIPT = "scripts/70_rebuild/e4_critic.py"

SPLIT_SCHEMA = "c3.e4.split/1"
PREPARE_SCHEMA = "c3.e4.prepare/1"
TRAIN_SCHEMA = "c3.e4.train/1"
SCORES_SCHEMA = "c3.e4.scores/1"
EVAL_SCHEMA = "c3.e4.eval/1"

#: Preregistration E4: the learning rates of the sweep.
PREREG_LRS = (1e-5, 3e-5, 1e-4)
#: Preregistration E4: replays per alternative of the reference.
REFERENCE_REPLAYS = 16
#: Preregistration E4, primary criterion.
CRITIC_BELOW = 0.5
C3_AT_LEAST = 0.8

#: The held-out questions as an evaluation suite, and the files that carry them.
HELDOUT_SUITE = "E4HELDOUT"
HELDOUT_TASK_FILE = "e4_heldout_task.yaml"
HELDOUT_POOL_FILE = "e4_heldout_pool.jsonl"
HELDOUT_REUSE_FILE = "e4_heldout_reuse.jsonl"
PREPARE_REPORT_FILE = "prepare_report.json"
TRAIN_REPORT_FILE = "train_report.json"
FINAL_DIR = "final"
FINAL_WEIGHTS = "critic.pt"
FINAL_CONFIG = "critic_config.json"

#: The role file of each workflow's task file (configs/roles/...), resolved by
#: the task loader against the repository's configs root.
WORKFLOW_ROLES = {"a2": "math/roles_duo.json"}

#: Manifest keys this script fills. The verdict key E4.critic_rank is the
#: maintainers' (results layout revision 2: no script writes a verdict).
KEY_N = "E4.n_points"
KEY_CRITIC = "E4.critic.rank_corr"
KEY_C3 = "E4.c3.rank_corr"
KEY_DIFF = "E4.paired_diff"
KEY_P = "E4.p"
KEY_MULT = {2: "E4.critic_2x.rank_corr", 4: "E4.critic_4x.rank_corr"}
VERDICT_KEY = "E4.critic_rank"
PRIMARY_KEYS = (KEY_N, KEY_CRITIC, KEY_C3, KEY_DIFF, KEY_P)
#: Units as the manifest declares them, for when the manifest is not at hand.
KEY_UNITS = {KEY_N: "points", KEY_CRITIC: "spearman", KEY_C3: "spearman",
             KEY_DIFF: "spearman", KEY_P: "p",
             KEY_MULT[2]: "spearman", KEY_MULT[4]: "spearman"}

#: The fields of restart.meta.task_env_cfg that decide the reward.
REWARD_FIELDS = ("env_name", "math_backend", "reward_mode", "use_math_verify", "episode_length")

#: How a rank correlation that is undefined (one side constant) is treated.
UNDEFINED_CHOICES = ("exclude", "zero")

#: The training settings the sweep does not vary. Used for
#: the 1x run when the flag is not given, and recorded in every report.
TRAIN_DEFAULTS: Dict[str, Any] = {
    "epochs": 3, "batch_size": 16, "max_len": 512, "weight_decay": 0.01,
    "head_dropout": 0.0, "pooling": "config", "val_frac": 0.2, "seed": 0,
    "bf16": False,
}
#: Flags a --data-multiplier 2|4 run takes from the 1x report instead.
TRAIN_HP_FLAGS = ("epochs", "batch_size", "max_len", "weight_decay", "head_dropout",
                  "pooling", "val_frac", "seed", "bf16", "heads", "lrs", "backbone")

DATA_DIR_PATTERN = re.compile(r"^\d{2}_data$")
BOOTSTRAP_RESAMPLES = 2000


class E4Error(RuntimeError):
    """A refusal: the inputs cannot support the requested step."""


# -----------------------------------------------------------------------------
# small utilities
# -----------------------------------------------------------------------------


def now_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_sha() -> Optional[str]:
    """HEAD of the repository holding this file, or None (read-only, never fatal)."""
    try:
        out = subprocess.run(["git", "-C", HERE, "rev-parse", "HEAD"],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    sha = out.stdout.decode("utf-8", "replace").strip()
    return sha if out.returncode == 0 and sha else None


def generated_by() -> Dict[str, Any]:
    return {"script": SCRIPT, "git_sha": git_sha(), "when": now_utc()}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_text_atomic(path: str, text: str) -> None:
    """Write UTF-8 with LF endings through a temporary file, so no half file is left."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".e4.", suffix=".tmp", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def write_json(path: str, obj: Any) -> None:
    write_text_atomic(path, json.dumps(obj, ensure_ascii=False, indent=1) + "\n")


def source_path(path: str, data_root: Optional[str] = None) -> str:
    """Path as written into a summary `source` (results layout section 3.4).

    The rule of `c3/analysis/rebuild/summary.py:source_path`, read for any
    two-digit data directory: cut at `data_root` when given, else at the last
    segment shaped like a data directory (`20_data`), else keep the path;
    forward slashes.
    """
    full = os.path.abspath(path).replace("\\", "/")
    if data_root is not None:
        root = os.path.abspath(data_root).replace("\\", "/").rstrip("/")
        folded, folded_root = os.path.normcase(full), os.path.normcase(root)
        if folded == folded_root or folded.startswith(os.path.normcase(root + "/")):
            parent = root.rsplit("/", 1)[0]
            return full[len(parent) + 1:]
        return path.replace("\\", "/")
    parts = full.split("/")
    for cut in range(len(parts) - 1, -1, -1):
        if DATA_DIR_PATTERN.match(parts[cut]):
            return "/".join(parts[cut:])
    return path.replace("\\", "/")


def format_p(p: Optional[float]) -> Optional[str]:
    """The manifest's p string, or None when p is undefined.

    Three tiers, plain text (results layout revision 6; same as
    `c3/analysis/rebuild/summary.py:format_p`): `< 0.001`, `< 0.01`, `= 0.30`.
    """
    if p is None:
        return None
    try:
        value = float(p)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    if value < 0.001:
        return "< 0.001"
    if value < 0.01:
        return "< 0.01"
    return "= %.2f" % value


def finite_or_none(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


# -----------------------------------------------------------------------------
# buckets
# -----------------------------------------------------------------------------


def read_buckets(path: str) -> List[Tuple[int, str, Dict[str, Any]]]:
    """(line number, raw line, record) for every bucket of a buckets.jsonl.

    Blank lines and lines starting with '#' are skipped, as the repository's
    reader does. The raw line is kept so a record can be copied byte for byte.
    """
    if not os.path.isfile(path):
        raise E4Error("no bucket file at %s" % path)
    out: List[Tuple[int, str, Dict[str, Any]]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as exc:
                raise E4Error("invalid JSON at %s:%d: %s" % (path, lineno, exc))
            if not isinstance(obj, dict):
                raise E4Error("%s:%d is not a JSON object" % (path, lineno))
            out.append((lineno, s, obj))
    return out


def question_of(bucket: Mapping[str, Any], where: str) -> str:
    restart = bucket.get("restart")
    q = restart.get("question") if isinstance(restart, Mapping) else None
    if not isinstance(q, str) or not q.strip():
        raise E4Error("%s: bucket has no restart.question" % where)
    return q


def context_of(bucket: Mapping[str, Any]) -> str:
    """The critic's input H_u, composed as the earlier critic script did.

    The question, then the outputs of the roles before the target role in the
    order the bucket stores them. At the first position (every E1 cell) the
    prefix is empty and the context is the question.
    """
    restart = bucket.get("restart") or {}
    ctx = str(restart.get("question", ""))
    prefix = restart.get("role_outputs_prefix") or {}
    if isinstance(prefix, Mapping) and prefix:
        ctx += "\n\n" + "\n\n".join(str(v) for v in prefix.values() if v)
    return ctx


def label_of(bucket: Mapping[str, Any]) -> Any:
    restart = bucket.get("restart") or {}
    meta = restart.get("meta") if isinstance(restart, Mapping) else None
    return meta.get("label") if isinstance(meta, Mapping) else None


def task_env_of(bucket: Mapping[str, Any]) -> Any:
    restart = bucket.get("restart") or {}
    meta = restart.get("meta") if isinstance(restart, Mapping) else None
    return meta.get("task_env_cfg") if isinstance(meta, Mapping) else None


def question_key(question: str) -> str:
    """Sixteen hex characters of the SHA256 of the question text.

    For MATH pool rows this is the row id `pool_ids.row_id` gives a row that
    has no unique_id, so a key can be looked up in pool_informative_ids.json.
    """
    return hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]


def candidates_of(bucket: Mapping[str, Any], where: str) -> List[Dict[str, Any]]:
    """The alternatives in j order: j, action_text, returns, next_actions."""
    raw = bucket.get("candidates")
    if not isinstance(raw, list):
        raise E4Error("%s: candidates is not a list" % where)
    out: List[Dict[str, Any]] = []
    for idx, cand in enumerate(raw):
        if not isinstance(cand, Mapping):
            raise E4Error("%s: candidate %d is not an object" % (where, idx))
        text = cand.get("action_text")
        if not isinstance(text, str):
            raise E4Error("%s: candidate %d has no string action_text" % (where, idx))
        j = cand.get("j", idx)
        if isinstance(j, bool) or not isinstance(j, int):
            raise E4Error("%s: candidate %d has a non-integer j" % (where, idx))
        rets = cand.get("returns")
        if not isinstance(rets, list):
            raise E4Error("%s: candidate %d has no returns list" % (where, idx))
        values = []
        for r in rets:
            v = finite_or_none(r) if not isinstance(r, bool) else None
            if v is None:
                raise E4Error("%s: candidate %d carries a return that is not a finite number" % (where, idx))
            values.append(v)
        nexts = cand.get("next_actions") or []
        out.append({"j": int(j), "action_text": text, "returns": values,
                    "next_actions": [str(x) for x in nexts] if isinstance(nexts, list) else []})
    out.sort(key=lambda c: c["j"])
    js = [c["j"] for c in out]
    if len(set(js)) != len(js):
        raise E4Error("%s: two candidates share one j" % where)
    return out


def decoding_without_seed(decoding: Any) -> Dict[str, Any]:
    if not isinstance(decoding, Mapping):
        return {}
    return {str(k): v for k, v in decoding.items() if k != "seed"}


def snapshot_of(pretrain: Any) -> Optional[str]:
    """The snapshot directory name of an HF cache path (the model revision), or None."""
    if not isinstance(pretrain, str):
        return None
    parts = [p for p in pretrain.replace("\\", "/").split("/") if p]
    if "snapshots" in parts:
        i = parts.index("snapshots")
        if i + 1 < len(parts):
            return parts[i + 1]
    return None


def cell_facts(records: Sequence[Tuple[int, str, Dict[str, Any]]], path: str) -> Dict[str, Any]:
    """The facts every bucket of one cell must agree on, plus descriptive meta.

    A cell whose buckets disagree on the seed, the decoding, the reward
    configuration, the roles or the model is refused: the reference is
    checked against these, so they have to be one value.
    """
    agreed: Optional[Dict[str, Any]] = None
    first_meta: Dict[str, Any] = {}
    for lineno, _raw, b in records:
        meta = b.get("meta") if isinstance(b.get("meta"), Mapping) else {}
        view = {
            "seed": meta.get("seed"),
            "decoding": meta.get("decoding"),
            "task_env_cfg": task_env_of(b),
            "target_role": b.get("target_role"),
            "next_role": meta.get("next_role"),
            "candidate_total_req": meta.get("candidate_total_req"),
            "v_extra_n": meta.get("v_extra_n", 0),
            "pretrain": meta.get("pretrain"),
            "seed_scheme": meta.get("seed_scheme"),
        }
        if agreed is None:
            agreed = view
            first_meta = dict(meta)
        elif view != agreed:
            fields = sorted(k for k in view if view[k] != agreed[k])
            raise E4Error("%s:%d disagrees with the cell's first bucket on %s"
                          % (path, lineno, ", ".join(fields)))
    if agreed is None:
        raise E4Error("%s holds no buckets" % path)
    facts = dict(agreed)
    for key in ("workflow", "model", "rule", "dataset", "split", "n_alternatives",
                "replays_per_alt", "policy_tag", "limit", "include_real_as_j0",
                "context_scope", "method"):
        facts[key] = first_meta.get(key)
    return facts


# -----------------------------------------------------------------------------
# rank statistics
# -----------------------------------------------------------------------------


def spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    """Spearman rank correlation with average ranks for ties.

    None when it is undefined: fewer than two items, or either side constant
    (no ordering to compare).
    """
    from scipy.stats import rankdata

    xa = np.asarray(list(x), dtype=float)
    ya = np.asarray(list(y), dtype=float)
    if xa.shape != ya.shape or xa.size < 2:
        return None
    rx = rankdata(xa)
    ry = rankdata(ya)
    dx = rx - rx.mean()
    dy = ry - ry.mean()
    sx = float(np.sqrt(np.sum(dx * dx)))
    sy = float(np.sqrt(np.sum(dy * dy)))
    if sx == 0.0 or sy == 0.0:
        return None
    r = float(np.sum(dx * dy) / (sx * sy))
    return max(-1.0, min(1.0, r))


def loo_advantages(means: Sequence[float]) -> List[float]:
    """C3's leave-one-out advantage: each mean minus the mean of the others.

    A_j = Q_j - (sum_k Q_k - Q_j) / (n - 1), an increasing affine map of Q_j
    within a decision point, so it ranks the alternatives as the means do.
    """
    q = [float(v) for v in means]
    n = len(q)
    if n < 2:
        raise E4Error("a leave-one-out advantage needs at least two alternatives")
    total = sum(q)
    return [qj - (total - qj) / (n - 1) for qj in q]


def paired_stats(c3: Sequence[float], critic: Sequence[float],
                 n_boot: int = BOOTSTRAP_RESAMPLES, seed: int = 0) -> Dict[str, Any]:
    """C3 minus critic over paired points: mean, paired t, Wilcoxon, bootstrap interval."""
    from scipy import stats

    a = np.asarray(list(c3), dtype=float)
    b = np.asarray(list(critic), dtype=float)
    if a.shape != b.shape:
        raise E4Error("paired vectors differ in length")
    n = int(a.size)
    out: Dict[str, Any] = {"n": n, "mean_c3": None, "mean_critic": None, "mean_diff": None,
                           "t": None, "p_t": None, "p_wilcoxon": None,
                           "boot_ci95": None, "boot_resamples": int(n_boot), "boot_seed": int(seed)}
    if n == 0:
        return out
    d = a - b
    out["mean_c3"] = float(a.mean())
    out["mean_critic"] = float(b.mean())
    out["mean_diff"] = float(d.mean())
    if n >= 2 and float(np.std(d)) > 0.0:
        res = stats.ttest_rel(a, b)
        out["t"] = finite_or_none(res.statistic)
        out["p_t"] = finite_or_none(res.pvalue)
    if n >= 2 and np.any(d != 0.0):
        try:
            out["p_wilcoxon"] = finite_or_none(stats.wilcoxon(d).pvalue)
        except ValueError:
            out["p_wilcoxon"] = None
    if n >= 2 and n_boot > 0:
        rng = np.random.RandomState(int(seed))
        idx = rng.randint(0, n, size=(int(n_boot), n))
        boot = d[idx].mean(axis=1)
        out["boot_ci95"] = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    return out


# -----------------------------------------------------------------------------
# split
# -----------------------------------------------------------------------------


def run_split(eval_buckets: str, out_path: str, heldout_frac: float = 0.2, seed: int = 0) -> Dict[str, Any]:
    """Draw the held-out points of the evaluation cell and write split.json.

    Eligible for holding out: at least two alternatives, each with at least one
    return (a ranking needs two items). The held-out points are the first
    round(heldout_frac * n_eligible) indices of
    numpy.random.RandomState(seed).permutation(n_eligible) over the eligible
    points in file order; they are written in file order.
    """
    if not 0.0 < float(heldout_frac) < 1.0:
        raise E4Error("--heldout-frac must lie strictly between 0 and 1")
    records = read_buckets(eval_buckets)
    facts = cell_facts(records, eval_buckets)
    points: List[Dict[str, Any]] = []
    seen_q: Dict[str, int] = {}
    seen_ctx: Dict[int, int] = {}
    for lineno, _raw, b in records:
        where = "%s:%d" % (eval_buckets, lineno)
        q = question_of(b, where)
        if q in seen_q:
            raise E4Error("%s: the question of line %d appears again; a cell holds one decision "
                          "point per question" % (where, seen_q[q]))
        seen_q[q] = lineno
        ch = b.get("ctx_hash")
        if isinstance(ch, bool) or not isinstance(ch, int):
            raise E4Error("%s: bucket has no integer ctx_hash" % where)
        if ch in seen_ctx:
            raise E4Error("%s: ctx_hash of line %d appears again" % (where, seen_ctx[ch]))
        seen_ctx[ch] = lineno
        cands = candidates_of(b, where)
        points.append({
            "line": lineno,
            "question_id": b.get("question_id"),
            "ctx_hash": ch,
            "bucket_id": b.get("bucket_id"),
            "question_key": question_key(q),
            "question": q,
            "context": context_of(b),
            "label": label_of(b),
            "candidates": cands,
            "_eligible": len(cands) >= 2 and all(c["returns"] for c in cands),
        })
    eligible = [i for i, p in enumerate(points) if p["_eligible"]]
    n_held = int(round(float(heldout_frac) * len(eligible)))
    if n_held < 1 or n_held >= len(eligible):
        raise E4Error("%d eligible points and held-out fraction %s leave %d held out; need at "
                      "least one on each side" % (len(eligible), heldout_frac, n_held))
    perm = np.random.RandomState(int(seed)).permutation(len(eligible))
    held_idx = sorted(eligible[int(k)] for k in perm[:n_held])
    held_set = set(held_idx)
    heldout = []
    for i in held_idx:
        p = dict(points[i])
        p.pop("_eligible")
        heldout.append(p)
    doc = {
        "schema": SPLIT_SCHEMA,
        "generated_by": generated_by(),
        "params": {
            "heldout_frac": float(heldout_frac),
            "seed": int(seed),
            "rule": ("held-out = first round(heldout_frac * n_eligible) indices of "
                     "numpy.random.RandomState(seed).permutation(n_eligible) over the eligible "
                     "points in file order; eligible = at least two alternatives, each with a "
                     "return; training excludes every record whose question is held out and "
                     "every alternative whose text is a held-out alternative, in every file"),
        },
        "eval_cell": {
            "path": os.path.abspath(eval_buckets).replace("\\", "/"),
            "source": source_path(eval_buckets),
            "sha256": sha256_file(eval_buckets),
            "n_records": len(records),
            "n_eligible": len(eligible),
            "facts": facts,
        },
        "n_heldout": len(heldout),
        "n_train_points": len(points) - len(heldout),
        "train_question_keys": [p["question_key"] for i, p in enumerate(points) if i not in held_set],
        "heldout": heldout,
    }
    write_json(out_path, doc)
    return doc


def load_split(path: str) -> Tuple[Dict[str, Any], str]:
    if not os.path.isfile(path):
        raise E4Error("no split file at %s" % path)
    doc = read_json(path)
    if not isinstance(doc, dict) or doc.get("schema") != SPLIT_SCHEMA:
        raise E4Error("%s is not a %s document" % (path, SPLIT_SCHEMA))
    return doc, sha256_file(path)


def heldout_sets(split: Mapping[str, Any]) -> Tuple[set, set]:
    questions = {p["question"] for p in split["heldout"]}
    texts = {c["action_text"] for p in split["heldout"] for c in p["candidates"]}
    return questions, texts


def resolve_eval_cell(split: Mapping[str, Any], override: Optional[str]) -> str:
    """The evaluation cell's bucket file, checked against the hash split recorded."""
    cell = split["eval_cell"]
    path = override or cell["path"]
    if not os.path.isfile(path):
        raise E4Error("the evaluation cell is not at %s (pass --eval-buckets with its path here)" % path)
    sha = sha256_file(path)
    if sha != cell["sha256"]:
        raise E4Error("%s is not the file split was drawn from (sha256 %s, split recorded %s)"
                      % (path, sha, cell["sha256"]))
    return path


# -----------------------------------------------------------------------------
# prepare-reference
# -----------------------------------------------------------------------------


def run_prepare(split_path: str, out_dir: str, eval_buckets: Optional[str] = None,
                pool_ids: Optional[str] = None, roles_path: Optional[str] = None) -> Dict[str, Any]:
    """Write what `build-buckets` needs to replay the held-out alternatives only.

    - e4_heldout_reuse.jsonl: the held-out buckets of the evaluation cell, the
      source lines copied byte for byte (question_id, ctx_hash and the
      alternatives are what --reuse_candidates_from reads);
    - e4_heldout_pool.jsonl: one row per held-out question, {"id": the cell's
      question_id, "input": the question, "answer": the label the cell
      recorded}. The runner takes a row's `id` as its question_id, so the
      replayed buckets carry the cell's own question ids and match the reuse
      file on (question_id, ctx_hash);
    - e4_heldout_task.yaml: the environment block the cell recorded in
      restart.meta.task_env_cfg, with eval_suites replaced by the one suite
      E4HELDOUT = the file above (resolved beside the task file). The body is
      JSON, which YAML reads as written;
    - prepare_report.json: hashes, counts and the build-buckets arguments.
    """
    split, split_sha = load_split(split_path)
    cell = split["eval_cell"]
    facts = cell["facts"]
    src = resolve_eval_cell(split, eval_buckets)
    records = read_buckets(src)
    by_line = {lineno: (raw, b) for lineno, raw, b in records}

    if int(facts.get("v_extra_n") or 0) != 0:
        raise E4Error("the evaluation cell carries V-extra alternatives; this preparation "
                      "handles cells without them only")
    total_req = facts.get("candidate_total_req")
    if isinstance(total_req, bool) or not isinstance(total_req, int) or total_req < 2:
        raise E4Error("the evaluation cell records no usable candidate_total_req")
    env = facts.get("task_env_cfg")
    if not isinstance(env, Mapping):
        raise E4Error("the evaluation cell recorded no restart.meta.task_env_cfg")
    workflow = facts.get("workflow")
    roles = roles_path or WORKFLOW_ROLES.get(str(workflow))
    if not roles:
        raise E4Error("no role file known for workflow %r; pass --roles-path" % workflow)

    ids = None
    if pool_ids:
        loaded = read_json(pool_ids)
        ids = loaded.get("unique_ids") if isinstance(loaded, Mapping) else loaded
        if not isinstance(ids, list):
            raise E4Error("%s holds no unique_ids list" % pool_ids)

    reuse_lines: List[str] = []
    pool_rows: List[Dict[str, Any]] = []
    for p in split["heldout"]:
        got = by_line.get(p["line"])
        if got is None:
            raise E4Error("line %d of %s is gone" % (p["line"], src))
        raw, b = got
        where = "%s:%d" % (src, p["line"])
        if question_of(b, where) != p["question"] or b.get("ctx_hash") != p["ctx_hash"]:
            raise E4Error("%s is not the held-out point split recorded" % where)
        qid = b.get("question_id")
        if isinstance(qid, bool) or not isinstance(qid, int):
            raise E4Error("%s: question_id %r is not an integer, so it cannot be carried as the "
                          "question file's id column" % (where, qid))
        label = p["label"]
        if not isinstance(label, str) or not label.strip():
            raise E4Error("%s: the recorded label %r is not a non-empty string" % (where, label))
        if ids is not None:
            if not 0 <= qid < len(ids) or question_key(p["question"]) != ids[qid]:
                raise E4Error("%s: the question does not hash to the pool id at position %d" % (where, qid))
        reuse_lines.append(raw)
        pool_rows.append({"id": qid, "input": p["question"], "answer": label})

    env_out = copy.deepcopy(dict(env))
    env_out["eval_suites"] = [{"name": HELDOUT_SUITE, "path": HELDOUT_POOL_FILE, "limit": None}]
    task = {"experiment_name": "c3_e4_heldout_reference",
            "environment": env_out,
            "mas": {"roles_path": roles}}
    task_text = (
        "# E4 held-out reference task, written by %s prepare-reference.\n"
        "# environment: restart.meta.task_env_cfg as the evaluation cell recorded it, with\n"
        "# eval_suites replaced by the held-out questions (suite %s, the file beside this one).\n"
        "# The body is JSON, which YAML reads as written.\n" % (SCRIPT, HELDOUT_SUITE)
        + json.dumps(task, ensure_ascii=False, indent=1) + "\n")

    os.makedirs(out_dir, exist_ok=True)
    reuse_path = os.path.join(out_dir, HELDOUT_REUSE_FILE)
    pool_path = os.path.join(out_dir, HELDOUT_POOL_FILE)
    task_path = os.path.join(out_dir, HELDOUT_TASK_FILE)
    write_text_atomic(reuse_path, "".join(line + "\n" for line in reuse_lines))
    write_text_atomic(pool_path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in pool_rows))
    write_text_atomic(task_path, task_text)

    report = {
        "schema": PREPARE_SCHEMA,
        "generated_by": generated_by(),
        "split": {"path": os.path.abspath(split_path).replace("\\", "/"), "sha256": split_sha},
        "source": {"path": os.path.abspath(src).replace("\\", "/"), "source": source_path(src),
                   "sha256": cell["sha256"]},
        "n_points": len(pool_rows),
        "suite": HELDOUT_SUITE,
        "pool_ids_checked": ids is not None,
        "files": {
            name: {"sha256": sha256_file(os.path.join(out_dir, name)), "lines": n}
            for name, n in ((HELDOUT_REUSE_FILE, len(reuse_lines)),
                            (HELDOUT_POOL_FILE, len(pool_rows)),
                            (HELDOUT_TASK_FILE, None))
        },
        "source_seed": facts.get("seed"),
        "build_buckets": {
            "--task": HELDOUT_TASK_FILE,
            "--split": HELDOUT_SUITE,
            "--limit": len(pool_rows),
            "--num_candidates": int(total_req),
            "--num_extra_v_samples": 0,
            "--include_real_as_j0": "false",
            "--target_role": facts.get("target_role"),
            "--next_role": facts.get("next_role"),
            "--reuse_candidates_from": HELDOUT_REUSE_FILE,
            "--decoding_json": decoding_without_seed(facts.get("decoding")),
        },
        "note": ("file names are relative to this directory; the replay seed (--seed) must differ "
                 "from source_seed, or the batched seeds repeat the cell's own replays"),
    }
    write_json(os.path.join(out_dir, PREPARE_REPORT_FILE), report)
    return report


# -----------------------------------------------------------------------------
# training rows
# -----------------------------------------------------------------------------


def collect_rows(paths: Sequence[str], held_q: set, held_c: set) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """(context, alternative, mean return) rows of every file, held-out points removed.

    A record whose question is held out is dropped whole; an alternative whose
    text equals a held-out alternative is dropped; a record with fewer than two
    alternatives is dropped (as the earlier critic script did). The target is the mean of
    the alternative's replay returns.
    """
    rows: List[Dict[str, Any]] = []
    accounts: List[Dict[str, Any]] = []
    for path in paths:
        records = read_buckets(path)
        acct = {"path": os.path.abspath(path).replace("\\", "/"), "source": source_path(path),
                "sha256": sha256_file(path), "n_records": len(records), "n_points": 0,
                "n_rows": 0, "n_rollouts": 0, "dropped_heldout_question": 0,
                "dropped_heldout_alternative": 0, "dropped_no_returns": 0,
                "dropped_single_alternative": 0}
        for lineno, _raw, b in records:
            where = "%s:%d" % (path, lineno)
            q = question_of(b, where)
            if q in held_q:
                acct["dropped_heldout_question"] += 1
                continue
            cands = candidates_of(b, where)
            if len(cands) < 2:
                acct["dropped_single_alternative"] += 1
                continue
            ctx = context_of(b)
            qk = question_key(q)
            used = 0
            for c in cands:
                if c["action_text"] in held_c:
                    acct["dropped_heldout_alternative"] += 1
                    continue
                if not c["returns"]:
                    acct["dropped_no_returns"] += 1
                    continue
                rows.append({"context": ctx, "action": c["action_text"],
                             "target": float(np.mean(c["returns"])),
                             "n_returns": len(c["returns"]), "question_key": qk})
                acct["n_rows"] += 1
                acct["n_rollouts"] += len(c["returns"])
                used += 1
            if used:
                acct["n_points"] += 1
        accounts.append(acct)
    return rows, accounts


def totals(accounts: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    return {k: int(sum(a[k] for a in accounts)) for k in ("n_rows", "n_rollouts", "n_points")}


def inner_split(rows: Sequence[Mapping[str, Any]], val_frac: float, seed: int) -> Tuple[List[int], List[int]]:
    """Row indices (train, validation), split by question so no question straddles them."""
    if not 0.0 < float(val_frac) < 1.0:
        raise E4Error("--val-frac must lie strictly between 0 and 1")
    order: List[str] = []
    seen = set()
    for r in rows:
        if r["question_key"] not in seen:
            seen.add(r["question_key"])
            order.append(r["question_key"])
    n_val = int(round(float(val_frac) * len(order)))
    if n_val < 1 or n_val >= len(order):
        raise E4Error("%d training points and --val-frac %s leave no room for both sides"
                      % (len(order), val_frac))
    perm = np.random.RandomState(int(seed)).permutation(len(order))
    val_keys = {order[int(k)] for k in perm[:n_val]}
    tr = [i for i, r in enumerate(rows) if r["question_key"] not in val_keys]
    va = [i for i, r in enumerate(rows) if r["question_key"] in val_keys]
    return tr, va


def parse_heads(specs: Sequence[str]) -> List[Tuple[str, List[int]]]:
    """`name=linear` or `name=w1,w2,...` (hidden widths of an MLP head)."""
    heads: List[Tuple[str, List[int]]] = []
    names = set()
    for spec in specs:
        if "=" not in spec:
            raise E4Error("head %r: write name=linear or name=w1,w2" % spec)
        name, rhs = spec.split("=", 1)
        name, rhs = name.strip(), rhs.strip()
        if not name or name in names:
            raise E4Error("head %r: the name is empty or repeated" % spec)
        if rhs.lower() == "linear":
            widths: List[int] = []
        else:
            try:
                widths = [int(w) for w in rhs.split(",") if w.strip()]
            except ValueError:
                raise E4Error("head %r: widths must be integers" % spec)
            if not widths or any(w < 1 for w in widths):
                raise E4Error("head %r: give positive widths, or linear" % spec)
        names.add(name)
        heads.append((name, widths))
    if not heads:
        raise E4Error("give at least one head")
    return heads


def within_point_rank_corr(preds: Sequence[float], rows: Sequence[Mapping[str, Any]]) -> Tuple[Optional[float], int]:
    """Mean over questions of spearman(prediction, target); a validation diagnostic only."""
    groups: Dict[str, List[int]] = {}
    for i, r in enumerate(rows):
        groups.setdefault(r["question_key"], []).append(i)
    vals = []
    for idx in groups.values():
        if len(idx) < 2:
            continue
        rho = spearman([preds[i] for i in idx], [rows[i]["target"] for i in idx])
        if rho is not None:
            vals.append(rho)
    return (float(np.mean(vals)) if vals else None), len(vals)


# -----------------------------------------------------------------------------
# the critic (torch and transformers are imported here, and only here)
# -----------------------------------------------------------------------------


def _torch():
    try:
        import torch  # noqa: F401
        from transformers import AutoModel, AutoTokenizer  # noqa: F401
    except ImportError as exc:
        raise E4Error("train and score need torch and transformers (%s)" % exc)
    import torch
    from transformers import AutoModel, AutoTokenizer
    return torch, AutoModel, AutoTokenizer


_CRITIC_CLASS = None


def _critic_class():
    """nn.Module subclass, defined on first use so the module imports without torch."""
    global _CRITIC_CLASS
    if _CRITIC_CLASS is None:
        torch, _auto_model, _auto_tok = _torch()

        class Critic(torch.nn.Module):
            def __init__(self, encoder, head, pooling):
                super().__init__()
                self.encoder = encoder
                self.head = head
                self.pooling = pooling

            def forward(self, input_ids, attention_mask):
                out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
                if self.pooling == "mean":
                    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
                    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                else:
                    pooled = hidden[:, 0]
                return self.head(pooled).squeeze(-1)

        _CRITIC_CLASS = Critic
    return _CRITIC_CLASS


def build_critic(backbone: str, widths: Sequence[int], head_dropout: float, pooling: str, seed: int):
    """Encoder in float32 plus a fresh head (linear, or MLP with GELU and dropout).

    pooling: "cls", "mean", or "config" (the encoder config's classifier_pooling,
    the pooling its own sequence-classification head uses; "cls" when absent).
    """
    torch, AutoModel, _ = _torch()
    encoder = AutoModel.from_pretrained(backbone)
    encoder = encoder.float()
    hidden = int(encoder.config.hidden_size)
    resolved = pooling
    if pooling == "config":
        resolved = str(getattr(encoder.config, "classifier_pooling", "") or "cls")
    if resolved not in ("cls", "mean"):
        raise E4Error("pooling %r is neither cls nor mean" % resolved)
    torch.manual_seed(int(seed))
    layers = []
    width_in = hidden
    for w in widths:
        layers += [torch.nn.Linear(width_in, int(w)), torch.nn.GELU(), torch.nn.Dropout(float(head_dropout))]
        width_in = int(w)
    layers.append(torch.nn.Linear(width_in, 1))
    head = torch.nn.Sequential(*layers)
    return _critic_class()(encoder, head, resolved)


def encode_pairs(tok, contexts: Sequence[str], actions: Sequence[str], max_len: int) -> Tuple[List[List[int]], Dict[str, Any]]:
    """Token ids of (context, alternative) pairs cut to max_len, and how many were cut."""
    enc = tok(list(contexts), list(actions), truncation=True, max_length=int(max_len))
    ids = [list(x) for x in enc["input_ids"]]
    full = tok(list(contexts), list(actions), truncation=False)["input_ids"]
    lengths = [len(x) for x in full]
    n_cut = sum(1 for n in lengths if n > int(max_len))
    info = {"max_len": int(max_len), "n": len(ids), "n_truncated": int(n_cut),
            "frac_truncated": (n_cut / len(ids)) if ids else None,
            "untruncated_len_max": int(max(lengths)) if lengths else None,
            "untruncated_len_p95": float(np.percentile(lengths, 95)) if lengths else None}
    return ids, info


def _batch(torch, ids_list: Sequence[Sequence[int]], pad_id: int, device: str):
    width = max(len(x) for x in ids_list)
    input_ids = torch.full((len(ids_list), width), int(pad_id), dtype=torch.long)
    attn = torch.zeros((len(ids_list), width), dtype=torch.long)
    for i, x in enumerate(ids_list):
        input_ids[i, :len(x)] = torch.tensor(list(x), dtype=torch.long)
        attn[i, :len(x)] = 1
    return input_ids.to(device), attn.to(device)


def _autocast(torch, device: str, bf16: bool):
    if bf16 and str(device).startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def predict(model, ids_list: Sequence[Sequence[int]], pad_id: int, device: str,
            batch_size: int, bf16: bool) -> List[float]:
    torch, _m, _t = _torch()
    model.eval()
    out: List[float] = []
    with torch.no_grad():
        for start in range(0, len(ids_list), int(batch_size)):
            chunk = ids_list[start:start + int(batch_size)]
            input_ids, attn = _batch(torch, chunk, pad_id, device)
            with _autocast(torch, device, bf16):
                pred = model(input_ids, attn)
            out.extend(float(v) for v in pred.float().cpu().tolist())
    return out


def fit(model, ids: Sequence[Sequence[int]], targets: Sequence[float], hp: Mapping[str, Any],
        lr: float, pad_id: int, device: str,
        val: Optional[Tuple[Sequence[Sequence[int]], Sequence[Mapping[str, Any]]]] = None) -> Dict[str, Any]:
    """AdamW, constant learning rate, MSE on the mean return, fixed epochs (the earlier critic script's recipe)."""
    torch, _m, _t = _torch()
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(hp["weight_decay"]))
    rng = np.random.RandomState(int(hp["seed"]))
    bs = int(hp["batch_size"])
    hist: Dict[str, Any] = {"train_mse": [], "val_mse": [], "val_rank_corr": [], "diverged": False}
    for _epoch in range(int(hp["epochs"])):
        model.train()
        order = rng.permutation(len(ids))
        total, count = 0.0, 0
        for start in range(0, len(order), bs):
            idx = [int(i) for i in order[start:start + bs]]
            input_ids, attn = _batch(torch, [ids[i] for i in idx], pad_id, device)
            target = torch.tensor([float(targets[i]) for i in idx], dtype=torch.float32, device=device)
            with _autocast(torch, device, bool(hp["bf16"])):
                pred = model(input_ids, attn)
            loss = torch.nn.functional.mse_loss(pred.float(), target)
            if not bool(torch.isfinite(loss)):
                hist["diverged"] = True
                break
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
            count += len(idx)
        if hist["diverged"]:
            break
        hist["train_mse"].append(total / max(count, 1))
        if val is not None:
            v_ids, v_rows = val
            preds = predict(model, v_ids, pad_id, device, bs, bool(hp["bf16"]))
            mse = float(np.mean([(p - r["target"]) ** 2 for p, r in zip(preds, v_rows)]))
            hist["val_mse"].append(mse if math.isfinite(mse) else None)
            hist["val_rank_corr"].append(within_point_rank_corr(preds, v_rows)[0])
    return hist


def _free(torch) -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _given(value: Any) -> bool:
    """Whether a train flag was given (None and an empty list mean it was not; 0 and False count)."""
    if value is None:
        return False
    if isinstance(value, list):
        return len(value) > 0
    return True


def _resolve_hp(args: argparse.Namespace) -> Dict[str, Any]:
    hp = {}
    for key, default in TRAIN_DEFAULTS.items():
        value = getattr(args, key, None)
        hp[key] = default if value is None else value
    return hp


def run_train(args: argparse.Namespace) -> Dict[str, Any]:
    split, split_sha = load_split(args.split)
    held_q, held_c = heldout_sets(split)
    mult = int(args.data_multiplier)
    if mult not in (1, 2, 4):
        raise E4Error("--data-multiplier is 1, 2 or 4")

    if mult == 1:
        if args.config_from or args.extra_buckets:
            raise E4Error("--config-from and --extra-buckets belong to a --data-multiplier 2|4 run")
        if not args.backbone:
            raise E4Error("--backbone is required (path of the mmBERT-base snapshot)")
        if not args.heads:
            raise E4Error("--heads is required: the head sizes are the caller's choice "
                          "(the paper's run: --heads small=linear large=768)")
        hp = _resolve_hp(args)
        hp["backbone"] = args.backbone
        heads = parse_heads(args.heads)
        lrs = [float(x) for x in (args.lrs or PREREG_LRS)]
        base_paths = [resolve_eval_cell(split, args.eval_buckets)] + list(args.train_buckets or [])
        extra_paths: List[str] = []
        base_ref = None
    else:
        given = [k for k in TRAIN_HP_FLAGS + ("eval_buckets",) if _given(getattr(args, k, None))]
        if given:
            raise E4Error("a --data-multiplier %d run takes every hyperparameter and its base data "
                          "from --config-from; drop %s"
                          % (mult, ", ".join("--" + g.replace("_", "-") for g in given)))
        if not args.config_from or not args.extra_buckets:
            raise E4Error("a --data-multiplier %d run needs --config-from (the 1x train_report.json) "
                          "and --extra-buckets" % mult)
        if args.train_buckets:
            raise E4Error("the base data of a retrain is the 1x run's; --train-buckets belongs to 1x")
        base_ref = read_json(args.config_from)
        if base_ref.get("schema") != TRAIN_SCHEMA or int(base_ref.get("data_multiplier", 0)) != 1:
            raise E4Error("%s is not a 1x train report" % args.config_from)
        if base_ref["split"]["sha256"] != split_sha:
            raise E4Error("the 1x report was trained on another split")
        hp = dict(base_ref["hyperparameters"])
        sel = base_ref["selected"]
        heads = [(sel["head"], list(sel["widths"]))]
        lrs = [float(sel["lr"])]
        base_paths = []
        for f in base_ref["data"]["base_files"]:
            path = f["path"]
            if not os.path.isfile(path) or sha256_file(path) != f["sha256"]:
                raise E4Error("base file %s of the 1x run is not here unchanged (run the retrain "
                              "where the 1x run ran)" % path)
            base_paths.append(path)
        extra_paths = list(args.extra_buckets)

    base_rows, base_acct = collect_rows(base_paths, held_q, held_c)
    extra_rows, extra_acct = collect_rows(extra_paths, held_q, held_c) if extra_paths else ([], [])
    rows = base_rows + extra_rows
    if args.max_rows:
        rows = rows[:int(args.max_rows)]
    if len(rows) < 8:
        raise E4Error("only %d training rows after the held-out exclusion" % len(rows))
    base_tot = totals(base_acct)
    all_tot = totals(base_acct + extra_acct)
    data = {
        "base_files": base_acct, "extra_files": extra_acct,
        "base": base_tot, "all": all_tot,
        "ratio_rows": all_tot["n_rows"] / base_tot["n_rows"] if base_tot["n_rows"] else None,
        "ratio_rollouts": all_tot["n_rollouts"] / base_tot["n_rollouts"] if base_tot["n_rollouts"] else None,
        "debug_max_rows": int(args.max_rows) if args.max_rows else None,
    }
    if mult > 1:
        print("[e4] %dx run: %d rows (%.3fx the base rows), %d rollouts (%.3fx the base rollouts)"
              % (mult, all_tot["n_rows"], data["ratio_rows"], all_tot["n_rollouts"], data["ratio_rollouts"]),
              file=sys.stderr)

    torch, _AutoModel, AutoTokenizer = _torch()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(hp["backbone"])
    pad_id = tok.pad_token_id
    if pad_id is None:
        raise E4Error("the tokenizer of %s has no pad token" % hp["backbone"])
    os.makedirs(args.out_dir, exist_ok=True)

    grid: List[Dict[str, Any]] = []
    truncation: Dict[str, Any] = {}
    if mult == 1:
        tr_idx, va_idx = inner_split(rows, hp["val_frac"], hp["seed"])
        tr_rows = [rows[i] for i in tr_idx]
        va_rows = [rows[i] for i in va_idx]
        tr_ids, truncation["inner_train"] = encode_pairs(tok, [r["context"] for r in tr_rows],
                                                         [r["action"] for r in tr_rows], hp["max_len"])
        va_ids, truncation["inner_val"] = encode_pairs(tok, [r["context"] for r in va_rows],
                                                       [r["action"] for r in va_rows], hp["max_len"])
        mean_tr = float(np.mean([r["target"] for r in tr_rows]))
        const_mse = float(np.mean([(mean_tr - r["target"]) ** 2 for r in va_rows]))
        for lr in lrs:
            for name, widths in heads:
                t0 = time.time()
                torch.manual_seed(int(hp["seed"]))
                model = build_critic(hp["backbone"], widths, hp["head_dropout"], hp["pooling"], hp["seed"]).to(device)
                hist = fit(model, tr_ids, [r["target"] for r in tr_rows], hp, lr, pad_id, device,
                           val=(va_ids, va_rows))
                final_val = hist["val_mse"][-1] if (hist["val_mse"] and not hist["diverged"]) else None
                grid.append({"lr": lr, "head": name, "widths": widths, "pooling": model.pooling,
                             "train_mse_by_epoch": hist["train_mse"], "val_mse_by_epoch": hist["val_mse"],
                             "val_rank_corr_by_epoch": hist["val_rank_corr"], "val_mse": final_val,
                             "diverged": hist["diverged"], "seconds": round(time.time() - t0, 1)})
                print("[e4] lr %g head %s: val MSE %s" % (lr, name, final_val), file=sys.stderr)
                del model
                _free(torch)
        usable = [g for g in grid if g["val_mse"] is not None]
        if not usable:
            raise E4Error("every configuration of the sweep diverged")
        best = min(usable, key=lambda g: g["val_mse"])
        selected = {"lr": best["lr"], "head": best["head"], "widths": best["widths"],
                    "val_mse": best["val_mse"], "rule": "lowest final-epoch validation MSE; ties go "
                    "to the first in grid order (lr outer, head inner)"}
        inner = {"n_train_rows": len(tr_rows), "n_val_rows": len(va_rows),
                 "n_val_points": len({r["question_key"] for r in va_rows}),
                 "val_mse_constant_baseline": const_mse}
    else:
        sel = base_ref["selected"]
        selected = {"lr": float(sel["lr"]), "head": sel["head"], "widths": list(sel["widths"]),
                    "val_mse": sel.get("val_mse"), "rule": "taken from the 1x run (%s)" % args.config_from}
        inner = None

    # The scored critic: the selected configuration trained on every training row.
    t0 = time.time()
    all_ids, truncation["final"] = encode_pairs(tok, [r["context"] for r in rows],
                                                [r["action"] for r in rows], hp["max_len"])
    torch.manual_seed(int(hp["seed"]))
    model = build_critic(hp["backbone"], selected["widths"], hp["head_dropout"], hp["pooling"], hp["seed"]).to(device)
    hist = fit(model, all_ids, [r["target"] for r in rows], hp, selected["lr"], pad_id, device)
    if hist["diverged"]:
        raise E4Error("the final fit of the selected configuration diverged")
    final_dir = os.path.join(args.out_dir, FINAL_DIR)
    os.makedirs(final_dir, exist_ok=True)
    weights = os.path.join(final_dir, FINAL_WEIGHTS)
    torch.save(model.state_dict(), weights)
    critic_cfg = {"backbone": hp["backbone"], "widths": selected["widths"], "head": selected["head"],
                  "head_dropout": hp["head_dropout"], "pooling": model.pooling,
                  "max_len": hp["max_len"], "bf16": hp["bf16"]}
    write_json(os.path.join(final_dir, FINAL_CONFIG), critic_cfg)

    report = {
        "schema": TRAIN_SCHEMA,
        "generated_by": generated_by(),
        "split": {"path": os.path.abspath(args.split).replace("\\", "/"), "sha256": split_sha},
        "data_multiplier": mult,
        "data": data,
        "hyperparameters": hp,
        "fixed": {"optimizer": "torch.optim.AdamW", "lr_schedule": "constant", "grad_clip": None,
                  "loss": "MSE between the prediction and the mean return of the alternative's replays",
                  "lrs_are_preregistered": [float(x) for x in lrs] == [float(x) for x in PREREG_LRS]
                  if mult == 1 else None,
                  "device": device, "torch": getattr(torch, "__version__", None),
                  "transformers": _transformers_version()},
        "heads": [{"head": n, "widths": w} for n, w in heads],
        "inner_split": inner,
        "grid": grid,
        "selected": selected,
        "final": {"n_rows": len(rows), "train_mse_by_epoch": hist["train_mse"],
                  "seconds": round(time.time() - t0, 1),
                  "weights": os.path.relpath(weights, args.out_dir).replace("\\", "/"),
                  "config": critic_cfg},
        "truncation": truncation,
    }
    write_json(os.path.join(args.out_dir, TRAIN_REPORT_FILE), report)
    return report


def _transformers_version() -> Optional[str]:
    try:
        import transformers
        return getattr(transformers, "__version__", None)
    except ImportError:
        return None


# -----------------------------------------------------------------------------
# score
# -----------------------------------------------------------------------------


def run_score(args: argparse.Namespace) -> Dict[str, Any]:
    report_path = os.path.join(args.model_dir, TRAIN_REPORT_FILE)
    if not os.path.isfile(report_path):
        raise E4Error("no %s in %s" % (TRAIN_REPORT_FILE, args.model_dir))
    report = read_json(report_path)
    split, split_sha = load_split(args.split)
    if report["split"]["sha256"] != split_sha:
        raise E4Error("the critic in %s was trained on another split" % args.model_dir)
    cfg = read_json(os.path.join(args.model_dir, FINAL_DIR, FINAL_CONFIG))
    torch, _AutoModel, AutoTokenizer = _torch()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(cfg["backbone"])
    pad_id = tok.pad_token_id
    model = build_critic(cfg["backbone"], cfg["widths"], cfg["head_dropout"], cfg["pooling"], 0)
    state = torch.load(os.path.join(args.model_dir, FINAL_DIR, FINAL_WEIGHTS), map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    contexts, actions, owners = [], [], []
    for pi, p in enumerate(split["heldout"]):
        for c in p["candidates"]:
            contexts.append(p["context"])
            actions.append(c["action_text"])
            owners.append(pi)
    ids, trunc = encode_pairs(tok, contexts, actions, cfg["max_len"])
    preds = predict(model, ids, pad_id, device, int(args.batch_size), bool(cfg["bf16"]))
    per_point: List[List[float]] = [[] for _ in split["heldout"]]
    for pi, v in zip(owners, preds):
        per_point[pi].append(v)
    doc = {
        "schema": SCORES_SCHEMA,
        "generated_by": generated_by(),
        "split": {"path": os.path.abspath(args.split).replace("\\", "/"), "sha256": split_sha},
        "model": {"train_report": os.path.abspath(report_path).replace("\\", "/"),
                  "train_report_sha256": sha256_file(report_path),
                  "data_multiplier": int(report["data_multiplier"]),
                  "selected": report["selected"], "hyperparameters": report["hyperparameters"],
                  "data": {k: report["data"].get(k) for k in ("base", "all", "ratio_rows",
                                                                "ratio_rollouts", "debug_max_rows")},
                  "files": [a["source"] for a in report["data"]["base_files"] + report["data"]["extra_files"]]},
        "truncation": trunc,
        "points": [{"question_key": p["question_key"], "question_id": p["question_id"],
                    "ctx_hash": p["ctx_hash"], "scores": s}
                   for p, s in zip(split["heldout"], per_point)],
    }
    write_json(args.out, doc)
    return doc


# -----------------------------------------------------------------------------
# evaluate
# -----------------------------------------------------------------------------


def _load_scores(path: str, split_sha: str, want_mult: int, split: Mapping[str, Any]) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise E4Error("no scores file at %s" % path)
    doc = read_json(path)
    if not isinstance(doc, dict) or doc.get("schema") != SCORES_SCHEMA:
        raise E4Error("%s is not a %s document" % (path, SCORES_SCHEMA))
    if doc["split"]["sha256"] != split_sha:
        raise E4Error("%s scores another split" % path)
    got = int(doc["model"]["data_multiplier"])
    if got != want_mult:
        raise E4Error("%s comes from a %dx critic, expected %dx" % (path, got, want_mult))
    if doc["model"]["data"].get("debug_max_rows"):
        raise E4Error("%s comes from a debug run (--max-rows); it cannot fill a key" % path)
    pts = doc.get("points") or []
    if len(pts) != len(split["heldout"]):
        raise E4Error("%s scores %d points, the split holds %d" % (path, len(pts), len(split["heldout"])))
    for sp, p in zip(pts, split["heldout"]):
        if sp["question_key"] != p["question_key"] or len(sp["scores"]) != len(p["candidates"]):
            raise E4Error("%s does not line up with the split at question_id %r" % (path, p["question_id"]))
        if any(finite_or_none(v) is None for v in sp["scores"]):
            raise E4Error("%s carries a non-finite score at question_id %r" % (path, p["question_id"]))
    doc["_path"] = path
    return doc


def _check_reference_cell(ref_records, facts: Mapping[str, Any], ref_path: str) -> Dict[str, Any]:
    """Refuse a reference that is not an independent replay of the same cell."""
    ref_facts = cell_facts(ref_records, ref_path)
    src_seed, ref_seed = facts.get("seed"), ref_facts.get("seed")
    if src_seed is None or ref_seed is None:
        raise E4Error("the replay seed is not recorded (cell %r, reference %r); independence "
                      "cannot be checked" % (src_seed, ref_seed))
    if int(src_seed) == int(ref_seed):
        raise E4Error("the reference was replayed under the cell's own seed %s: the batched seeds "
                      "repeat the cell's replays, so the reference is not independent" % src_seed)
    if decoding_without_seed(ref_facts.get("decoding")) != decoding_without_seed(facts.get("decoding")):
        raise E4Error("the reference decoding %r differs from the cell's %r"
                      % (ref_facts.get("decoding"), facts.get("decoding")))
    env_s, env_r = facts.get("task_env_cfg") or {}, ref_facts.get("task_env_cfg") or {}
    diff = [k for k in REWARD_FIELDS if env_s.get(k) != env_r.get(k)]
    if diff:
        raise E4Error("the reference's reward configuration differs from the cell's on %s" % ", ".join(diff))
    for k in ("target_role", "next_role"):
        if ref_facts.get(k) != facts.get(k):
            raise E4Error("the reference's %s %r differs from the cell's %r" % (k, ref_facts.get(k), facts.get(k)))
    snap_s, snap_r = snapshot_of(facts.get("pretrain")), snapshot_of(ref_facts.get("pretrain"))
    if snap_s and snap_r and snap_s != snap_r:
        raise E4Error("the reference policy snapshot %s differs from the cell's %s" % (snap_r, snap_s))
    return ref_facts


def _boxed(text: str) -> bool:
    return "\\boxed" in text


def run_evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    if args.undefined_rho not in UNDEFINED_CHOICES:
        raise E4Error("--undefined-rho is one of %s" % ", ".join(UNDEFINED_CHOICES))
    split, split_sha = load_split(args.split)
    facts = split["eval_cell"]["facts"]
    if not os.path.isfile(args.reference):
        raise E4Error("no reference bucket file at %s" % args.reference)
    ref_records = read_buckets(args.reference)
    if not ref_records:
        raise E4Error("the reference file %s holds no buckets" % args.reference)
    ref_facts = _check_reference_cell(ref_records, facts, args.reference)
    want = int(args.reference_replays)

    ref_by_ctx: Dict[int, Tuple[int, Dict[str, Any]]] = {}
    for lineno, _raw, b in ref_records:
        ch = b.get("ctx_hash")
        if isinstance(ch, bool) or not isinstance(ch, int):
            raise E4Error("%s:%d has no integer ctx_hash" % (args.reference, lineno))
        if ch in ref_by_ctx:
            raise E4Error("%s: ctx_hash %d appears on lines %d and %d"
                          % (args.reference, ch, ref_by_ctx[ch][0], lineno))
        ref_by_ctx[ch] = (lineno, b)

    missing = [p["question_id"] for p in split["heldout"] if p["ctx_hash"] not in ref_by_ctx]
    if missing:
        raise E4Error("the reference covers %d of %d held-out points; missing question_id %s%s"
                      % (len(split["heldout"]) - len(missing), len(split["heldout"]),
                         ", ".join(str(m) for m in missing[:10]), " ..." if len(missing) > 10 else ""))

    scores = {1: _load_scores(args.scores, split_sha, 1, split)}
    if args.scores_2x:
        scores[2] = _load_scores(args.scores_2x, split_sha, 2, split)
    if args.scores_4x:
        scores[4] = _load_scores(args.scores_4x, split_sha, 4, split)

    points_out: List[Dict[str, Any]] = []
    dup_total = dup_any = prefix_sig = n_ref_replays = ref_boxed = c3_boxed = n_c3_replays = 0
    for pi, p in enumerate(split["heldout"]):
        lineno, rb = ref_by_ctx[p["ctx_hash"]]
        where = "%s:%d" % (args.reference, lineno)
        if question_of(rb, where) != p["question"]:
            raise E4Error("%s: same ctx_hash, different question than held-out question_id %r"
                          % (where, p["question_id"]))
        if label_of(rb) != p["label"]:
            raise E4Error("%s: label %r, the cell recorded %r" % (where, label_of(rb), p["label"]))
        rc = candidates_of(rb, where)
        if [c["action_text"] for c in rc] != [c["action_text"] for c in p["candidates"]]:
            raise E4Error("%s: the reference replayed other alternatives than the held-out point "
                          "question_id %r holds (or another j order)" % (where, p["question_id"]))
        for c in rc:
            if len(c["returns"]) != want:
                raise E4Error("%s: an alternative carries %d reference replays, expected %d"
                              % (where, len(c["returns"]), want))
        ref_means = [float(np.mean(c["returns"])) for c in rc]
        c3_means = [float(np.mean(c["returns"])) for c in p["candidates"]]
        c3_loo = loo_advantages(c3_means)
        rho_c3 = spearman(c3_loo, ref_means)
        ref_defined = len(set(ref_means)) > 1
        crit: Dict[str, Any] = {}
        for mult, doc in scores.items():
            s = [float(v) for v in doc["points"][pi]["scores"]]
            crit["%dx" % mult] = {"scores": s, "rho": spearman(s, ref_means)}
        for rc_c, sc in zip(rc, p["candidates"]):
            own = set(sc["next_actions"])
            hits = sum(1 for t in rc_c["next_actions"] if t in own)
            dup_total += hits
            dup_any += 1 if hits else 0
            k = len(sc["next_actions"])
            if k and rc_c["next_actions"][:k] == sc["next_actions"]:
                prefix_sig += 1
            n_ref_replays += len(rc_c["next_actions"])
            ref_boxed += sum(1 for t in rc_c["next_actions"] if _boxed(t))
            n_c3_replays += len(sc["next_actions"])
            c3_boxed += sum(1 for t in sc["next_actions"] if _boxed(t))
        points_out.append({
            "question_key": p["question_key"], "question_id": p["question_id"],
            "ctx_hash": p["ctx_hash"], "n_alternatives": len(rc),
            "reference_means": ref_means, "reference_defined": ref_defined,
            "c3_means": c3_means, "c3_replays": [len(c["returns"]) for c in p["candidates"]],
            "c3_loo": c3_loo, "rho_c3": rho_c3, "critic": crit,
        })

    def paired(mult: int, convention: str) -> Tuple[Dict[str, Any], Dict[str, int], List[int]]:
        tag = "%dx" % mult
        c3v, crv, used = [], [], []
        why = {"reference_constant": 0, "c3_undefined": 0, "critic_undefined": 0}
        for i, po in enumerate(points_out):
            if not po["reference_defined"]:
                why["reference_constant"] += 1
                continue
            a, b = po["rho_c3"], po["critic"][tag]["rho"]
            if a is None:
                why["c3_undefined"] += 1
            if b is None:
                why["critic_undefined"] += 1
            if convention == "exclude" and (a is None or b is None):
                continue
            c3v.append(0.0 if a is None else a)
            crv.append(0.0 if b is None else b)
            used.append(i)
        return paired_stats(c3v, crv, int(args.bootstrap), int(args.bootstrap_seed)), why, used

    results: Dict[str, Any] = {}
    for convention in UNDEFINED_CHOICES:
        results[convention] = {}
        for mult in scores:
            st, why, used = paired(mult, convention)
            results[convention]["%dx" % mult] = {"stats": st, "counts": why, "n_used": len(used)}
    primary_used = paired(1, args.undefined_rho)[2]
    for i, po in enumerate(points_out):
        po["in_primary_set"] = i in set(primary_used)

    conv = args.undefined_rho
    main = results[conv]["1x"]
    st = main["stats"]
    first = scores[1]["model"]
    sources = [split["eval_cell"]["source"], source_path(args.reference), source_path(args.split),
               source_path(args.scores)]
    ref_seed = ref_facts.get("seed")
    counts = main["counts"]
    keys: Dict[str, Dict[str, Any]] = {}
    if st["n"] > 0:
        common = ("held-out decision points of %s (%d drawn; %d with a constant reference, not "
                  "rankable; C3 undefined on %d, critic on %d; undefined rank correlations: %s); "
                  "reference %d replays per alternative, seed %s"
                  % (split["eval_cell"]["source"], len(points_out), counts["reference_constant"],
                     counts["c3_undefined"], counts["critic_undefined"], conv, want, ref_seed))
        sel = first["selected"]
        keys[KEY_N] = {"value": int(st["n"]), "n": int(st["n"]), "source": sources, "note": common}
        keys[KEY_CRITIC] = {"value": st["mean_critic"], "n": int(st["n"]), "source": sources,
                            "note": ("best critic: lr %g, head %s %s, chosen on inner validation MSE, "
                                     "refit on %d training rows; %s"
                                     % (sel["lr"], sel["head"], sel["widths"] or "linear",
                                        first["data"]["all"]["n_rows"], common))}
        keys[KEY_C3] = {"value": st["mean_c3"], "n": int(st["n"]), "source": sources,
                        "note": ("leave-one-out advantages from the cell's own %s replays per "
                                 "alternative; %s" % (facts.get("replays_per_alt"), common))}
        keys[KEY_DIFF] = {"value": st["mean_diff"], "n": int(st["n"]), "source": sources,
                          "note": ("C3 minus critic, mean over %d paired points; bootstrap 95%% "
                                   "interval %s (%d resamples over points)"
                                   % (st["n"], st["boot_ci95"], st["boot_resamples"]))}
        p_str = format_p(st["p_t"])
        if p_str is not None:
            keys[KEY_P] = {"value": p_str, "n": int(st["n"]), "source": sources,
                           "note": ("two-sided paired t test over %d points (p = %s); Wilcoxon "
                                    "signed-rank p = %s" % (st["n"], st["p_t"], st["p_wilcoxon"]))}
    for mult in (2, 4):
        if mult not in scores:
            continue
        stm = results[conv]["%dx" % mult]["stats"]
        if stm["n"] == 0:
            continue
        dm = scores[mult]["model"]["data"]
        keys[KEY_MULT[mult]] = {
            "value": stm["mean_critic"], "n": int(stm["n"]),
            "source": sources[:3] + [source_path(scores[mult]["_path"])],
            "note": ("the 1x selected configuration retrained on %.3fx the training rollouts "
                     "(%.3fx the rows): %s; undefined rank correlations: %s"
                     % (dm["ratio_rollouts"], dm["ratio_rows"],
                        ", ".join(scores[mult]["model"]["files"]), conv))}

    manifest_checked = False
    if args.manifest and os.path.isfile(args.manifest):
        manifest = read_json(args.manifest)
        manifest_checked = True
    else:
        manifest = None
        print("[e4] no manifest at %s: keys not checked against one" % args.manifest, file=sys.stderr)
    out_keys: Dict[str, Any] = {}
    for key in sorted(keys):
        if manifest is not None:
            entry = manifest.get(key)
            if entry is None:
                print("skip %s: not a manifest key" % key, file=sys.stderr)
                continue
            if str(entry.get("unit")) == "verdict":
                print("skip %s: verdict key, the maintainers decide it" % key, file=sys.stderr)
                continue
            unit = entry.get("unit")
        else:
            unit = KEY_UNITS.get(key)
        rec = keys[key]
        out_keys[key] = {"value": rec["value"], "n": rec["n"], "unit": unit,
                         "source": list(rec["source"]), "note": rec["note"]}

    wanted = list(PRIMARY_KEYS) + [KEY_MULT[m] for m in (2, 4) if m in scores]
    missing_keys = [k for k in wanted if k not in out_keys]
    crit_1 = st["mean_critic"]
    c3_1 = st["mean_c3"]
    criterion = {
        "primary": "best critic < %.1f and C3 >= %.1f (preregistration E4)" % (CRITIC_BELOW, C3_AT_LEAST),
        "critic": crit_1, "c3": c3_1,
        "critic_below": (crit_1 < CRITIC_BELOW) if crit_1 is not None else None,
        "c3_at_least": (c3_1 >= C3_AT_LEAST) if c3_1 is not None else None,
        "undefined_rho": conv,
        "decided_by": "the maintainers; %s is not written by this script" % VERDICT_KEY,
    }
    diagnostics = {
        "reference_points_not_heldout": len(ref_by_ctx) - len(points_out),
        "reference_candidate_reuse": all(bool((b.get("meta") or {}).get("candidate_reuse"))
                                         for _l, _r, b in ref_records),
        "reference_replay_texts_equal_to_a_c3_replay": dup_total,
        "alternatives_with_any_such_replay": dup_any,
        "alternatives_whose_first_c_reference_replays_are_the_c3_replays": prefix_sig,
        "reference_replays": n_ref_replays,
        "reference_replays_with_boxed": ref_boxed,
        "c3_replays": n_c3_replays,
        "c3_replays_with_boxed": c3_boxed,
        "reference_seed": ref_seed, "cell_seed": facts.get("seed"),
        "reference_decoding": ref_facts.get("decoding"),
        "critic_truncation": {"%dx" % m: d.get("truncation") for m, d in scores.items()},
    }
    eval_doc = {
        "schema": EVAL_SCHEMA,
        "generated_by": generated_by(),
        "undefined_rho": conv,
        "inputs": {
            "split": {"path": source_path(args.split), "sha256": split_sha},
            "eval_cell": {"path": split["eval_cell"]["source"], "sha256": split["eval_cell"]["sha256"]},
            "reference": {"path": source_path(args.reference), "sha256": sha256_file(args.reference)},
            "scores": {"%dx" % m: {"path": source_path(d["_path"]), "sha256": sha256_file(d["_path"])}
                       for m, d in scores.items()},
            "reference_replays_expected": want,
        },
        "critics": {"%dx" % m: {"selected": d["model"]["selected"], "data": d["model"]["data"],
                                 "files": d["model"]["files"], "hyperparameters": d["model"]["hyperparameters"]}
                    for m, d in scores.items()},
        "results": results,
        "criterion": criterion,
        "keys": out_keys,
        "missing_keys": missing_keys,
        "manifest_checked": manifest_checked,
        "diagnostics": diagnostics,
        "points": points_out,
    }
    summary = {
        "experiment": "E4",
        "undefined_rho": conv,
        "generated_by": eval_doc["generated_by"],
        "keys": out_keys,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    write_json(os.path.join(args.out_dir, "e4_eval.json"), eval_doc)
    write_json(os.path.join(args.out_dir, "summary.json"), summary)
    return eval_doc


# -----------------------------------------------------------------------------
# command line
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("split", help="draw the held-out decision points of the evaluation cell")
    s.add_argument("--eval-buckets", required=True, help="buckets.jsonl of the evaluation cell")
    s.add_argument("--heldout-frac", type=float, default=0.2)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", required=True, help="split.json to write")

    p = sub.add_parser("prepare-reference", help="held-out points in the form build-buckets reads")
    p.add_argument("--split", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--eval-buckets", default=None, help="the evaluation cell here, if not at the path split recorded")
    p.add_argument("--pool-ids", default=None, help="configs/data/pool_informative_ids.json: check every question against its pool id")
    p.add_argument("--roles-path", default=None, help="role file for the task file (default by workflow)")

    t = sub.add_parser("train", help="sweep, select and refit the critic; or retrain at 2x/4x")
    t.add_argument("--split", required=True)
    t.add_argument("--out-dir", required=True)
    t.add_argument("--eval-buckets", default=None, help="the evaluation cell here, if not at the path split recorded")
    t.add_argument("--train-buckets", nargs="*", default=None, help="1x: more bucket files for the base training data")
    t.add_argument("--data-multiplier", type=int, default=1, choices=(1, 2, 4))
    t.add_argument("--config-from", default=None, help="2x/4x: the 1x run's train_report.json")
    t.add_argument("--extra-buckets", nargs="*", default=None, help="2x/4x: bucket files added to the base data")
    t.add_argument("--backbone", default=None, help="mmBERT-base snapshot directory (or hub id under HF_HUB_OFFLINE)")
    t.add_argument("--heads", nargs="*", default=None, help="name=linear or name=w1,w2 (two sizes per the preregistration)")
    t.add_argument("--lrs", nargs="*", type=float, default=None, help="default: the preregistered 1e-5 3e-5 1e-4")
    t.add_argument("--epochs", type=int, default=None)
    t.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    t.add_argument("--max-len", dest="max_len", type=int, default=None)
    t.add_argument("--weight-decay", dest="weight_decay", type=float, default=None)
    t.add_argument("--head-dropout", dest="head_dropout", type=float, default=None)
    t.add_argument("--pooling", choices=("config", "cls", "mean"), default=None)
    t.add_argument("--val-frac", dest="val_frac", type=float, default=None)
    t.add_argument("--seed", type=int, default=None)
    t.add_argument("--bf16", action="store_true", default=None)
    t.add_argument("--device", default=None)
    t.add_argument("--max-rows", dest="max_rows", type=int, default=None,
                   help="debug only: cap the training rows; scores from such a run cannot fill a key")

    c = sub.add_parser("score", help="score every held-out alternative with a trained critic")
    c.add_argument("--model-dir", required=True)
    c.add_argument("--split", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--batch-size", dest="batch_size", type=int, default=32)
    c.add_argument("--device", default=None)

    e = sub.add_parser("evaluate", help="rank correlations against the reference, paired test, summary.json")
    e.add_argument("--split", required=True)
    e.add_argument("--reference", required=True, help="buckets.jsonl of the reference replays")
    e.add_argument("--scores", required=True, help="scores of the 1x critic")
    e.add_argument("--scores-2x", dest="scores_2x", default=None)
    e.add_argument("--scores-4x", dest="scores_4x", default=None)
    e.add_argument("--undefined-rho", dest="undefined_rho", required=True, choices=UNDEFINED_CHOICES,
                   help="a point where C3's or the critic's ranking is constant: drop it from the "
                        "pair (exclude) or count its correlation as 0 (zero); both are written")
    e.add_argument("--reference-replays", dest="reference_replays", type=int, default=REFERENCE_REPLAYS)
    e.add_argument("--bootstrap", type=int, default=BOOTSTRAP_RESAMPLES)
    e.add_argument("--bootstrap-seed", dest="bootstrap_seed", type=int, default=0)
    e.add_argument("--manifest", default=DEFAULT_MANIFEST)
    e.add_argument("--out-dir", required=True)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.cmd == "split":
            doc = run_split(args.eval_buckets, args.out, args.heldout_frac, args.seed)
            print("[e4] split: %d held out, %d training points of %s"
                  % (doc["n_heldout"], doc["n_train_points"], doc["eval_cell"]["source"]))
        elif args.cmd == "prepare-reference":
            rep = run_prepare(args.split, args.out_dir, args.eval_buckets, args.pool_ids, args.roles_path)
            print("[e4] prepared %d held-out points in %s" % (rep["n_points"], args.out_dir))
        elif args.cmd == "train":
            rep = run_train(args)
            print("[e4] trained: selected %s; final fit on %d rows"
                  % (rep["selected"], rep["final"]["n_rows"]))
        elif args.cmd == "score":
            doc = run_score(args)
            print("[e4] scored %d held-out points" % len(doc["points"]))
        elif args.cmd == "evaluate":
            doc = run_evaluate(args)
            print(json.dumps({"keys": {k: v["value"] for k, v in doc["keys"].items()},
                              "criterion": doc["criterion"]}, ensure_ascii=False, indent=1))
            if doc["missing_keys"]:
                # The per-point record is written so the reason can be read; the exit code
                # says the summary is short of keys (no paired points, or an undefined p).
                print("[e4] KEYS MISSING from summary.json: %s (see results and points in e4_eval.json)"
                      % ", ".join(doc["missing_keys"]), file=sys.stderr)
                return 3
    except E4Error as exc:
        print("REFUSED: %s" % exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
