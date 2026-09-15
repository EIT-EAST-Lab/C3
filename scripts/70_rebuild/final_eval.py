#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the evaluation of record on one trained policy and write its record.

This is the evaluation the main table is read off, run once per model after
training has finished. It is `eval_probe.py` with one thing added: the number of
samples per problem is per suite here, and one evaluation run has exactly one
sample count, so the suites are split into one run per sample count.

The estimator is avg@k. Every problem is sampled k times, its score is the mean
of those k rewards, and the accuracy of a suite is the mean of its problem
scores. k comes from the sampling-noise arithmetic, not from taste, and it is
stated per suite on the command line:

    MATH500 4, Minerva-Math 4, GSM8K-test 4, CMATH-test 4
    AMC23 16, AIME24 16, AIME25 16

Why one run per sample count: the trainer evaluates the whole concatenated
evaluation set with a single `--eval_n_samples_per_prompt`, and reshapes the
rewards to (problems, k) with that one k. Two suites at different k cannot ride
in one run. So this driver groups the suites by k, writes one generated task
file per group that carries only that group's suites and is otherwise the task
file it came from, and runs one evaluation per group through the same path
`eval_probe.py` uses: `scripts/50_eval/paper_main_results.sh one` in its direct
mode, which runs `openrlhf.cli.train_ppo_ray --eval_only` with the learning
rates at zero.

Decoding: temperature 0.7, top-p 0.8, top-k 20, which is the sampling setting of
the training rollouts, and a generation cap of 2048 tokens for training and
evaluation alike. Every suite of the protocol is sampled, the two saturation
controls included. A group at k=1 would be decoded greedily (temperature 0),
where the nucleus and top-k cut-offs do not apply; the shipped sample counts
have no such group, and k=1 stays available as a value rather than as the
reading of any suite.

What comes out, per suite, in the evaluation record:

    accuracy        avg@k: mean over problems of the mean of their k rewards
    k               samples per problem of this suite
    n_questions     problems the run actually covered
    boxed_rate      share of scored generations whose answer was extracted from
                    a `\\boxed{...}`, read off `reward_info_json.per_role[0]`
    wilson95        95 percent Wilson interval over the draws, the same
                    convention `eval_probe.py` reports and with the same caveat:
                    it is binomial over problem times sample draws, so it is
                    narrower than a cluster-aware interval whenever k > 1
    per_question    the k rewards of every problem, in dump order

AIME24 and AIME25 are also merged into one `AIME` entry of 60 problems, which is
the suite the main table reports. The merge is the mean over the 60 problem
scores, so it is not the mean of the two suite accuracies unless the two suites
have the same number of problems.

Examples:

    # Print the commands of the groups and stop.
    python scripts/70_rebuild/final_eval.py --policy /models/Qwen3-4B-Instruct-2507 --dry-run

    # Run every group and write the record.
    python scripts/70_rebuild/final_eval.py --policy ckpt/_runs/paper_C3_math_seed0/final_hf \\
        --run_id paper_C3_math_seed0 --out /abs/path/final_eval_record.json

    # Re-derive the record from artifacts that are already on disk.
    python scripts/70_rebuild/final_eval.py --policy /models/Qwen3-4B-Instruct-2507 \\
        --summarize_only --out /abs/path/final_eval_record.json
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

# eval_probe.py sits in this directory, which is not a package, so it is
# imported the way the tests import it: by putting the directory on the path.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import eval_probe  # noqa: E402

EVAL_SCRIPT = eval_probe.EVAL_SCRIPT

DEFAULT_TASK = "configs/tasks/math_final_eval.yaml"
DEFAULT_RUN_ID = "final_eval"
DEFAULT_OUT_SUBDIR = "final_eval"
DEFAULT_CKPT_ROOT = "ckpt"

# SFT maps to marl_algorithm=none in the evaluation script, which is what an
# evaluation with no credit assignment means there. The policy under evaluation
# is whatever `--policy` points at, a base model or the `final_hf` directory a
# training run wrote.
DEFAULT_METHOD = "SFT"

DEFAULT_K_BY_SUITE = "MATH500=4,Minerva-Math=4,AMC23=16,AIME24=16,AIME25=16,GSM8K-test=4,CMATH-test=4"

DEFAULT_GENERATE_MAX_LEN = 2048
DEFAULT_PROMPT_MAX_LEN = 2560
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.8
DEFAULT_TOP_K = 20
DEFAULT_SEED = 0
DEFAULT_TIMEOUT_H = 8.0

# The merged entries of the record: a reported suite that is more than one file.
MERGED_SUITES: Dict[str, Tuple[str, ...]] = {"AIME": ("AIME24", "AIME25")}

# A reward at or above this counts as a solved draw, as in eval_probe.py. The
# math environment scores vote_binary, so rewards are 0 or 1 and the threshold
# never bites; the record reports how many rewards were neither.
SUCCESS_THRESHOLD = eval_probe.SUCCESS_THRESHOLD

# Directory of the generated per-group task files, under the run root.
GROUP_TASK_SUBDIR = "_task_groups"

RECORD_SCHEMA = "c3.final_eval.record/1"


# ---------------------------------------------------------------------------
# Group description
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalGroup:
    """One evaluation run: the suites that share a sample count."""

    label: str
    k: int
    profile: str
    temperature: float
    top_p: Optional[float]
    top_k: Optional[int]
    suites: Tuple[str, ...]
    task_path: str
    eval_dir: str
    argv: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)

    @property
    def eval_jsonl(self) -> str:
        return eval_probe.posix_join(self.eval_dir, "eval_only.jsonl")

    @property
    def metrics_jsonl(self) -> str:
        return eval_probe.posix_join(self.eval_dir, "eval_only.jsonl.metrics.jsonl")

    def command_line(self) -> str:
        prefix = "".join(f"{k}={shlex.quote(v)} " for k, v in sorted(self.env.items()))
        return prefix + " ".join(shlex.quote(a) for a in self.argv)


# ---------------------------------------------------------------------------
# Planning the groups
# ---------------------------------------------------------------------------


def parse_k_by_suite(text: str) -> Dict[str, int]:
    """`Name=k,Name=k` as a mapping, refusing anything it cannot read."""
    out: Dict[str, int] = {}
    for chunk in str(text).split(","):
        item = chunk.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"--k_by_suite entry {item!r} is not of the form <suite>=<k>")
        name, _, raw = item.partition("=")
        name = name.strip()
        raw = raw.strip()
        if not name:
            raise ValueError(f"--k_by_suite entry {item!r} has an empty suite name")
        try:
            k = int(raw)
        except ValueError:
            raise ValueError(f"--k_by_suite: the sample count of {name!r} is not an integer: {raw!r}") from None
        if k <= 0:
            raise ValueError(f"--k_by_suite: the sample count of {name!r} must be positive, got {k}")
        if name in out and out[name] != k:
            raise ValueError(f"--k_by_suite gives {name!r} two different sample counts: {out[name]} and {k}")
        out[name] = k
    if not out:
        raise ValueError("--k_by_suite is empty")
    return out


def group_task_dir(*, ckpt_root: str, run_id: str, out_subdir: str) -> str:
    """Where the generated per-group task files are written."""
    root = ckpt_root if eval_probe.is_rooted(ckpt_root) else eval_probe.posix_join(REPO_ROOT.as_posix(), ckpt_root)
    return eval_probe.posix_join(root, "_runs", "_sft_main_results", run_id, out_subdir, GROUP_TASK_SUBDIR)


def group_task_path(*, ckpt_root: str, run_id: str, out_subdir: str, task_path: str, k: int) -> str:
    """The generated task file of one group, named after the task it came from."""
    stem = eval_probe.task_tag(task_path)
    directory = group_task_dir(ckpt_root=ckpt_root, run_id=run_id, out_subdir=out_subdir)
    return eval_probe.posix_join(directory, f"{stem}_k{int(k)}.yaml")


def plan_groups(args: argparse.Namespace) -> List[EvalGroup]:
    """
    The evaluation runs, cheapest sample count first.

    Every suite the task declares has to have a sample count and every named
    sample count has to belong to a suite the task declares: a typo on either
    side would otherwise evaluate a different set of problems than the record
    claims.
    """
    task_path = eval_probe.absolute_repo_path(args.task)
    declared = eval_probe.expected_suites(Path(task_path))
    if not declared:
        raise ValueError(f"the task file declares no evaluation suite: {task_path}")

    k_by_suite = parse_k_by_suite(args.k_by_suite)

    unknown = [name for name in k_by_suite if name not in declared]
    if unknown:
        raise ValueError(f"--k_by_suite names suites the task does not declare: {sorted(unknown)}; task has {declared}")
    missing = [name for name in declared if name not in k_by_suite]
    if missing:
        raise ValueError(f"--k_by_suite gives no sample count for {missing}")

    by_k: Dict[int, List[str]] = {}
    for name in declared:
        by_k.setdefault(k_by_suite[name], []).append(name)

    script = eval_probe.absolute_repo_path(EVAL_SCRIPT)
    env = {
        "GENERATE_MAX_LEN": str(int(args.generate_max_len)),
        "PROMPT_MAX_LEN": str(int(args.prompt_max_len)),
    }

    groups: List[EvalGroup] = []
    for k in sorted(by_k):
        suites = tuple(by_k[k])
        label = f"k{k}"
        if args.only and args.only != label:
            continue

        # A single sample per problem is the greedy reading; the sampling
        # profile is for everything else. No suite of the shipped protocol asks
        # for k=1, so this branch is what a k of 1 would mean rather than the
        # reading of anything in the main table. The profile only picks the
        # directory the artifacts land in, so both numbers are written into the
        # command either way.
        if k == 1:
            profile, temperature = "greedy", 0.0
            top_p: Optional[float] = None
            top_k: Optional[int] = None
        else:
            profile, temperature = "n10", float(args.temperature)
            top_p = float(args.top_p)
            top_k = int(args.top_k)

        this_task = group_task_path(
            ckpt_root=args.ckpt_root,
            run_id=args.run_id,
            out_subdir=args.out_subdir,
            task_path=task_path,
            k=k,
        )

        extra = f"--eval_n_samples_per_prompt {int(k)} --eval_temperature {float(temperature)}"
        if top_p is not None and top_k is not None:
            extra = f"{extra} --top_p {float(top_p)} --top_k {int(top_k)}"
        # Platform specific trainer flags (for example --attn_implementation sdpa where
        # flash-attn is not installed, or the GPU counts of a one GPU run) are appended
        # from the environment, so this driver stays free of machine details.
        extra_env = os.environ.get("FINAL_EVAL_EXTRA_ARGS", "").strip()
        if extra_env:
            extra = f"{extra} {extra_env}"

        argv = [
            args.bash,
            script,
            "one",
            "--id",
            args.run_id,
            "--profile",
            profile,
            "--method",
            args.method,
            "--task",
            this_task,
            "--source_type",
            "hf_base",
            "--hf_base",
            args.policy,
            "--seed",
            str(int(args.seed)),
            "--ckpt_root",
            args.ckpt_root,
            "--out_subdir",
            args.out_subdir,
            "--extra_eval_args",
            extra,
        ]

        groups.append(
            EvalGroup(
                label=label,
                k=int(k),
                profile=profile,
                temperature=float(temperature),
                top_p=top_p,
                top_k=top_k,
                suites=suites,
                task_path=this_task,
                eval_dir=eval_probe.eval_dir_for(
                    ckpt_root=args.ckpt_root,
                    run_id=args.run_id,
                    out_subdir=args.out_subdir,
                    task_path=this_task,
                    profile=profile,
                ),
                argv=argv,
                env=dict(env),
            )
        )

    if not groups:
        raise ValueError(f"--only {args.only!r} selected no group")
    return groups


# ---------------------------------------------------------------------------
# The generated task file of a group
# ---------------------------------------------------------------------------


def absolute_roles_path(main_task: Path, roles_path: Any) -> Optional[str]:
    """
    The roles file of the source task as an absolute path, or None.

    The loader resolves a relative `mas.roles_path` against the directory of the
    task file it is reading, and a generated group task file does not sit beside
    the task it came from: `../roles/math/roles_duo.json` means something else
    under the run root than it does under `configs/tasks/`. Pinning the absolute
    path keeps the generated file pointing at the roles the source task named.
    None means the value could not be resolved to a file here, in which case it
    is left exactly as the source task wrote it rather than guessed at.
    """
    raw = str(roles_path or "").strip()
    if not raw:
        return None
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = main_task.parent / candidate
    resolved = candidate.resolve()
    return resolved.as_posix() if resolved.is_file() else None


def group_task_document(
    main_spec: Dict[str, Any],
    suites: Sequence[str],
    *,
    roles_path: Optional[str] = None,
) -> Dict[str, Any]:
    """The task document of one group: the main task with its suites narrowed."""
    spec = copy.deepcopy(main_spec)
    environment = spec.get("environment")
    if not isinstance(environment, dict):
        raise ValueError("the task file has no environment block")

    by_name: Dict[str, Any] = {}
    for entry in environment.get("eval_suites") or []:
        if isinstance(entry, dict) and entry.get("name"):
            by_name[str(entry["name"])] = entry

    narrowed = []
    for name in suites:
        if name not in by_name:
            raise ValueError(f"the task file declares no evaluation suite named {name!r}")
        narrowed.append(by_name[name])

    environment["eval_suites"] = narrowed

    if roles_path is not None:
        mas = spec.get("mas")
        if not isinstance(mas, dict):
            raise ValueError("the task file has no mas block to pin the roles path in")
        mas["roles_path"] = roles_path

    return spec


def write_group_task(group: EvalGroup, *, main_task: Path) -> str:
    """Write the generated task file of one group and return its path."""
    with main_task.open("r", encoding="utf-8") as handle:
        main_spec = yaml.safe_load(handle) or {}

    declared_roles = (main_spec.get("mas") or {}).get("roles_path") if isinstance(main_spec.get("mas"), dict) else None
    roles_path = absolute_roles_path(main_task, declared_roles)
    if roles_path is None and declared_roles:
        print(
            f"[final_eval] WARNING: could not resolve the roles path {declared_roles!r} of {main_task.as_posix()}; "
            "the generated task file keeps it as written, which the loader resolves against the generated "
            "file's own directory",
            file=sys.stderr,
        )
    document = group_task_document(main_spec, group.suites, roles_path=roles_path)

    header = (
        "# Generated by scripts/70_rebuild/final_eval.py. Do not edit; it is rewritten on every run.\n"
        f"# Source task: {main_task.as_posix()}\n"
        f"# Evaluation group: {group.label}, {len(group.suites)} suite(s) at {group.k} sample(s) per problem.\n"
        f"# Suites: {', '.join(group.suites)}\n"
        "#\n"
        "# One evaluation run has one sample count, so the suites of the source task are\n"
        "# split by sample count and each group gets a task file of its own. Everything\n"
        "# outside the evaluation suites is the source task, unchanged.\n"
    )

    out_path = Path(group.task_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(document, sort_keys=False, allow_unicode=True, default_flow_style=False)
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(header)
        handle.write(body)
    return out_path.as_posix()


# ---------------------------------------------------------------------------
# Reading what the evaluation left behind
# ---------------------------------------------------------------------------


def row_pred_method(row: Dict[str, Any]) -> Optional[str]:
    """
    How the scorer extracted the answer of one generation, or None.

    The multi-agent branch of the evaluation dump carries the reward detail of
    the answering role as `reward_info_json`, a JSON string written by
    `c3.envs.math.reward.score_math`, whose `per_role` list holds one entry per
    scored role. The answering role is the first entry, and its `pred_method` is
    `boxed` when the answer came out of a `\\boxed{...}`.
    """
    raw: Any = row.get("reward_info_json")
    if isinstance(raw, (list, tuple)):
        raw = raw[0] if len(raw) == 1 else None
    if raw is None:
        return None

    if isinstance(raw, str):
        text = raw.strip()
        if not text or text == "null":
            return None
        try:
            info: Any = json.loads(text)
        except ValueError:
            return None
    elif isinstance(raw, dict):
        info = raw
    else:
        return None

    if not isinstance(info, dict):
        return None
    per_role = info.get("per_role")
    if not isinstance(per_role, (list, tuple)) or not per_role:
        return None
    first = per_role[0]
    if not isinstance(first, dict):
        return None
    method = first.get("pred_method")
    return None if method is None else str(method)


def summarize_dump(rows: Sequence[Dict[str, Any]], *, k: int) -> Dict[str, Dict[str, Any]]:
    """Per-suite avg@k, interval, boxed rate and per-question rewards of one dump."""
    counters: Dict[str, Dict[str, Any]] = {}
    by_question: Dict[str, Dict[Any, List[float]]] = {}
    question_order: Dict[str, List[Any]] = {}

    for index, row in enumerate(rows):
        suite = row.get("datasource")
        suite = "unknown" if suite is None else str(suite)
        bucket = counters.setdefault(
            suite,
            {
                "n_draws": 0,
                "n_missing_reward": 0,
                "n_missing_question_id": 0,
                "n_non_binary_reward": 0,
                "successes": 0,
                "n_boxed": 0,
                "n_with_pred_method": 0,
                "n_without_pred_method": 0,
            },
        )
        questions = by_question.setdefault(suite, {})
        order = question_order.setdefault(suite, [])

        bucket["n_draws"] += 1

        reward = eval_probe.row_reward(row)
        if reward is None:
            bucket["n_missing_reward"] += 1
        else:
            if reward not in (0.0, 1.0):
                bucket["n_non_binary_reward"] += 1
            if reward >= SUCCESS_THRESHOLD:
                bucket["successes"] += 1
            # A row without a question id is kept as a problem of its own rather
            # than folded into one bucket, so a dump that lost the id understates
            # nothing and shows up as n_missing_question_id instead.
            question_id = row.get("question_id")
            if question_id is None:
                bucket["n_missing_question_id"] += 1
                question_id = ("row", index)
            if question_id not in questions:
                questions[question_id] = []
                order.append(question_id)
            questions[question_id].append(reward)

            method = row_pred_method(row)
            if method is None:
                bucket["n_without_pred_method"] += 1
            else:
                bucket["n_with_pred_method"] += 1
                if method == "boxed":
                    bucket["n_boxed"] += 1

    out: Dict[str, Dict[str, Any]] = {}
    for suite, bucket in sorted(counters.items()):
        questions = by_question.get(suite, {})
        order = question_order.get(suite, [])
        n_questions = len(order)
        scored_draws = sum(len(questions[q]) for q in order)

        accuracy = (sum(sum(questions[q]) / len(questions[q]) for q in order) / n_questions) if n_questions else None
        low, high = eval_probe.wilson_interval(bucket["successes"], scored_draws)
        off_k = sum(1 for q in order if len(questions[q]) != int(k))

        with_method = int(bucket["n_with_pred_method"])
        boxed_rate = (bucket["n_boxed"] / with_method) if with_method else None

        out[suite] = {
            "k": int(k),
            "n_questions": n_questions,
            "n_draws": bucket["n_draws"],
            "n_scored_draws": scored_draws,
            "n_questions_off_k": off_k,
            "n_missing_reward": bucket["n_missing_reward"],
            "n_missing_question_id": bucket["n_missing_question_id"],
            "n_non_binary_reward": bucket["n_non_binary_reward"],
            "successes": bucket["successes"],
            "accuracy": accuracy,
            "wilson95": {"low": low, "high": high},
            "wilson95_basis": (
                "binomial over problem times sample draws; samples of one problem are correlated, "
                "so this interval is narrower than a cluster-aware one when more than one sample "
                "per problem is drawn"
            ),
            "success_threshold": SUCCESS_THRESHOLD,
            "boxed_rate": boxed_rate,
            "boxed": {
                "n_boxed": bucket["n_boxed"],
                "n_with_pred_method": with_method,
                "n_without_pred_method": bucket["n_without_pred_method"],
                "definition": (
                    "share of scored generations whose answer was extracted from a boxed expression, "
                    "that is reward_info_json.per_role[0].pred_method == 'boxed'; the denominator is "
                    "the generations that carried a pred_method at all"
                ),
            },
            "per_question": [
                {"suite": suite, "question_id": q, "rewards": list(questions[q])} for q in order
            ],
        }
    return out


def merged_block(name: str, parts: Sequence[str], suites: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    One reported suite out of several files, merged problem by problem.

    The accuracy is the mean over the problems of all parts, which is not the
    mean of the part accuracies unless the parts have the same problem count.
    """
    present = [part for part in parts if part in suites]
    if not present:
        return None

    ks = {int(suites[part]["k"]) for part in present}
    if len(ks) != 1:
        raise ValueError(f"{name}: its parts were evaluated at different sample counts: {sorted(ks)}")
    k = ks.pop()

    per_question: List[Dict[str, Any]] = []
    for part in present:
        per_question.extend(suites[part]["per_question"])

    n_questions = len(per_question)
    scored_draws = sum(len(item["rewards"]) for item in per_question)
    accuracy = (
        sum(sum(item["rewards"]) / len(item["rewards"]) for item in per_question if item["rewards"]) / n_questions
        if n_questions
        else None
    )
    successes = sum(int(suites[part]["successes"]) for part in present)
    low, high = eval_probe.wilson_interval(successes, scored_draws)

    n_boxed = sum(int(suites[part]["boxed"]["n_boxed"]) for part in present)
    with_method = sum(int(suites[part]["boxed"]["n_with_pred_method"]) for part in present)
    without_method = sum(int(suites[part]["boxed"]["n_without_pred_method"]) for part in present)

    block = {
        "k": k,
        "merged_from": list(present),
        "n_questions": n_questions,
        "n_draws": sum(int(suites[part]["n_draws"]) for part in present),
        "n_scored_draws": scored_draws,
        "n_questions_off_k": sum(int(suites[part]["n_questions_off_k"]) for part in present),
        "n_missing_reward": sum(int(suites[part]["n_missing_reward"]) for part in present),
        "n_missing_question_id": sum(int(suites[part]["n_missing_question_id"]) for part in present),
        "n_non_binary_reward": sum(int(suites[part]["n_non_binary_reward"]) for part in present),
        "successes": successes,
        "accuracy": accuracy,
        "wilson95": {"low": low, "high": high},
        "wilson95_basis": suites[present[0]]["wilson95_basis"],
        "success_threshold": SUCCESS_THRESHOLD,
        "boxed_rate": (n_boxed / with_method) if with_method else None,
        "boxed": {
            "n_boxed": n_boxed,
            "n_with_pred_method": with_method,
            "n_without_pred_method": without_method,
            "definition": suites[present[0]]["boxed"]["definition"],
        },
        "per_question": per_question,
    }
    if len(present) != len(parts):
        block["note"] = f"merged from {present}; {[p for p in parts if p not in present]} produced no rows"
    return block


def summarize_group(group: EvalGroup) -> Dict[str, Any]:
    """One group's block of the record, including why it is empty when it is."""
    block: Dict[str, Any] = {
        "label": group.label,
        "k": group.k,
        "profile": group.profile,
        "decoding": {
            "temperature": group.temperature,
            "top_p": group.top_p,
            "top_k": group.top_k,
            "note": (
                "temperature 0 is greedy decoding, where the nucleus and top-k cut-offs do not apply"
                if group.k == 1
                else "the sampling setting of the training rollouts"
            ),
        },
        "declared_suites": list(group.suites),
        "task": group.task_path,
        "eval_jsonl": group.eval_jsonl,
        "metrics_jsonl": group.metrics_jsonl,
        "suites": {},
        "problems": [],
    }

    dump_path = Path(group.eval_jsonl)
    if not dump_path.is_file():
        block["problems"].append(f"evaluation dump not found: {group.eval_jsonl}")
        return block

    rows, bad = eval_probe.read_jsonl(dump_path)
    if bad:
        block["problems"].append(f"{bad} line(s) of the evaluation dump did not parse")
    block["n_rows"] = len(rows)
    block["suites"] = summarize_dump(rows, k=group.k)

    metrics_path = Path(group.metrics_jsonl)
    if metrics_path.is_file():
        metric_rows, _ = eval_probe.read_jsonl(metrics_path)
        logged = eval_probe.metrics_by_suite(metric_rows)
        for suite, values in block["suites"].items():
            pass1 = logged.get(suite, {}).get("pass1")
            values["accuracy_logged_by_trainer"] = pass1
            if pass1 is None or values["accuracy"] is None:
                values["accuracy_matches_trainer"] = None
            else:
                values["accuracy_matches_trainer"] = abs(pass1 - values["accuracy"]) <= 1e-6
    else:
        block["problems"].append(f"metrics file not found: {group.metrics_jsonl}")

    observed = sorted(block["suites"])
    missing = [name for name in group.suites if name not in observed]
    unexpected = [name for name in observed if name not in group.suites]
    block["observed_suites"] = observed
    block["missing_suites"] = missing
    block["unexpected_suites"] = unexpected
    if missing:
        block["problems"].append(
            f"the run covered {len(observed)} of the {len(group.suites)} suites of this group; missing {missing}. "
            "The trainer concatenates the suites into one evaluation set after aligning their columns "
            "(c3.integration.task_datasets.concatenate_datasets_aligned); a suite that still produced no "
            "rows points at the trainer log for this run."
        )
    for name, values in sorted(block["suites"].items()):
        if values["n_questions_off_k"]:
            block["problems"].append(
                f"{name}: {values['n_questions_off_k']} of {values['n_questions']} problems did not carry "
                f"{group.k} rewards, so their avg@k is over a different number of samples"
            )
    return block


def build_record(groups: Sequence[EvalGroup], args: argparse.Namespace) -> Dict[str, Any]:
    """The evaluation record: one entry per reported suite, plus what produced it."""
    task_path = Path(eval_probe.absolute_repo_path(args.task))
    record: Dict[str, Any] = {
        "schema": RECORD_SCHEMA,
        "policy": args.policy,
        "method": args.method,
        "task": task_path.as_posix(),
        "seed": int(args.seed),
        "estimator": "avg@k: per problem the mean reward of its k samples, per suite the mean over problems",
        "generate_max_len": int(args.generate_max_len),
        "prompt_max_len": int(args.prompt_max_len),
        "k_by_suite": {name: int(k) for name, k in sorted(parse_k_by_suite(args.k_by_suite).items())},
        "groups": {},
        "suites": {},
    }

    problems: List[str] = []
    suites: Dict[str, Dict[str, Any]] = {}
    for group in groups:
        block = summarize_group(group)
        record["groups"][group.label] = block
        problems.extend(f"{group.label}: {p}" for p in block.get("problems", []))
        for name, values in block["suites"].items():
            if name in suites:
                problems.append(f"{name} was produced by more than one group; the later one wins")
            entry = dict(values)
            entry["group"] = group.label
            suites[name] = entry

    for name, parts in sorted(MERGED_SUITES.items()):
        if name in suites:
            problems.append(f"{name} is both a merged entry and a suite of the task; the merge is not written")
            continue
        try:
            merged = merged_block(name, parts, suites)
        except ValueError as exc:
            problems.append(str(exc))
            continue
        if merged is not None:
            suites[name] = merged

    record["suites"] = suites
    record["problems"] = problems
    record["ok"] = not problems
    return record


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def run_one(group: EvalGroup, *, timeout_s: float) -> Optional[str]:
    """Run one group. Returns a failure reason, or None when it succeeded."""
    print(f"[final_eval] {group.label} run: {group.command_line()}", flush=True)

    env = dict(os.environ)
    env.update(group.env)

    try:
        completed = subprocess.run(
            group.argv,
            cwd=str(REPO_ROOT),
            env=env,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return f"timeout after {timeout_s:.0f}s"
    except OSError as exc:
        return f"could not start: {exc}"

    if completed.returncode != 0:
        return f"exit code {completed.returncode}"
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/70_rebuild/final_eval.py",
        description="Run the evaluation of record on one policy and write its evaluation record.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--policy", required=True, help="Model directory or hub id of the policy under evaluation.")
    parser.add_argument("--task", default=DEFAULT_TASK, help="Task file that lists the evaluation suites.")
    parser.add_argument(
        "--k_by_suite",
        default=DEFAULT_K_BY_SUITE,
        help="Samples per problem, per suite, as Name=k pairs. Every declared suite needs one.",
    )
    parser.add_argument("--method", default=DEFAULT_METHOD, help="Method label; SFT means no credit algorithm.")
    parser.add_argument("--run_id", default=DEFAULT_RUN_ID, help="Run id the evaluation artifacts are filed under.")
    parser.add_argument("--out_subdir", default=DEFAULT_OUT_SUBDIR, help="Artifact subdirectory of the run.")
    parser.add_argument("--ckpt_root", default=DEFAULT_CKPT_ROOT, help="Root that holds ckpt/_runs.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Evaluation seed.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature of the groups with more than one sample per problem.",
    )
    parser.add_argument("--top_p", type=float, default=DEFAULT_TOP_P, help="Nucleus cut-off of the sampling groups.")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K, help="Top-k cut-off of the sampling groups.")
    parser.add_argument(
        "--generate_max_len",
        type=int,
        default=DEFAULT_GENERATE_MAX_LEN,
        help="Generation cap passed to the evaluation script.",
    )
    parser.add_argument(
        "--prompt_max_len",
        type=int,
        default=DEFAULT_PROMPT_MAX_LEN,
        help="Prompt cap passed to the evaluation script.",
    )
    parser.add_argument("--out", default="final_eval_record.json", help="Where the evaluation record is written.")
    parser.add_argument("--only", default="", help="Run one group only, by its label (k1, k4, k16).")
    parser.add_argument("--bash", default="bash", help="Shell used to invoke the evaluation script.")
    parser.add_argument("--timeout_h", type=float, default=DEFAULT_TIMEOUT_H, help="Per group wall clock limit in hours.")
    parser.add_argument("--dry-run", "--dry_run", dest="dry_run", action="store_true", help="Print commands only.")
    parser.add_argument(
        "--summarize_only",
        action="store_true",
        help="Skip the evaluations and summarize artifacts that are already on disk.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.generate_max_len <= 0:
        print("[final_eval] ERROR: --generate_max_len must be positive", file=sys.stderr)
        return 2
    if args.prompt_max_len <= 0:
        print("[final_eval] ERROR: --prompt_max_len must be positive", file=sys.stderr)
        return 2
    if args.timeout_h <= 0:
        print("[final_eval] ERROR: --timeout_h must be positive", file=sys.stderr)
        return 2

    task_path = Path(eval_probe.absolute_repo_path(args.task))
    if not task_path.is_file():
        print(f"[final_eval] ERROR: task file not found: {task_path}", file=sys.stderr)
        return 2

    try:
        groups = plan_groups(args)
    except ValueError as exc:
        print(f"[final_eval] ERROR: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        for group in groups:
            print(group.command_line())
        print(f"# groups: {len(groups)}")
        for group in groups:
            print(
                f"# {group.label}: k={group.k} t={group.temperature} profile={group.profile} "
                f"suites={','.join(group.suites)} -> {group.eval_jsonl}"
            )
            print(f"# {group.label}: task file to generate: {group.task_path}")
        return 0

    failures: List[Tuple[str, str]] = []
    if not args.summarize_only:
        for group in groups:
            try:
                written = write_group_task(group, main_task=task_path)
            except (OSError, ValueError) as exc:
                failures.append((group.label, f"could not write the group task file: {exc}"))
                print(f"[final_eval] {group.label} FAILED: {exc}", file=sys.stderr)
                continue
            print(f"[final_eval] {group.label} task file: {written}", flush=True)

            reason = run_one(group, timeout_s=args.timeout_h * 3600.0)
            if reason:
                failures.append((group.label, reason))
                print(f"[final_eval] {group.label} FAILED: {reason}", file=sys.stderr)

    record = build_record(groups, args)
    record["failed_groups"] = [{"label": label, "reason": reason} for label, reason in failures]
    if failures:
        record["ok"] = False

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"[final_eval] wrote {out_path}")

    for problem in record.get("problems", []):
        print(f"[final_eval] PROBLEM {problem}", file=sys.stderr)

    return 0 if record.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
