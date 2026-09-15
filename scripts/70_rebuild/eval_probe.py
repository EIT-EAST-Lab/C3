#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Measure the start accuracy of a frozen policy on the candidate benchmarks.

A benchmark is only worth running if the policy we start from can neither solve
nearly all of it nor almost none of it. This script measures exactly that, on
one frozen model, with no training and no new evaluator: it drives the existing
evaluation path, `scripts/50_eval/paper_main_results.sh one` in its direct mode,
which runs `openrlhf.cli.train_ppo_ray --eval_only` with the learning rates at
zero, and then reads the artifacts that run leaves behind.

Two decodings, the same suites in both:

    greedy   one sample per problem at temperature 0
    sample   four samples per problem at temperature 0.7

The sample count and the temperature are always written into the command, so
the profile name only picks the directory the artifacts land in. `greedy` and
`n10` are the two names that script accepts, and the `n10` directory of the
sampling run therefore holds four samples per problem, not ten. The summary
records the real numbers.

What comes out, per suite, in `probe_summary.json`:

    n_questions       problems the run actually covered
    n_draws           generations scored, that is problems times samples
    accuracy          mean over problems of the mean reward of their samples,
                      the same quantity the trainer logs as pass1
    wilson95          95 percent Wilson interval over the draws
    length            mean generation length and truncation rate, when the
                      evaluation dump carries token counts at all

Examples:

    # Print the two commands and stop.
    python scripts/70_rebuild/eval_probe.py --policy /models/Qwen3-4B-Instruct-2507 --dry-run

    # Run both decodings and write the summary.
    python scripts/70_rebuild/eval_probe.py --policy /models/Qwen3-4B-Instruct-2507 \\
        --out /abs/path/probe_summary.json

    # Re-derive the summary from artifacts that are already on disk.
    python scripts/70_rebuild/eval_probe.py --policy /models/Qwen3-4B-Instruct-2507 \\
        --summarize_only --out /abs/path/probe_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

EVAL_SCRIPT = "scripts/50_eval/paper_main_results.sh"

DEFAULT_TASK = "configs/tasks/math_eval_probe.yaml"
DEFAULT_RUN_ID = "eval_probe"
DEFAULT_OUT_SUBDIR = "eval_probe"
DEFAULT_CKPT_ROOT = "ckpt"

# SFT maps to marl_algorithm=none in the evaluation script, which is what a
# frozen policy with no credit assignment means there.
DEFAULT_METHOD = "SFT"

DEFAULT_SAMPLE_N = 4
DEFAULT_SAMPLE_TEMPERATURE = 0.7
DEFAULT_GENERATE_MAX_LEN = 512
DEFAULT_PROMPT_MAX_LEN = 2560
DEFAULT_SEED = 0
DEFAULT_TIMEOUT_H = 8.0

# A reward at or above this counts as a solved draw. The math environment scores
# vote_binary, so rewards are 0 or 1 and the threshold never bites; the summary
# reports how many rewards were neither, so a change in reward mode is visible
# instead of silently thresholded.
SUCCESS_THRESHOLD = 0.5

# Two-sided 95 percent normal quantile.
Z95 = 1.959963984540054

_METRIC_KEY = re.compile(r"^eval_(?P<name>.+)_pass(?P<k>\d+)$")


# ---------------------------------------------------------------------------
# Run description
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbeRun:
    """One decoding of the probe: how it is invoked and where it writes."""

    label: str
    profile: str
    n_samples_per_prompt: int
    temperature: float
    eval_dir: str
    argv: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)

    @property
    def eval_jsonl(self) -> str:
        return posix_join(self.eval_dir, "eval_only.jsonl")

    @property
    def metrics_jsonl(self) -> str:
        return posix_join(self.eval_dir, "eval_only.jsonl.metrics.jsonl")

    def command_line(self) -> str:
        prefix = "".join(f"{k}={shlex.quote(v)} " for k, v in sorted(self.env.items()))
        return prefix + " ".join(shlex.quote(a) for a in self.argv)


def posix_join(*parts: str) -> str:
    """
    Join path parts with forward slashes, keeping the first part as it came.

    The leading slash of an absolute posix root has to survive, which is why
    this is not a plain join on stripped parts: the platform runs on Linux, and
    an artifact root of `/data/ckpt` that came back as `data/ckpt` would send
    every path in the summary somewhere relative to the working directory.
    """
    cleaned = [str(p) for p in parts if str(p) != ""]
    if not cleaned:
        return ""
    joined = PurePosixPath(cleaned[0])
    for part in cleaned[1:]:
        trimmed = part.strip("/")
        if trimmed:
            joined = joined / trimmed
    return joined.as_posix()


def is_rooted(value: str) -> bool:
    """
    Whether a path already has a root.

    The evaluation script decides this with a leading slash, and the platform is
    Linux, so that is the rule that matters. `Path.is_absolute()` would answer it
    with the rules of whichever machine is running, which says no to `/data/ckpt`
    on Windows; the Windows drive form is accepted as well so a local dry run
    prints paths a person can read.
    """
    text = str(value)
    return text.startswith("/") or PureWindowsPath(text).is_absolute()


def absolute_repo_path(value: str) -> str:
    """A repository-relative path as an absolute one, in posix form."""
    if is_rooted(value):
        return Path(value).as_posix()
    return (REPO_ROOT / value).as_posix()


# ---------------------------------------------------------------------------
# Command building
# ---------------------------------------------------------------------------


def task_tag(task_path: str) -> str:
    """The directory name the evaluation script derives from an absolute task path."""
    return Path(task_path).name[: -len(".yaml")] if task_path.endswith(".yaml") else Path(task_path).name


def eval_dir_for(
    *,
    ckpt_root: str,
    run_id: str,
    out_subdir: str,
    task_path: str,
    profile: str,
) -> str:
    """
    Where `paper_main_results.sh one --source_type hf_base` writes its artifacts.

    Mirrors _sft_run_root and the eval_dir line of that script. The task path has
    to be absolute for the tag to be a bare name rather than a nested path, which
    is why build_runs resolves it.
    """
    root = ckpt_root if is_rooted(ckpt_root) else posix_join(REPO_ROOT.as_posix(), ckpt_root)
    run_root = posix_join(root, "_runs", "_sft_main_results", run_id)
    return posix_join(run_root, out_subdir, task_tag(task_path).lower(), profile)


def build_runs(args: argparse.Namespace) -> List[ProbeRun]:
    """The two decodings of the probe, in the order they are run."""
    task_path = absolute_repo_path(args.task)
    script = absolute_repo_path(EVAL_SCRIPT)

    env = {
        "GENERATE_MAX_LEN": str(int(args.generate_max_len)),
        "PROMPT_MAX_LEN": str(int(args.prompt_max_len)),
    }

    plan: List[Tuple[str, str, int, float]] = [
        ("greedy", "greedy", 1, 0.0),
        ("sample", "n10", int(args.sample_n), float(args.sample_temperature)),
    ]

    runs: List[ProbeRun] = []
    for label, profile, n_samples, temperature in plan:
        if args.only and args.only != label:
            continue

        eval_dir = eval_dir_for(
            ckpt_root=args.ckpt_root,
            run_id=args.run_id,
            out_subdir=args.out_subdir,
            task_path=task_path,
            profile=profile,
        )

        # The profile already fixes a sample count and a temperature. Both are
        # repeated here so the command states what it really does, and because a
        # later flag of the same name wins in the argparse parser on the far end.
        extra = f"--eval_n_samples_per_prompt {int(n_samples)} --eval_temperature {float(temperature)}"
        # Platform specific trainer flags (for example --attn_implementation sdpa where flash-attn
        # is not installed, or the GPU counts of a one GPU run) are appended from the environment,
        # so the probe itself stays free of machine details.
        extra_env = os.environ.get("EVAL_PROBE_EXTRA_ARGS", "").strip()
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
            task_path,
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

        runs.append(
            ProbeRun(
                label=label,
                profile=profile,
                n_samples_per_prompt=int(n_samples),
                temperature=float(temperature),
                eval_dir=eval_dir,
                argv=argv,
                env=dict(env),
            )
        )

    if not runs:
        raise ValueError(f"--only {args.only!r} selected no run")
    return runs


# ---------------------------------------------------------------------------
# Reading what the evaluation left behind
# ---------------------------------------------------------------------------


def read_jsonl(path: Path) -> Tuple[List[Dict[str, Any]], int]:
    """Every object of a JSONL file, plus the number of lines that did not parse."""
    rows: List[Dict[str, Any]] = []
    bad = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                bad += 1
                continue
            if isinstance(obj, dict):
                rows.append(obj)
            else:
                bad += 1
    return rows, bad


def _scalar_number(value: Any) -> Optional[float]:
    """A number that may have been dumped as a one-element list or a string."""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (list, tuple)):
        return _scalar_number(value[0]) if len(value) == 1 else None
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def row_reward(row: Dict[str, Any]) -> Optional[float]:
    """
    The reward of one generation.

    The evaluation dump has two shapes. A multi-agent run writes one row per
    (problem, sample) with the answering role's reward under `answer_reward`; a
    single-agent run writes `reward`. Both are read, neither is assumed.
    """
    for key in ("answer_reward", "reward"):
        if key in row:
            value = _scalar_number(row.get(key))
            if value is not None:
                return value
    return None


def row_length(row: Dict[str, Any]) -> Tuple[Optional[float], Optional[bool]]:
    """Response length in tokens and the truncation flag, when the row carries them."""
    info = row.get("info")
    if not isinstance(info, dict):
        return None, None
    length = _scalar_number(info.get("response_length"))
    clipped = _scalar_number(info.get("response_clip_ratio"))
    return length, (None if clipped is None else bool(clipped))


def wilson_interval(successes: float, trials: int, z: float = Z95) -> Tuple[Optional[float], Optional[float]]:
    """
    Wilson score interval for a binomial proportion, clipped to [0, 1].

    Chosen over the normal approximation because the probe reports proportions
    near 0 and near 1, where the normal interval runs outside the unit interval
    and is badly calibrated at these sample sizes.
    """
    n = int(trials)
    if n <= 0:
        return None, None
    p = float(successes) / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + (z * z) / (4 * n * n))
    return max(0.0, center - half), min(1.0, center + half)


def metrics_by_suite(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Per-suite pass rates as the trainer itself logged them.

    The trainer writes one `eval_metrics` record per evaluation, whose metrics
    map holds `eval_<suite>_pass1` and `eval_<suite>_pass<K>`. The last record
    wins, which for an eval-only run is the only one.
    """
    out: Dict[str, Dict[str, float]] = {}
    for row in rows:
        if str(row.get("kind", "")) != "eval_metrics":
            continue
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            continue
        parsed: Dict[str, Dict[str, float]] = {}
        for key, value in metrics.items():
            match = _METRIC_KEY.match(str(key))
            number = _scalar_number(value)
            if match is None or number is None:
                continue
            name = match.group("name")
            k = int(match.group("k"))
            entry = parsed.setdefault(name, {})
            entry["pass1" if k == 1 else f"pass{k}"] = number
        if parsed:
            out = parsed
    return out


def summarize_dump(
    rows: Sequence[Dict[str, Any]],
    *,
    generate_max_len: int,
    success_threshold: float = SUCCESS_THRESHOLD,
) -> Dict[str, Dict[str, Any]]:
    """Per-suite counts, accuracy, interval and length statistics of one dump."""
    per_suite: Dict[str, Dict[str, Any]] = {}
    by_question: Dict[str, Dict[Any, List[float]]] = {}

    for index, row in enumerate(rows):
        suite = row.get("datasource")
        if suite is None:
            suite = "unknown"
        suite = str(suite)
        bucket = per_suite.setdefault(
            suite,
            {
                "n_draws": 0,
                "n_missing_reward": 0,
                "n_missing_question_id": 0,
                "n_non_binary_reward": 0,
                "successes": 0,
                "_lengths": [],
                "_clipped": [],
            },
        )
        questions = by_question.setdefault(suite, {})

        bucket["n_draws"] += 1

        reward = row_reward(row)
        if reward is None:
            bucket["n_missing_reward"] += 1
        else:
            if reward not in (0.0, 1.0):
                bucket["n_non_binary_reward"] += 1
            if reward >= success_threshold:
                bucket["successes"] += 1
            # A row without a question id is kept as a problem of its own rather
            # than folded into one bucket, so a dump that lost the id understates
            # nothing and shows up as n_missing_question_id instead.
            question_id = row.get("question_id")
            if question_id is None:
                bucket["n_missing_question_id"] += 1
                question_id = ("row", index)
            questions.setdefault(question_id, []).append(reward)

        length, clipped = row_length(row)
        if length is not None:
            bucket["_lengths"].append(length)
            bucket["_clipped"].append(bool(clipped) if clipped is not None else length >= generate_max_len)

    out: Dict[str, Dict[str, Any]] = {}
    for suite, bucket in sorted(per_suite.items()):
        questions = by_question.get(suite, {})
        n_questions = len(questions)
        scored_draws = sum(len(v) for v in questions.values())

        if n_questions:
            accuracy = sum(sum(v) / len(v) for v in questions.values()) / n_questions
        else:
            accuracy = None

        low, high = wilson_interval(bucket["successes"], scored_draws)

        lengths = bucket.pop("_lengths")
        clipped = bucket.pop("_clipped")
        if lengths:
            length_block: Dict[str, Any] = {
                "source": "eval dump info.response_length",
                "n": len(lengths),
                "mean_tokens": sum(lengths) / len(lengths),
                "max_tokens": max(lengths),
                "truncation_rate": (sum(1 for c in clipped if c) / len(clipped)) if clipped else None,
                "generate_max_len": int(generate_max_len),
            }
        else:
            length_block = {
                "source": "unavailable",
                "n": 0,
                "mean_tokens": None,
                "max_tokens": None,
                "truncation_rate": None,
                "generate_max_len": int(generate_max_len),
                "note": (
                    "No row carried info.response_length. The multi-agent branch of the evaluation "
                    "dump writes the per-role texts and the answer reward, not the token counts the "
                    "generator recorded, so generation length and truncation cannot be derived from "
                    "this file. They would come from the same info block the single-agent branch "
                    "already dumps."
                ),
            }

        out[suite] = {
            "n_questions": n_questions,
            "n_draws": bucket["n_draws"],
            "n_scored_draws": scored_draws,
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
            "success_threshold": success_threshold,
            "length": length_block,
        }
    return out


def expected_suites(task_path: Path) -> List[str]:
    """The suite names the task file declares, in file order."""
    with task_path.open("r", encoding="utf-8") as handle:
        spec = yaml.safe_load(handle) or {}
    env = spec.get("environment") if isinstance(spec, dict) else None
    suites = (env or {}).get("eval_suites") if isinstance(env, dict) else None
    if isinstance(suites, dict):
        # The loader accepts a mapping of name to entry as well as a list.
        return [str(key) for key in suites]
    names: List[str] = []
    for index, entry in enumerate(suites or []):
        if isinstance(entry, dict) and entry.get("name"):
            names.append(str(entry["name"]))
        elif isinstance(entry, str):
            names.append(Path(entry).name.replace(".", "_"))
        else:
            names.append(f"eval_{index}")
    return names


def summarize_run(run: ProbeRun, *, task_path: Path, generate_max_len: int) -> Dict[str, Any]:
    """One decoding's block of the summary, including why it is empty when it is."""
    block: Dict[str, Any] = {
        "label": run.label,
        "profile": run.profile,
        "n_samples_per_prompt": run.n_samples_per_prompt,
        "temperature": run.temperature,
        "eval_jsonl": run.eval_jsonl,
        "metrics_jsonl": run.metrics_jsonl,
        "suites": {},
        "problems": [],
    }

    dump_path = Path(run.eval_jsonl)
    if not dump_path.is_file():
        block["problems"].append(f"evaluation dump not found: {run.eval_jsonl}")
        return block

    rows, bad = read_jsonl(dump_path)
    if bad:
        block["problems"].append(f"{bad} line(s) of the evaluation dump did not parse")
    block["n_rows"] = len(rows)
    block["suites"] = summarize_dump(rows, generate_max_len=generate_max_len)

    metrics_path = Path(run.metrics_jsonl)
    if metrics_path.is_file():
        metric_rows, _ = read_jsonl(metrics_path)
        logged = metrics_by_suite(metric_rows)
        for suite, values in block["suites"].items():
            pass1 = logged.get(suite, {}).get("pass1")
            values["accuracy_logged_by_trainer"] = pass1
            if pass1 is None or values["accuracy"] is None:
                values["accuracy_matches_trainer"] = None
            else:
                values["accuracy_matches_trainer"] = abs(pass1 - values["accuracy"]) <= 1e-6
    else:
        block["problems"].append(f"metrics file not found: {run.metrics_jsonl}")

    declared = expected_suites(task_path)
    observed = sorted(block["suites"].keys())
    missing = [name for name in declared if name not in observed]
    unexpected = [name for name in observed if name not in declared]
    block["expected_suites"] = declared
    block["observed_suites"] = observed
    block["missing_suites"] = missing
    block["unexpected_suites"] = unexpected
    if missing:
        block["problems"].append(
            "the run covered "
            f"{len(observed)} of the {len(declared)} declared suites; missing {missing}. "
            "The trainer concatenates the suites into one evaluation set and falls back to the "
            "first suite alone when that concatenation raises, so a suite whose prepared columns "
            "differ from the others disappears from the run without an error."
        )
    return block


def build_summary(runs: Sequence[ProbeRun], args: argparse.Namespace) -> Dict[str, Any]:
    task_path = Path(absolute_repo_path(args.task))
    summary: Dict[str, Any] = {
        "schema": "c3.eval_probe.summary/1",
        "policy": args.policy,
        "method": args.method,
        "task": task_path.as_posix(),
        "seed": int(args.seed),
        "generate_max_len": int(args.generate_max_len),
        "prompt_max_len": int(args.prompt_max_len),
        "runs": {},
    }
    problems: List[str] = []
    for run in runs:
        block = summarize_run(run, task_path=task_path, generate_max_len=int(args.generate_max_len))
        summary["runs"][run.label] = block
        problems.extend(f"{run.label}: {p}" for p in block.get("problems", []))
    summary["problems"] = problems
    summary["ok"] = not problems
    return summary


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def run_one(run: ProbeRun, *, timeout_s: float) -> Optional[str]:
    """Run one decoding. Returns a failure reason, or None when it succeeded."""
    print(f"[eval_probe] {run.label} run: {run.command_line()}", flush=True)

    env = dict(os.environ)
    env.update(run.env)

    try:
        completed = subprocess.run(
            run.argv,
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
        prog="python scripts/70_rebuild/eval_probe.py",
        description="Measure start accuracy of a frozen policy on the candidate benchmarks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--policy", required=True, help="Model directory or hub id of the frozen policy.")
    parser.add_argument("--task", default=DEFAULT_TASK, help="Task file that lists the evaluation suites.")
    parser.add_argument("--method", default=DEFAULT_METHOD, help="Method label; SFT means no credit algorithm.")
    parser.add_argument("--run_id", default=DEFAULT_RUN_ID, help="Run id the evaluation artifacts are filed under.")
    parser.add_argument("--out_subdir", default=DEFAULT_OUT_SUBDIR, help="Artifact subdirectory of the run.")
    parser.add_argument("--ckpt_root", default=DEFAULT_CKPT_ROOT, help="Root that holds ckpt/_runs.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Evaluation seed.")
    parser.add_argument("--sample_n", type=int, default=DEFAULT_SAMPLE_N, help="Samples per problem of the sampling run.")
    parser.add_argument(
        "--sample_temperature",
        type=float,
        default=DEFAULT_SAMPLE_TEMPERATURE,
        help="Temperature of the sampling run.",
    )
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
    parser.add_argument("--out", default="probe_summary.json", help="Where the summary is written.")
    parser.add_argument("--only", default="", choices=["", "greedy", "sample"], help="Run one decoding only.")
    parser.add_argument("--bash", default="bash", help="Shell used to invoke the evaluation script.")
    parser.add_argument("--timeout_h", type=float, default=DEFAULT_TIMEOUT_H, help="Per run wall clock limit in hours.")
    parser.add_argument("--dry-run", "--dry_run", dest="dry_run", action="store_true", help="Print commands only.")
    parser.add_argument(
        "--summarize_only",
        action="store_true",
        help="Skip the evaluations and summarize artifacts that are already on disk.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.sample_n <= 0:
        print("[eval_probe] ERROR: --sample_n must be positive", file=sys.stderr)
        return 2
    if args.generate_max_len <= 0:
        print("[eval_probe] ERROR: --generate_max_len must be positive", file=sys.stderr)
        return 2
    if args.timeout_h <= 0:
        print("[eval_probe] ERROR: --timeout_h must be positive", file=sys.stderr)
        return 2

    task_path = Path(absolute_repo_path(args.task))
    if not task_path.is_file():
        print(f"[eval_probe] ERROR: task file not found: {task_path}", file=sys.stderr)
        return 2

    try:
        runs = build_runs(args)
    except ValueError as exc:
        print(f"[eval_probe] ERROR: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        for run in runs:
            print(run.command_line())
        print(f"# runs: {len(runs)}")
        for run in runs:
            print(f"# {run.label}: n={run.n_samples_per_prompt} t={run.temperature} -> {run.eval_jsonl}")
        return 0

    failures: List[Tuple[str, str]] = []
    if not args.summarize_only:
        for run in runs:
            reason = run_one(run, timeout_s=args.timeout_h * 3600.0)
            if reason:
                failures.append((run.label, reason))
                print(f"[eval_probe] {run.label} FAILED: {reason}", file=sys.stderr)

    summary = build_summary(runs, args)
    summary["failed_runs"] = [{"label": label, "reason": reason} for label, reason in failures]
    if failures:
        summary["ok"] = False

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"[eval_probe] wrote {out_path}")

    for problem in summary.get("problems", []):
        print(f"[eval_probe] PROBLEM {problem}", file=sys.stderr)

    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
