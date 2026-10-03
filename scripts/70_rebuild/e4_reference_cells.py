#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Emit, and optionally run, the reference cells of the E4 critic comparison.

The critic comparison (paper arXiv:2603.06859v3, Section 4 and Appendix B.7)
scores a learned decision-value critic and C3's own low-budget measurement on
held-out decision points against one independent reference: sixteen fresh
continuations of every held-out alternative. This driver produces that
reference. It replays the alternatives the evaluation cell recorded
(`build-buckets --reuse_candidates_from`), samples nothing for the measured
role, and draws only the downstream continuations, under a replay seed the cell
did not use.

Why the seed must differ. The batched replay path seeds continuation r of
alternative j from (run seed, question id, j, r, role), not from the
continuation count, so sixteen continuations under the cell's own seed would
repeat the cell's four as their first four. A seed equal to the cell's is
refused here, and `e4_critic.py evaluate` refuses such a reference again.

Input: the directory `scripts/70_rebuild/e4_critic.py prepare-reference` wrote:

    e4_heldout_task.yaml    task file whose only evaluation suite is the file below
    e4_heldout_pool.jsonl   one row per held-out question
    e4_heldout_reuse.jsonl  the held-out buckets of the evaluation cell, byte for byte
    prepare_report.json     counts, hashes, the cell's seed and the build-buckets arguments

Output, one cell per replay seed:

    <results_root>/E4/reference_heldout/<workflow>/<model>/n<n>_c<c>_seed<seed>/buckets.jsonl

The defaults are the paper's: 16 continuations, replay seed 3003 for the
reference and 4004 for the second reference, Qwen3-4B-Instruct-2507.

`--all-points --source <the cell's buckets.jsonl>` replays every decision point
of the evaluation cell instead of the held-out ones, on the cell's own task file
and suite, and writes under `reference_all/`; `evaluate` accepts either.

Examples:

    # Print the two commands.
    python scripts/70_rebuild/e4_reference_cells.py --prep <prep dir> --model_root /models --dry-run

    # Run the reference only.
    python scripts/70_rebuild/e4_reference_cells.py --prep <prep dir> --model_root /models \\
        --results_root /abs/path/20_data/results --seeds 3003
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from e1_cells import (  # noqa: E402
    ANALYSIS_MODULE,
    POLICY_TAG,
    TASK_YAML_BY_WORKFLOW,
    _already_done,
    parse_int_csv,
    posix_join,
)

PREPARE_REPORT_FILE = "prepare_report.json"
DEFAULT_SEEDS = "3003,4004"
DEFAULT_REPLAYS = 16
DEFAULT_WORKFLOW = "a2"
DEFAULT_MODEL = "4b=Qwen3-4B-Instruct-2507"
DEFAULT_RESULTS_ROOT = "20_data/results"
DEFAULT_ANALYSIS_YAML = "configs/analysis.yaml"
DEFAULT_TIMEOUT_H = 12.0

#: The suite of the evaluation cell, for --all-points.
ALL_POINTS_SPLIT = "MATHPOOL"


class RefusedError(ValueError):
    """The inputs cannot support the requested reference."""


@dataclass(frozen=True)
class ReferenceCell:
    seed: int
    out_path: str
    expected: int
    argv: List[str]

    def command_line(self) -> str:
        return " ".join(shlex.quote(a) for a in self.argv)


def read_report(prep: str) -> Dict[str, Any]:
    path = os.path.join(prep, PREPARE_REPORT_FILE)
    if not os.path.isfile(path):
        raise RefusedError(f"no {path} (run e4_critic.py prepare-reference first)")
    with open(path, "r", encoding="utf-8") as fh:
        report = json.load(fh)
    bb = report.get("build_buckets")
    if not isinstance(bb, Mapping):
        raise RefusedError(f"{path} carries no build_buckets block")
    for key in ("--task", "--split", "--limit", "--num_candidates", "--target_role", "--next_role",
                "--reuse_candidates_from", "--decoding_json"):
        if key not in bb:
            raise RefusedError(f"{path}: build_buckets has no {key}")
    seed = report.get("source_seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise RefusedError(f"{path}: source_seed {seed!r} is not an integer")
    return report


def _count_lines(path: str) -> int:
    with open(path, "r", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def build_cells(args: argparse.Namespace) -> List[ReferenceCell]:
    prep = os.path.abspath(args.prep)
    report = read_report(prep)
    bb = report["build_buckets"]
    source_seed = int(report["source_seed"])
    n_alt = int(bb["--num_candidates"])
    decoding = bb["--decoding_json"]
    if not isinstance(decoding, Mapping) or not all(k in decoding for k in ("temperature", "top_p", "top_k")):
        raise RefusedError("the prepare report's decoding carries no temperature, top_p and top_k")

    name, _, directory = str(args.model).partition("=")
    if not name or not directory:
        raise RefusedError(f"--model must look like name=directory, got {args.model!r}")
    model_path = posix_join(args.model_root, directory)

    if args.all_points:
        if not args.source:
            raise RefusedError("--all-points needs --source, the evaluation cell's buckets.jsonl")
        if not os.path.isfile(args.source):
            raise RefusedError(f"no {args.source}")
        task = TASK_YAML_BY_WORKFLOW[args.workflow]
        split = ALL_POINTS_SPLIT
        reuse = args.source
        limit = _count_lines(args.source)
        scope = "all"
    else:
        task = posix_join(prep, str(bb["--task"]))
        split = str(bb["--split"])
        reuse = posix_join(prep, str(bb["--reuse_candidates_from"]))
        limit = int(bb["--limit"])
        scope = "heldout"
    split_sha = (report.get("split") or {}).get("sha256")

    cells: List[ReferenceCell] = []
    for seed in parse_int_csv(args.seeds):
        if seed == source_seed:
            raise RefusedError(f"seed {seed} is the evaluation cell's own seed: the reference "
                               f"would repeat its continuations")
        out_path = posix_join(args.results_root, "E4", f"reference_{scope}", args.workflow, name,
                              f"n{n_alt}_c{args.replays}_seed{seed}", "buckets.jsonl")
        meta = {
            "workflow": args.workflow,
            "model": name,
            "rule": f"e4_reference_{scope}_n{n_alt}_c{args.replays}",
            "n_alternatives": n_alt,
            "replays_per_alt": int(args.replays),
            "temperature": decoding["temperature"],
            "top_p": decoding["top_p"],
            "top_k": decoding["top_k"],
            "seed": int(seed),
            "source_seed": source_seed,
            "policy_tag": POLICY_TAG,
            "dataset": split,
            "question_order_seed": 0,
            "include_real_as_j0": False,
            "context_scope": "ancestors",
            "limit": int(limit),
            "e4_split_sha256": split_sha,
            "purpose": "E4 high-budget Monte Carlo reference",
        }
        argv = [
            args.python, "-m", ANALYSIS_MODULE, "build-buckets",
            "--task", task,
            "--split", split,
            "--policy_ckpt", model_path,
            "--method", POLICY_TAG,
            "--target_role", str(bb["--target_role"]),
            "--next_role", str(bb["--next_role"]),
            "--limit", str(int(limit)),
            "--num_candidates", str(n_alt),
            "--num_completions", str(int(args.replays)),
            "--seed", str(int(seed)),
            "--include_real_as_j0", "false",
            "--num_extra_v_samples", "0",
            "--analysis_yaml", args.analysis_yaml,
            "--defaults_section", "influence",
            "--decoding_json", json.dumps(decoding, sort_keys=True, separators=(",", ":")),
            "--engine", args.engine,
            "--batched",
            "--reuse_candidates_from", reuse,
            "--out", out_path,
            "--overwrite",
            "--meta_json", json.dumps(meta, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        ]
        cells.append(ReferenceCell(seed=int(seed), out_path=out_path, expected=int(limit), argv=argv))
    if not cells:
        raise RefusedError("--seeds names no seed")
    return cells


def run_cells(cells: Sequence[ReferenceCell], *, timeout_s: float, overwrite: bool) -> List[Tuple[str, str]]:
    failures: List[Tuple[str, str]] = []
    for index, cell in enumerate(cells, start=1):
        head = f"[e4_reference_cells] ({index}/{len(cells)}) seed {cell.seed}"
        if not overwrite and _already_done(cell.out_path):
            print(f"{head} skip, buckets already present: {cell.out_path}", flush=True)
            continue
        Path(cell.out_path).parent.mkdir(parents=True, exist_ok=True)
        print(f"{head} run: {cell.command_line()}", flush=True)
        try:
            completed = subprocess.run(cell.argv, cwd=str(REPO_ROOT), timeout=timeout_s, check=False)
        except subprocess.TimeoutExpired:
            failures.append((str(cell.seed), f"timeout after {timeout_s:.0f}s"))
            continue
        except OSError as exc:
            failures.append((str(cell.seed), f"could not start: {exc}"))
            continue
        if completed.returncode != 0:
            failures.append((str(cell.seed), f"exit code {completed.returncode}"))
            continue
        got = _count_lines(cell.out_path) if os.path.isfile(cell.out_path) else 0
        if got != cell.expected:
            failures.append((str(cell.seed), f"{got} of {cell.expected} buckets"))
            print(f"{head} FAILED: {got} of {cell.expected} buckets", flush=True)
        else:
            print(f"{head} done: {cell.out_path}", flush=True)
    return failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/70_rebuild/e4_reference_cells.py",
        description="Emit or run the reference cells of the E4 critic comparison.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--prep", required=True, help="Directory e4_critic.py prepare-reference wrote.")
    parser.add_argument("--model_root", required=True, help="Directory that holds the base model directories.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="One name=directory pair.")
    parser.add_argument("--workflow", default=DEFAULT_WORKFLOW, choices=sorted(TASK_YAML_BY_WORKFLOW),
                        help="Workflow of the evaluation cell.")
    parser.add_argument("--results_root", default=DEFAULT_RESULTS_ROOT, help="Root of the results tree.")
    parser.add_argument("--seeds", default=DEFAULT_SEEDS, help="Comma separated replay seeds, one cell each.")
    parser.add_argument("--replays", type=int, default=DEFAULT_REPLAYS, help="Continuations per alternative.")
    parser.add_argument("--all-points", "--all_points", dest="all_points", action="store_true",
                        help="Replay every decision point of the evaluation cell.")
    parser.add_argument("--source", default="", help="With --all-points: the evaluation cell's buckets.jsonl.")
    parser.add_argument("--analysis_yaml", default=DEFAULT_ANALYSIS_YAML, help="Analysis defaults file.")
    parser.add_argument("--engine", default="vllm", choices=["auto", "hf", "vllm"], help="Policy engine.")
    parser.add_argument("--python", default="python", help="Python executable used in the emitted commands.")
    parser.add_argument("--timeout_h", type=float, default=DEFAULT_TIMEOUT_H, help="Per cell wall clock limit in hours.")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild cells whose bucket file already exists.")
    parser.add_argument("--dry-run", "--dry_run", dest="dry_run", action="store_true", help="Print commands only.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.replays <= 0:
        print("[e4_reference_cells] ERROR: --replays must be positive", file=sys.stderr)
        return 2
    try:
        cells = build_cells(args)
    except (ValueError, OSError) as exc:
        print(f"[e4_reference_cells] ERROR: {exc}", file=sys.stderr)
        return 2
    if args.dry_run:
        for cell in cells:
            print(cell.command_line())
        print(f"# cells: {len(cells)}")
        return 0
    failures = run_cells(cells, timeout_s=args.timeout_h * 3600.0, overwrite=args.overwrite)
    print(f"# cells: {len(cells)}")
    if failures:
        print(f"# failed: {len(failures)}")
        for name, reason in failures:
            print(f"#   seed {name}: {reason}")
        return 1
    print("# failed: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
