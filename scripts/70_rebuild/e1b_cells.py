#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Emit, and optionally run, the bucket-generation cells of the E1b noise-law grid.

E1b measures how the noise of a credit vector scales with the branching factor
n and the continuations per alternative c, at one decision point. In the paper
(arXiv:2603.06859v3) it is the upper block of Table 3: the measured noise over
the noise the law predicts, on a 3 x 3 grid of n and c.

A grid point is measured twice on the same alternatives. Seed A draws n
alternatives at each decision point and c continuations of each; seed B reuses
the alternatives seed A recorded (`build-buckets --reuse_candidates_from`) and
draws only the continuations again, under its own seed. The difference between
the two advantage vectors is therefore continuation noise and nothing else, and
`c3.analysis.rebuild.aggregate_e1b` pairs the two seeds by question id. A seed B
run without the reuse would measure the difference between two draws of
alternatives instead, which is not the quantity of the table.

Layout of the products, which is the layout `aggregate_e1b` reads:

    <results_root>/<tree>/<workflow>/<model>/n<n>_c<c>/seedA/buckets.jsonl
    <results_root>/<tree>/<workflow>/<model>/n<n>_c<c>/seedB/buckets.jsonl

Defaults are the grid of the paper: the 2-agent workflow at its Reasoner,
Qwen3-4B-Instruct-2507, n and c in {2, 4, 8}, 500 decision points per cell from
the screened question pool `MATHPOOL`, continuation seeds 1001 (A) and 2002 (B).
Decoding, the measured position and the task file are the ones the E1 cells use
for the same workflow (`e1_cells.py`), so a grid cell sits at the decision point
of the sweep cell it is compared against.

Examples:

    # Print the 18 commands of the grid and the cell count.
    python scripts/70_rebuild/e1b_cells.py --model_root /models --dry-run

    # Run one grid point, seed A then seed B.
    python scripts/70_rebuild/e1b_cells.py --model_root /models \\
        --results_root /abs/path/20_data/results --only 4/4

    # Aggregate the grid.
    python -m c3.analysis.rebuild.aggregate_e1b --results /abs/path/20_data/results/E1b \\
        --manifest results/manifest.json --out /abs/path/20_data/results/E1b/summary.json
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Shared with the E1 driver on purpose: the workflow table, the measured
# position, the sampling defaults and the command shape must be read the same
# way in both, or a grid cell and the sweep cell it is compared against drift
# apart.
from e1_cells import (  # noqa: E402
    POLICY_TAG,
    _already_done,
    build_argv,
    build_meta,
    defaults_section_for,
    measured_positions,
    parse_int_csv,
    parse_models,
    posix_join,
    read_sampling_defaults,
)

DEFAULT_WORKFLOW = "a2"
DEFAULT_MODELS = "4b=Qwen3-4B-Instruct-2507"
DEFAULT_NS = "2,4,8"
DEFAULT_CS = "2,4,8"
DEFAULT_RESULTS_ROOT = "20_data/results"
DEFAULT_TREE = "E1b"
DEFAULT_ANALYSIS_YAML = "configs/analysis.yaml"
DEFAULT_SPLIT = "MATHPOOL"
DEFAULT_LIMIT = 500
DEFAULT_TIMEOUT_H = 6.0

#: The two continuation seeds of a grid point. B reuses A's alternatives.
SEEDS: Dict[str, int] = {"A": 1001, "B": 2002}

#: The decision points are the first `limit` questions of the pool in pool
#: order, the same for both seeds, so the question order is not tied to the
#: continuation seed.
QUESTION_ORDER_SEED = 0


@dataclass(frozen=True)
class GridCell:
    workflow: str
    model: str
    n: int
    c: int
    seed_tag: str
    out_path: str
    reuse_from: Optional[str]
    argv: List[str]

    @property
    def name(self) -> str:
        return f"{self.workflow}/{self.model}/n{self.n}_c{self.c}/seed{self.seed_tag}"

    def command_line(self) -> str:
        return " ".join(shlex.quote(a) for a in self.argv)


def parse_only(value: str) -> Tuple[int, int]:
    """Parse the `<n>/<c>` selector of --only."""
    parts = [p.strip() for p in str(value).split("/")]
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"--only must look like <n>/<c>, got {value!r}")
    try:
        return int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise ValueError(f"--only must name two integers, got {value!r}") from exc


def cell_dir(results_root: str, tree: str, workflow: str, model: str, n: int, c: int) -> str:
    return posix_join(results_root, tree, workflow, model, f"n{n}_c{c}")


def build_cells(args: argparse.Namespace) -> List[GridCell]:
    models = parse_models(args.models)
    if len(models) != 1:
        # aggregate_e1b names a grid cell by (n, c) alone, so one tree holds one model.
        raise ValueError("--models must name exactly one model; the grid keys do not carry a model")
    ns = parse_int_csv(args.ns)
    cs = parse_int_csv(args.cs)
    selector = parse_only(args.only) if args.only else None
    if selector is not None and (selector[0] not in ns or selector[1] not in cs):
        raise ValueError(f"--only {args.only!r} is not a point of the grid n={ns} c={cs}")

    target_role, next_role = measured_positions(args.workflow)
    sampling = read_sampling_defaults(args.analysis_yaml, defaults_section_for(target_role))
    (model, directory), = models.items()
    model_path = posix_join(args.model_root, directory)

    cells: List[GridCell] = []
    for n in ns:
        for c in cs:
            if selector is not None and (n, c) != selector:
                continue
            base = cell_dir(args.results_root, args.tree, args.workflow, model, n, c)
            a_path = posix_join(base, "seedA", "buckets.jsonl")
            for tag in ("A", "B"):
                seed = SEEDS[tag]
                out_path = posix_join(base, f"seed{tag}", "buckets.jsonl")
                meta: Dict[str, Any] = build_meta(
                    workflow=args.workflow,
                    model=model,
                    n=n,
                    completions=c,
                    seed=seed,
                    split=args.split,
                    sampling=sampling,
                )
                meta["rule"] = f"e1b_n{n}_c{c}"
                meta["seed_tag"] = tag
                meta["question_order_seed"] = QUESTION_ORDER_SEED
                meta["limit"] = int(args.limit)
                argv = build_argv(
                    python_bin=args.python,
                    workflow=args.workflow,
                    model_path=model_path,
                    target_role=target_role,
                    next_role=next_role,
                    n=n,
                    completions=c,
                    limit=args.limit,
                    seed=seed,
                    split=args.split,
                    analysis_yaml=args.analysis_yaml,
                    engine=args.engine,
                    method=args.method,
                    tensor_parallel_size=args.tensor_parallel_size,
                    out_path=out_path,
                    meta=meta,
                    overwrite=args.overwrite,
                    batched=True,
                )
                reuse = a_path if tag == "B" else None
                if reuse is not None:
                    argv += ["--reuse_candidates_from", reuse]
                cells.append(GridCell(args.workflow, model, n, c, tag, out_path, reuse, argv))
    if not cells:
        raise ValueError("the grid selected no cell")
    return cells


def run_cells(cells: Sequence[GridCell], *, timeout_s: float, overwrite: bool) -> List[Tuple[str, str]]:
    """Run the cells in order, seed A of a point before its seed B.

    A seed B cell whose seed A buckets are missing is refused rather than run:
    without them it would draw fresh alternatives and measure the wrong noise.
    """
    failures: List[Tuple[str, str]] = []
    for index, cell in enumerate(cells, start=1):
        head = f"[e1b_cells] ({index}/{len(cells)}) {cell.name}"
        if not overwrite and _already_done(cell.out_path):
            print(f"{head} skip, buckets already present: {cell.out_path}", flush=True)
            continue
        if cell.reuse_from is not None and not _already_done(cell.reuse_from):
            failures.append((cell.name, f"seed A buckets missing: {cell.reuse_from}"))
            print(f"{head} REFUSED: seed B needs the finished seed A cell at {cell.reuse_from}", flush=True)
            continue
        print(f"{head} run: {cell.command_line()}", flush=True)
        try:
            completed = subprocess.run(cell.argv, cwd=str(REPO_ROOT), timeout=timeout_s, check=False)
        except subprocess.TimeoutExpired:
            failures.append((cell.name, f"timeout after {timeout_s:.0f}s"))
            print(f"{head} FAILED: timeout after {timeout_s:.0f}s", flush=True)
            continue
        except OSError as exc:
            failures.append((cell.name, f"could not start: {exc}"))
            print(f"{head} FAILED: could not start: {exc}", flush=True)
            continue
        if completed.returncode != 0:
            failures.append((cell.name, f"exit code {completed.returncode}"))
            print(f"{head} FAILED: exit code {completed.returncode}", flush=True)
        else:
            print(f"{head} done: {cell.out_path}", flush=True)
    return failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/70_rebuild/e1b_cells.py",
        description="Emit or run the bucket-generation cells of the E1b noise-law grid.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results_root", default=DEFAULT_RESULTS_ROOT,
                        help="Root of the results tree. Pass an absolute path when actually running cells.")
    parser.add_argument("--tree", default=DEFAULT_TREE, help="Directory of the grid under the results root.")
    parser.add_argument("--model_root", required=True, help="Directory that holds the base model directories.")
    parser.add_argument("--models", default=DEFAULT_MODELS, help="One name=directory pair.")
    parser.add_argument("--workflow", default=DEFAULT_WORKFLOW, help="Workflow of the grid.")
    parser.add_argument("--ns", default=DEFAULT_NS, help="Comma separated branching factors.")
    parser.add_argument("--cs", default=DEFAULT_CS, help="Comma separated continuations per alternative.")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Number of decision points per cell.")
    parser.add_argument("--split", default=DEFAULT_SPLIT, help="Evaluation suite the decision points come from.")
    parser.add_argument("--analysis_yaml", default=DEFAULT_ANALYSIS_YAML, help="Analysis defaults file.")
    parser.add_argument("--engine", default="auto", choices=["auto", "hf", "vllm"], help="Policy engine.")
    parser.add_argument("--method", default=POLICY_TAG, help="Method label written into bucket meta.")
    parser.add_argument("--tensor_parallel_size", "--tp", dest="tensor_parallel_size", type=int, default=0,
                        help="vLLM tensor parallel size. 0 leaves the flag out of the command.")
    parser.add_argument("--python", default="python", help="Python executable used in the emitted commands.")
    parser.add_argument("--timeout_h", type=float, default=DEFAULT_TIMEOUT_H, help="Per cell wall clock limit in hours.")
    parser.add_argument("--only", default="", help="Run one grid point, given as <n>/<c>, both seeds.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Rebuild cells whose bucket file already exists, and pass --overwrite through.")
    parser.add_argument("--dry-run", "--dry_run", dest="dry_run", action="store_true", help="Print commands only.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.limit <= 0:
        print("[e1b_cells] ERROR: --limit must be positive", file=sys.stderr)
        return 2
    if args.timeout_h <= 0:
        print("[e1b_cells] ERROR: --timeout_h must be positive", file=sys.stderr)
        return 2
    try:
        cells = build_cells(args)
    except (ValueError, FileNotFoundError, OSError) as exc:
        print(f"[e1b_cells] ERROR: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        for cell in cells:
            print(cell.command_line())
        print(f"# cells: {len(cells)}")
        return 0

    if not Path(args.results_root).is_absolute():
        print(
            f"[e1b_cells] WARNING: --results_root {args.results_root!r} is relative, so bucket files "
            f"land under the repository at {REPO_ROOT}. Pass an absolute path to write elsewhere.",
            file=sys.stderr,
        )
    failures = run_cells(cells, timeout_s=args.timeout_h * 3600.0, overwrite=args.overwrite)
    print(f"# cells: {len(cells)}")
    if failures:
        print(f"# failed: {len(failures)}")
        for name, reason in failures:
            print(f"#   {name}: {reason}")
        return 1
    print("# failed: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
