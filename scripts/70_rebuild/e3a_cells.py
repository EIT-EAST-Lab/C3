#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Emit, and optionally run, the bucket-generation cells of the E3a bias map.

E3a asks what an ablation signal measures. At every decision point of the three
agent workflow it puts one extra arm next to the sampled alternatives: a null
action, the message the measured role did not send. The bias of the ablation
signal at that point is the mean return of the sampled alternatives minus the
return of the null arm.

Two cells, one per reported form of the null action:

    <results_root>/E3a/a3/4b/empty/buckets.jsonl
    <results_root>/E3a/a3/4b/deleted/buckets.jsonl

Both inject the empty string. In this code base a role whose output is the
empty string contributes nothing to the downstream context, no paragraph, no
separator and no label, so the empty message and the deleted paragraph are the
same construction; `meta.null_form_note` records that on every bucket. The two
cells are kept apart because the paper reports them as two readings, and
because a later change to the assembly would make them differ.

Measured position: reasoner, with the actor recorded as the downstream role,
which is the position the preregistration fixes for E3a.

Examples:

    # Print the two commands and the cell count.
    python scripts/70_rebuild/e3a_cells.py --model_root /models --dry-run

    # Run one cell.
    python scripts/70_rebuild/e3a_cells.py --model_root /models \\
        --results_root /abs/path/20_data/results --only empty
"""

from __future__ import annotations

import argparse
import json
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
# position and the sampling defaults must be read the same way in both, or the
# recorded meta of the two experiments drifts apart.
from e1_cells import (  # noqa: E402
    ANALYSIS_MODULE,
    POLICY_TAG,
    defaults_section_for,
    measured_positions,
    parse_models,
    posix_join,
    read_sampling_defaults,
    task_yaml_for,
)

WORKFLOW = "a3"
MODEL = "4b"
FORMS = ("empty", "placeholder")

# The null action itself, per form (preregistration revision 15): the empty string
# is dropped by the context assembly, so the role vanishes from the downstream
# prompts (the leave-one-agent-out counterfactual); the placeholder keeps the role
# present with a fixed message that contributes nothing.
NULL_TEXTS = {"empty": "", "placeholder": "No message."}

DEFAULT_MODELS = "4b=Qwen3-4B-Instruct-2507"
DEFAULT_RESULTS_ROOT = "20_data/results"
DEFAULT_ANALYSIS_YAML = "configs/analysis.yaml"
DEFAULT_SPLIT = "MATH500"
DEFAULT_ALTERNATIVES = 4
DEFAULT_COMPLETIONS = 4
DEFAULT_LIMIT = 150
DEFAULT_SEED = 0
DEFAULT_TIMEOUT_H = 6.0

# Every role reads its transitive ancestors and nothing else (results contract,
# revision 2). The value is stamped on the buckets so the tree records it.
CONTEXT_SCOPE = "ancestors"


# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    form: str
    target_role: str
    next_role: str
    out_path: str
    meta: Dict[str, Any]
    argv: List[str]

    @property
    def name(self) -> str:
        return f"{WORKFLOW}/{MODEL}/{self.form}"

    def command_line(self) -> str:
        return " ".join(shlex.quote(a) for a in self.argv)


def bucket_out_path(results_root: str, form: str) -> str:
    return posix_join(results_root, "E3a", WORKFLOW, MODEL, form, "buckets.jsonl")


def build_meta(
    *,
    form: str,
    alternatives: int,
    completions: int,
    seed: int,
    split: str,
    sampling: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the meta block the results contract requires on every bucket.

    `n_alternatives` counts the sampled alternatives only. The null arm is the
    thing they are compared against, not one of them; the run itself records it
    as `null_arm`, `null_form` and `null_text`.
    """
    return {
        "workflow": WORKFLOW,
        "model": MODEL,
        "rule": f"e3a_{form}",
        "n_alternatives": int(alternatives),
        "replays_per_alt": int(completions),
        "temperature": sampling["temperature"],
        "top_p": sampling["top_p"],
        "top_k": sampling["top_k"],
        "seed": int(seed),
        "policy_tag": POLICY_TAG,
        "dataset": split,
        "question_order_seed": int(seed),
        "include_real_as_j0": bool(sampling["include_real_as_j0"]),
        "context_scope": CONTEXT_SCOPE,
    }


def build_argv(
    *,
    python_bin: str,
    form: str,
    model_path: str,
    target_role: str,
    next_role: str,
    alternatives: int,
    completions: int,
    limit: int,
    seed: int,
    split: str,
    analysis_yaml: str,
    engine: str,
    method: str,
    tensor_parallel_size: int,
    out_path: str,
    meta: Dict[str, Any],
    overwrite: bool,
    batched: bool = True,
) -> List[str]:
    argv = [
        python_bin,
        "-m",
        ANALYSIS_MODULE,
        "build-buckets",
        "--task",
        task_yaml_for(WORKFLOW),
        "--split",
        split,
        "--policy_ckpt",
        model_path,
        "--method",
        method,
        "--target_role",
        target_role,
        "--next_role",
        next_role,
        "--num_candidates",
        str(int(alternatives)),
        "--num_completions",
        str(int(completions)),
        "--limit",
        str(int(limit)),
        "--seed",
        str(int(seed)),
        "--analysis_yaml",
        analysis_yaml,
        "--engine",
        engine,
        # The null arm. The empty argument is what makes the alternative empty,
        # so it has to survive the shell: the printed command quotes it.
        "--inject_literal_candidate",
        NULL_TEXTS[form],
        "--null_form",
        form,
    ]
    if batched:
        # Cross bucket batching. Without it this workflow issues one engine
        # call per downstream role per replay, which no engine can pack.
        argv.append("--batched")
    if tensor_parallel_size > 0:
        argv += ["--tensor_parallel_size", str(int(tensor_parallel_size))]
    argv += [
        "--out",
        out_path,
        "--meta_json",
        json.dumps(meta, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    ]
    if overwrite:
        argv.append("--overwrite")
    return argv


def build_cells(args: argparse.Namespace) -> List[Cell]:
    models = parse_models(args.models)
    if MODEL not in models:
        raise ValueError(f"--models must name the {MODEL!r} base model, got {sorted(models)}")

    if args.only and args.only not in FORMS:
        raise ValueError(f"--only must be one of {list(FORMS)}, got {args.only!r}")

    target_role, next_role = measured_positions(WORKFLOW)
    if (target_role, next_role) != (args.target_role, args.next_role):
        raise ValueError(
            f"workflow {WORKFLOW!r} measures {target_role}->{next_role}, but the command line asks for "
            f"{args.target_role}->{args.next_role}"
        )

    sampling = read_sampling_defaults(args.analysis_yaml, defaults_section_for(target_role))
    model_path = posix_join(args.model_root, models[MODEL])

    cells: List[Cell] = []
    for form in FORMS:
        if args.only and form != args.only:
            continue

        out_path = bucket_out_path(args.results_root, form)
        meta = build_meta(
            form=form,
            alternatives=args.alternatives,
            completions=args.completions,
            seed=args.seed,
            split=args.split,
            sampling=sampling,
        )
        argv = build_argv(
            python_bin=args.python,
            form=form,
            model_path=model_path,
            target_role=target_role,
            next_role=next_role,
            alternatives=args.alternatives,
            completions=args.completions,
            limit=args.limit,
            seed=args.seed,
            split=args.split,
            analysis_yaml=args.analysis_yaml,
            engine=args.engine,
            method=args.method,
            tensor_parallel_size=args.tensor_parallel_size,
            out_path=out_path,
            meta=meta,
            overwrite=args.overwrite,
            batched=bool(args.batched),
        )
        cells.append(
            Cell(
                form=form,
                target_role=target_role,
                next_role=next_role,
                out_path=out_path,
                meta=meta,
                argv=argv,
            )
        )

    if args.only and not cells:
        raise ValueError(f"--only {args.only!r} selected no cell")
    return cells


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def _already_done(out_path: str) -> bool:
    """A cell is done when its bucket file exists and is not empty."""
    path = Path(out_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def run_cells(cells: Sequence[Cell], *, timeout_s: float, overwrite: bool) -> List[Tuple[str, str]]:
    """Run the cells in order. One failure never stops the rest."""
    failures: List[Tuple[str, str]] = []

    for index, cell in enumerate(cells, start=1):
        head = f"[e3a_cells] ({index}/{len(cells)}) {cell.name}"

        if not overwrite and _already_done(cell.out_path):
            print(f"{head} skip, buckets already present: {cell.out_path}", flush=True)
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/70_rebuild/e3a_cells.py",
        description="Emit or run the E3a bucket-generation cells of the rebuild study.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results_root",
        default=DEFAULT_RESULTS_ROOT,
        help="Root of the results tree. Pass an absolute path when actually running cells.",
    )
    parser.add_argument("--model_root", required=True, help="Directory that holds the base model directories.")
    parser.add_argument("--models", default=DEFAULT_MODELS, help="Comma separated name=directory pairs.")
    parser.add_argument(
        "--alternatives",
        type=int,
        default=DEFAULT_ALTERNATIVES,
        help="Sampled alternatives per decision point. The null arm comes on top of these.",
    )
    parser.add_argument("--completions", type=int, default=DEFAULT_COMPLETIONS, help="Replays per alternative.")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Number of decision points per cell.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Sampling and question order seed.")
    parser.add_argument("--split", default=DEFAULT_SPLIT, help="Evaluation suite the decision points come from.")
    parser.add_argument("--analysis_yaml", default=DEFAULT_ANALYSIS_YAML, help="Analysis defaults file.")
    parser.add_argument("--engine", default="auto", choices=["auto", "hf", "vllm"], help="Policy engine.")
    parser.add_argument("--method", default=POLICY_TAG, help="Method label written into bucket meta.")
    parser.add_argument("--target_role", default="reasoner", help="Role whose message is ablated.")
    parser.add_argument("--next_role", default="actor", help="Role whose answer is recorded downstream.")
    parser.add_argument(
        "--tensor_parallel_size",
        "--tp",
        dest="tensor_parallel_size",
        type=int,
        default=0,
        help="vLLM tensor parallel size. 0 leaves the flag out of the command.",
    )
    parser.add_argument(
        "--batched",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Emit --batched, which batches every generation stage across buckets.",
    )
    parser.add_argument("--python", default="python", help="Python executable used in the emitted commands.")
    parser.add_argument(
        "--timeout_h",
        type=float,
        default=DEFAULT_TIMEOUT_H,
        help="Per cell wall clock limit in hours.",
    )
    parser.add_argument("--only", default="", help=f"Run one cell, given as one of {list(FORMS)}.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild cells whose bucket file already exists, and pass --overwrite through.",
    )
    parser.add_argument("--dry-run", "--dry_run", dest="dry_run", action="store_true", help="Print commands only.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.alternatives <= 0:
        print("[e3a_cells] ERROR: --alternatives must be positive", file=sys.stderr)
        return 2
    if args.completions <= 0:
        print("[e3a_cells] ERROR: --completions must be positive", file=sys.stderr)
        return 2
    if args.limit <= 0:
        print("[e3a_cells] ERROR: --limit must be positive", file=sys.stderr)
        return 2
    if args.timeout_h <= 0:
        print("[e3a_cells] ERROR: --timeout_h must be positive", file=sys.stderr)
        return 2

    try:
        cells = build_cells(args)
    except (ValueError, FileNotFoundError, OSError) as exc:
        print(f"[e3a_cells] ERROR: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        for cell in cells:
            print(cell.command_line())
        print(f"# cells: {len(cells)}")
        return 0

    if not Path(args.results_root).is_absolute():
        print(
            f"[e3a_cells] WARNING: --results_root {args.results_root!r} is relative, so bucket files "
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
