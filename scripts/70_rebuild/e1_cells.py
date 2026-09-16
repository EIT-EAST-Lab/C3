#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Emit, and optionally run, the bucket-generation cells of the E1 depth study.

E1 measures the estimator itself on a frozen policy: for one workflow, one base
model and one branching factor, it replays the alternatives already collected at
the measured decision point and writes them as one bucket file. A cell is that
triple (workflow, model, branching factor), and the full sweep is the cross
product of the three lists below.

Layout of the products, fixed by the results layout:

    <results_root>/E1/<workflow>/<model>/sweep_n<n>/buckets.jsonl

Only sweep cells are produced. The fixed-budget and constant-per-decision
readings are the same measurements under a different label, so the aggregation
step derives them from these files instead of re-running them.

Measured position: the first role of the workflow in topological order, with its
first successor recorded as the downstream role, so that influence and the
correction rate can be computed from the same buckets.

Decision points come from `MATHPOOL`, the screened question pool, which is what
the depth study measures on. Every workflow task file read here declares that
suite, the two-agent arm included: `configs/tasks/math_a2.yaml` is the paper's
own `configs/tasks/math.yaml` with the pool suite appended, so the task file a
reader of the paper runs stays free of a suite that exists only after the
screening pass.

The appendix comparison over the full MATH500 suite is the same sweep with two
flags changed, and it writes its own tree so the two key sets cannot collide:

    python scripts/70_rebuild/e1_cells.py --model_root /models \\
        --split MATH500 --results_root <root>/E1_math500all

Examples:

    # Print the 60 commands of the full sweep and the cell count.
    python scripts/70_rebuild/e1_cells.py --model_root /models --dry-run

    # Run one cell.
    python scripts/70_rebuild/e1_cells.py --model_root /models \\
        --results_root /abs/path/20_data/results --only a3/4b/3
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

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from c3.task.config import load_task, topo_sort_roles  # noqa: E402


ANALYSIS_MODULE = "c3.analysis.analysis"

DEFAULT_WORKFLOWS = "a2,a3,mt4,branch,c5,c10"
DEFAULT_MODELS = "4b=Qwen3-4B-Instruct-2507,m2=Qwen3-8B"
DEFAULT_NS = "2,3,4,6,8"
DEFAULT_RESULTS_ROOT = "20_data/results"
DEFAULT_ANALYSIS_YAML = "configs/analysis.yaml"
DEFAULT_SPLIT = "MATHPOOL"
DEFAULT_COMPLETIONS = 4
DEFAULT_LIMIT = 500
DEFAULT_SEED = 0
DEFAULT_TIMEOUT_H = 6.0

# The policy under measurement is the frozen supervised-fine-tuned checkpoint.
POLICY_TAG = "sft"

# Every workflow has its own task file, and every one of them declares MATHPOOL.
# a2's file is configs/tasks/math.yaml with the pool suite appended; see the
# module docstring for why the paper's own task file is not read here.
TASK_YAML_BY_WORKFLOW = {
    "a2": "configs/tasks/math_a2.yaml",
    "a3": "configs/tasks/math_a3.yaml",
    "mt4": "configs/tasks/math_mt4.yaml",
    "branch": "configs/tasks/math_branch.yaml",
    "c5": "configs/tasks/math_c5.yaml",
    "c10": "configs/tasks/math_c10.yaml",
}


# ---------------------------------------------------------------------------
# Small parsers
# ---------------------------------------------------------------------------


def parse_csv(value: str) -> List[str]:
    """Split a comma separated list, dropping empty fields."""
    return [part.strip() for part in str(value).split(",") if part.strip()]


def parse_int_csv(value: str) -> List[int]:
    out: List[int] = []
    for part in parse_csv(value):
        try:
            out.append(int(part))
        except ValueError as exc:
            raise ValueError(f"not an integer in --ns: {part!r}") from exc
    return out


def parse_models(value: str) -> Dict[str, str]:
    """Parse `name=directory,name=directory` into an ordered mapping."""
    models: Dict[str, str] = {}
    for part in parse_csv(value):
        if "=" not in part:
            raise ValueError(f"--models entries must look like name=directory, got {part!r}")
        name, _, directory = part.partition("=")
        name = name.strip()
        directory = directory.strip()
        if not name or not directory:
            raise ValueError(f"--models entry has an empty side: {part!r}")
        if name in models:
            raise ValueError(f"--models names a model twice: {name!r}")
        models[name] = directory
    if not models:
        raise ValueError("--models is empty")
    return models


def parse_only(value: str) -> Tuple[str, str, int]:
    """Parse the `<workflow>/<model>/<n>` selector of --only."""
    parts = [p.strip() for p in str(value).split("/")]
    if len(parts) != 3 or not all(parts):
        raise ValueError(f"--only must look like <workflow>/<model>/<n>, got {value!r}")
    try:
        n = int(parts[2])
    except ValueError as exc:
        raise ValueError(f"--only branching factor must be an integer, got {parts[2]!r}") from exc
    return parts[0], parts[1], n


def posix_join(*parts: str) -> str:
    """Join path pieces with forward slashes, which is what the commands carry."""
    cleaned = [str(p).replace("\\", "/").rstrip("/") for p in parts[:-1]]
    cleaned.append(str(parts[-1]).replace("\\", "/"))
    return "/".join(p for p in cleaned if p != "")


# ---------------------------------------------------------------------------
# Repository facts read at command build time
# ---------------------------------------------------------------------------


def task_yaml_for(workflow: str) -> str:
    try:
        return TASK_YAML_BY_WORKFLOW[workflow]
    except KeyError as exc:
        known = ",".join(sorted(TASK_YAML_BY_WORKFLOW))
        raise ValueError(f"unknown workflow {workflow!r}; known workflows: {known}") from exc


def measured_positions(workflow: str) -> Tuple[str, str]:
    """Return (target_role, next_role) for one workflow.

    The target is the first role in topological order. The next role is its
    first successor in that same order, which for the branching workflow is
    solver_a rather than the other parallel solver.
    """
    spec = load_task(str(REPO_ROOT / task_yaml_for(workflow)))
    roles = topo_sort_roles(spec.roles)
    topo = [r.name for r in roles]
    target = topo[0]

    by_name = {r.name: r for r in roles}
    successors = [name for name in topo if target in (by_name[name].depends_on or ())]
    if not successors:
        raise ValueError(f"workflow {workflow!r}: role {target!r} has no successor to record")
    return target, successors[0]


def defaults_section_for(target_role: str) -> str:
    """Mirror the `--defaults_section auto` rule of c3.analysis.analysis."""
    return "credit" if str(target_role).lower() == "actor" else "influence"


def read_sampling_defaults(analysis_yaml: str, section: str) -> Dict[str, Any]:
    """Read the sampling parameters that the run will actually use.

    The section is the one `--defaults_section auto` resolves to for the target
    role, so the recorded meta cannot drift from the decoding the run applies.
    """
    path = Path(analysis_yaml)
    if not path.is_absolute():
        path = REPO_ROOT / path
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top level must be a mapping")

    block = raw.get(section) or {}
    if not isinstance(block, dict):
        raise ValueError(f"{path}: section {section!r} must be a mapping")

    decoding = block.get("decoding") or {}
    if not isinstance(decoding, dict):
        raise ValueError(f"{path}: {section}.decoding must be a mapping")

    fidelity = block.get("fidelity") or {}
    if not isinstance(fidelity, dict):
        raise ValueError(f"{path}: {section}.fidelity must be a mapping")

    missing = [k for k in ("temperature", "top_p", "top_k") if k not in decoding]
    if missing:
        raise ValueError(f"{path}: {section}.decoding is missing {missing}")

    return {
        "section": section,
        "temperature": decoding["temperature"],
        "top_p": decoding["top_p"],
        "top_k": decoding["top_k"],
        "include_real_as_j0": bool(fidelity.get("include_real_as_j0", False)),
    }


# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    workflow: str
    model: str
    n: int
    target_role: str
    next_role: str
    out_path: str
    meta: Dict[str, Any]
    argv: List[str]

    @property
    def name(self) -> str:
        return f"{self.workflow}/{self.model}/{self.n}"

    def command_line(self) -> str:
        return " ".join(shlex.quote(a) for a in self.argv)


def bucket_out_path(results_root: str, workflow: str, model: str, n: int) -> str:
    return posix_join(results_root, "E1", workflow, model, f"sweep_n{n}", "buckets.jsonl")


def build_meta(
    *,
    workflow: str,
    model: str,
    n: int,
    completions: int,
    seed: int,
    split: str,
    sampling: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the meta block the results layout requires on every bucket."""
    return {
        "workflow": workflow,
        "model": model,
        "rule": f"sweep_n{n}",
        "n_alternatives": int(n),
        "replays_per_alt": int(completions),
        "temperature": sampling["temperature"],
        "top_p": sampling["top_p"],
        "top_k": sampling["top_k"],
        "seed": int(seed),
        "policy_tag": POLICY_TAG,
        "dataset": split,
        "question_order_seed": int(seed),
        "include_real_as_j0": bool(sampling["include_real_as_j0"]),
        # generation-time context is the transitive ancestors of a role (results layout revision 2)
        "context_scope": "ancestors",
    }


def build_argv(
    *,
    python_bin: str,
    workflow: str,
    model_path: str,
    target_role: str,
    next_role: str,
    n: int,
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
        task_yaml_for(workflow),
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
        str(int(n)),
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
    ]
    if batched:
        # Cross bucket batching. Without it a deep workflow issues one engine
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
    workflows = parse_csv(args.workflows)
    models = parse_models(args.models)
    ns = parse_int_csv(args.ns)

    selector: Optional[Tuple[str, str, int]] = parse_only(args.only) if args.only else None
    if selector is not None:
        wf, model, n = selector
        if wf not in workflows:
            raise ValueError(f"--only names workflow {wf!r}, which is not in --workflows {workflows}")
        if model not in models:
            raise ValueError(f"--only names model {model!r}, which is not in --models {sorted(models)}")
        if n not in ns:
            raise ValueError(f"--only names branching factor {n}, which is not in --ns {ns}")

    cells: List[Cell] = []
    for workflow in workflows:
        target_role, next_role = measured_positions(workflow)
        sampling = read_sampling_defaults(args.analysis_yaml, defaults_section_for(target_role))

        for model, directory in models.items():
            model_path = posix_join(args.model_root, directory)
            for n in ns:
                if selector is not None and (workflow, model, n) != selector:
                    continue

                out_path = bucket_out_path(args.results_root, workflow, model, n)
                meta = build_meta(
                    workflow=workflow,
                    model=model,
                    n=n,
                    completions=args.completions,
                    seed=args.seed,
                    split=args.split,
                    sampling=sampling,
                )
                argv = build_argv(
                    python_bin=args.python,
                    workflow=workflow,
                    model_path=model_path,
                    target_role=target_role,
                    next_role=next_role,
                    n=n,
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
                        workflow=workflow,
                        model=model,
                        n=n,
                        target_role=target_role,
                        next_role=next_role,
                        out_path=out_path,
                        meta=meta,
                        argv=argv,
                    )
                )

    if selector is not None and not cells:
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
        head = f"[e1_cells] ({index}/{len(cells)}) {cell.name}"

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
        prog="python scripts/70_rebuild/e1_cells.py",
        description="Emit or run the E1 bucket-generation cells of the rebuild study.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results_root",
        default=DEFAULT_RESULTS_ROOT,
        help="Root of the results tree. Pass an absolute path when actually running cells.",
    )
    parser.add_argument("--model_root", required=True, help="Directory that holds the base model directories.")
    parser.add_argument("--models", default=DEFAULT_MODELS, help="Comma separated name=directory pairs.")
    parser.add_argument("--workflows", default=DEFAULT_WORKFLOWS, help="Comma separated workflow names.")
    parser.add_argument("--ns", default=DEFAULT_NS, help="Comma separated branching factors.")
    parser.add_argument("--completions", type=int, default=DEFAULT_COMPLETIONS, help="Replays per alternative.")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Number of decision points per cell.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Sampling and question order seed.")
    parser.add_argument("--split", default=DEFAULT_SPLIT, help="Evaluation suite the decision points come from.")
    parser.add_argument("--analysis_yaml", default=DEFAULT_ANALYSIS_YAML, help="Analysis defaults file.")
    parser.add_argument("--engine", default="auto", choices=["auto", "hf", "vllm"], help="Policy engine.")
    parser.add_argument("--method", default=POLICY_TAG, help="Method label written into bucket meta.")
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
    parser.add_argument("--only", default="", help="Run one cell, given as <workflow>/<model>/<n>.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild cells whose bucket file already exists, and pass --overwrite through.",
    )
    parser.add_argument("--dry-run", "--dry_run", dest="dry_run", action="store_true", help="Print commands only.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.completions <= 0:
        print("[e1_cells] ERROR: --completions must be positive", file=sys.stderr)
        return 2
    if args.limit <= 0:
        print("[e1_cells] ERROR: --limit must be positive", file=sys.stderr)
        return 2
    if args.timeout_h <= 0:
        print("[e1_cells] ERROR: --timeout_h must be positive", file=sys.stderr)
        return 2

    try:
        cells = build_cells(args)
    except (ValueError, FileNotFoundError, OSError) as exc:
        print(f"[e1_cells] ERROR: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        for cell in cells:
            print(cell.command_line())
        print(f"# cells: {len(cells)}")
        return 0

    if not Path(args.results_root).is_absolute():
        print(
            f"[e1_cells] WARNING: --results_root {args.results_root!r} is relative, so bucket files "
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
