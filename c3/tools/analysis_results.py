#!/usr/bin/env python3
# c3/tools/analysis_results.py
#
# Aggregate C3 analysis artifacts produced by:
#   examples/c3/analysis/run_credit_influence.sh
#
# Expected layout (flexible; we scan recursively):
#   <analysis_root>/seed*/metrics/credit_<method>_<split>.json
#   <analysis_root>/seed*/metrics/influence_<method>_<split>.json
#
# Each credit json is produced by: python -m c3.analysis.c3_analysis credit
# Each influence json is produced by: python -m c3.analysis.c3_analysis influence
#
# Outputs:
#   <out_dir>/<out_prefix>.raw.json
#   <out_dir>/<out_prefix>.summary.json
#   <out_dir>/<out_prefix>.credit_table.tex
#   <out_dir>/<out_prefix>.influence_table.tex
#   <out_dir>/<out_prefix>.combined_table.tex

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


# -------------------------
# small utils
# -------------------------


def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr)


def _j(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False)


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def _mean(xs: Sequence[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


def _std_sample(xs: Sequence[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    return float(statistics.stdev(xs))


def _fmt_num(x: Any, ndigits: int) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x):.{ndigits}f}"
    except Exception:
        return "—"


def _fmt_cell(mean: float, std: float, *, ndigits: int, pm: bool) -> str:
    if not pm:
        return _fmt_num(mean, ndigits)
    return f"{_fmt_num(mean, ndigits)}$\\pm${_fmt_num(std, ndigits)}"


# -------------------------
# normalization
# -------------------------


_METHOD_MAP = {
    "c3": "C3",
    "mappo": "MAPPO",
    "magrpo": "MAGRPO",
    "sft": "SFT",
}

# Accept a bunch of historical split spellings / shorthands and map to paper-facing names.
_SPLIT_ALIASES = {
    "math500": "MATH500",
    "cmath-test": "CMATH-test",
    "cmath_test": "CMATH-test",
    "gsm8k-test": "GSM8K-test",
    "gsm8k_test": "GSM8K-test",
    "gsm8k": "GSM8K-test",
    "mbpp+": "MBPP+",
    "mbppplus": "MBPP+",
    "mbppplus+": "MBPP+",
    "mbpp_plus": "MBPP+",
    "mbpp-test": "MBPP-test",
    "mbpp_test": "MBPP-test",
    "humaneval+": "HumanEval+",
    "humanevalplus": "HumanEval+",
    "humanevalplus+": "HumanEval+",
    "humaneval_plus": "HumanEval+",
}


def _norm_method(s: str, user_map: Optional[Mapping[str, str]] = None) -> str:
    raw = (s or "").strip()
    key = raw.lower()
    if user_map and key in user_map:
        return str(user_map[key])
    if key in _METHOD_MAP:
        return _METHOD_MAP[key]
    return raw


def _norm_split(s: str, user_map: Optional[Mapping[str, str]] = None) -> str:
    raw = (s or "").strip()
    key = raw.lower().replace(" ", "")
    key = key.replace("__", "_")
    if user_map and key in user_map:
        return str(user_map[key])
    if key in _SPLIT_ALIASES:
        return _SPLIT_ALIASES[key]
    return raw


# -------------------------
# scanning + parsing
# -------------------------


_SEED_RE_DEFAULT = r"seed(\d+)"
_STRICT_SEED_DIR_RE = re.compile(r"^seed(\d+)$")


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"{path}: expected dict json")
    return obj


def _extract_seed_from_path(path: Path, seed_re: re.Pattern) -> Optional[int]:
    """
    Extract seed from a *path component* to avoid false positives from run names like:
      .../abl_no_replay_seed0/analysis/metrics/...
    We only accept:
      1) an explicit directory component "seed{N}"
      2) a user-provided regex that matches an entire directory component (not a substring)
    """
    # 1) Prefer explicit ".../seed{N}/..." directory.
    for part in path.parts:
        m = _STRICT_SEED_DIR_RE.fullmatch(part)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None

    # 2) Fallback: user regex must match the whole component (not a substring).
    for part in path.parts:
        m = seed_re.search(part)
        if m and m.start() == 0 and m.end() == len(part):
            try:
                return int(m.group(1))
            except Exception:
                return None

    return None


def _seed_sort_key(sd: Optional[int]) -> Tuple[int, int]:
    # Sort ints first, then None last (stable across Python versions).
    return (1, 0) if sd is None else (0, int(sd))


def _parse_method_split_from_filename(
    name: str,
    *,
    prefix: str,
    suffix: str,
    known_splits: Sequence[str],
) -> Tuple[str, str]:
    """
    Parse: f"{prefix}{method}_{split}{suffix}"
    Prefer matching split by suffix against known_splits (longest match).
    Fallback: split at last underscore.
    """
    if not (name.startswith(prefix) and name.endswith(suffix)):
        raise ValueError(f"unexpected filename pattern: {name}")
    core = name[len(prefix) : -len(suffix)]  # method_split
    core = core.strip("_")

    # Try known splits first (robust when method contains underscores).
    best: Optional[Tuple[str, str]] = None
    for sp in sorted(known_splits, key=lambda x: len(x), reverse=True):
        token = "_" + sp
        if core.endswith(token):
            method = core[: -len(token)]
            best = (method, sp)
            break
    if best is not None:
        return best

    # Fallback: last underscore.
    if "_" not in core:
        return core, "UNKNOWN"
    method, sp = core.rsplit("_", 1)
    return method, sp


@dataclass(frozen=True)
class Point:
    kind: str  # "credit" | "influence"
    method: str
    split: str
    seed: Optional[int]
    value: float
    metric: str  # fidelity_real | fidelity_all | var | influence
    path: str


def _discover_points(
    analysis_roots: Sequence[str],
    *,
    seed_re: re.Pattern,
    method_map: Optional[Mapping[str, str]],
    split_map: Optional[Mapping[str, str]],
    suite_splits: Sequence[str],
    strict_kinds: bool,
) -> Tuple[List[Point], List[Dict[str, Any]]]:
    points: List[Point] = []
    warnings: List[Dict[str, Any]] = []

    known_splits = list(suite_splits)
    # Also allow discovery to parse any split; we still normalize later.
    # known_splits helps parse "method" safely when underscores exist.

    for root in analysis_roots:
        rp = Path(root)
        if not rp.exists():
            warnings.append({"type": "missing_root", "root": root})
            continue

        for p in rp.rglob("metrics/*.json"):
            base = p.name
            seed = _extract_seed_from_path(p, seed_re)

            try:
                obj = _load_json(p)
            except Exception as e:
                warnings.append({"type": "bad_json", "path": str(p), "error": str(e)})
                continue

            # Decide kind from filename (more stable than json "kind" in practice).
            if base.startswith("credit_") and base.endswith(".json"):
                kind = "credit"
                prefix = "credit_"
                suffix = ".json"
            elif base.startswith("influence_") and base.endswith(".json"):
                kind = "influence"
                prefix = "influence_"
                suffix = ".json"
            else:
                continue

            # Optional: validate json kind field (can be disabled).
            if strict_kinds:
                want = "credit_metrics" if kind == "credit" else "influence_metrics"
                if str(obj.get("kind", "")).strip() != want:
                    warnings.append(
                        {
                            "type": "kind_mismatch",
                            "path": str(p),
                            "filename_kind": kind,
                            "json_kind": obj.get("kind"),
                            "expected": want,
                        }
                    )
                    continue

            raw_method, raw_split = _parse_method_split_from_filename(
                base, prefix=prefix, suffix=suffix, known_splits=known_splits
            )
            method = _norm_method(raw_method, method_map)
            split = _norm_split(raw_split, split_map)

            if kind == "credit":
                fidelity = obj.get("fidelity", {})
                var = obj.get("var", {})

                fid_real = None
                fid_all = None
                if isinstance(fidelity, Mapping):
                    fid_real = _safe_float(fidelity.get("spearman_real_only"))
                    fid_all = _safe_float(fidelity.get("spearman_all_candidates"))

                var_mean = None
                if isinstance(var, Mapping):
                    var_mean = _safe_float(var.get("mean"))

                if fid_real is not None:
                    points.append(
                        Point(
                            kind="credit",
                            method=method,
                            split=split,
                            seed=seed,
                            value=fid_real,
                            metric="fidelity_real",
                            path=str(p),
                        )
                    )
                if fid_all is not None:
                    points.append(
                        Point(
                            kind="credit",
                            method=method,
                            split=split,
                            seed=seed,
                            value=fid_all,
                            metric="fidelity_all",
                            path=str(p),
                        )
                    )
                if var_mean is not None:
                    points.append(
                        Point(
                            kind="credit",
                            method=method,
                            split=split,
                            seed=seed,
                            value=var_mean,
                            metric="var",
                            path=str(p),
                        )
                    )

            else:  # influence
                mi = obj.get("mi", {})
                mi_mean = None
                if isinstance(mi, Mapping):
                    mi_mean = _safe_float(mi.get("mean"))
                if mi_mean is not None:
                    points.append(
                        Point(
                            kind="influence",
                            method=method,
                            split=split,
                            seed=seed,
                            value=mi_mean,
                            metric="influence",
                            path=str(p),
                        )
                    )

    return points, warnings


# -------------------------
# aggregation
# -------------------------


def _group_points(points: Sequence[Point]) -> Dict[Tuple[str, str, str], List[Point]]:
    """
    (method, split, metric) -> points
    """
    out: Dict[Tuple[str, str, str], List[Point]] = {}
    for pt in points:
        out.setdefault((pt.method, pt.split, pt.metric), []).append(pt)
    return out


def _aggregate_group(points: Sequence[Point]) -> Dict[str, Any]:
    vals = [p.value for p in points]
    return {"mean": _mean(vals), "std": _std_sample(vals), "n": len(vals)}


def _suite_by_seed(
    points: Sequence[Point],
    *,
    suite_splits: Sequence[str],
    metric: str,
) -> Dict[Tuple[str, Optional[int]], Dict[str, float]]:
    """
    For each (method, seed), compute suite average across splits:
      suite_value(method, seed) = mean_{split in suite_splits} value(method, split, seed)
    We only include splits that are present for that (method, seed).
    """
    # index: (method, split, seed) -> value
    idx: Dict[Tuple[str, str, Optional[int]], float] = {}
    for p in points:
        if p.metric != metric:
            continue
        idx[(p.method, p.split, p.seed)] = p.value

    out: Dict[Tuple[str, Optional[int]], Dict[str, float]] = {}
    # Only consider points for the requested metric when enumerating methods/seeds.
    methods = sorted({p.method for p in points if p.metric == metric})
    seeds = sorted({p.seed for p in points if p.metric == metric}, key=_seed_sort_key)

    for m in methods:
        for sd in seeds:
            vals: List[float] = []
            for sp in suite_splits:
                key = (m, sp, sd)
                if key in idx:
                    vals.append(idx[key])
            if vals:
                out[(m, sd)] = {"value": float(sum(vals) / len(vals)), "n_splits": float(len(vals))}
    return out


def _aggregate_suite_across_seeds(
    suite_seed_vals: Mapping[Tuple[str, Optional[int]], Dict[str, float]]
) -> Dict[str, Dict[str, Any]]:
    by_method: Dict[str, List[float]] = {}
    for (m, _sd), rec in suite_seed_vals.items():
        v = rec.get("value")
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            by_method.setdefault(m, []).append(float(v))
    out: Dict[str, Dict[str, Any]] = {}
    for m, vals in by_method.items():
        out[m] = {"mean": _mean(vals), "std": _std_sample(vals), "n": len(vals)}
    return out


# -------------------------
# LaTeX rendering
# -------------------------


def _write_tex(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_credit_table(
    *,
    suite_credit: Mapping[str, Mapping[str, Any]],
    method_order: Sequence[str],
    out_path: Path,
    ndigits: int,
    pm: bool,
    fidelity_variant: str,
) -> None:
    """
    Rows: METHOD & Fidelity & Var \\
    """
    lines: List[str] = []
    lines.append(f"% Auto-generated by analysis_results.py | credit | fidelity={fidelity_variant}")
    for m in list(method_order) + sorted([x for x in suite_credit.keys() if x not in method_order]):
        rec = suite_credit.get(m)
        if not isinstance(rec, Mapping):
            continue
        fid = rec.get(fidelity_variant)
        var = rec.get("var")
        if not (isinstance(fid, Mapping) and isinstance(var, Mapping)):
            continue
        cell_f = _fmt_cell(float(fid["mean"]), float(fid["std"]), ndigits=ndigits, pm=pm)
        cell_v = _fmt_cell(float(var["mean"]), float(var["std"]), ndigits=ndigits, pm=pm)
        lines.append(f"{m} & {cell_f} & {cell_v} \\\\")
    _write_tex(out_path, lines)


def _render_influence_table(
    *,
    suite_infl: Mapping[str, Mapping[str, Any]],
    method_order: Sequence[str],
    out_path: Path,
    ndigits: int,
    pm: bool,
) -> None:
    """
    Rows: METHOD & Influence \\
    """
    lines: List[str] = []
    lines.append("% Auto-generated by analysis_results.py | influence")
    for m in list(method_order) + sorted([x for x in suite_infl.keys() if x not in method_order]):
        rec = suite_infl.get(m)
        if not isinstance(rec, Mapping):
            continue
        infl = rec.get("influence")
        if not isinstance(infl, Mapping):
            continue
        cell = _fmt_cell(float(infl["mean"]), float(infl["std"]), ndigits=ndigits, pm=pm)
        lines.append(f"{m} & {cell} \\\\")
    _write_tex(out_path, lines)


def _render_combined_table(
    *,
    suite_credit: Mapping[str, Mapping[str, Any]],
    suite_infl: Mapping[str, Mapping[str, Any]],
    method_order: Sequence[str],
    out_path: Path,
    ndigits: int,
    pm: bool,
    fidelity_variant: str,
) -> None:
    """
    Rows: METHOD & Fidelity & Var & Influence \\
    """
    lines: List[str] = []
    lines.append(f"% Auto-generated by analysis_results.py | combined | fidelity={fidelity_variant}")
    methods = set(suite_credit.keys()) | set(suite_infl.keys())
    ordered = list(method_order) + sorted([m for m in methods if m not in method_order])
    for m in ordered:
        cred = suite_credit.get(m, {})
        infl = suite_infl.get(m, {})
        fid = cred.get(fidelity_variant, None)
        var = cred.get("var", None)
        inf = infl.get("influence", None)

        fid_s = "--"
        var_s = "--"
        inf_s = "--"
        if isinstance(fid, Mapping):
            fid_s = _fmt_cell(float(fid["mean"]), float(fid["std"]), ndigits=ndigits, pm=pm)
        if isinstance(var, Mapping):
            var_s = _fmt_cell(float(var["mean"]), float(var["std"]), ndigits=ndigits, pm=pm)
        if isinstance(inf, Mapping):
            inf_s = _fmt_cell(float(inf["mean"]), float(inf["std"]), ndigits=ndigits, pm=pm)
        lines.append(f"{m} & {fid_s} & {var_s} & {inf_s} \\\\")
    _write_tex(out_path, lines)


# -------------------------
# CLI
# -------------------------


def _load_map_json(path: Optional[str]) -> Optional[Dict[str, str]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise ValueError(f"map json not found: {path}")
    obj = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"map json must be dict: {path}")
    # normalize keys to lowercase for matching
    out: Dict[str, str] = {}
    for k, v in obj.items():
        if isinstance(k, str) and isinstance(v, str):
            out[k.lower()] = v
    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="analysis_results.py", description="Aggregate credit/influence analysis metrics.")
    sub = p.add_subparsers(dest="cmd", required=True)

    agg = sub.add_parser("aggregate", help="scan analysis_out roots and aggregate metrics")
    agg.add_argument(
        "--analysis_root",
        action="append",
        required=True,
        help="Analysis output root (repeatable). Example: /path/to/analysis_out",
    )
    agg.add_argument("--out_dir", required=True, type=str, help="Output directory")
    agg.add_argument("--out_prefix", default="analysis_results", type=str, help="Output filename prefix")

    agg.add_argument("--seed_re", default=_SEED_RE_DEFAULT, type=str, help="Regex to extract seed from path")

    agg.add_argument(
        "--suite",
        default="math",
        choices=["math", "code", "custom"],
        help="Which suite splits to average over",
    )
    agg.add_argument(
        "--suite_splits",
        default="",
        type=str,
        help='Custom suite splits, comma/space separated. Example: "MATH500,CMATH-test,GSM8K-test"',
    )

    agg.add_argument(
        "--fidelity_variant",
        default="fidelity_real",
        choices=["fidelity_real", "fidelity_all"],
        help="Which fidelity to report in LaTeX",
    )

    agg.add_argument("--strict_kinds", default=1, type=int, help="Require json.kind matches (default: 1)")
    agg.add_argument("--latex_digits", default=3, type=int, help="Decimal digits for LaTeX numbers")
    agg.add_argument("--latex_pm", default=0, type=int, help="Render mean\\pmstd (default: 0, paper-style)")

    agg.add_argument(
        "--method_map_json",
        default=None,
        type=str,
        help="Optional JSON dict to map raw method->display name (keys matched case-insensitive).",
    )
    agg.add_argument(
        "--split_map_json",
        default=None,
        type=str,
        help="Optional JSON dict to map raw split->display name (keys matched case-insensitive).",
    )

    agg.add_argument(
        "--method_order",
        default="SFT,MAPPO,MAGRPO,C3",
        type=str,
        help="Comma-separated order for LaTeX rows",
    )

    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.cmd != "aggregate":
        raise RuntimeError(f"unknown cmd: {args.cmd}")

    method_map = _load_map_json(args.method_map_json)
    split_map = _load_map_json(args.split_map_json)

    seed_re = re.compile(str(args.seed_re))

    suite = str(args.suite)
    if suite == "math":
        suite_splits = ["MATH500", "CMATH-test", "GSM8K-test"]
    elif suite == "code":
        suite_splits = ["MBPP+", "MBPP-test"]
    else:
        raw = str(args.suite_splits or "").strip()
        if not raw:
            raise ValueError("--suite custom requires --suite_splits")
        suite_splits = [x.strip() for x in raw.replace(",", " ").split() if x.strip()]

    # Normalize suite splits too (handles user passing gsm8k, mbppplus, etc.)
    suite_splits = [_norm_split(x, split_map) for x in suite_splits]

    method_order = [x.strip() for x in str(args.method_order).split(",") if x.strip()]

    points, warnings = _discover_points(
        analysis_roots=list(args.analysis_root),
        seed_re=seed_re,
        method_map=method_map,
        split_map=split_map,
        suite_splits=suite_splits,
        strict_kinds=bool(int(args.strict_kinds)),
    )

    if not points:
        eprint("[WARN] No metrics points discovered. Check --analysis_root and file layout.")
        # still write raw diagnostics
    else:
        eprint(f"[OK] discovered points={len(points)} from roots={len(args.analysis_root)}")

    # Raw dump (points + warnings)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = out_dir / f"{args.out_prefix}.raw.json"
    summary_path = out_dir / f"{args.out_prefix}.summary.json"
    credit_tex = out_dir / f"{args.out_prefix}.credit_table.tex"
    infl_tex = out_dir / f"{args.out_prefix}.influence_table.tex"
    comb_tex = out_dir / f"{args.out_prefix}.combined_table.tex"

    raw_payload = {
        "meta": {
            "analysis_root": list(args.analysis_root),
            "suite": suite,
            "suite_splits": suite_splits,
            "seed_re": str(args.seed_re),
            "strict_kinds": bool(int(args.strict_kinds)),
        },
        "warnings": warnings,
        "points": [p.__dict__ for p in points],
    }
    raw_path.write_text(json.dumps(raw_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Per-split aggregation (method x split x metric)
    grouped = _group_points(points)
    per_split: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for (m, sp, metric), pts in grouped.items():
        per_split.setdefault(m, {}).setdefault(sp, {})[metric] = _aggregate_group(pts)

    # Suite-by-seed then aggregate across seeds (paper-like: equal weight across splits)
    suite_credit: Dict[str, Dict[str, Any]] = {}
    suite_infl: Dict[str, Dict[str, Any]] = {}

    # Credit: fidelity_real / fidelity_all / var
    for metric in ("fidelity_real", "fidelity_all", "var"):
        suite_vals = _suite_by_seed(points, suite_splits=suite_splits, metric=metric)
        agg = _aggregate_suite_across_seeds(suite_vals)
        for m, rec in agg.items():
            suite_credit.setdefault(m, {})[metric] = rec

    # Influence
    suite_vals = _suite_by_seed(points, suite_splits=suite_splits, metric="influence")
    agg = _aggregate_suite_across_seeds(suite_vals)
    for m, rec in agg.items():
        suite_infl.setdefault(m, {})["influence"] = rec

    summary = {
        "meta": {
            "suite": suite,
            "suite_splits": suite_splits,
            "method_order": method_order,
            "latex_digits": int(args.latex_digits),
            "latex_pm": bool(int(args.latex_pm)),
            "fidelity_variant": str(args.fidelity_variant),
        },
        "per_split": per_split,
        "suite": {
            "credit": suite_credit,
            "influence": suite_infl,
        },
        "diagnostics": {
            "n_points": len(points),
            "n_warnings": len(warnings),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # LaTeX
    nd = int(args.latex_digits)
    pm = bool(int(args.latex_pm))
    fid_variant = str(args.fidelity_variant)

    _render_credit_table(
        suite_credit=suite_credit,
        method_order=method_order,
        out_path=credit_tex,
        ndigits=nd,
        pm=pm,
        fidelity_variant=fid_variant,
    )
    _render_influence_table(
        suite_infl=suite_infl,
        method_order=method_order,
        out_path=infl_tex,
        ndigits=nd,
        pm=pm,
    )
    _render_combined_table(
        suite_credit=suite_credit,
        suite_infl=suite_infl,
        method_order=method_order,
        out_path=comb_tex,
        ndigits=nd,
        pm=pm,
        fidelity_variant=fid_variant,
    )

    eprint(f"[OK] wrote: {raw_path}")
    eprint(f"[OK] wrote: {summary_path}")
    eprint(f"[OK] wrote: {credit_tex}")
    eprint(f"[OK] wrote: {infl_tex}")
    eprint(f"[OK] wrote: {comb_tex}")

    if warnings:
        eprint(f"[WARN] warnings={len(warnings)} example={_j(warnings[0])}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
