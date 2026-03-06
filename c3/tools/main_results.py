#!/usr/bin/env python3
# c3/tools/main_results.py
#
# Aggregate "main results" from eval artifacts produced by examples/c3 sweep.
#
# Contract:
#   - Each run (id, method, task, seed) has two profiles:
#       greedy: temperature=0, n=1   -> Greedy
#       n10:    temperature=0.7, n=10 -> P@1 / P@10
#   - Artifacts:
#       <run_root>/<out_subdir>/<task>/<profile>/eval_only.jsonl
#       <run_root>/<out_subdir>/<task>/<profile>/eval_only.jsonl.metrics.jsonl
#   - run_root:
#       source.type=train_run_dir -> source.train_run_dir
#       source.type=hf_base       -> <ckpt_root>/_runs/_sft_main_results/<id>

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple

try:
    import yaml  # type: ignore
except Exception as e:  # pragma: no cover
    yaml = None  # type: ignore
    _yaml_import_error = e


SourceType = Literal["train_run_dir", "hf_base"]
TaskType = Literal["math", "code"]
ProfileType = Literal["greedy", "n10"]


# Expected benchmark suite per task.
EXPECTED_DATASOURCES: Dict[TaskType, List[str]] = {
    "math": ["MATH500", "CMATH-test", "GSM8K-test"],
    "code": ["MBPP-test", "MBPP+"],
}

# Table column order (paper-facing order).
TABLE_COLUMNS: List[str] = ["MATH500", "CMATH-test", "GSM8K-test", "MBPP+", "MBPP-test"]
METHOD_ORDER: List[str] = ["SFT", "MAPPO", "MAGRPO", "C3"]


# -------------------------
# small utils
# -------------------------


def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr)


def _as_float(x: Any) -> float:
    if x is None:
        raise ValueError("metric value is None")
    if isinstance(x, bool):
        return 1.0 if x else 0.0
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        if not s:
            raise ValueError("empty metric string")
        return float(s)
    raise TypeError(f"unsupported metric type: {type(x)}")


def _mean(xs: Sequence[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


def _std_sample(xs: Sequence[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    return float(statistics.stdev(xs))


def _j(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


# -------------------------
# schema
# -------------------------


@dataclass(frozen=True)
class RunRecord:
    id: str
    method: str  # normalized to {SFT, MAPPO, MAGRPO, C3} where possible
    task: TaskType
    seed: int

    source_type: SourceType
    train_run_dir: Optional[str] = None
    hf_base: Optional[str] = None  # kept for provenance; run_root is fixed by contract

    # Important: per-run output subdir to avoid overwrite across pseudo-seeds.
    out_subdir: Optional[str] = None

    # For eval harness compatibility (kept for completeness; not used in aggregation).
    alg_for_eval: str = "c3"


@dataclass(frozen=True)
class EvalArtifact:
    run_id: str
    task: TaskType
    profile: ProfileType
    run_dir: str
    metrics_path: str
    samples_path: str


@dataclass(frozen=True)
class DatasourceMetrics:
    greedy: float
    pass1: float
    pass10: float
    n_questions: int


# -------------------------
# registry
# -------------------------


def _normalize_method(m: str) -> str:
    s = (m or "").strip()
    up = s.upper()
    if up == "C3":
        return "C3"
    if up == "SFT":
        return "SFT"
    if up == "MAPPO":
        return "MAPPO"
    if up == "MAGRPO":
        return "MAGRPO"
    return up or s


def _load_registry(path: str) -> Tuple[Dict[str, Any], List[RunRecord]]:
    if yaml is None:  # pragma: no cover
        raise RuntimeError(f"PyYAML required. Import error: {_yaml_import_error}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if int(cfg.get("version", 1)) != 1:
        raise ValueError(f"unsupported registry version: {cfg.get('version')}")

    defaults = cfg.get("defaults") or {}
    runs_cfg = cfg.get("runs") or []
    if not isinstance(runs_cfg, list):
        raise ValueError("registry 'runs' must be a list")

    seen_ids: set[str] = set()
    runs: List[RunRecord] = []

    for r in runs_cfg:
        if not isinstance(r, dict):
            raise ValueError("each run entry must be a dict")

        rid = str(r.get("id", "")).strip()
        if not rid:
            raise ValueError("run.id is required")
        if rid in seen_ids:
            raise ValueError(f"duplicate run.id: {rid}")
        seen_ids.add(rid)

        method = _normalize_method(str(r.get("method", "")).strip())
        task = str(r.get("task", "")).strip()
        if task not in ("math", "code"):
            raise ValueError(f"{rid}: task must be 'math'|'code', got {task!r}")
        task_t: TaskType = task  # type: ignore[assignment]

        seed = r.get("seed")
        if not isinstance(seed, int):
            raise ValueError(f"{rid}: seed must be int, got {seed!r}")

        out_subdir = r.get("out_subdir")
        out_subdir_s = str(out_subdir).strip() if out_subdir is not None else None
        out_subdir_s = out_subdir_s or None

        alg_for_eval = str(r.get("alg_for_eval", "c3")).strip() or "c3"

        src = r.get("source") or {}
        if not isinstance(src, dict):
            raise ValueError(f"{rid}: source must be a dict")

        st = str(src.get("type", "")).strip()
        if st not in ("train_run_dir", "hf_base"):
            raise ValueError(f"{rid}: unknown source.type {st!r}")
        st_t: SourceType = st  # type: ignore[assignment]

        train_run_dir: Optional[str] = None
        hf_base: Optional[str] = None
        if st_t == "train_run_dir":
            train_run_dir = str(src.get("train_run_dir", "")).strip()
            if not train_run_dir:
                raise ValueError(f"{rid}: source.train_run_dir is required")
        else:
            hf_base = str(src.get("hf_base", "")).strip()
            if not hf_base:
                raise ValueError(f"{rid}: source.hf_base is required")

        runs.append(
            RunRecord(
                id=rid,
                method=method,
                task=task_t,
                seed=seed,
                source_type=st_t,
                train_run_dir=train_run_dir,
                hf_base=hf_base,
                out_subdir=out_subdir_s,
                alg_for_eval=alg_for_eval,
            )
        )

    return defaults, runs


# -------------------------
# artifact paths
# -------------------------


def _resolve_run_root(run: RunRecord, ckpt_root: str) -> str:
    if run.source_type == "train_run_dir":
        assert run.train_run_dir
        return run.train_run_dir
    return str(Path(ckpt_root) / "_runs" / "_sft_main_results" / run.id)


def _pick_out_subdir(
    *,
    defaults: Mapping[str, Any],
    run: RunRecord,
    out_subdir_override: Optional[str],
) -> str:
    # Priority: CLI override > run.out_subdir > defaults.out_subdir > "main_results"
    if out_subdir_override and str(out_subdir_override).strip():
        return str(out_subdir_override).strip()
    if run.out_subdir and run.out_subdir.strip():
        return run.out_subdir.strip()
    d = defaults.get("out_subdir")
    d = str(d).strip() if d is not None else ""
    return d or "main_results"


def _resolve_artifacts(
    run: RunRecord,
    *,
    defaults: Mapping[str, Any],
    ckpt_root: str,
    out_subdir_override: Optional[str],
) -> Dict[ProfileType, EvalArtifact]:
    out_subdir = _pick_out_subdir(defaults=defaults, run=run, out_subdir_override=out_subdir_override)
    run_root = _resolve_run_root(run, ckpt_root)

    out: Dict[ProfileType, EvalArtifact] = {}
    for profile in ("greedy", "n10"):
        run_dir = str(Path(run_root) / out_subdir / run.task / profile)
        samples_path = str(Path(run_dir) / "eval_only.jsonl")
        metrics_path = samples_path + ".metrics.jsonl"
        out[profile] = EvalArtifact(
            run_id=run.id,
            task=run.task,
            profile=profile,  # type: ignore[assignment]
            run_dir=run_dir,
            metrics_path=metrics_path,
            samples_path=samples_path,
        )
    return out


# -------------------------
# IO helpers
# -------------------------


def _read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception as e:
                raise ValueError(f"bad jsonl: {path}:{ln}: {e}") from e
            if isinstance(obj, dict):
                yield obj


def _read_metrics_jsonl(metrics_path: str) -> Dict[str, Any]:
    """
    metrics.jsonl may contain multiple records; prefer the max(global_step) when available.
    Accepts either:
      {"metrics": {...}, "global_step": ...}
    or:
      {"payload": {"metrics": {...}, "global_step": ...}}
    """
    best_step: Optional[int] = None
    best_metrics: Optional[Dict[str, Any]] = None
    last_metrics: Optional[Dict[str, Any]] = None

    for obj in _read_jsonl(metrics_path):
        metrics: Optional[Dict[str, Any]] = None
        if isinstance(obj.get("metrics"), dict):
            metrics = obj["metrics"]
        elif isinstance(obj.get("payload"), dict) and isinstance(obj["payload"].get("metrics"), dict):
            metrics = obj["payload"]["metrics"]
        if not isinstance(metrics, dict):
            continue

        last_metrics = metrics

        step = obj.get("global_step")
        if step is None and isinstance(obj.get("payload"), dict):
            step = obj["payload"].get("global_step")

        step_i = step if isinstance(step, int) else None
        if step_i is None:
            # Keep last_metrics as fallback.
            continue

        if best_step is None or step_i > best_step:
            best_step = step_i
            best_metrics = metrics

    if best_metrics is not None:
        return best_metrics
    if last_metrics is not None:
        return last_metrics
    raise ValueError(f"no usable metrics in {metrics_path}")


# -------------------------
# metric extraction
# -------------------------


_PASS_KEY_RE = re.compile(r"^eval_(?P<ds>.+)_pass(?P<k>\d+|K)$")


def _extract_pass_metrics(metrics: Mapping[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    Parse eval_*_pass1/pass10/passK style keys into:
      out[datasource]["pass1"]  = ...
      out[datasource]["pass10"] = ...
      out[datasource]["passK"]  = ...
    """
    out: Dict[str, Dict[str, float]] = {}
    for k, v in metrics.items():
        if not isinstance(k, str):
            continue
        m = _PASS_KEY_RE.match(k)
        if not m:
            continue
        ds = m.group("ds")
        kk = m.group("k")
        try:
            fv = _as_float(v)
        except Exception:
            continue
        out.setdefault(ds, {})[f"pass{kk}"] = fv
    return out


def _expected_n_for_profile(profile: ProfileType) -> Optional[int]:
    return 1 if profile == "greedy" else (10 if profile == "n10" else None)


def _compute_from_samples_jsonl(samples_path: str, *, profile: ProfileType, strict: bool) -> Dict[str, DatasourceMetrics]:
    """
    Streaming aggregation over eval_only.jsonl.

    For each (datasource, question_id), aggregate:
      - count K
      - mean over K
      - max over K

    Then:
      Greedy = mean_q( reward )          (K must be 1)
      P@1    = mean_q( mean_K(reward) )  (K must be 10)
      P@10   = mean_q( max_K(reward) )   (K must be 10)
    """
    exp_n = _expected_n_for_profile(profile)

    # stats[(ds,qid)] = [count, sum, max]
    stats: Dict[Tuple[str, str], List[float]] = {}

    for obj in _read_jsonl(samples_path):
        ds = obj.get("datasource")
        qid = obj.get("question_id")
        r = obj.get("answer_reward")

        if not isinstance(ds, str) or not isinstance(qid, (str, int)):
            continue

        try:
            rv = _as_float(r)
        except Exception:
            continue

        key = (ds, str(qid))
        st = stats.get(key)
        if st is None:
            stats[key] = [1.0, rv, rv]
        else:
            st[0] += 1.0
            st[1] += rv
            if rv > st[2]:
                st[2] = rv

    # regroup by datasource
    by_ds: Dict[str, List[Tuple[int, float, float]]] = {}
    bad: List[Tuple[str, str, int]] = []

    for (ds, qid_s), (cnt_f, s, mx) in stats.items():
        cnt = int(cnt_f)
        if exp_n is not None and cnt != exp_n:
            bad.append((ds, qid_s, cnt))
        mean_k = (s / cnt) if cnt > 0 else float("nan")
        by_ds.setdefault(ds, []).append((cnt, mean_k, mx))

    if strict and exp_n is not None and bad:
        preview = ", ".join([f"{ds}:{qid}(n={cnt})" for ds, qid, cnt in bad[:20]])
        raise ValueError(
            f"[samples] {samples_path}: expected {exp_n} samples/question for profile={profile}, "
            f"mismatches (up to 20): {preview}"
        )

    out: Dict[str, DatasourceMetrics] = {}
    for ds, rows in by_ds.items():
        n_q = len(rows)
        mean_of_means = _mean([rk[1] for rk in rows])
        mean_of_max = _mean([rk[2] for rk in rows])
        out[ds] = DatasourceMetrics(
            greedy=mean_of_means,  # for greedy profile this is the scalar reward
            pass1=mean_of_means,   # P@1 definition for n10 is mean of K rewards
            pass10=mean_of_max,    # P@10 definition is max over K
            n_questions=n_q,
        )
    return out


# -------------------------
# LaTeX
# -------------------------


def _format_pm(mean: float, std: float, *, scale: float, digits: int) -> str:
    m = mean * scale
    s = std * scale
    fmt = f"{{:.{digits}f}}"
    return f"{fmt.format(m)}$\\pm${fmt.format(s)}"


def _render_latex_table(*, summary: Mapping[str, Any], out_path: str, scale: float, digits: int) -> None:
    """
    Writes a LaTeX-ready body (rows only).

    Cell format: "Greedy / P@1 / P@10" per benchmark.
    (This avoids baking in a specific paper column layout.)
    """
    lines: List[str] = []
    lines.append(f"% Auto-generated by main_results.py (scale={scale}, digits={digits})")
    lines.append("% Columns: " + " | ".join(TABLE_COLUMNS))

    methods = summary.get("methods", {})
    known = [m for m in METHOD_ORDER if m in methods]
    extras = sorted([m for m in methods.keys() if m not in METHOD_ORDER])
    for method in known + extras:
        mrec = methods.get(method)
        if not isinstance(mrec, dict):
            continue

        row_cells: List[str] = []
        for ds in TABLE_COLUMNS:
            cell = "--"
            ds_rec = mrec.get(ds)
            if isinstance(ds_rec, dict):
                g = ds_rec.get("Greedy")
                p1 = ds_rec.get("P@1")
                p10 = ds_rec.get("P@10")
                if all(isinstance(x, dict) for x in (g, p1, p10)):
                    try:
                        cell = " / ".join(
                            [
                                _format_pm(float(g["mean"]), float(g["std"]), scale=scale, digits=digits),
                                _format_pm(float(p1["mean"]), float(p1["std"]), scale=scale, digits=digits),
                                _format_pm(float(p10["mean"]), float(p10["std"]), scale=scale, digits=digits),
                            ]
                        )
                    except Exception:
                        cell = "--"
            row_cells.append(cell)

        lines.append(f"{method} & " + " & ".join(row_cells) + r" \\")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


# -------------------------
# aggregation
# -------------------------


def _aggregate(
    *,
    registry: str,
    ckpt_root: str,
    out_dir: str,
    out_prefix: str,
    strict: bool,
    prefer_metrics: bool,
    validate_samples_when_strict: bool,
    expected_runs_per_method_task: int,
    out_subdir_override: Optional[str],
    latex_scale: float,
    latex_digits: int,
) -> int:
    defaults, runs = _load_registry(registry)

    # Coverage: per (method, task)
    by_method_task: Dict[Tuple[str, str], List[RunRecord]] = {}
    for r in runs:
        by_method_task.setdefault((r.method, r.task), []).append(r)
    if strict:
        for (m, t), rr in sorted(by_method_task.items()):
            if len(rr) != expected_runs_per_method_task:
                raise ValueError(
                    f"strict coverage: method={m} task={t} has {len(rr)} runs, expected {expected_runs_per_method_task}"
                )

    missing_artifacts: List[Dict[str, Any]] = []
    missing_keys: List[Dict[str, Any]] = []
    raw_rows: List[Dict[str, Any]] = []

    per_run_meta: Dict[str, Dict[str, Any]] = {}

    for run in runs:
        artifacts = _resolve_artifacts(run, defaults=defaults, ckpt_root=ckpt_root, out_subdir_override=out_subdir_override)
        exp_ds = EXPECTED_DATASOURCES[run.task]

        prof_data: Dict[ProfileType, Dict[str, DatasourceMetrics]] = {}

        for profile in ("greedy", "n10"):
            art = artifacts[profile]
            metrics_exists = os.path.isfile(art.metrics_path)
            samples_exists = os.path.isfile(art.samples_path)

            if not metrics_exists and not samples_exists:
                missing_artifacts.append(
                    {
                        "run_id": run.id,
                        "task": run.task,
                        "seed": run.seed,
                        "profile": profile,
                        "metrics_path": art.metrics_path,
                        "samples_path": art.samples_path,
                    }
                )
                continue

            used: Optional[str] = None
            ds_metrics: Dict[str, DatasourceMetrics] = {}

            # Prefer metrics.
            if prefer_metrics and metrics_exists:
                try:
                    m = _read_metrics_jsonl(art.metrics_path)
                    pm = _extract_pass_metrics(m)

                    for ds in exp_ds:
                        drec = pm.get(ds, {})

                        if profile == "greedy":
                            if "pass1" not in drec:
                                missing_keys.append({"path": art.metrics_path, "missing": f"eval_{ds}_pass1"})
                                continue
                            v = float(drec["pass1"])
                            ds_metrics[ds] = DatasourceMetrics(greedy=v, pass1=v, pass10=v, n_questions=-1)
                        else:
                            if "pass1" not in drec:
                                missing_keys.append({"path": art.metrics_path, "missing": f"eval_{ds}_pass1"})
                                continue
                            v1 = float(drec["pass1"])
                            if "pass10" in drec:
                                v10 = float(drec["pass10"])
                            elif "passK" in drec:
                                v10 = float(drec["passK"])
                            else:
                                missing_keys.append({"path": art.metrics_path, "missing": f"eval_{ds}_pass10"})
                                continue
                            ds_metrics[ds] = DatasourceMetrics(greedy=float("nan"), pass1=v1, pass10=v10, n_questions=-1)

                    used = "metrics"
                except Exception as e:
                    missing_keys.append({"path": art.metrics_path, "error": str(e)})
                    used = None
                    ds_metrics = {}

            # Fallback to samples.
            if used is None and samples_exists:
                try:
                    ds_metrics = _compute_from_samples_jsonl(art.samples_path, profile=profile, strict=strict)
                    used = "samples"
                except Exception as e:
                    missing_keys.append({"path": art.samples_path, "error": str(e)})
                    used = None
                    ds_metrics = {}

            if used is None:
                missing_artifacts.append(
                    {
                        "run_id": run.id,
                        "task": run.task,
                        "seed": run.seed,
                        "profile": profile,
                        "detail": "failed to load metrics/samples",
                        "metrics_path": art.metrics_path,
                        "samples_path": art.samples_path,
                    }
                )
                continue

            # In strict mode, if we used metrics and samples exist, also validate sample counts (cheap sanity).
            if strict and validate_samples_when_strict and used == "metrics" and samples_exists:
                _ = _compute_from_samples_jsonl(art.samples_path, profile=profile, strict=True)

            # Enforce presence of expected datasources.
            if strict:
                for ds in exp_ds:
                    if ds not in ds_metrics:
                        missing_artifacts.append(
                            {
                                "run_id": run.id,
                                "task": run.task,
                                "seed": run.seed,
                                "profile": profile,
                                "detail": f"missing datasource {ds} ({used})",
                                "run_dir": art.run_dir,
                            }
                        )

            prof_data[profile] = ds_metrics

        if "greedy" not in prof_data or "n10" not in prof_data:
            continue

        per_run_meta[run.id] = dataclasses.asdict(run)

        for ds in exp_ds:
            g = prof_data["greedy"].get(ds)
            n = prof_data["n10"].get(ds)
            if g is None or n is None:
                continue

            raw_rows.append(
                {
                    "run_id": run.id,
                    "method": run.method,
                    "task": run.task,
                    "seed": run.seed,
                    "datasource": ds,
                    "Greedy": g.greedy,  # greedy profile (pass1 / scalar reward)
                    "P@1": n.pass1,       # n10 profile pass1
                    "P@10": n.pass10,     # n10 profile pass10 (or passK fallback)
                    "n_questions_greedy": g.n_questions,
                    "n_questions_n10": n.n_questions,
                }
            )

    if strict and (missing_artifacts or missing_keys):
        msg = []
        if missing_artifacts:
            msg.append(f"missing_artifacts={len(missing_artifacts)} example={_j(missing_artifacts[0])}")
        if missing_keys:
            msg.append(f"missing_keys={len(missing_keys)} example={_j(missing_keys[0])}")
        raise ValueError("strict aggregation failed: " + " | ".join(msg))

    # Summary: method x datasource x metric -> mean/std/n
    summary: Dict[str, Any] = {
        "meta": {
            "registry": registry,
            "ckpt_root": ckpt_root,
            "out_subdir_override": out_subdir_override,
            "expected_runs_per_method_task": expected_runs_per_method_task,
            "strict": strict,
            "prefer_metrics": prefer_metrics,
        },
        "methods": {},
    }

    # Collect values.
    method_ds_values: Dict[Tuple[str, str], Dict[str, List[float]]] = {}
    for row in raw_rows:
        key = (str(row["method"]), str(row["datasource"]))
        d = method_ds_values.setdefault(key, {})
        d.setdefault("Greedy", []).append(float(row["Greedy"]))
        d.setdefault("P@1", []).append(float(row["P@1"]))
        d.setdefault("P@10", []).append(float(row["P@10"]))

    for (method, ds), d in sorted(method_ds_values.items()):
        mrec = summary["methods"].setdefault(method, {})
        dsrec = mrec.setdefault(ds, {})
        for metric_name in ("Greedy", "P@1", "P@10"):
            vals = d.get(metric_name, [])
            dsrec[metric_name] = {"mean": _mean(vals), "std": _std_sample(vals), "n": len(vals)}

    # Write outputs.
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    raw_path = out_dir_p / f"{out_prefix}.raw.json"
    summary_path = out_dir_p / f"{out_prefix}.summary.json"
    tex_path = out_dir_p / f"{out_prefix}.table.tex"

    raw_payload = {
        "runs": per_run_meta,
        "rows": raw_rows,
        "diagnostics": {"missing_artifacts": missing_artifacts, "missing_keys": missing_keys},
        "contract": {
            "run_root_hf_base": "<ckpt_root>/_runs/_sft_main_results/<id>",
            "artifact_relpath": "<out_subdir>/<task>/<profile>/eval_only.jsonl(.metrics.jsonl)",
            "expected_datasources": EXPECTED_DATASOURCES,
            "table_columns": TABLE_COLUMNS,
        },
    }

    raw_path.write_text(json.dumps(raw_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _render_latex_table(summary=summary, out_path=str(tex_path), scale=float(latex_scale), digits=int(latex_digits))

    eprint(f"[OK] wrote: {raw_path}")
    eprint(f"[OK] wrote: {summary_path}")
    eprint(f"[OK] wrote: {tex_path}")

    if missing_artifacts or missing_keys:
        eprint(f"[WARN] missing_artifacts={len(missing_artifacts)} missing_keys={len(missing_keys)}")
        return 1
    return 0


# -------------------------
# CLI
# -------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="main_results.py", description="Aggregate main results from eval artifacts.")
    sub = p.add_subparsers(dest="cmd", required=True)

    agg = sub.add_parser("aggregate", help="aggregate main results from registry + artifacts")
    agg.add_argument("--registry", required=True, type=str, help="Path to registry YAML")
    agg.add_argument("--ckpt_root", required=True, type=str, help="CKPT root (used for hf_base container dirs)")
    agg.add_argument("--out_dir", required=True, type=str, help="Output directory")
    agg.add_argument("--out_prefix", default="main_results", type=str, help="Output filename prefix")

    agg.add_argument("--strict", default=1, type=int, help="Fail-fast on missing artifacts/keys (default: 1)")
    agg.add_argument("--prefer_metrics", default=1, type=int, help="Prefer *.metrics.jsonl (default: 1)")
    agg.add_argument(
        "--validate_samples_when_strict",
        default=1,
        type=int,
        help="If strict and metrics exist, also validate sample counts when eval_only.jsonl exists (default: 1)",
    )
    agg.add_argument(
        "--expected_runs_per_method_task",
        default=5,
        type=int,
        help="Strict coverage: expected #runs per (method, task) (default: 5)",
    )

    # Global override; normal behavior is per-run out_subdir (then defaults.out_subdir).
    agg.add_argument("--out_subdir", default=None, type=str, help="Override out_subdir for all runs (optional)")

    # LaTeX formatting knobs.
    agg.add_argument("--latex_scale", default=100.0, type=float, help="Scale for LaTeX numbers (default: 100)")
    agg.add_argument("--latex_digits", default=1, type=int, help="Decimal digits for LaTeX numbers (default: 1)")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.cmd == "aggregate":
        return _aggregate(
            registry=args.registry,
            ckpt_root=args.ckpt_root,
            out_dir=args.out_dir,
            out_prefix=args.out_prefix,
            strict=bool(int(args.strict)),
            prefer_metrics=bool(int(args.prefer_metrics)),
            validate_samples_when_strict=bool(int(args.validate_samples_when_strict)),
            expected_runs_per_method_task=int(args.expected_runs_per_method_task),
            out_subdir_override=(str(args.out_subdir).strip() if args.out_subdir else None),
            latex_scale=float(args.latex_scale),
            latex_digits=int(args.latex_digits),
        )

    raise RuntimeError(f"unknown cmd: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
