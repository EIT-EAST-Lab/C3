#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare math datasets into canonical JSONL files (robust + reproducible)."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import yaml

# The download stack is optional at import time. The pure helpers in this file
# (normalize_question, subtract_by_question, _pick_repo_files, _source_configs)
# carry the split and deduplication rules, so they must stay importable and
# unit-testable on a machine that has no `datasets` installed. Every code path
# that really downloads calls _require() first and fails with the same message
# this module used to raise at import time.
try:
    from datasets import load_dataset  # type: ignore
except Exception:  # pragma: no cover
    load_dataset = None  # type: ignore

try:
    from huggingface_hub import HfApi, hf_hub_download  # type: ignore
except Exception:  # pragma: no cover
    HfApi = None  # type: ignore
    hf_hub_download = None  # type: ignore

try:
    import pyarrow.parquet as pq  # type: ignore
except Exception:  # pragma: no cover
    pq = None  # type: ignore


def _require(obj: Any, package: str) -> Any:
    if obj is None:
        raise SystemExit(f"`{package}` is required. Please install dependencies.")
    return obj


# -----------------------------
# Manifest helpers
# -----------------------------


def _load_manifest(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Manifest not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if not isinstance(obj, dict) or "outputs" not in obj:
        raise SystemExit(f"Invalid manifest format: {path}")
    return obj


def _save_manifest(path: Path, obj: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def _index_outputs(manifest: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    outputs = manifest.get("outputs", [])
    if not isinstance(outputs, list):
        raise SystemExit("Manifest `outputs` must be a list")
    out: Dict[str, Dict[str, Any]] = {}
    for item in outputs:
        if isinstance(item, dict) and isinstance(item.get("name"), str):
            out[item["name"]] = item
    return out


_HEX_RE = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)


def _is_pinned_revision(rev: Optional[str]) -> bool:
    if rev is None:
        return False
    r = str(rev).strip()
    if r.lower() in {"main", "master", "head", "latest", "default"}:
        return False
    return bool(_HEX_RE.match(r))


def _resolve_output_path(output_path: str, out_dir: Path) -> Path:
    p = Path(output_path)
    if len(p.parts) >= 1 and p.parts[0] == "data":
        return out_dir / Path(*p.parts[1:])
    return out_dir / p


# -----------------------------
# IO helpers
# -----------------------------


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _count_jsonl_rows(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _write_jsonl_atomic(path: Path, rows: Iterable[Dict[str, Any]], *, overwrite: bool) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return -1

    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
                n += 1
        tmp.replace(path)
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        raise
    return n


def _verify_or_record_sha(
    *,
    name: str,
    out_path: Path,
    expected_sha: Optional[str],
    computed_sha: str,
    strict: bool,
    optional: bool,
    missing_sha: List[Tuple[str, str]],
) -> None:
    # An entry whose pin was deliberately cleared (null, or an empty string)
    # counts as unpinned, not as a mismatch against the empty string.
    if not _has_sha_pin(expected_sha):
        missing_sha.append((name, computed_sha))
        if strict and not optional:
            return
        return

    expected = str(expected_sha).strip().lower()
    if expected != computed_sha.lower():
        raise SystemExit(
            "\n".join(
                [
                    "[FAIL] sha256 mismatch for prepared artifact:",
                    f"  name       : {name}",
                    f"  path       : {out_path}",
                    f"  expected   : {expected}",
                    f"  computed   : {computed_sha}",
                ]
            )
        )


def _has_sha_pin(expected_sha: Any) -> bool:
    return expected_sha is not None and bool(str(expected_sha).strip())


def _check_expected_rows(
    *,
    name: str,
    spec: Dict[str, Any],
    out_path: Path,
    strict: bool,
    expected_sha: Any,
) -> None:
    """
    Row-count guard for a manifest entry that carries `expected_rows`.

    A pinned sha256 is the stronger check, so the count is only verified while
    the pin is still empty, which is exactly the window after an upstream
    revision moved and a wrong split would otherwise pass unnoticed.
    """
    expected_rows = spec.get("expected_rows", None)
    if expected_rows is None or not strict or _has_sha_pin(expected_sha):
        return

    try:
        want = int(expected_rows)
    except Exception:
        raise SystemExit(f"[FAIL] {name}: expected_rows must be an integer, got {expected_rows!r}")

    got = _count_jsonl_rows(out_path)
    if got != want:
        raise SystemExit(
            "\n".join(
                [
                    "[FAIL] row count mismatch for prepared artifact:",
                    f"  name       : {name}",
                    f"  path       : {out_path}",
                    f"  expected   : {want}",
                    f"  computed   : {got}",
                ]
            )
        )
    print(f"[OK] {name}: {got} rows match expected_rows.")


# -----------------------------
# HF robust loading helpers
# -----------------------------


def _norm_cfg(cfg: Any) -> Optional[str]:
    if cfg is None:
        return None
    s = str(cfg).strip()
    if not s or s.lower() in {"none", "null", "default"}:
        return None
    return s


def _source_configs(src: Dict[str, Any]) -> List[Optional[str]]:
    """
    Configs to load for one manifest entry, in manifest order.

    `configs: [a, b, ...]` means "load each of these configs and concatenate";
    it is how the canonical MATH train split is assembled from its seven subject
    configs. A single `config:` keeps the historical one-config behaviour.
    """
    raw = src.get("configs", None)
    if raw is None:
        return [_norm_cfg(src.get("config"))]

    if not isinstance(raw, list) or not raw:
        raise SystemExit("[FAIL] source.configs must be a non-empty list of config names.")

    out: List[Optional[str]] = []
    for item in raw:
        cfg = _norm_cfg(item)
        if cfg is None:
            raise SystemExit(f"[FAIL] source.configs holds an unusable config name: {item!r}")
        if cfg in out:
            raise SystemExit(f"[FAIL] source.configs lists {cfg!r} twice.")
        out.append(cfg)
    return out


def _source_for_config(src: Dict[str, Any], cfg: Optional[str]) -> Dict[str, Any]:
    """One single-config view of a (possibly multi-config) source block."""
    sub = dict(src)
    sub.pop("configs", None)
    sub["config"] = cfg
    return sub


def _source_for_split(src: Dict[str, Any], split: str) -> Dict[str, Any]:
    """Same repo and revision, another split (used by the subtraction rule)."""
    sub = dict(src)
    sub.pop("subtract_split", None)
    sub["split"] = str(split)
    return sub


def _source_string(src: Dict[str, Any], cfg: Optional[str] = None) -> str:
    if cfg:
        return f"{src['id']}:{cfg}:{src['split']}@{src.get('revision')}"
    return f"{src['id']}:{src['split']}@{src.get('revision')}"


def _load_dataset_rows(repo_id: str, cfg: Optional[str], split: str, rev: Optional[str]) -> List[Dict[str, Any]]:
    _require(load_dataset, "datasets")
    if cfg:
        ds = load_dataset(repo_id, cfg, split=split, revision=rev)
    else:
        ds = load_dataset(repo_id, split=split, revision=rev)
    return [dict(x) for x in ds]


def _known_data_file(path: str) -> bool:
    p = path.lower()
    return p.endswith(".parquet") or p.endswith(".jsonl") or p.endswith(".jsonl.gz") or p.endswith(".json") or p.endswith(".gz")


def _split_hit(path: str, split: str) -> bool:
    p = path.lower()
    s = split.lower()
    base = p.split("/")[-1]
    return (
        base.startswith(f"{s}-")
        or base == f"{s}.parquet"
        or base == f"{s}.jsonl"
        or base == f"{s}.jsonl.gz"
        or f"/{s}/" in p
        or f"_{s}_" in base
        or f"-{s}-" in base
    )


def _pick_repo_files(files: List[str], *, split: str, cfg: Optional[str]) -> List[str]:
    """
    Repository files that hold `split`, or nothing at all.

    A requested split that no file matches returns an empty list, and the caller
    turns that into a hard failure. It used to fall back to `data/*` and then to
    every data file in the repository, which silently prepared some other split
    under the requested name: that is how a CMATH revision without a test split
    produced a `test.jsonl` holding the validation rows.
    """
    cands = sorted([f for f in files if _known_data_file(f)])
    if not cands:
        return []

    if cfg:
        cfg_hits = [f for f in cands if f.startswith(f"{cfg}/") or f"/{cfg}/" in f or f"_{cfg}_" in f.lower()]
        if cfg_hits:
            cands = cfg_hits

    return sorted([f for f in cands if _split_hit(f, split)])


def _iter_rows_from_repo_files(repo_id: str, revision: str, split: str, cfg: Optional[str]) -> Iterator[Dict[str, Any]]:
    _require(HfApi, "huggingface_hub")
    endpoint = os.environ.get("HF_ENDPOINT", "").strip() or None
    api = HfApi(endpoint=endpoint)
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision)
    selected = _pick_repo_files(files, split=split, cfg=cfg)
    if not selected:
        raise SystemExit(
            f"[FAIL] {repo_id}@{revision}: no data file matches split='{split}'"
            + (f" (config='{cfg}')" if cfg else "")
            + ". Repin the revision to one that really has this split; "
            "this loader never substitutes another split."
        )

    for fn in selected:
        local = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=fn, revision=revision)
        low = fn.lower()

        if low.endswith(".parquet"):
            _require(pq, "pyarrow")
            table = pq.read_table(local)
            if "__index_level_0__" in table.column_names:
                table = table.drop(["__index_level_0__"])
            for batch in table.to_batches(max_chunksize=4096):
                for row in batch.to_pylist():
                    if isinstance(row, dict):
                        yield row
            continue

        if low.endswith(".jsonl"):
            with open(local, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        yield obj
            continue

        if low.endswith(".jsonl.gz") or low.endswith(".gz"):
            with gzip.open(local, "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(obj, dict):
                        yield obj
            continue

        if low.endswith(".json"):
            with open(local, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, list):
                for x in obj:
                    if isinstance(x, dict):
                        yield x
            elif isinstance(obj, dict):
                # Common shapes: {"train": [...], "test": [...]}, or a single dict.
                val = obj.get(split)
                if isinstance(val, list):
                    for x in val:
                        if isinstance(x, dict):
                            yield x
                elif isinstance(val, dict):
                    yield val
                else:
                    yield obj


def _iter_hf_rows(src: Dict[str, Any], *, logical_name: str) -> Iterator[Dict[str, Any]]:
    repo_id = str(src["id"])
    split = str(src["split"])
    rev = src.get("revision")
    cfg = _norm_cfg(src.get("config"))

    first_err: Optional[Exception] = None
    try:
        rows = _load_dataset_rows(repo_id, cfg, split, rev)
        for r in rows:
            yield r
        return
    except Exception as e:
        first_err = e
        print(f"[WARN] {logical_name}: load_dataset failed ({type(e).__name__}); trying repo-file fallback.")

    try:
        for row in _iter_rows_from_repo_files(repo_id=repo_id, revision=str(rev), split=split, cfg=cfg):
            yield row
        return
    except Exception as e2:
        raise SystemExit(
            "\n".join(
                [
                    f"[FAIL] {logical_name}: both primary and fallback loaders failed.",
                    f"  primary:  {type(first_err).__name__}: {first_err}" if first_err else "  primary:  <not-run>",
                    f"  fallback: {type(e2).__name__}: {e2}",
                ]
            )
        )


# -----------------------------
# Canonicalization
# -----------------------------


def _extract_boxed_answer(text: str) -> Optional[str]:
    if not text:
        return None
    key = r"\boxed{"
    i = text.rfind(key)
    if i < 0:
        return None
    j = i + len(key)
    depth = 1
    out: List[str] = []
    while j < len(text):
        ch = text[j]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out).strip()
        out.append(ch)
        j += 1
    return None


def _first_non_empty(*vals: Any) -> str:
    for v in vals:
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return ""


def _canon_math_row(
    *,
    problem: str,
    answer: str,
    solution: Optional[str] = None,
    subject: Optional[str] = None,
    level: Optional[Any] = None,
    unique_id: Optional[str] = None,
    source: str,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "input": str(problem).strip(),
        "answer": str(answer).strip(),
        "source": source,
    }
    if solution is not None:
        row["solution"] = str(solution)
    if subject is not None:
        row["subject"] = str(subject)
    if level is not None:
        try:
            row["level"] = int(level)
        except Exception:
            row["level"] = str(level)
    if unique_id is not None:
        row["unique_id"] = str(unique_id)
    return row


# -----------------------------
# Question-level set operations
# -----------------------------

# Prepared rows carry the problem statement under `input`; the raw upstream
# rows use the names below. Same order as _PROMPT_KEY_CANDIDATES in
# c3/integration/task_datasets.py, so "the question" means the same string here,
# in the overlap gate and in what the trainer feeds the model.
QUESTION_KEY_CANDIDATES: Tuple[str, ...] = ("input", "question", "problem", "prompt", "text")


def normalize_question(text: Any) -> str:
    """Whitespace-normal form of a problem statement: trimmed, runs collapsed."""
    return " ".join(str(text or "").split())


def question_key(row: Dict[str, Any], key_fields: Iterable[str] = QUESTION_KEY_CANDIDATES) -> str:
    """Normalized problem statement of one row, or '' when the row has none."""
    for field in key_fields:
        if field in row:
            key = normalize_question(row.get(field))
            if key:
                return key
    return ""


def subtract_by_question(
    rows: Iterable[Dict[str, Any]],
    other_rows: Iterable[Dict[str, Any]],
    key_fields: Iterable[str] = QUESTION_KEY_CANDIDATES,
) -> List[Dict[str, Any]]:
    """
    `rows` minus every row whose problem statement also occurs in `other_rows`.

    Comparison is on the whitespace-normalized statement only, never on answers
    or ids, because the same problem can be stored with a different id in two
    splits. Order of the surviving rows is preserved. A row with no usable
    statement cannot be matched, so it is kept.
    """
    fields = tuple(key_fields)
    drop = {k for k in (question_key(r, fields) for r in other_rows) if k}
    return [r for r in rows if question_key(r, fields) not in drop]


# -----------------------------
# Dataset builders
# -----------------------------


def _prepare_math_train(spec: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """
    Canonical MATH train split: every config in `source.configs`, concatenated.

    The upstream repository stores one config per subject, so the train split of
    the benchmark is the union of the seven subject train splits. They are read
    in manifest order and deduplicated, so the output is deterministic.
    """
    src = spec["source"]

    # Some mirrors may expose duplicate shards for this dataset.
    # Dedup with a stable key so output stays canonical and deterministic.
    seen: set[str] = set()

    for cfg in _source_configs(src):
        sub = _source_for_config(src, cfg)
        src_str = _source_string(sub, cfg)
        logical = f"MATH-train[{cfg}]" if cfg else "MATH-train"

        for x in _iter_hf_rows(sub, logical_name=logical):
            sol = x.get("solution")
            ans = _first_non_empty(
                x.get("answer"),
                x.get("final_answer"),
                x.get("target"),
                x.get("label"),
                _extract_boxed_answer(str(sol or "")),
            )
            # Upstream calls the subject column `type`; the canonical row calls
            # it `subject`.
            subject = x.get("subject")
            if subject is None:
                subject = x.get("type")
            row = _canon_math_row(
                problem=_first_non_empty(x.get("problem"), x.get("question"), x.get("input"), x.get("prompt")),
                answer=ans,
                solution=sol,
                subject=subject,
                level=x.get("level"),
                unique_id=x.get("unique_id") or x.get("id") or x.get("uid"),
                source=src_str,
            )
            key = str(row.get("unique_id") or "").strip()
            if not key:
                key = "␟".join([
                    str(row.get("input", "")).strip(),
                    str(row.get("answer", "")).strip(),
                    str(row.get("solution", "")).strip(),
                ])
            if key in seen:
                continue
            seen.add(key)
            yield row


def _prepare_math500(spec: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    src = spec["source"]
    src_str = f"{src['id']}:{src['split']}@{src.get('revision')}"
    for x in _iter_hf_rows(src, logical_name="MATH500"):
        sol = x.get("solution")
        ans = _first_non_empty(x.get("answer"), x.get("final_answer"), x.get("target"), _extract_boxed_answer(str(sol or "")))
        yield _canon_math_row(
            problem=_first_non_empty(x.get("problem"), x.get("question"), x.get("input"), x.get("prompt")),
            answer=ans,
            solution=sol,
            subject=x.get("subject"),
            level=x.get("level"),
            unique_id=x.get("unique_id") or x.get("id") or x.get("uid"),
            source=src_str,
        )


def _cmath_rows(src: Dict[str, Any]) -> List[Dict[str, Any]]:
    src_str = _source_string(src)
    out: List[Dict[str, Any]] = []
    for x in _iter_hf_rows(src, logical_name=f"CMATH-{src.get('split')}"):
        q = _first_non_empty(x.get("question"), x.get("problem"), x.get("input"), x.get("prompt"))
        a = _first_non_empty(x.get("answer"), x.get("output"), x.get("label"), x.get("target"))
        r: Dict[str, Any] = {"input": q, "answer": a, "source": src_str}
        for k in ("id", "uid", "grade", "difficulty", "subject", "type"):
            if k in x and x.get(k) is not None:
                r[k] = x.get(k)
        out.append(r)
    return out


def _prepare_cmath(spec: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """
    One CMATH split, optionally minus another split of the same revision.

    `source.subtract_split` is what keeps the training artifact disjoint from
    the evaluation artifact: upstream ships a validation split and a test split
    that share a few problems, so the training artifact is validation minus the
    problems that also occur in test.
    """
    src = spec["source"]
    rows = _cmath_rows(src)

    subtract = str(src.get("subtract_split") or "").strip()
    if subtract:
        if subtract == str(src.get("split") or "").strip():
            raise SystemExit(f"[FAIL] CMATH: source.subtract_split equals source.split ({subtract!r}).")
        other = _cmath_rows(_source_for_split(src, subtract))
        kept = subtract_by_question(rows, other, ("input",))
        print(
            f"[INFO] CMATH-{src.get('split')}: {len(rows)} rows minus "
            f"{len(rows) - len(kept)} shared with split='{subtract}' -> {len(kept)} rows"
        )
        rows = kept

    return rows


def _prepare_gsm8k(spec: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    src = spec["source"]
    cfg = _norm_cfg(src.get("config"))
    src_str = f"{src['id']}:{cfg}:{src['split']}@{src.get('revision')}" if cfg else f"{src['id']}:{src['split']}@{src.get('revision')}"

    for x in _iter_hf_rows(src, logical_name="GSM8K-test"):
        q = x.get("question", "") or ""
        a = x.get("answer", "") or ""
        yield {"input": str(q).strip(), "answer": str(a).strip(), "source": src_str}


# -----------------------------
# Candidate evaluation benchmarks (start-accuracy probe)
# -----------------------------

# These five artifacts exist so the start accuracy of a frozen policy can be
# measured on benchmarks that may replace the ones with no headroom left. They
# are evaluation only, nothing trains on them, and they are prepared only when
# --prepare_eval_probe_sets 1 is passed, so the default preparation run and the
# release checks are unchanged.

# A manifest policy field still waiting for a ruling. The builder refuses to run
# while it holds this value, which is the same contract as the PIN-ME revision
# placeholder: an unanswered question fails loudly instead of picking a default.
POLICY_UNSET = "DECIDE-ME"

# What the prepared answer of an OlympiadBench row with several answers is.
OLYMPIAD_MULTI_ANSWER_POLICIES = ("drop", "first", "join_comma")

# What the prepared answer of an OlympiadBench row that carries a unit is.
OLYMPIAD_UNIT_POLICIES = ("drop", "ignore", "append")


def _read_policy(src: Dict[str, Any], key: str, allowed: Tuple[str, ...], *, name: str) -> str:
    """One manifest policy field, or a hard failure naming the choices."""
    raw = src.get(key, None)
    value = "" if raw is None else str(raw).strip()
    if not value or value == POLICY_UNSET:
        raise SystemExit(
            f"[FAIL] {name}: source.{key} is {raw!r}. Set it to one of {list(allowed)} in "
            "configs/data_manifest.yaml. This builder never picks a default, because the "
            "choice changes both the answers and the row count of the prepared file."
        )
    if value not in allowed:
        raise SystemExit(f"[FAIL] {name}: source.{key}={value!r} is not one of {list(allowed)}.")
    return value


def _answer_list(raw: Any) -> List[str]:
    """`final_answer` as a list of non-empty strings, whatever shape it arrives in."""
    if raw is None:
        return []
    if isinstance(raw, str):
        s = raw.strip()
        return [s] if s else []
    if isinstance(raw, (list, tuple)):
        out: List[str] = []
        for item in raw:
            if item is None:
                continue
            s = str(item).strip()
            if s:
                out.append(s)
        return out
    s = str(raw).strip()
    return [s] if s else []


def _joined_solution(raw: Any) -> Optional[str]:
    """A solution column that upstream stores as a list of strings, as one string."""
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        parts = [str(x) for x in raw if x is not None and str(x).strip()]
        return "\n\n".join(parts) if parts else None
    s = str(raw)
    return s if s.strip() else None


def _require_scorable(*, name: str, problem: str, answer: str, row_id: Any) -> None:
    """A row the reward path could never score is a preparation failure, not a row."""
    if not str(problem).strip():
        raise SystemExit(f"[FAIL] {name}: row id={row_id!r} has an empty problem statement.")
    if not str(answer).strip():
        raise SystemExit(
            f"[FAIL] {name}: row id={row_id!r} has an empty answer. The math reward compares the "
            "last \\boxed{} of the response against this field, so an empty one can never be "
            "scored. Fix the field mapping or repin the revision."
        )


def _prepare_minerva(spec: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """Minerva Math: upstream ships exactly two string columns, question and answer."""
    src = spec["source"]
    src_str = _source_string(src)

    for x in _iter_hf_rows(src, logical_name="Minerva-Math"):
        problem = _first_non_empty(x.get("question"), x.get("problem"), x.get("input"), x.get("prompt"))
        answer = _first_non_empty(x.get("answer"), x.get("final_answer"), x.get("target"))
        _require_scorable(name="Minerva-Math", problem=problem, answer=answer, row_id=x.get("id"))
        yield _canon_math_row(problem=problem, answer=answer, source=src_str)


def _prepare_olympiadbench(spec: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """
    OlympiadBench, open-ended text-only English maths.

    Upstream keeps the answer in `final_answer`, a list of strings, and keeps any
    unit out of it in a separate `unit` column. The reward path compares one
    string, so a row with several answers and a row that carries a unit each need
    a rule, and both rules are read from the manifest rather than assumed.
    """
    src = spec["source"]
    src_str = _source_string(src)
    multi_policy = _read_policy(src, "multi_answer_policy", OLYMPIAD_MULTI_ANSWER_POLICIES, name="OlympiadBench")
    unit_policy = _read_policy(src, "unit_policy", OLYMPIAD_UNIT_POLICIES, name="OlympiadBench")

    kept = 0
    dropped_multi = 0
    dropped_unit = 0

    for x in _iter_hf_rows(src, logical_name="OlympiadBench"):
        row_id = x.get("id")

        context = str(x.get("context") or "").strip()
        if context:
            raise SystemExit(
                f"[FAIL] OlympiadBench: row id={row_id!r} carries a non-empty `context`, which the "
                "674 rows of the pinned revision never do. The question alone may then be "
                "incomplete, so the driver has to rule on how context is joined to it before this "
                "file can be prepared."
            )

        answers = _answer_list(x.get("final_answer"))
        if not answers:
            raise SystemExit(f"[FAIL] OlympiadBench: row id={row_id!r} has an empty final_answer.")

        if len(answers) > 1:
            if multi_policy == "drop":
                dropped_multi += 1
                continue
            answer = answers[0] if multi_policy == "first" else ", ".join(answers)
        else:
            answer = answers[0]

        unit = str(x.get("unit") or "").strip()
        if unit:
            if unit_policy == "drop":
                dropped_unit += 1
                continue
            if unit_policy == "append":
                answer = f"{answer} {unit}"

        problem = _first_non_empty(x.get("question"), x.get("problem"), x.get("input"), x.get("prompt"))
        _require_scorable(name="OlympiadBench", problem=problem, answer=answer, row_id=row_id)

        kept += 1
        yield _canon_math_row(
            problem=problem,
            answer=answer,
            solution=_joined_solution(x.get("solution")),
            subject=x.get("subfield"),
            level=x.get("difficulty"),
            unique_id=row_id,
            source=src_str,
        )

    print(
        f"[INFO] OlympiadBench: kept {kept} rows "
        f"(multi_answer_policy={multi_policy} dropped {dropped_multi}, "
        f"unit_policy={unit_policy} dropped {dropped_unit})"
    )


def _prepare_amc23(spec: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """AMC 2023: upstream columns are id, question, answer and url."""
    src = spec["source"]
    src_str = _source_string(src)

    for x in _iter_hf_rows(src, logical_name="AMC23"):
        problem = _first_non_empty(x.get("question"), x.get("problem"), x.get("input"), x.get("prompt"))
        answer = _first_non_empty(x.get("answer"), x.get("final_answer"), x.get("target"))
        _require_scorable(name="AMC23", problem=problem, answer=answer, row_id=x.get("id"))
        yield _canon_math_row(
            problem=problem,
            answer=answer,
            unique_id=x.get("id"),
            source=src_str,
        )


def _prepare_aime24(spec: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """
    AIME 2024: upstream has no answer column at all.

    Its columns are id, problem, solution and url, and the solution of the pinned
    revision is the bare string `\\boxed{<answer>}`, so the answer is whatever the
    last box holds. An empty extraction is a failure, never an empty answer field.
    """
    src = spec["source"]
    src_str = _source_string(src)

    for x in _iter_hf_rows(src, logical_name="AIME24"):
        solution = x.get("solution")
        problem = _first_non_empty(x.get("problem"), x.get("question"), x.get("input"), x.get("prompt"))
        answer = _first_non_empty(
            x.get("answer"),
            x.get("final_answer"),
            _extract_boxed_answer(str(solution or "")),
        )
        _require_scorable(name="AIME24", problem=problem, answer=answer, row_id=x.get("id"))
        yield _canon_math_row(
            problem=problem,
            answer=answer,
            solution=solution,
            unique_id=x.get("id"),
            source=src_str,
        )


def _prepare_aime25(spec: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """AIME 2025: upstream columns are problem, answer and id."""
    src = spec["source"]
    src_str = _source_string(src)

    for x in _iter_hf_rows(src, logical_name="AIME25"):
        problem = _first_non_empty(x.get("problem"), x.get("question"), x.get("input"), x.get("prompt"))
        answer = _first_non_empty(x.get("answer"), x.get("final_answer"), x.get("target"))
        _require_scorable(name="AIME25", problem=problem, answer=answer, row_id=x.get("id"))
        yield _canon_math_row(
            problem=problem,
            answer=answer,
            unique_id=x.get("id"),
            source=src_str,
        )


# Manifest name and builder of every candidate benchmark, in preparation order.
EVAL_PROBE_OUTPUTS = (
    ("Minerva-Math", _prepare_minerva),
    ("OlympiadBench", _prepare_olympiadbench),
    ("AMC23", _prepare_amc23),
    ("AIME24", _prepare_aime24),
    ("AIME25", _prepare_aime25),
)


# -----------------------------
# Main
# -----------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, default="configs/data_manifest.yaml", help="Path to data manifest")
    ap.add_argument("--out_dir", type=str, default="", help="Output root override")
    ap.add_argument("--data_dir", type=str, default="", help="Alias of --out_dir")
    ap.add_argument("--overwrite", type=int, default=0, nargs="?", const=1, choices=[0, 1])
    ap.add_argument("--strict", type=int, default=0, nargs="?", const=1, choices=[0, 1])
    ap.add_argument("--update_manifest_sha256", type=int, default=0, nargs="?", const=1, choices=[0, 1])

    ap.add_argument("--prepare_math_train", type=int, default=1)
    ap.add_argument("--prepare_math500", type=int, default=1)
    ap.add_argument("--prepare_cmath", type=int, default=1)
    ap.add_argument("--prepare_gsm8k", type=int, default=1)
    # Candidate benchmarks for the start-accuracy probe. Off by default, so the
    # default preparation run and the release checks stay exactly as they were.
    ap.add_argument("--prepare_eval_probe_sets", type=int, default=0, nargs="?", const=1, choices=[0, 1])

    args = ap.parse_args()

    out_dir = Path((args.out_dir or args.data_dir or "data").strip()).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    overwrite = bool(args.overwrite)
    strict = bool(args.strict)
    update_manifest_sha256 = bool(args.update_manifest_sha256)

    manifest_path = Path(args.manifest).resolve()
    manifest = _load_manifest(manifest_path)
    idx = _index_outputs(manifest)

    missing_sha: List[Tuple[str, str]] = []
    computed_by_name: Dict[str, str] = {}

    def _run(name: str, builder_fn) -> None:
        if name not in idx:
            raise SystemExit(f"Manifest missing output entry: {name}")
        spec = idx[name]
        src = spec.get("source", {})
        optional = bool(spec.get("optional", False))

        if strict and src.get("kind") == "huggingface" and not _is_pinned_revision(src.get("revision")):
            raise SystemExit(
                f"[FAIL] {name}: source.revision must be an immutable HF commit SHA (not '{src.get('revision')}')."
            )

        out_path = _resolve_output_path(spec["output_path"], out_dir)
        expected_sha = spec.get("sha256", None)

        force_overwrite = overwrite
        if out_path.exists() and not overwrite:
            if out_path.stat().st_size == 0:
                print(f"[WARN] {out_path} is empty (0 bytes); regenerating.")
                force_overwrite = True
            else:
                sha = _sha256(out_path)
                computed_by_name[name] = sha
                print(f"[SKIP] {out_path} exists (sha256={sha})")
                _verify_or_record_sha(
                    name=name,
                    out_path=out_path,
                    expected_sha=expected_sha,
                    computed_sha=sha,
                    strict=strict,
                    optional=optional,
                    missing_sha=missing_sha,
                )
                _check_expected_rows(
                    name=name,
                    spec=spec,
                    out_path=out_path,
                    strict=strict,
                    expected_sha=expected_sha,
                )
                return

        n = _write_jsonl_atomic(out_path, builder_fn(spec), overwrite=force_overwrite)
        sha = _sha256(out_path)
        computed_by_name[name] = sha
        print(f"[OK] wrote {n if n >= 0 else '?'} rows -> {out_path} (sha256={sha})")
        _verify_or_record_sha(
            name=name,
            out_path=out_path,
            expected_sha=expected_sha,
            computed_sha=sha,
            strict=strict,
            optional=optional,
            missing_sha=missing_sha,
        )
        _check_expected_rows(
            name=name,
            spec=spec,
            out_path=out_path,
            strict=strict,
            expected_sha=expected_sha,
        )

    if args.prepare_math_train:
        _run("MATH-train", _prepare_math_train)
    if args.prepare_math500:
        _run("MATH500", _prepare_math500)
    if args.prepare_cmath:
        _run("CMATH-train", _prepare_cmath)
        _run("CMATH-test", _prepare_cmath)
    if args.prepare_gsm8k:
        _run("GSM8K-test", _prepare_gsm8k)
    if args.prepare_eval_probe_sets:
        for probe_name, probe_builder in EVAL_PROBE_OUTPUTS:
            _run(probe_name, probe_builder)

    if update_manifest_sha256:
        for item in manifest.get("outputs", []):
            if not isinstance(item, dict):
                continue
            n = item.get("name")
            if isinstance(n, str) and n in computed_by_name:
                item["sha256"] = computed_by_name[n]
        _save_manifest(manifest_path, manifest)
        print(f"[OK] updated sha256 pins in {manifest_path}")

    if strict:
        missing_required = [(n, s) for (n, s) in missing_sha if not bool(idx.get(n, {}).get("optional", False))]
        if missing_required:
            lines = [
                "[FAIL] Missing sha256 pins in configs/data_manifest.yaml for prepared artifacts.",
                "\nRun once with --update_manifest_sha256 1, then commit the manifest.",
                "",
            ]
            for n, s in missing_required:
                lines.append(f"  - name: {n}\n    sha256: {s}")
            raise SystemExit("\n".join(lines))

    print("[DONE] math dataset preparation complete.")


if __name__ == "__main__":
    main()
