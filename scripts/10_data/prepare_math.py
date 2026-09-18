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
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import yaml

# The pool builder below and the E1 screening pass have to agree on what the id
# of a prepared row is, so both call the same function instead of each carrying
# its own rule. Importing it needs the repository root on sys.path, the same
# bootstrap scripts/70_rebuild/e1_cells.py uses.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from c3.analysis.rebuild.pool_ids import row_id  # noqa: E402

# The write door asks whether a row carries a problem statement at all. It asks with
# the key table the trainer reads, imported rather than copied, so the two cannot
# drift apart. scripts/10_data/prepare_code.py imports the same table.
from c3.task.datasets import _PROMPT_KEY_CANDIDATES as PROMPT_KEY_CANDIDATES  # noqa: E402

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
    except BaseException:
        # SystemExit from the row gate and KeyboardInterrupt included: no half-written artifact stays
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
    allow_mismatch: bool = False,
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
        if allow_mismatch:
            # A run that re-pins (--update_manifest_sha256 1) rewrites the artifact on purpose;
            # the old pin is reported, not enforced, and the new one lands in the manifest.
            print(f"[INFO] {name}: sha256 pin changes {expected[:12]}... -> {computed_sha[:12]}... (re-pinning)")
            return
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


def _has_problem_statement(row: Dict[str, Any]) -> bool:
    """True when one of the keys a consumer reads as the problem statement is filled."""
    return any(str(row.get(key, "") or "").strip() for key in PROMPT_KEY_CANDIDATES)


def _require_complete_rows(name: str, rows: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    """
    Refuse to write a row without a problem text or without an answer.

    The CMATH artifacts of 2026-09-15 were written with every answer empty (the upstream column
    is `golden`, a name the row builder did not know) and still passed the row-count and sha
    checks. An empty label is a wrong number waiting to happen, so it fails at the one door
    every artifact goes through, not in a warning. The MBPP+ artifact of 2026-09-07 was the
    same mistake on the other side, a whole file of rows with no problem statement, which is
    why the statement is checked against the same key table the trainer reads and not against
    the single name this builder happens to write.
    """
    for i, r in enumerate(rows):
        if not _has_problem_statement(r):
            raise SystemExit(
                f"[FAIL] {name}: row {i} has an empty input (problem text); none of "
                f"{list(PROMPT_KEY_CANDIDATES)} carries one."
            )
        if not str(r.get("answer", "") or "").strip():
            raise SystemExit(
                f"[FAIL] {name}: row {i} has an empty answer; the source columns are probably misnamed "
                f"(row keys: {sorted(r.keys())})."
            )
        yield r


def _check_rows_complete_on_disk(name: str, out_path: Path) -> None:
    """The same gate for an artifact that already exists and is verified instead of written."""
    with out_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not _has_problem_statement(r) or not str(r.get("answer", "") or "").strip():
                raise SystemExit(f"[FAIL] {name}: {out_path} row {i} has an empty input or answer; regenerate it.")
    print(f"[OK] {name}: every row carries an input and an answer.")


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


_RE_BOXED_BARE = re.compile(r"\\boxed\s+([^\s${}\\]+)")


def _extract_boxed_answer(text: str) -> Optional[str]:
    """
    The content of the last \\boxed{...} in a solution, or None.

    MATH solutions also write a one-token box without braces (`\\boxed 2$.`, two rows of the
    training split), which the brace scanner used to miss, leaving those rows without an
    answer. An empty box (`\\boxed{}`, two upstream rows) has no answer to extract and returns
    None; the caller decides what to do with such a row.
    """
    if not text:
        return None
    key = r"\boxed"
    i = text.rfind(key)
    if i < 0:
        return None
    j = i + len(key)
    if j < len(text) and text[j] == "{":
        j += 1
        depth = 1
        out: List[str] = []
        while j < len(text):
            ch = text[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    inner = "".join(out).strip()
                    return inner or None
            out.append(ch)
            j += 1
        return None
    m = _RE_BOXED_BARE.match(text, i)
    if m:
        return m.group(1).rstrip(".,;:") or None
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
# c3/task/datasets.py, so "the question" means the same string here,
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


def _prepare_math_subject_union(spec: Dict[str, Any], *, name: str) -> Iterable[Dict[str, Any]]:
    """
    One MATH split: every config in `source.configs`, concatenated.

    The upstream repository stores one config per subject, so a split of the
    benchmark is the union of the seven subject splits. They are read in manifest
    order and deduplicated, so the output is deterministic.

    `MATH-train` and `MATH-test` differ in `source.split` and in nothing else, so
    they share this builder: the row format, the deduplication key and the answer
    gate cannot drift apart between the two artifacts.
    """
    src = spec["source"]

    # Some mirrors may expose duplicate shards for this dataset.
    # Dedup with a stable key so output stays canonical and deterministic.
    seen: set[str] = set()
    dropped_without_answer = 0

    for cfg in _source_configs(src):
        sub = _source_for_config(src, cfg)
        src_str = _source_string(sub, cfg)
        logical = f"{name}[{cfg}]" if cfg else name

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
            if not str(row.get("answer", "") or "").strip():
                # No label, no reward: a row whose solution ends in an empty \boxed{} cannot be
                # trained on or scored. Dropped here, counted below, and the manifest carries the
                # resulting row count.
                dropped_without_answer += 1
                continue
            yield row
    if dropped_without_answer:
        print(f"[INFO] {name}: dropped {dropped_without_answer} row(s) whose solution has no extractable answer")


def _ascii(text: Any) -> str:
    """A log line that survives any console encoding."""
    return str(text).encode("ascii", "backslashreplace").decode("ascii")


def _statement_preview(row: Dict[str, Any], width: int = 80) -> str:
    """The head of one problem statement with its answer, for a log line about a dropped row."""
    text = normalize_question(row.get("input"))
    if len(text) > width:
        text = text[: width - 3] + "..."
    return _ascii(f"{text} | answer: {row.get('answer')}")


def _dedupe_by_question(name: str, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    `rows` with the second and later copy of a repeated problem statement removed.

    The deduplication inside `_prepare_math_subject_union` keys on `unique_id` when the
    row has one, so a problem upstream ships twice under two ids survives it. This pass
    keys on the normalized statement, which is the key the overlap gate compares, and it
    keeps the first occurrence.
    """
    seen: set[str] = set()
    kept: List[Dict[str, Any]] = []
    repeats: List[Dict[str, Any]] = []
    for row in rows:
        key = question_key(row, ("input",))
        if key and key in seen:
            repeats.append(row)
            continue
        if key:
            seen.add(key)
        kept.append(row)

    for row in repeats:
        print(f"[INFO] {name}: repeated statement dropped: {_statement_preview(row)}")
    print(f"[INFO] {name}: {len(rows)} rows minus {len(repeats)} repeated statement(s) -> {len(kept)} rows")
    return kept


def _prepare_math_train(spec: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """
    Canonical MATH train split: the union of the seven subject train splits, with the
    repeated statements removed and minus every problem that also occurs in the test split.

    `EleutherAI/hendrycks_math` at the pinned revision ships one problem in both the train
    and the test split, and one problem twice inside the train split, so the union as
    shipped is neither internally distinct nor disjoint from `MATH-test`, and the overlap
    gate fails on it. The rule is the one `CMATH-train` and `MBPP-train` already follow:
    subtract by normalized problem statement, the same key the gate compares, so the
    preparation and the gate cannot disagree about what a shared problem is.

    The subtraction is unconditional rather than a manifest field, because there is no
    configuration of this repository in which a MATH training artifact should keep a test
    problem. It builds the test side with this same builder, so what it subtracts is
    exactly what `MATH-test` is prepared as, answer gate included.
    """
    rows = _dedupe_by_question("MATH-train", list(_prepare_math_subject_union(spec, name="MATH-train")))

    test_spec = dict(spec)
    test_spec["source"] = _source_for_split(spec["source"], "test")
    test_rows = list(_prepare_math_subject_union(test_spec, name="MATH-test (read for the subtraction)"))

    drop_keys = {k for k in (question_key(r, ("input",)) for r in test_rows) if k}
    for row in rows:
        if question_key(row, ("input",)) in drop_keys:
            print(f"[INFO] MATH-train: also in MATH-test, dropped: {_statement_preview(row)}")

    kept = subtract_by_question(rows, test_rows, ("input",))
    print(
        f"[INFO] MATH-train: {len(rows)} rows minus {len(rows) - len(kept)} shared with "
        f"the {len(test_rows)} row test split -> {len(kept)} rows"
    )
    return kept


def _prepare_math_test(spec: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """
    Canonical MATH test split, the union of the seven subject test splits.

    It is the set MATH500 is drawn from, so it is an evaluation artifact and the
    overlap gate checks it against every training file like any other.
    """
    return _prepare_math_subject_union(spec, name="MATH-test")


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
        # upstream weitianwen/cmath names the answer column `golden`
        a = _first_non_empty(x.get("answer"), x.get("golden"), x.get("output"), x.get("label"), x.get("target"))
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
# The informative-question pool (MATHPOOL)
# -----------------------------

# The E1 screening pass selects the questions whose alternatives differ enough to
# carry a measurable credit signal. The repository does not redistribute dataset
# rows, so what it ships is the id list of that selection; this builder turns the
# list back into a file by taking those rows out of the prepared MATH-test
# artifact. Until the screening pass has run the list is empty, and then the
# builder writes nothing instead of writing an empty artifact.

POOL_SKIP_MESSAGE = "[SKIP] MATHPOOL: pool_informative_ids.json is empty; run the E1 screening pass first"


def _pool_ids_path(spec: Dict[str, Any]) -> Path:
    """Where the id list lives, from `source.id`, resolved against the repository root."""
    raw = str(spec.get("source", {}).get("id") or "").strip()
    if not raw:
        raise SystemExit("[FAIL] MATH-pool: source.id must name the id list, for example configs/data/pool_informative_ids.json.")
    path = Path(raw)
    return path if path.is_absolute() else REPO_ROOT / path


def load_pool_ids(path: Path) -> List[str]:
    """
    The `unique_ids` of the pool id list, in file order.

    An id that repeats, an empty id, or a `screen.taken` count that disagrees
    with the length of the list is a failure here rather than a surprise in the
    prepared file. An empty list is not a failure; it is the state the repository
    ships in, and the caller turns it into a skip.
    """
    if not path.exists():
        raise SystemExit(f"[FAIL] MATH-pool: id list not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise SystemExit(f"[FAIL] MATH-pool: {path} must hold a JSON object.")

    raw = obj.get("unique_ids", None)
    if not isinstance(raw, list):
        raise SystemExit(f"[FAIL] MATH-pool: {path} has no `unique_ids` list.")

    ids: List[str] = []
    seen: set[str] = set()
    for item in raw:
        value = str(item).strip()
        if not value:
            raise SystemExit(f"[FAIL] MATH-pool: {path} lists an empty id.")
        if value in seen:
            raise SystemExit(f"[FAIL] MATH-pool: {path} lists the id {value!r} twice.")
        seen.add(value)
        ids.append(value)

    screen = obj.get("screen", None)
    taken = screen.get("taken", None) if isinstance(screen, dict) else None
    if taken is not None and int(taken) != len(ids):
        raise SystemExit(f"[FAIL] MATH-pool: {path} says screen.taken={taken} but lists {len(ids)} ids.")

    return ids


def _pool_ids_or_skip(spec: Dict[str, Any]) -> Optional[List[str]]:
    """The pool ids, or None (with the skip line printed) while the list is empty."""
    ids = load_pool_ids(_pool_ids_path(spec))
    if not ids:
        print(POOL_SKIP_MESSAGE)
        return None
    return ids


def index_rows_by_id(path: Path, *, name: str) -> Dict[str, Dict[str, Any]]:
    """Every row of a prepared JSONL file, keyed by `row_id`."""
    out: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = row_id(row)
            if key in out:
                raise SystemExit(f"[FAIL] {name}: {path} line {lineno} repeats the row id {key!r}.")
            out[key] = row
    return out


def _prepare_math_pool(
    spec: Dict[str, Any],
    *,
    ids: List[str],
    out_dir: Path,
    outputs: Dict[str, Dict[str, Any]],
) -> Iterable[Dict[str, Any]]:
    """
    The rows of the parent artifact named by the id list, in the order it gives.

    A missing id is a failure and never a shorter file: the pool is what the
    measurement is defined over, so a silently smaller one would change the
    measurement without saying so.
    """
    parent_name = str(spec["source"].get("parent") or "").strip()
    if parent_name not in outputs:
        raise SystemExit(f"[FAIL] MATH-pool: source.parent={parent_name!r} is not a manifest output entry.")

    parent_path = _resolve_output_path(outputs[parent_name]["output_path"], out_dir)
    if not parent_path.exists():
        raise SystemExit(
            f"[FAIL] MATH-pool: {parent_name} is not prepared yet ({parent_path}). "
            "Prepare it first; the pool is a subset of it."
        )

    by_id = index_rows_by_id(parent_path, name=parent_name)
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise SystemExit(
            f"[FAIL] MATH-pool: {len(missing)} of {len(ids)} ids are not in {parent_name} "
            f"({parent_path}). First few: {missing[:5]}. The id list and the prepared parent "
            "must come from the same revision."
        )

    print(f"[INFO] MATH-pool: {len(ids)} of {len(by_id)} {parent_name} rows selected by the id list")
    for wanted in ids:
        yield by_id[wanted]


# -----------------------------
# Candidate evaluation benchmarks
# -----------------------------

# These five artifacts are evaluation only; nothing trains on them. The main
# task evaluates on four of them (Minerva-Math, AMC23, AIME24, AIME25) since the
# evaluation protocol of 2026-09-15, so they are prepared by default; the fifth,
# OlympiadBench, is prepared with them and is read by the start-accuracy probe
# only. --prepare_candidate_benchmarks turns the group off.

# A manifest policy field still waiting for a decision. The builder refuses to
# run while it holds this value, which is the same rule as the PIN-ME revision
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
                "incomplete, so the maintainers have to decide how context is joined to it before this "
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
    ap.add_argument("--prepare_math_test", type=int, default=1)
    # The pool is a subset of MATH-test named by configs/data/pool_informative_ids.json.
    # While that list is empty this prepares nothing, so the flag can default to 1.
    ap.add_argument("--prepare_math_pool", type=int, default=1)
    ap.add_argument("--prepare_math500", type=int, default=1)
    ap.add_argument("--prepare_cmath", type=int, default=1)
    ap.add_argument("--prepare_gsm8k", type=int, default=1)
    # The candidate benchmarks. On by default since the evaluation protocol of
    # 2026-09-15 made four of them evaluation suites of the main task.
    # --prepare_eval_probe_sets is the former name of this flag, kept as an alias
    # for one release; both spellings write the same destination.
    ap.add_argument("--prepare_candidate_benchmarks", "--prepare_eval_probe_sets",
                    dest="prepare_candidate_benchmarks",
                    type=int, default=1, nargs="?", const=1, choices=[0, 1])

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
                if strict:
                    _check_rows_complete_on_disk(name, out_path)
                _verify_or_record_sha(
                    name=name,
                    out_path=out_path,
                    expected_sha=expected_sha,
                    computed_sha=sha,
                    strict=strict,
                    optional=optional,
                    missing_sha=missing_sha,
                    allow_mismatch=update_manifest_sha256,
                )
                _check_expected_rows(
                    name=name,
                    spec=spec,
                    out_path=out_path,
                    strict=strict,
                    expected_sha=expected_sha,
                )
                return

        n = _write_jsonl_atomic(out_path, _require_complete_rows(name, builder_fn(spec)), overwrite=force_overwrite)
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
            allow_mismatch=update_manifest_sha256,
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
    if args.prepare_math_test:
        _run("MATH-test", _prepare_math_test)
    if args.prepare_math_pool:
        pool_spec = idx.get("MATH-pool")
        if pool_spec is None:
            raise SystemExit("Manifest missing output entry: MATH-pool")
        pool_ids = _pool_ids_or_skip(pool_spec)
        if pool_ids is not None:
            _run("MATH-pool", lambda s: _prepare_math_pool(s, ids=pool_ids, out_dir=out_dir, outputs=idx))
    if args.prepare_math500:
        _run("MATH500", _prepare_math500)
    if args.prepare_cmath:
        _run("CMATH-train", _prepare_cmath)
        _run("CMATH-test", _prepare_cmath)
    if args.prepare_gsm8k:
        _run("GSM8K-test", _prepare_gsm8k)
    if args.prepare_candidate_benchmarks:
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
