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

try:
    from datasets import load_dataset  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit("`datasets` is required. Please install dependencies.") from e

try:
    from huggingface_hub import HfApi, hf_hub_download  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit("`huggingface_hub` is required. Please install dependencies.") from e

try:
    import pyarrow.parquet as pq  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit("`pyarrow` is required. Please install dependencies.") from e


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
    if expected_sha is None:
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


def _load_dataset_rows(repo_id: str, cfg: Optional[str], split: str, rev: Optional[str]) -> List[Dict[str, Any]]:
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
    cands = sorted([f for f in files if _known_data_file(f)])
    if not cands:
        return []

    if cfg:
        cfg_hits = [f for f in cands if f.startswith(f"{cfg}/") or f"/{cfg}/" in f or f"_{cfg}_" in f.lower()]
        if cfg_hits:
            cands = cfg_hits

    split_hits = [f for f in cands if _split_hit(f, split)]
    if split_hits:
        return sorted(split_hits)

    data_hits = [f for f in cands if f.startswith("data/")]
    if data_hits:
        return sorted(data_hits)

    return cands


def _iter_rows_from_repo_files(repo_id: str, revision: str, split: str, cfg: Optional[str]) -> Iterator[Dict[str, Any]]:
    endpoint = os.environ.get("HF_ENDPOINT", "").strip() or None
    api = HfApi(endpoint=endpoint)
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision)
    selected = _pick_repo_files(files, split=split, cfg=cfg)
    if not selected:
        raise SystemExit(f"[FAIL] {repo_id}@{revision}: no usable data files found for split='{split}'.")

    for fn in selected:
        local = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=fn, revision=revision)
        low = fn.lower()

        if low.endswith(".parquet"):
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
                # 常见结构: {"train": [...], "test": [...]} 或单条 dict
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
# Dataset builders
# -----------------------------


def _prepare_math_train(spec: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    src = spec["source"]
    src_str = f"{src['id']}:{src['split']}@{src.get('revision')}"

    # Some mirrors may expose duplicate shards for this dataset.
    # Dedup with a stable key so output stays canonical and deterministic.
    seen: set[str] = set()

    for x in _iter_hf_rows(src, logical_name="MATH-train"):
        sol = x.get("solution")
        ans = _first_non_empty(
            x.get("answer"),
            x.get("final_answer"),
            x.get("target"),
            x.get("label"),
            _extract_boxed_answer(str(sol or "")),
        )
        row = _canon_math_row(
            problem=_first_non_empty(x.get("problem"), x.get("question"), x.get("input"), x.get("prompt")),
            answer=ans,
            solution=sol,
            subject=x.get("subject"),
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


def _prepare_cmath(spec: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    src = spec["source"]
    src_str = f"{src['id']}:{src['split']}@{src.get('revision')}"

    for x in _iter_hf_rows(src, logical_name=f"CMATH-{src.get('split')}"):
        q = _first_non_empty(x.get("question"), x.get("problem"), x.get("input"), x.get("prompt"))
        a = _first_non_empty(x.get("answer"), x.get("output"), x.get("label"), x.get("target"))
        r: Dict[str, Any] = {"input": q, "answer": a, "source": src_str}
        for k in ("id", "uid", "grade", "difficulty", "subject", "type"):
            if k in x and x.get(k) is not None:
                r[k] = x.get(k)
        yield r


def _prepare_gsm8k(spec: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    src = spec["source"]
    cfg = _norm_cfg(src.get("config"))
    src_str = f"{src['id']}:{cfg}:{src['split']}@{src.get('revision')}" if cfg else f"{src['id']}:{src['split']}@{src.get('revision')}"

    for x in _iter_hf_rows(src, logical_name="GSM8K-test"):
        q = x.get("question", "") or ""
        a = x.get("answer", "") or ""
        yield {"input": str(q).strip(), "answer": str(a).strip(), "source": src_str}


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

    if args.prepare_math_train:
        _run("MATH-train", _prepare_math_train)
    if args.prepare_math500:
        _run("MATH500", _prepare_math500)
    if args.prepare_cmath:
        _run("CMATH-train", _prepare_cmath)
        _run("CMATH-test", _prepare_cmath)
    if args.prepare_gsm8k:
        _run("GSM8K-test", _prepare_gsm8k)

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
