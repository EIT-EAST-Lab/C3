#!/usr/bin/env python3
"""Prepare code benchmarks into canonical JSONL for C3.

Default behavior in this version:
- HumanEval/APPS are disabled unless explicitly enabled.
- MBPP/MBPP+ are enabled.

This avoids unnecessary downloads for users who only need MBPP-family data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import yaml

try:
    from datasets import load_dataset  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "[FAIL] `datasets` is required. Install dependencies first."
    ) from e

try:
    from huggingface_hub import HfApi, hf_hub_download  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "[FAIL] `huggingface_hub` is required. Install dependencies first."
    ) from e

try:
    import pyarrow.parquet as pq  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "[FAIL] `pyarrow` is required. Install dependencies first."
    ) from e


_HEX_RE = re.compile(r"^[0-9a-f]{40}$")
_EVALPLUS_REV_RE = re.compile(
    r"^(?:evalplus@[0-9]+\.[0-9]+\.[0-9]+(?:[a-zA-Z0-9.+-]*)?|fallback_mbpp@[0-9a-f]{40})$"
)


# -----------------------------
# Generic helpers
# -----------------------------


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_manifest(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if not isinstance(obj, dict) or not isinstance(obj.get("outputs"), list):
        raise SystemExit(f"[FAIL] Invalid manifest format: {path}")
    return obj


def _write_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False, allow_unicode=True)
    tmp.replace(path)


def _is_pinned_revision(rev: Optional[str]) -> bool:
    return bool(rev and _HEX_RE.match(str(rev).strip()))


def _is_pinned_evalplus_revision(rev: Optional[str]) -> bool:
    return bool(rev and _EVALPLUS_REV_RE.match(str(rev).strip()))


def _get_evalplus_version() -> Optional[str]:
    try:
        import evalplus  # type: ignore

        v = getattr(evalplus, "__version__", None)
        if v:
            return str(v)
    except Exception:
        pass

    try:
        from importlib.metadata import version

        return version("evalplus")
    except Exception:
        return None


def _resolve_output_path(p: str, out_dir: Optional[str]) -> Path:
    pp = Path(p)
    if out_dir:
        root = Path(out_dir)
        if pp.is_absolute():
            return root / pp.name
        if len(pp.parts) >= 1 and pp.parts[0] == "data":
            return root / Path(*pp.parts[1:])
        return root / pp
    return pp


def _dump_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    tmp.replace(path)
    return n


def _verify_or_record(
    name: str,
    expected_sha: Optional[str],
    actual_sha: str,
    *,
    missing_sha: List[Tuple[str, str]],
    computed_by_name: Dict[str, str],
    out_path: Path,
    allow_repin_on_mismatch: bool = False,
) -> None:
    computed_by_name[name] = actual_sha
    if expected_sha:
        exp = str(expected_sha).strip().lower()
        act = str(actual_sha).strip().lower()
        if exp != act:
            if allow_repin_on_mismatch:
                print(
                    f"[WARN] {name}: sha256 changed, repinning due to --update_manifest_sha256 1.\n"
                    f"       old={exp}\n"
                    f"       new={act}"
                )
            else:
                raise SystemExit(
                    f"[FAIL] {name}: sha256 mismatch for {out_path}\n"
                    f"  expected: {expected_sha}\n"
                    f"  actual:   {actual_sha}\n"
                    "Delete output and rerun with --overwrite 1 if needed."
                )
    else:
        missing_sha.append((name, str(out_path)))


# -----------------------------
# HF robust row loading
# -----------------------------


def _is_script_not_supported_error(exc: Exception) -> bool:
    s = f"{type(exc).__name__}: {exc}"
    return "Dataset scripts are no longer supported" in s


def _is_schema_cast_error(exc: Exception) -> bool:
    s = f"{type(exc).__name__}: {exc}"
    return ("CastError" in s) and ("column names don't match" in s)


def _normalize_cfg(cfg: Any) -> Optional[str]:
    if cfg is None:
        return None
    s = str(cfg).strip()
    if s == "" or s.lower() == "null":
        return None
    return s


def _known_data_file(path: str) -> bool:
    p = path.lower()
    if p.endswith("/readme.md") or p.endswith("readme.md"):
        return False
    if p.endswith("dataset_infos.json"):
        return False
    if p.endswith(".md") or p.endswith(".txt"):
        return False
    return p.endswith(".parquet") or p.endswith(".jsonl") or p.endswith(".json")


def _split_aliases(split: str) -> List[str]:
    s = split.lower().strip()
    aliases = [s]
    if s == "test":
        aliases += ["validation", "val", "dev"]
    elif s in {"validation", "val", "dev"}:
        aliases += ["test"]
    return aliases


def _pick_repo_files_for_split(files: List[str], split: str) -> List[str]:
    cands = sorted([f for f in files if _known_data_file(f)])
    if not cands:
        return []

    def _hits(tag: str) -> List[str]:
        tag = tag.lower()
        out: List[str] = []
        for f in cands:
            ff = f.lower()
            base = ff.rsplit("/", 1)[-1]
            if (
                base.startswith(f"{tag}-")
                or base == f"{tag}.parquet"
                or base == f"{tag}.jsonl"
                or base == f"{tag}.json"
                or f"/{tag}/" in ff
                or f"/{tag}-" in ff
                or f"_{tag}_" in ff
            ):
                out.append(f)
        return sorted(set(out))

    for a in _split_aliases(split):
        h = _hits(a)
        if h:
            return h

    # Safe fallback only when there is a single obvious data file.
    if len(cands) == 1:
        return cands
    return []


def _iter_rows_from_repo_files(repo_id: str, revision: str, split: str) -> Iterator[Dict[str, Any]]:
    endpoint = None
    # respect HF_ENDPOINT if present
    import os

    env_endpoint = os.environ.get("HF_ENDPOINT", "").strip()
    if env_endpoint:
        endpoint = env_endpoint

    api = HfApi(endpoint=endpoint)
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision)
    picked = _pick_repo_files_for_split(files, split)
    if not picked:
        raise RuntimeError(
            f"No unambiguous data files found for {repo_id}@{revision} split={split}. "
            "Please verify source.id/source.split in manifest."
        )

    for fn in picked:
        local = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=fn,
            revision=revision,
        )
        lf = fn.lower()
        if lf.endswith(".parquet"):
            table = pq.read_table(local)
            drop_cols = [c for c in ("__index_level_0__", "index") if c in table.column_names]
            if drop_cols:
                table = table.drop(drop_cols)
            for batch in table.to_batches(max_chunksize=4096):
                for row in batch.to_pylist():
                    if isinstance(row, dict):
                        yield row
            continue

        # json/jsonl
        with open(local, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if not content:
            continue
        if lf.endswith(".jsonl"):
            for line in content.splitlines():
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield obj
        else:
            obj = json.loads(content)
            if isinstance(obj, list):
                for x in obj:
                    if isinstance(x, dict):
                        yield x
            elif isinstance(obj, dict):
                # common wrappers: {"data": [...]} / {"train": [...]} etc.
                yielded = False
                for k in ("data", split, split.lower(), "train", "test", "validation"):
                    v = obj.get(k) if isinstance(obj, dict) else None
                    if isinstance(v, list):
                        for x in v:
                            if isinstance(x, dict):
                                yield x
                        yielded = True
                        break
                if not yielded:
                    yield obj


def _load_hf_rows(src: Dict[str, Any], logical_name: str) -> Iterator[Dict[str, Any]]:
    repo_id = src["id"]
    split = src.get("split", "test")
    rev = src.get("revision")
    cfg = _normalize_cfg(src.get("config"))

    try:
        ds = load_dataset(repo_id, cfg, split=split, revision=rev)
        for x in ds:
            yield dict(x)
        return
    except Exception as e:
        if _is_script_not_supported_error(e) or _is_schema_cast_error(e):
            print(f"[WARN] {logical_name}: load_dataset failed ({type(e).__name__}); trying repo-file fallback.")
        else:
            raise

    for row in _iter_rows_from_repo_files(repo_id=repo_id, revision=rev, split=split):
        yield row


# -----------------------------
# Canonicalization
# -----------------------------


def _canon_humaneval_row(x: Dict[str, Any], source: str) -> Dict[str, Any]:
    return {
        "task_id": x.get("task_id"),
        "prompt": x.get("prompt"),
        "canonical_solution": x.get("canonical_solution"),
        "test": x.get("test"),
        "entry_point": x.get("entry_point"),
        "source": source,
    }


def _canon_apps_row(x: Dict[str, Any], source: str) -> Dict[str, Any]:
    return {
        "problem_id": x.get("problem_id"),
        "question": x.get("question"),
        "starter_code": x.get("starter_code"),
        "input_output": x.get("input_output"),
        "solutions": x.get("solutions"),
        "difficulty": x.get("difficulty"),
        "source": source,
    }


def _canon_mbpp_row(x: Dict[str, Any], source: str) -> Dict[str, Any]:
    return {
        "task_id": x.get("task_id"),
        "text": x.get("text"),
        "code": x.get("code"),
        "test_list": x.get("test_list"),
        "test_setup_code": x.get("test_setup_code"),
        "source": source,
    }


def _problem_key(row: Dict[str, Any]) -> str:
    """Whitespace-normal form of an MBPP problem text; the same form check_overlap.py compares."""
    return " ".join(str(row.get("text") or "").split())


def _problem_keys_from_file(path: Path) -> set:
    keys: set = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                keys.add(_problem_key(json.loads(line)))
    keys.discard("")
    return keys


def _subtract_test_problems(name: str, rows: Iterable[Dict[str, Any]], test_keys: set) -> Iterator[Dict[str, Any]]:
    """
    Yield the training rows whose problem does not occur in the test split.

    Upstream MBPP (full config, revision 4bb6404) ships three problems in both the train and
    the test split, so a training artifact taken as shipped fails the overlap gate against
    MBPP-test and MBPP+. The subtraction is by normalized problem text, the key the gate uses.
    """
    dropped = 0
    for r in rows:
        if _problem_key(r) in test_keys:
            dropped += 1
            continue
        yield r
    print(f"[INFO] {name}: dropped {dropped} row(s) whose problem also occurs in MBPP-test")


def _canon_mbpp_plus_row(x: Dict[str, Any], source: str, seed: int, max_tests: int) -> Dict[str, Any]:
    tests: List[str] = list(x.get("test_list") or [])
    if max_tests > 0 and len(tests) > max_tests:
        task_id = x.get("task_id", 0)
        try:
            task_i = int(task_id)
        except Exception:
            task_i = 0
        rng = random.Random(seed + task_i)
        idx = list(range(len(tests)))
        rng.shuffle(idx)
        idx = sorted(idx[:max_tests])
        tests = [tests[i] for i in idx]

    return {
        "task_id": x.get("task_id"),
        "text": x.get("text"),
        "code": x.get("code"),
        "test_list": tests,
        "test_setup_code": x.get("test_setup_code"),
        "source": source,
    }


def _try_load_evalplus_mbpp_plus() -> Optional[List[Dict[str, Any]]]:
    try:
        from evalplus.data import get_mbpp_plus  # type: ignore

        tasks = get_mbpp_plus()
        rows: List[Dict[str, Any]] = []
        for k, v in tasks.items():
            row = dict(v)
            try:
                row["task_id"] = int(k)
            except Exception:
                row["task_id"] = k
            rows.append(row)
        rows.sort(key=lambda r: r.get("task_id"))
        return rows
    except Exception:
        return None


# -----------------------------
# Main
# -----------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".")
    ap.add_argument("--manifest", type=str, default="configs/data_manifest.yaml")
    ap.add_argument("--data_dir", type=str, default="")
    ap.add_argument("--out_dir", type=str, default="")
    ap.add_argument("--overwrite", type=int, default=0, choices=[0, 1])
    ap.add_argument("--strict", type=int, default=0, choices=[0, 1])
    ap.add_argument("--update_manifest_sha256", type=int, default=0, choices=[0, 1])

    ap.add_argument("--prepare_humaneval", type=int, default=0, choices=[0, 1])
    ap.add_argument("--prepare_apps", type=int, default=0, choices=[0, 1])
    ap.add_argument("--prepare_mbpp", type=int, default=1, choices=[0, 1])
    ap.add_argument("--prepare_mbpp_plus", type=int, default=1, choices=[0, 1])

    ap.add_argument("--mbpp_plus_seed", type=int, default=1234)
    ap.add_argument("--mbpp_plus_max_tests", type=int, default=20)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    manifest_path = (root / args.manifest).resolve()
    manifest = _load_manifest(manifest_path)

    outputs = manifest.get("outputs") or []
    idx: Dict[str, Dict[str, Any]] = {o["name"]: o for o in outputs if isinstance(o, dict) and "name" in o}

    computed_by_name: Dict[str, str] = {}
    computed_src_revision_by_name: Dict[str, str] = {}
    missing_sha: List[Tuple[str, str]] = []

    out_dir = args.out_dir.strip() if args.out_dir else ""
    data_dir = args.data_dir.strip() if args.data_dir else ""
    out_base = out_dir or data_dir

    overwrite = bool(args.overwrite)
    strict = bool(args.strict)
    allow_repin_on_mismatch = bool(args.update_manifest_sha256) and not strict

    def _selected(name: str) -> bool:
        if name == "HumanEval":
            return bool(args.prepare_humaneval)
        if name == "APPS":
            return bool(args.prepare_apps)
        if name in {"MBPP-train", "MBPP-test"}:
            return bool(args.prepare_mbpp)
        if name == "MBPP+":
            return bool(args.prepare_mbpp_plus)
        return False

    def _handle_existing(name: str, spec: Dict[str, Any], out_path: Path) -> bool:
        if out_path.exists() and not overwrite:
            if out_path.stat().st_size == 0:
                print(f"[WARN] {name}: {out_path} is empty (0 bytes); regenerating.")
                return False
            sha = _sha256(out_path)
            _verify_or_record(
                name,
                spec.get("sha256"),
                sha,
                missing_sha=missing_sha,
                computed_by_name=computed_by_name,
                out_path=out_path,
                allow_repin_on_mismatch=allow_repin_on_mismatch,
            )
            print(f"[SKIP] {name}: {out_path} exists (verified sha256).")
            return True
        return False

    def _check_pinned(name: str, spec: Dict[str, Any]) -> None:
        src = spec.get("source", {})
        if strict and src.get("kind") == "huggingface" and not _is_pinned_revision(src.get("revision")):
            raise SystemExit(
                f"[FAIL] {name}: source.revision must be an immutable HF commit SHA "
                f"(got '{src.get('revision')}')."
            )

        if strict and src.get("kind") == "evalplus":
            rev = src.get("revision")
            if not _is_pinned_evalplus_revision(rev):
                raise SystemExit(
                    f"[FAIL] {name}: source.revision must be pinned as "
                    f"'evalplus@<VERSION>' or 'fallback_mbpp@<HF_COMMIT_SHA>' "
                    f"(got '{rev}')."
                )

            rev = str(rev).strip()
            if rev.startswith("evalplus@"):
                expected_ver = rev.split("@", 1)[1]
                installed_ver = _get_evalplus_version()
                if not installed_ver:
                    raise SystemExit(
                        "[FAIL] MBPP+: manifest pins EvalPlus, but `evalplus` is not importable."
                    )
                if installed_ver != expected_ver:
                    raise SystemExit(
                        "[FAIL] MBPP+: EvalPlus version mismatch.\n"
                        f"  expected: {expected_ver}\n"
                        f"  actual:   {installed_ver}"
                    )

            if rev.startswith("fallback_mbpp@"):
                expected_base = rev.split("@", 1)[1]
                base_spec = idx.get("MBPP-test") or idx.get("MBPP-train")
                base_src = (base_spec or {}).get("source", {})
                base_rev = base_src.get("revision")
                if base_rev and base_rev != expected_base:
                    raise SystemExit(
                        "[FAIL] MBPP+: fallback pin does not match MBPP HF revision.\n"
                        f"  fallback pin: {expected_base}\n"
                        f"  MBPP rev:     {base_rev}"
                    )

    def _require(name: str) -> Dict[str, Any]:
        spec = idx.get(name)
        if spec is None:
            raise SystemExit(f"[FAIL] Manifest missing output entry: {name}")
        return spec

    # ---------------- HumanEval ----------------
    if _selected("HumanEval"):
        spec = _require("HumanEval")
        _check_pinned("HumanEval", spec)
        src = spec.get("source", {})
        out_path = _resolve_output_path(spec["output_path"], out_base)
        if not _handle_existing("HumanEval", spec, out_path):
            src_str = f"{src['id']}:{src.get('split', 'test')}@{src.get('revision')}"
            rows = (_canon_humaneval_row(x, src_str) for x in _load_hf_rows(src, "HumanEval"))
            n = _dump_jsonl(out_path, rows)
            sha = _sha256(out_path)
            _verify_or_record(
                "HumanEval", spec.get("sha256"), sha,
                missing_sha=missing_sha, computed_by_name=computed_by_name, out_path=out_path,
                allow_repin_on_mismatch=allow_repin_on_mismatch,
            )
            print(f"[OK] HumanEval -> {out_path} ({n} rows, sha256 {sha})")
    else:
        print("[SKIP] HumanEval disabled (--prepare_humaneval 0).")

    # ---------------- APPS ----------------
    if _selected("APPS"):
        spec = _require("APPS")
        _check_pinned("APPS", spec)
        src = spec.get("source", {})
        out_path = _resolve_output_path(spec["output_path"], out_base)
        if not _handle_existing("APPS", spec, out_path):
            src_str = f"{src['id']}:{src.get('split', 'test')}@{src.get('revision')}"
            rows = (_canon_apps_row(x, src_str) for x in _load_hf_rows(src, "APPS"))
            n = _dump_jsonl(out_path, rows)
            sha = _sha256(out_path)
            _verify_or_record(
                "APPS", spec.get("sha256"), sha,
                missing_sha=missing_sha, computed_by_name=computed_by_name, out_path=out_path,
                allow_repin_on_mismatch=allow_repin_on_mismatch,
            )
            print(f"[OK] APPS -> {out_path} ({n} rows, sha256 {sha})")
    else:
        print("[SKIP] APPS disabled (--prepare_apps 0).")

    # ---------------- MBPP ----------------
    if bool(args.prepare_mbpp):
        # MBPP-test first: MBPP-train is written minus the problems that also occur in the
        # test split, so the training artifact and the evaluation artifacts stay disjoint.
        test_keys: set = set()
        for name in ("MBPP-test", "MBPP-train"):
            if name not in idx:
                continue
            spec = _require(name)
            _check_pinned(name, spec)
            src = spec.get("source", {})
            out_path = _resolve_output_path(spec["output_path"], out_base)
            if _handle_existing(name, spec, out_path):
                if name == "MBPP-test":
                    test_keys = _problem_keys_from_file(out_path)
                continue

            src_str = f"{src['id']}:{src.get('split')}@{src.get('revision')}"
            rows: Iterable[Dict[str, Any]] = (_canon_mbpp_row(x, src_str) for x in _load_hf_rows(src, name))
            if name == "MBPP-test":
                test_rows = list(rows)
                test_keys = {_problem_key(r) for r in test_rows}
                test_keys.discard("")
                rows = test_rows
            else:
                if "MBPP-test" in idx and not test_keys:
                    raise SystemExit("[FAIL] MBPP-train: the MBPP-test problems are not available for subtraction.")
                rows = _subtract_test_problems(name, rows, test_keys)
            n = _dump_jsonl(out_path, rows)
            sha = _sha256(out_path)
            _verify_or_record(
                name, spec.get("sha256"), sha,
                missing_sha=missing_sha, computed_by_name=computed_by_name, out_path=out_path,
                allow_repin_on_mismatch=allow_repin_on_mismatch,
            )
            print(f"[OK] {name} -> {out_path} ({n} rows, sha256 {sha})")
    else:
        print("[SKIP] MBPP disabled (--prepare_mbpp 0).")

    # ---------------- MBPP+ ----------------
    if bool(args.prepare_mbpp_plus):
        plus_spec = idx.get("MBPP+")
        if plus_spec is None:
            raise SystemExit("[FAIL] Manifest missing output entry: MBPP+")
        _check_pinned("MBPP+", plus_spec)
        plus_out = _resolve_output_path(plus_spec["output_path"], out_base)

        if not _handle_existing("MBPP+", plus_spec, plus_out):
            expected_rev = (plus_spec.get("source") or {}).get("revision")
            expected_rev_str = str(expected_rev).strip() if expected_rev is not None else ""

            force_fallback = bool(strict and expected_rev_str.startswith("fallback_mbpp@"))
            force_evalplus = bool(strict and expected_rev_str.startswith("evalplus@"))

            rows: Optional[List[Dict[str, Any]]] = None
            used_evalplus = False

            if not force_fallback:
                tasks = _try_load_evalplus_mbpp_plus()
                if tasks is not None:
                    v = _get_evalplus_version()
                    if not v and strict:
                        raise SystemExit(
                            "[FAIL] MBPP+: could not determine EvalPlus version in strict mode."
                        )
                    v = v or "unknown"
                    src_str = f"evalplus:mbpp_plus@{v}"
                    rows = [
                        _canon_mbpp_plus_row(
                            x,
                            source=src_str,
                            seed=args.mbpp_plus_seed,
                            max_tests=args.mbpp_plus_max_tests,
                        )
                        for x in tasks
                    ]
                    computed_src_revision_by_name["MBPP+"] = f"evalplus@{v}"
                    used_evalplus = True

            if force_evalplus and not used_evalplus:
                raise SystemExit(
                    "[FAIL] MBPP+: manifest pins EvalPlus, but EvalPlus MBPP+ could not be loaded.\n"
                    "  EvalPlus downloads MBPP+ from a GitHub release.\n"
                    "  If your network cannot reach github.com, set GITHUB_MIRROR_PREFIX "
                    "(see docs/31_network_mirrors.md) and rerun through "
                    "scripts/10_data/prepare_all.sh, which seeds the EvalPlus cache "
                    "through the mirror."
                )

            if rows is None:
                base_spec = idx.get("MBPP-test") or idx.get("MBPP-train")
                if base_spec is None:
                    raise SystemExit("[FAIL] MBPP+: requires MBPP-test or MBPP-train for fallback path.")
                src = base_spec.get("source", {})
                base_rev = src.get("revision")

                if strict and not _is_pinned_revision(base_rev):
                    raise SystemExit("[FAIL] MBPP+: fallback requires pinned MBPP source revision.")

                if strict and expected_rev_str.startswith("fallback_mbpp@"):
                    expected_base = expected_rev_str.split("@", 1)[1]
                    if base_rev and base_rev != expected_base:
                        raise SystemExit(
                            "[FAIL] MBPP+: fallback pin does not match MBPP HF revision.\n"
                            f"  expected: {expected_base}\n"
                            f"  actual:   {base_rev}"
                        )

                src_str = f"{src['id']}:{src.get('split', 'test')}@{src.get('revision')} (fallback-mbpp-plus)"
                rows = [_canon_mbpp_row(x, src_str) for x in _load_hf_rows(src, "MBPP+ fallback")]
                if base_rev:
                    computed_src_revision_by_name["MBPP+"] = f"fallback_mbpp@{base_rev}"

            n = _dump_jsonl(plus_out, rows)
            sha = _sha256(plus_out)
            _verify_or_record(
                "MBPP+", plus_spec.get("sha256"), sha,
                missing_sha=missing_sha, computed_by_name=computed_by_name, out_path=plus_out,
                allow_repin_on_mismatch=allow_repin_on_mismatch,
            )
            print(f"[OK] MBPP+ -> {plus_out} ({n} rows, sha256 {sha})")
    else:
        print("[SKIP] MBPP+ disabled (--prepare_mbpp_plus 0).")

    # Write back pins for artifacts processed in this run.
    if args.update_manifest_sha256:
        updated = 0
        for o in outputs:
            if not isinstance(o, dict):
                continue
            name = o.get("name")
            if not isinstance(name, str):
                continue

            if name in computed_by_name:
                new_sha = computed_by_name[name]
                if o.get("sha256") != new_sha:
                    o["sha256"] = new_sha
                    updated += 1

            if name in computed_src_revision_by_name:
                src = o.setdefault("source", {})
                new_rev = computed_src_revision_by_name[name]
                if src.get("revision") != new_rev:
                    src["revision"] = new_rev
                    updated += 1

        if updated:
            _write_manifest(manifest_path, manifest)
            print(f"[OK] Updated {updated} pin(s) in {manifest_path}")
        else:
            print("[OK] No pins needed updating.")

    # Strict mode: only enforce for artifacts actually processed in this run.
    if strict:
        optional = {o.get("name"): bool(o.get("optional", False)) for o in outputs if isinstance(o, dict)}
        missing_required = [x for x in missing_sha if not optional.get(x[0], False)]
        if missing_required:
            msg = "\n".join([f"  - {n}: {p}" for n, p in missing_required])
            raise SystemExit(
                "[FAIL] Strict mode requires sha256 pins for all processed required artifacts.\n"
                "Missing sha256 pins for:\n"
                f"{msg}\n"
                "Run once with --update_manifest_sha256 1, commit manifest, then rerun --strict 1."
            )

    print("[OK] prepare_code.py finished successfully.")


if __name__ == "__main__":
    main()
