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
import keyword
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import yaml

# The write door below asks whether a row carries a problem statement at all, and it
# has to ask with the same key table the trainer reads. That table lives in
# c3/task/datasets.py and is imported rather than copied, which needs the repository
# root on sys.path: the same bootstrap scripts/10_data/prepare_math.py uses.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from c3.task.datasets import _PROMPT_KEY_CANDIDATES as PROMPT_KEY_CANDIDATES  # noqa: E402

# The MBPP+ builder generates a test_setup_code that depends on how the evaluator splits
# and runs it, so it asks the evaluator itself rather than keeping a second copy of the
# rule. Both names are private to that module and imported deliberately.
from c3.envs.code.executor import _RE_IMPORT_LINE, _split_setup_imports  # noqa: E402

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
# MBPP+ has one provenance form. `fallback_mbpp@<HF_COMMIT_SHA>` was the second, and it
# meant "write the MBPP test problems under the MBPP+ name when EvalPlus is unavailable";
# that is a silent substitution of one benchmark for another and it is gone.
_EVALPLUS_REV_RE = re.compile(r"^evalplus@[0-9]+\.[0-9]+\.[0-9]+(?:[a-zA-Z0-9.+-]*)?$")


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
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
                n += 1
        tmp.replace(path)
    except BaseException:
        # The row gate raises SystemExit from inside this loop. Whatever was written
        # so far is a fragment of a refused artifact, so it leaves nothing behind.
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        raise
    return n


# -----------------------------
# The write door
# -----------------------------

# What the CodeEnv evaluator reads as the test payload of one row. The first three
# are the keys c3/envs/code/executor.py:_assemble_tests reads (test_list is executed
# one assert at a time, test is a whole script and runs only when test_list is
# empty); APPS carries its cases in input_output instead and no other artifact does.
TEST_KEY_CANDIDATES: Dict[str, Tuple[str, ...]] = {
    "HumanEval": ("test",),
    "APPS": ("input_output",),
    "MBPP-train": ("test_list", "challenge_test_list", "test"),
    "MBPP-test": ("test_list", "challenge_test_list", "test"),
    "MBPP+": ("test_list", "challenge_test_list", "test"),
}


def _is_filled(value: Any) -> bool:
    """True when a row field carries something a consumer can use."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) > 0
    return True


def _require_complete_rows(
    name: str,
    rows: Iterable[Dict[str, Any]],
    *,
    test_keys: Tuple[str, ...],
) -> Iterator[Dict[str, Any]]:
    """
    Refuse to write a row with no problem statement, or with no tests to score it by.

    A sha256 pin says "this file is the one we produced", never "this file is usable".
    The MBPP+ artifact of 2026-09-07 was 378 rows in which only task_id and source
    carried a value, because the row builder read MBPP column names from a source that
    uses EvalPlus ones, and the pin matched that empty file byte for byte on every run.
    The check below is the one door every artifact of this script goes through, so an
    empty artifact fails at preparation instead of becoming an evaluation that silently
    scores nothing.
    """
    for i, row in enumerate(rows):
        if not any(_is_filled(row.get(key)) for key in PROMPT_KEY_CANDIDATES):
            raise SystemExit(
                f"[FAIL] {name}: row {i} has an empty problem statement; none of "
                f"{list(PROMPT_KEY_CANDIDATES)} carries one (row keys: {sorted(row.keys())})."
            )
        if test_keys and not any(_is_filled(row.get(key)) for key in test_keys):
            raise SystemExit(
                f"[FAIL] {name}: row {i} has no tests; none of {list(test_keys)} carries any "
                f"(row keys: {sorted(row.keys())}). A row the evaluator cannot score is not a row."
            )
        yield row


def _write_checked(name: str, out_path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    """Every write of this script goes through the row gate."""
    return _dump_jsonl(out_path, _require_complete_rows(name, rows, test_keys=TEST_KEY_CANDIDATES.get(name, ())))


def _check_rows_complete_on_disk(name: str, out_path: Path) -> None:
    """The same gate for an artifact that is verified instead of written."""
    test_keys = TEST_KEY_CANDIDATES.get(name, ())
    with out_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not any(_is_filled(row.get(key)) for key in PROMPT_KEY_CANDIDATES):
                raise SystemExit(
                    f"[FAIL] {name}: {out_path} row {i} has an empty problem statement; regenerate it "
                    "with --overwrite 1."
                )
            if test_keys and not any(_is_filled(row.get(key)) for key in test_keys):
                raise SystemExit(
                    f"[FAIL] {name}: {out_path} row {i} has no tests; regenerate it with --overwrite 1."
                )
    print(f"[OK] {name}: every row carries a problem statement and tests.")


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


def _subtract_eval_problems(name: str, rows: Iterable[Dict[str, Any]], eval_keys: set) -> Iterator[Dict[str, Any]]:
    """
    Yield the training rows whose problem occurs in no evaluation artifact.

    Two upstream facts make this necessary. MBPP (full config, revision 4bb6404) ships three
    problems in both the train and the test split. MBPP+ is drawn from the sanitized MBPP,
    which spans the whole id range, so 108 of its 378 task ids are MBPP train-split ids and
    54 of its problem statements occur in the prepared training file. Either way the overlap
    gate fails on the training artifact as shipped. The subtraction is by normalized problem
    text, the key the gate compares, so the two cannot disagree.
    """
    dropped = 0
    for r in rows:
        if _problem_key(r) in eval_keys:
            dropped += 1
            continue
        yield r
    print(f"[INFO] {name}: dropped {dropped} row(s) whose problem also occurs in an evaluation file")


# -----------------------------
# MBPP+ (EvalPlus)
# -----------------------------

_MBPP_PLUS_TASK_ID_RE = re.compile(r"^(?:mbpp/)?([0-9]+)$")


def mbpp_plus_task_id(raw: Any) -> int:
    """
    The integer task id of one EvalPlus MBPP+ task.

    EvalPlus 0.3.1 keys its tasks `Mbpp/100`, so `int()` on that key raises, and both
    call sites used to swallow it: one kept the string, the other substituted 0, which
    gave every row of the file the same sampling seed. An id this function cannot read
    is a failure and never a default, because a silent 0 is a wrong number wearing the
    shape of a right one.
    """
    text = str(raw).strip().lower()
    match = _MBPP_PLUS_TASK_ID_RE.match(text)
    if match is None:
        raise SystemExit(
            f"[FAIL] MBPP+: cannot read an integer task id from {raw!r}. EvalPlus 0.3.1 keys "
            "its tasks 'Mbpp/<n>'; a different key shape needs this parser extended, not a default."
        )
    return int(match.group(1))


def _evalplus_mbpp_plus_tasks() -> Optional[List[Dict[str, Any]]]:
    """
    Every EvalPlus MBPP+ task, ordered by integer task id, or None when EvalPlus cannot load.

    The keys of one task in EvalPlus 0.3.1 are prompt, canonical_solution, assertion,
    contract, entry_point, atol, base_input, plus_input and task_id. None of them is an
    MBPP column name, which is the whole of the bug this function's caller now refuses
    to reproduce.
    """
    try:
        from evalplus.data import get_mbpp_plus  # type: ignore

        tasks = get_mbpp_plus()
    except Exception:
        return None

    rows: List[Dict[str, Any]] = []
    for key, task in tasks.items():
        row = dict(task)
        row["task_id"] = mbpp_plus_task_id(key)
        rows.append(row)
    rows.sort(key=lambda r: int(r["task_id"]))
    return rows


# The wall clock one MBPP+ task is given, carried on the row and read by
# c3/envs/code/reward.py. It is what EvalPlus allows a task
# (EVALPLUS_TIMEOUT_PER_TASK, 60 seconds); this benchmark needs it because the
# reference solution of one of its problems runs for close to 30 seconds by itself,
# and the evaluator's default of 15 would score that problem 0 for every candidate.
MBPP_PLUS_TIMEOUT_S = 60

# The default of C3_CODE_MAX_ASSERT_CHARS in c3/envs/code/executor.py, which refuses a
# task whose test_list text is longer. The generated tests are one short call each, so
# the longest task lands near 3000 characters; this is checked, not assumed.
# tests/contract/test_prepare_data_gates.py pins the two numbers together.
MBPP_PLUS_ASSERT_BUDGET = 4000

# EvalPlus judges these entry points with something other than equality. The lists are
# MBPP_OUTPUT_SET_EQ_TASKS and MBPP_OUTPUT_NOT_NONE_TASKS in
# evalplus/eval/_special_oracle.py, plus the two problems it re-implements there.
MBPP_PLUS_SET_EQ_ENTRY_POINTS = (
    "similar_elements",
    "find_char_long",
    "common_in_nested_lists",
    "extract_singly",
    "larg_nnum",
    "intersection_array",
    "find_dissimilar",
    "Diff",
)
MBPP_PLUS_NOT_NONE_ENTRY_POINTS = ("check_str", "text_match_three", "text_starta_endb")

# Two more EvalPlus special oracles. Neither entry point occurs in MbppPlus v0.2.0, so
# neither has a mode below; if a future release brings one back, the builder stops
# instead of judging those problems by plain equality.
MBPP_PLUS_UNHANDLED_ENTRY_POINTS = ("are_equivalent", "sum_div")

_SIMPLE_IMPORT_RE = re.compile(r"^import\s+([A-Za-z_][A-Za-z0-9_]*)(?:\s+as\s+([A-Za-z_][A-Za-z0-9_]*))?$")
_SIMPLE_FROM_RE = re.compile(r"^from\s+([A-Za-z_][A-Za-z0-9_]*)\s+import\s+(.+)$")
_SIMPLE_FROM_NAME_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)(?:\s+as\s+([A-Za-z_][A-Za-z0-9_]*))?$")

# The judge, verbatim in every row. It is the comparison of
# evalplus/eval/__init__.py:unsafe_execute written for the CodeEnv sandbox, which
# whitelists the standard library and has no numpy: set comparison and the
# output-is-not-None rule for the entry points that need them, then EvalPlus's float
# tolerance, which starts at the task's atol, becomes 1e-06 the first time an expected
# value is a float, and stays there for the rest of the task, exactly as the loop it
# comes from does it.
_MBPP_PLUS_HELPERS = '''
def _c3_copy(value):
    if isinstance(value, list):
        return [_c3_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple([_c3_copy(item) for item in value])
    if isinstance(value, dict):
        return dict([(key, _c3_copy(item)) for key, item in value.items()])
    if isinstance(value, set):
        return set([_c3_copy(item) for item in value])
    if isinstance(value, frozenset):
        return frozenset([_c3_copy(item) for item in value])
    return value


def _c3_is_number(value):
    if isinstance(value, bool):
        return False
    return isinstance(value, (int, float, complex))


def _c3_is_floats(value):
    if isinstance(value, float):
        return True
    if isinstance(value, (list, tuple)) and len(value) > 0:
        for item in value:
            if not isinstance(item, float):
                return False
        return True
    return False


def _c3_allclose(got, want, atol):
    if isinstance(want, (list, tuple)):
        if not isinstance(got, (list, tuple)) or len(got) != len(want):
            return False
        for left, right in zip(got, want):
            if not _c3_allclose(left, right, atol):
                return False
        return True
    if not _c3_is_number(got) or not _c3_is_number(want):
        return False
    if got == want:
        return True
    try:
        delta = abs(got - want)
        limit = atol + 1e-07 * abs(want)
    except Exception:
        return False
    if delta != delta or limit != limit:
        return False
    return delta <= limit


def _c3_judge(got, want, args):
    global _C3_ATOL_STATE
    try:
        exact = bool(got == want)
    except Exception:
        exact = False
    if _C3_MODE == "set_eq":
        try:
            exact = set(got) == set(want)
        except Exception:
            exact = False
    elif _C3_MODE == "not_none":
        if isinstance(got, bool):
            exact = bool(got == want)
        else:
            exact = bool(want == (got is not None))
    elif _C3_MODE == "surface_area" and not exact:
        try:
            exact = abs(got - _c3_oracle(*args)) <= _C3_ATOL_STATE
        except Exception:
            exact = False
    elif _C3_MODE == "digit_distance" and not exact:
        try:
            exact = bool(got == _c3_oracle(*args))
        except Exception:
            exact = False
    if _C3_ATOL_STATE == 0 and _c3_is_floats(want):
        _C3_ATOL_STATE = 1e-06
    if not exact and _C3_ATOL_STATE != 0:
        if type(got) is not type(want):
            return False
        if isinstance(want, (list, tuple)) and len(got) != len(want):
            return False
        return _c3_allclose(got, want, _C3_ATOL_STATE)
    return bool(exact)


def _c3_case(index):
    args = _C3_INPUTS[index]
    try:
        want = _C3_REF(*_c3_copy(args))
    except Exception:
        return False
    if _C3_MODE == "not_none":
        want = want is not None
    try:
        got = _C3_FN(*_c3_copy(args))
    except Exception:
        return False
    return _c3_judge(got, want, args)
'''

# The two problems EvalPlus judges against a second implementation of its own.
_MBPP_PLUS_ORACLES = {
    "surface_Area": '''
_c3_math = __import__("math")


def _c3_oracle(base_edge, height):
    slant_height = _c3_math.sqrt((base_edge / 2) ** 2 + height ** 2)
    base_area = base_edge ** 2
    lateral_area = 4 * (base_edge * slant_height) / 2
    return round(base_area + lateral_area)
''',
    "digit_distance_nums": '''
def _c3_oracle(num1, num2):
    text1 = str(num1)
    text2 = str(num2)
    width = max(len(text1), len(text2))
    text1 = text1.zfill(width)
    text2 = text2.zfill(width)
    total = 0
    for left, right in zip(text1, text2):
        total = total + abs(int(left) - int(right))
    return total
''',
}


def mbpp_plus_statement(prompt: Any) -> str:
    """
    The problem statement of one EvalPlus task, taken out of its docstring prompt.

    EvalPlus writes the prompt as a docstring holding the MBPP statement and the first
    of the MBPP assertions. The statement is stored on its own because it is what the
    contamination gate compares: writing the docstring in as the statement would leave
    the gate comparing a docstring against a sentence, which matches nothing at all and
    reports a clean file whatever the file holds.
    """
    text = str(prompt or "")
    marker = '"""'
    start = text.find(marker)
    if start >= 0:
        rest = text[start + len(marker):]
        end = rest.find(marker)
        text = rest if end < 0 else rest[:end]

    kept: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("assert"):
            continue
        kept.append(stripped)
    return " ".join(" ".join(kept).split())


def python_literal(value: Any) -> str:
    """
    One input argument as Python source the sandbox can evaluate.

    `repr` is not enough on its own: a non-finite float reprs as the bare name `inf`,
    which is not a literal and is not defined in the sandbox. Every type that occurs in
    the MbppPlus v0.2.0 inputs is handled here and an unknown one is a failure, because
    the alternative is a test file that parses and means something else.
    """
    if value is None or value is True or value is False:
        return repr(value)
    if isinstance(value, float):
        if value != value:
            return "float('nan')"
        if value == float("inf"):
            return "float('inf')"
        if value == float("-inf"):
            return "float('-inf')"
        return repr(value)
    if isinstance(value, (int, str, bytes, complex)):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(python_literal(item) for item in value) + "]"
    if isinstance(value, tuple):
        inner = ", ".join(python_literal(item) for item in value)
        return "(" + inner + ("," if len(value) == 1 else "") + ")"
    if isinstance(value, (set, frozenset)):
        name = "set" if isinstance(value, set) else "frozenset"
        if not value:
            return name + "()"
        items = sorted(value, key=repr)
        return name + "([" + ", ".join(python_literal(item) for item in items) + "])"
    if isinstance(value, dict):
        pairs = ", ".join(f"{python_literal(k)}: {python_literal(v)}" for k, v in value.items())
        return "{" + pairs + "}"
    raise SystemExit(
        f"[FAIL] MBPP+: no literal form for an input of type {type(value).__name__}. "
        "Extend python_literal rather than letting the test file mean something else."
    )


def _same_value(left: Any, right: Any) -> bool:
    """Equality that also requires the same type, for the literal round trip check."""
    if type(left) is not type(right):
        return False
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(_same_value(a, b) for a, b in zip(left, right))
    if isinstance(left, dict):
        return list(left.keys()) == list(right.keys()) and all(_same_value(left[k], right[k]) for k in left)
    if isinstance(left, float) and left != left:
        return right != right
    return bool(left == right)


_LITERAL_NAMESPACE: Dict[str, Any] = {
    "__builtins__": {"set": set, "frozenset": frozenset, "complex": complex, "float": float}
}


def _check_literal_round_trip(name: str, value: Any, text: str) -> None:
    """The emitted source has to read back as the value it was emitted from."""
    try:
        back = eval(text, dict(_LITERAL_NAMESPACE))  # noqa: S307  (our own emitted literal)
    except Exception as exc:
        raise SystemExit(f"[FAIL] {name}: the emitted input literal does not parse: {type(exc).__name__}: {exc}")
    if not _same_value(value, back):
        raise SystemExit(f"[FAIL] {name}: the emitted input literal reads back as a different value.")


def _rebind_import_lines(code: str) -> List[str]:
    """
    Statements that restore, after the candidate has run, what the reference imports.

    The executor hoists every import line of `test_setup_code` and runs it before the
    candidate, so the modules the reference uses are bound before the candidate gets a
    chance to rebind those names. These lines bind them again inside the part that runs
    after the candidate, and they are written with `__import__` so that the hoist does
    not take them too. Only the three import forms the MbppPlus canonical solutions use
    are understood; anything else stops the preparation.
    """
    out: List[str] = []
    for raw in str(code or "").splitlines():
        line = raw.strip()
        if not _RE_IMPORT_LINE.match(line):
            continue

        plain = _SIMPLE_IMPORT_RE.match(line)
        if plain is not None:
            module, alias = plain.group(1), plain.group(2)
            out.append(f'{alias or module} = __import__("{module}")')
            continue

        from_form = _SIMPLE_FROM_RE.match(line)
        if from_form is not None:
            module, names = from_form.group(1), from_form.group(2)
            for piece in names.split(","):
                name_form = _SIMPLE_FROM_NAME_RE.match(piece.strip())
                if name_form is None:
                    raise SystemExit(f"[FAIL] MBPP+: cannot rebind the import {line!r}.")
                symbol, alias = name_form.group(1), name_form.group(2)
                out.append(f'{alias or symbol} = __import__("{module}").{symbol}')
            continue

        raise SystemExit(
            f"[FAIL] MBPP+: the import {line!r} is not one of the forms this builder can "
            "restore after the candidate code. Extend _rebind_import_lines."
        )
    return out


def _mbpp_plus_mode(entry_point: str) -> str:
    """Which of EvalPlus's comparisons this task is judged by."""
    if entry_point in MBPP_PLUS_UNHANDLED_ENTRY_POINTS:
        raise SystemExit(
            f"[FAIL] MBPP+: entry point {entry_point!r} needs an EvalPlus special oracle this "
            "builder does not implement. Judging it by equality would report a wrong number."
        )
    if entry_point in MBPP_PLUS_SET_EQ_ENTRY_POINTS:
        return "set_eq"
    if entry_point in MBPP_PLUS_NOT_NONE_ENTRY_POINTS:
        return "not_none"
    if entry_point == "surface_Area":
        return "surface_area"
    if entry_point == "digit_distance_nums":
        return "digit_distance"
    return "plain"


def mbpp_plus_setup_code(task: Dict[str, Any], inputs: List[Any], name: str) -> str:
    """
    The `test_setup_code` of one MBPP+ row: a differential test against the reference.

    Order is the whole design. The executor runs `test_setup_code` before the candidate,
    and when that raises NameError it runs the same block again after the candidate
    instead. The first two statements here capture the candidate's function and then
    delete the name, which raises NameError while no candidate has run yet, so the whole
    block lands after the candidate on the second attempt. Everything the tests depend
    on is therefore defined after the candidate and cannot be forged by it: a candidate
    that defines its own `_c3_case` is simply overwritten. The `del` is what makes this
    work for `Mbpp/126` as well, whose entry point is `sum`: a bare reference to it
    would find the builtin and raise nothing.

    Expected values are not stored. They are computed here by running the reference on
    the same input, which is how the whole of MBPP+ fits in a few megabytes: writing
    them out comes to 1.3 GB, because one problem returns a value that serializes to
    1.29 GB by itself.
    """
    entry = str(task["entry_point"])
    if not entry.isidentifier() or keyword.iskeyword(entry):
        raise SystemExit(f"[FAIL] {name}: {entry!r} is not usable as a function name.")

    solution = str(task.get("prompt") or "") + str(task.get("canonical_solution") or "")
    literal = python_literal(inputs)
    _check_literal_round_trip(name, inputs, literal)

    mode = _mbpp_plus_mode(entry)
    atol = task.get("atol", 0)

    parts: List[str] = [
        "# Generated by scripts/10_data/prepare_code.py. The two statements below raise",
        "# NameError until the candidate has run, which is what moves this whole block",
        "# after it; see the MBPP+ section of docs/30_data_sources.md.",
        f"_C3_FN = {entry}",
        f"del {entry}",
    ]
    parts.extend(_rebind_import_lines(solution))
    parts.append(solution.strip("\n"))
    parts.append(f"_C3_REF = {entry}")
    parts.append(f"_C3_MODE = {mode!r}")
    parts.append(f"_C3_ATOL = {atol!r}")
    parts.append(f"_C3_ATOL_STATE = {atol!r}")
    parts.append(f"_C3_INPUTS = {literal}")
    oracle = _MBPP_PLUS_ORACLES.get(entry)
    if oracle:
        parts.append(oracle.strip("\n"))
    parts.append(_MBPP_PLUS_HELPERS.strip("\n"))

    setup = "\n".join(parts) + "\n"
    _check_setup_compiles(name, setup)
    return setup


def _check_setup_compiles(name: str, setup: str) -> None:
    """
    Both halves of the executor's import split have to compile.

    The executor moves every import line of the setup to the front and runs the rest as
    one block. A canonical solution with an indented import therefore loses that line
    from its block, which can leave an empty one. None of the MbppPlus v0.2.0 solutions
    does, and this check is what keeps that a fact rather than an assumption.
    """
    imports, rest = _split_setup_imports(setup)
    for half, text in (("hoisted imports", imports), ("the rest", rest)):
        if not text.strip():
            continue
        try:
            compile(text, f"<{name} setup>", "exec")
        except SyntaxError as exc:
            raise SystemExit(
                f"[FAIL] {name}: {half} of the generated test_setup_code does not compile "
                f"after the executor's import split: {exc}"
            )


def canon_mbpp_plus_row(task: Dict[str, Any], source: str) -> Dict[str, Any]:
    """
    One EvalPlus MBPP+ task as a row of `data/MBPP_PLUS/test.jsonl`.

    EvalPlus key on the left, the key a consumer of this repository reads on the right:

      prompt              -> text              (the statement, out of the docstring)
      canonical_solution  -> code              (with the prompt, as EvalPlus runs it)
      entry_point         -> entry_point
      atol                -> atol
      base_input          -> test_setup_code and one entry of test_list each
      plus_input          -> test_setup_code and one entry of test_list each
      task_id             -> task_id           (the integer, not 'Mbpp/100')

    `assertion` and `contract` are not carried: the first is the three original MBPP
    assertions, which are the base tests this row already runs against the reference,
    and the second is input validation for EvalPlus's own input generator.
    """
    entry = str(task["entry_point"])
    task_id = mbpp_plus_task_id(task.get("task_id"))
    name = f"MBPP+ Mbpp/{task_id}"

    base_inputs = list(task.get("base_input") or [])
    plus_inputs = list(task.get("plus_input") or [])
    inputs = base_inputs + plus_inputs
    if not inputs:
        raise SystemExit(f"[FAIL] {name}: the task carries no inputs at all.")

    tests = [f"assert _c3_case({index})" for index in range(len(inputs))]
    budget = sum(len(test) for test in tests)
    if budget > MBPP_PLUS_ASSERT_BUDGET:
        raise SystemExit(
            f"[FAIL] {name}: its {len(tests)} tests are {budget} characters, over the "
            f"{MBPP_PLUS_ASSERT_BUDGET} the evaluator allows one task."
        )

    return {
        "task_id": task_id,
        "text": mbpp_plus_statement(task.get("prompt")),
        "code": str(task.get("prompt") or "") + str(task.get("canonical_solution") or ""),
        "entry_point": entry,
        "atol": task.get("atol", 0),
        "n_base_inputs": len(base_inputs),
        "n_plus_inputs": len(plus_inputs),
        "test_setup_code": mbpp_plus_setup_code(task, inputs=inputs, name=name),
        "test_list": tests,
        "timeout_s": MBPP_PLUS_TIMEOUT_S,
        "source": source,
    }


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

    # --mbpp_plus_seed and --mbpp_plus_max_tests are gone with the row builder they
    # belonged to: they sampled a subset of an MBPP+ test list that was never read.
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
            if strict:
                # A file that is skipped is still a file the evaluation will read, so it
                # faces the same gate a written one does. The empty MBPP+ artifact was
                # skipped and verified on every rerun after the first.
                _check_rows_complete_on_disk(name, out_path)
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
                    f"[FAIL] {name}: source.revision must be pinned as 'evalplus@<VERSION>' "
                    f"(got '{rev}')."
                )

            expected_ver = str(rev).strip().split("@", 1)[1]
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
            n = _write_checked("HumanEval", out_path, rows)
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
            n = _write_checked("APPS", out_path, rows)
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
    # The order below is the contamination rule rather than a convenience: both
    # evaluation artifacts are prepared first, and the training artifact is written
    # minus every problem that occurs in either of them. MBPP+ is drawn from the
    # sanitized MBPP and 108 of its 378 task ids are MBPP train-split ids, so leaving it
    # out of the subtraction leaves real contamination in the training file.
    eval_problem_keys: set = set()

    def _prepare_mbpp_split(name: str) -> Optional[Tuple[Dict[str, Any], Path, str]]:
        """The manifest entry, output path and source string of one MBPP split, or None."""
        if name not in idx:
            return None
        spec = _require(name)
        _check_pinned(name, spec)
        src = spec.get("source", {})
        out_path = _resolve_output_path(spec["output_path"], out_base)
        src_str = f"{src['id']}:{src.get('split')}@{src.get('revision')}"
        return spec, out_path, src_str

    def _record_written(name: str, spec: Dict[str, Any], out_path: Path, n: int) -> None:
        sha = _sha256(out_path)
        _verify_or_record(
            name, spec.get("sha256"), sha,
            missing_sha=missing_sha, computed_by_name=computed_by_name, out_path=out_path,
            allow_repin_on_mismatch=allow_repin_on_mismatch,
        )
        print(f"[OK] {name} -> {out_path} ({n} rows, sha256 {sha})")

    # ---------------- MBPP-test ----------------
    if bool(args.prepare_mbpp):
        prepared = _prepare_mbpp_split("MBPP-test")
        if prepared is not None:
            spec, out_path, src_str = prepared
            if _handle_existing("MBPP-test", spec, out_path):
                eval_problem_keys |= _problem_keys_from_file(out_path)
            else:
                rows = [_canon_mbpp_row(x, src_str) for x in _load_hf_rows(spec.get("source", {}), "MBPP-test")]
                eval_problem_keys |= {k for k in (_problem_key(r) for r in rows) if k}
                _record_written("MBPP-test", spec, out_path, _write_checked("MBPP-test", out_path, rows))
    else:
        print("[SKIP] MBPP disabled (--prepare_mbpp 0).")

    # ---------------- MBPP+ ----------------
    plus_spec = idx.get("MBPP+")
    if plus_spec is None:
        raise SystemExit("[FAIL] Manifest missing output entry: MBPP+")
    plus_out = _resolve_output_path(plus_spec["output_path"], out_base)

    if bool(args.prepare_mbpp_plus):
        _check_pinned("MBPP+", plus_spec)
        if _handle_existing("MBPP+", plus_spec, plus_out):
            eval_problem_keys |= _problem_keys_from_file(plus_out)
        else:
            tasks = _evalplus_mbpp_plus_tasks()
            if tasks is None:
                raise SystemExit(
                    "[FAIL] MBPP+: EvalPlus could not be loaded, and there is no substitute.\n"
                    "  EvalPlus downloads MBPP+ from a GitHub release. If your network cannot\n"
                    "  reach github.com, set GITHUB_MIRROR_PREFIX (see docs/31_network_mirrors.md)\n"
                    "  and rerun through scripts/10_data/prepare_all.sh, which seeds the EvalPlus\n"
                    "  cache through the mirror. Preparing the MBPP test problems under the MBPP+\n"
                    "  name was the former fallback; it is gone, because a benchmark that holds\n"
                    "  something else is worse than a benchmark that is missing."
                )
            version = _get_evalplus_version()
            if not version:
                raise SystemExit("[FAIL] MBPP+: EvalPlus is importable but reports no version.")

            src_str = f"evalplus:mbpp_plus@{version}"
            rows = [canon_mbpp_plus_row(task, src_str) for task in tasks]
            n_base = sum(int(r["n_base_inputs"]) for r in rows)
            n_plus = sum(int(r["n_plus_inputs"]) for r in rows)
            print(
                f"[INFO] MBPP+: {len(rows)} tasks, {n_base} base inputs and {n_plus} plus inputs, "
                f"one test each, judged against the reference solution at evaluation time"
            )
            eval_problem_keys |= {k for k in (_problem_key(r) for r in rows) if k}
            computed_src_revision_by_name["MBPP+"] = f"evalplus@{version}"
            _record_written("MBPP+", plus_spec, plus_out, _write_checked("MBPP+", plus_out, rows))
    else:
        print("[SKIP] MBPP+ disabled (--prepare_mbpp_plus 0).")
        if plus_out.exists():
            # Not preparing it does not make its problems safe to train on.
            eval_problem_keys |= _problem_keys_from_file(plus_out)

    # ---------------- MBPP-train ----------------
    if bool(args.prepare_mbpp):
        prepared = _prepare_mbpp_split("MBPP-train")
        if prepared is not None:
            spec, out_path, src_str = prepared
            if not _handle_existing("MBPP-train", spec, out_path):
                if not eval_problem_keys:
                    raise SystemExit(
                        "[FAIL] MBPP-train: the evaluation problems are not available for "
                        "subtraction. Prepare MBPP-test and MBPP+ first."
                    )
                rows = (_canon_mbpp_row(x, src_str) for x in _load_hf_rows(spec.get("source", {}), "MBPP-train"))
                kept = _subtract_eval_problems("MBPP-train", rows, eval_problem_keys)
                _record_written("MBPP-train", spec, out_path, _write_checked("MBPP-train", out_path, kept))

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
