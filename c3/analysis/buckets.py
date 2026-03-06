# c3/analysis/buckets.py
"""
Bucket JSONL IO + schema validation.

Design goals:
- Stable, explicit on-disk schema for replay / rollout-derived buckets.
- Streaming read/write (large files), deterministic compact JSON.
- Validation with actionable error paths (fails fast, no silent coercion).
- Accepts either dict-like buckets or dataclass buckets from analysis.replay.

Schema (one JSON object per line):
{
  "bucket_id": "...",
  "ctx_hash": 123,
  "target_role": "Actor",
  "question_id": "...",
  "restart": {
    "roles_topo": ["Reasoner","Actor"],
    "role_outputs_prefix": {"Reasoner":"..."}
  },
  "candidates": [
    {"j": 0, "action_text": "...", "returns": [1.0], "next_actions": ["..."]}
  ],
  "meta": {...}
}

Notes:
- We allow extra keys anywhere for forward-compat, but required keys must exist.
- Writer normalizes buckets into the schema (adds missing j/bucket_id/question_id).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Sequence, Tuple, Union

import numpy as np

from c3.utils.collision_guard import ContextKeyCollisionError, global_guard
from c3.utils.context_key import fingerprint

JsonDict = Dict[str, Any]
BucketLike = Union[Mapping[str, Any], Any]  # dict-like or dataclass bucket


__all__ = [
    "BucketValidationError",
    "write_buckets_jsonl",
    "read_buckets_jsonl",
    "validate_bucket",
    "aggregate_candidate_returns",
]


# -------------------------
# Errors
# -------------------------


class BucketValidationError(ValueError):
    """Schema validation error with a JSON-pointer-ish path."""

    def __init__(self, message: str, *, path: str = "$") -> None:
        super().__init__(f"{message} @ {path}")
        self.path = path


# -------------------------
# JSON helpers
# -------------------------


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash16_hex(text: str) -> str:
    h = hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()
    return h


def _compute_bucket_id(d: Mapping[str, Any]) -> str:
    """
    Deterministic bucket_id to avoid relying on upstream naming.
    Includes only "identity" fields, not returns, so it's stable across reruns.
    """
    meta = d.get("meta") if isinstance(d.get("meta"), Mapping) else {}
    payload = {
        "ctx_hash": d.get("ctx_hash"),
        "question_id": d.get("question_id"),
        "target_role": d.get("target_role"),
        "seed": meta.get("seed"),
        "actions": [c.get("action_text") for c in d.get("candidates", []) if isinstance(c, Mapping)],
    }
    return f"bkt_{_hash16_hex(_json_dumps(payload))}"


def _context_identity_string(d: Mapping[str, Any]) -> str:
    """Derive a stable identity string for a bucket context.

    We intentionally exclude candidate actions/returns so that multiple bucket
    files (or different sampling configs) for the same context share the same
    identity.
    """

    restart = d.get("restart") if isinstance(d.get("restart"), Mapping) else {}
    role_outputs_prefix = restart.get("role_outputs_prefix") if isinstance(restart.get("role_outputs_prefix"), Mapping) else {}

    payload: Dict[str, Any] = {
        "question_id": d.get("question_id"),
        "target_role": d.get("target_role"),
        "roles_topo": restart.get("roles_topo"),
        "role_outputs_prefix": dict(sorted((str(k), str(v)) for k, v in role_outputs_prefix.items())),
    }

    # If question text is available (e.g., dataclass RestartState), include it.
    q = restart.get("question")
    if isinstance(q, str) and q:
        payload["question"] = q

    return _json_dumps(payload)


def _to_plain_dict(obj: BucketLike) -> JsonDict:
    """
    Convert bucket-like objects into a plain dict.
    Supports:
    - dict / mapping
    - dataclass instances
    - objects exposing .to_dict()
    """
    if isinstance(obj, Mapping):
        return dict(obj)

    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        out = to_dict()
        if not isinstance(out, Mapping):
            raise TypeError("bucket.to_dict() must return a mapping")
        return dict(out)

    if is_dataclass(obj):
        return asdict(obj)

    # Best-effort: public attrs to dict (avoid pulling in huge objects).
    if hasattr(obj, "__dict__"):
        return dict(getattr(obj, "__dict__"))

    raise TypeError("Unsupported bucket type; expected mapping/dataclass/.to_dict()")


def _normalize_bucket_for_write(bucket: BucketLike) -> JsonDict:
    """
    Normalize a bucket into the stable JSONL schema.

    - Ensures top-level question_id exists (pull from restart if needed).
    - Ensures each candidate has integer j (enumerate if missing).
    - Ensures meta exists (dict).
    - Ensures bucket_id exists (compute if missing).
    """
    d = _to_plain_dict(bucket)

    # Ensure restart is dict-like.
    restart = d.get("restart")
    if is_dataclass(restart):
        restart = asdict(restart)
    elif restart is None:
        restart = {}
    elif not isinstance(restart, Mapping):
        # Some upstream may store restart_state under another name.
        raise TypeError("bucket.restart must be a mapping or dataclass")
    d["restart"] = dict(restart)

    # question_id: prefer explicit, else restart.question_id.
    if "question_id" not in d:
        if "question_id" in d["restart"]:
            d["question_id"] = d["restart"]["question_id"]
        else:
            # Try common legacy key
            qid = d.get("qid") or d.get("instance_id")
            if qid is not None:
                d["question_id"] = qid

    # target_role: keep if present; else try bucket.target_role, or meta hint.
    if "target_role" not in d:
        tr = getattr(bucket, "target_role", None)
        if isinstance(tr, str) and tr:
            d["target_role"] = tr
        else:
            meta = d.get("meta")
            if isinstance(meta, Mapping) and isinstance(meta.get("target_role"), str):
                d["target_role"] = meta["target_role"]

    # meta: must be a dict for schema stability.
    meta = d.get("meta")
    if meta is None:
        meta = {}
    if not isinstance(meta, Mapping):
        meta = {"meta_raw": meta}
    d["meta"] = dict(meta)

    # candidates: ensure list[dict] with j.
    cands = d.get("candidates")
    if cands is None:
        cands = []
    if not isinstance(cands, Sequence) or isinstance(cands, (str, bytes, bytearray)):
        raise TypeError("bucket.candidates must be a sequence")
    norm_cands: List[JsonDict] = []
    for idx, c in enumerate(cands):
        if is_dataclass(c):
            c = asdict(c)
        elif not isinstance(c, Mapping):
            raise TypeError("each candidate must be a mapping or dataclass")

        cd = dict(c)
        if "j" not in cd:
            cd["j"] = idx
        # Default optional fields.
        cd.setdefault("next_actions", [])
        norm_cands.append(cd)
    d["candidates"] = norm_cands

    # bucket_id
    bid = d.get("bucket_id")
    if not isinstance(bid, str) or not bid:
        d["bucket_id"] = _compute_bucket_id(d)

    return d


# -------------------------
# Public API
# -------------------------


def write_buckets_jsonl(path: str | os.PathLike[str], buckets_iter: Iterable[BucketLike], *, overwrite: bool = False) -> None:
    """
    Write buckets as JSONL, one bucket per line.

    - Normalizes each bucket into the stable schema.
    - Writes compact deterministic JSON (sorted keys, no extra whitespace).
    """
    p = Path(path)
    if p.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {p}")

    p.parent.mkdir(parents=True, exist_ok=True)

    with p.open("w", encoding="utf-8") as f:
        for b in buckets_iter:
            d = _normalize_bucket_for_write(b)
            validate_bucket(d)  # fail early with precise path
            f.write(_json_dumps(d))
            f.write("\n")


def read_buckets_jsonl(path: str | os.PathLike[str]) -> Iterator[JsonDict]:
    """
    Stream buckets from JSONL.

    Skips empty lines and lines starting with '#'.
    Raises ValueError with line number for invalid JSON.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {p}:{lineno}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object at {p}:{lineno}, got {type(obj).__name__}")
            yield obj


def validate_bucket(d: Mapping[str, Any]) -> None:
    """
    Validate required schema fields. Allows extra keys for forward-compat.
    Raises BucketValidationError on violation.
    """

    def req(cond: bool, msg: str, path: str) -> None:
        if not cond:
            raise BucketValidationError(msg, path=path)

    # bucket_id
    bid = d.get("bucket_id")
    req(isinstance(bid, str) and bool(bid.strip()), "bucket_id must be a non-empty string", "$.bucket_id")

    # ctx_hash
    ctx_hash = d.get("ctx_hash")
    req(isinstance(ctx_hash, int), "ctx_hash must be int", "$.ctx_hash")
    req(0 <= ctx_hash < (1 << 63), "ctx_hash must be in [0, 2^63)", "$.ctx_hash")

    # target_role
    tr = d.get("target_role")
    req(isinstance(tr, str) and bool(tr.strip()), "target_role must be a non-empty string", "$.target_role")

    # question_id
    qid = d.get("question_id")
    req(isinstance(qid, (str, int)) and str(qid) != "", "question_id must be str|int and non-empty", "$.question_id")

    # restart
    restart = d.get("restart")
    req(isinstance(restart, Mapping), "restart must be an object", "$.restart")
    roles_topo = restart.get("roles_topo")
    req(isinstance(roles_topo, list) and all(isinstance(x, str) and x for x in roles_topo), "restart.roles_topo must be list[str]", "$.restart.roles_topo")
    role_outputs_prefix = restart.get("role_outputs_prefix")
    req(isinstance(role_outputs_prefix, Mapping), "restart.role_outputs_prefix must be an object", "$.restart.role_outputs_prefix")
    for k, v in role_outputs_prefix.items():
        req(isinstance(k, str) and k, "role_outputs_prefix keys must be non-empty strings", "$.restart.role_outputs_prefix")
        req(isinstance(v, str), "role_outputs_prefix values must be strings", f"$.restart.role_outputs_prefix.{k}")

    # Context-key collision guard.
    # If two different contexts ever share the same ctx_hash, downstream
    # aggregation becomes ambiguous; we fail fast with a clear error.
    try:
        ident = _context_identity_string(d)
        global_guard().observe(int(ctx_hash), fingerprint(ident), where="buckets.validate_bucket")
    except ContextKeyCollisionError as e:
        raise BucketValidationError(str(e), path="$.ctx_hash") from e

    # candidates
    cands = d.get("candidates")
    req(isinstance(cands, list) and len(cands) > 0, "candidates must be a non-empty list", "$.candidates")

    js: List[int] = []
    for i, c in enumerate(cands):
        path_i = f"$.candidates[{i}]"
        req(isinstance(c, Mapping), "candidate must be an object", path_i)

        j = c.get("j")
        req(isinstance(j, int) and j >= 0, "candidate.j must be int >= 0", f"{path_i}.j")
        js.append(j)

        a = c.get("action_text")
        req(isinstance(a, str), "candidate.action_text must be string", f"{path_i}.action_text")

        r = c.get("returns")
        req(isinstance(r, list) and len(r) > 0, "candidate.returns must be non-empty list[float]", f"{path_i}.returns")
        for k, rv in enumerate(r):
            try:
                fv = float(rv)
            except Exception as e:
                raise BucketValidationError("returns entries must be numeric", path=f"{path_i}.returns[{k}]") from e
            req(np.isfinite(fv).item(), "returns entries must be finite", f"{path_i}.returns[{k}]")

        na = c.get("next_actions", [])
        req(isinstance(na, list), "candidate.next_actions must be list[str] (or omitted)", f"{path_i}.next_actions")
        for k, y in enumerate(na):
            req(isinstance(y, str), "next_actions entries must be strings", f"{path_i}.next_actions[{k}]")

    # Enforce contiguous j = 0..n-1 to make downstream aggregation unambiguous.
    n = len(cands)
    req(len(set(js)) == n, "candidate.j must be unique within a bucket", "$.candidates[*].j")
    req(sorted(js) == list(range(n)), "candidate.j must be contiguous 0..n-1", "$.candidates[*].j")

    # meta
    meta = d.get("meta")
    req(isinstance(meta, Mapping), "meta must be an object", "$.meta")


def aggregate_candidate_returns(bucket: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregate per-candidate mean return and sample count.

    Returns:
      barR: shape [N], barR[j] = mean(returns_j)
      counts: shape [N], counts[j] = len(returns_j)

    Preconditions: validate_bucket(bucket) passes.
    """
    validate_bucket(bucket)

    cands = bucket["candidates"]
    n = len(cands)
    barR = np.zeros((n,), dtype=np.float64)
    counts = np.zeros((n,), dtype=np.int64)

    # j is guaranteed contiguous, so direct indexing is safe.
    for c in cands:
        j = int(c["j"])
        rets = c["returns"]
        counts[j] = len(rets)
        barR[j] = float(np.mean(np.asarray(rets, dtype=np.float64)))

    return barR, counts
