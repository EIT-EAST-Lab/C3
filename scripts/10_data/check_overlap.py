#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/10_data/check_overlap.py

Fail when an evaluation problem also occurs in a training file.

The manifest lists every prepared artifact. This gate reads the prepared JSONL
files it points at, normalizes each problem statement, and intersects every
training file with every evaluation file. Any non-empty intersection is a
contamination bug: the reported numbers would then be measured on problems the
model was trained on.

It exists because that bug already happened here, twice, and both times a
silent fallback hid it: a training artifact pinned to a mirror that shipped the
whole benchmark under the name of its train split, and an evaluation artifact
prepared from a revision that had no test split at all. A mechanical check is
cheap; reading the numbers wrong is not.

Train and evaluation files are told apart by the manifest naming contract in
docs/30_data_sources.md: an artifact whose name ends in `-train` is a training
file, every other artifact is an evaluation file.

Standard library only, so it runs anywhere the preparation scripts run and
needs nothing installed.

Usage:
  python scripts/10_data/check_overlap.py [--manifest configs/data_manifest.yaml]
                                          [--out_dir data] [--max_examples 3]

Exit codes:
  0  no evaluation problem occurs in any training file
  1  at least one does
  2  the check could not be run (manifest unreadable, nothing to compare)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


# Same order as QUESTION_KEY_CANDIDATES in scripts/10_data/prepare_math.py and
# _PROMPT_KEY_CANDIDATES in c3/task/datasets.py: "the question" must
# mean the same string in the gate and in what the trainer feeds the model.
QUESTION_KEY_CANDIDATES: Tuple[str, ...] = ("input", "question", "problem", "prompt", "text")

TRAIN_NAME_SUFFIX = "-train"


# -----------------------------
# Question normalization
# -----------------------------


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


# -----------------------------
# Manifest reading (no third-party YAML)
# -----------------------------

_LIST_ITEM = re.compile(r"^(?P<indent>[ ]*)-[ ]+(?P<rest>\S.*)$")
_KEY_VALUE = re.compile(r"^(?P<indent>[ ]*)(?P<key>[A-Za-z_][A-Za-z0-9_]*)[ ]*:[ ]*(?P<value>.*)$")


def _scalar(raw: str) -> Optional[str]:
    """One plain or quoted YAML scalar, as text; null and empty become None."""
    value = raw.strip()
    if value[:1] in {"'", '"'} and value[-1:] == value[:1] and len(value) >= 2:
        quote = value[0]
        inner = value[1:-1]
        return inner.replace(quote * 2, quote) if quote == "'" else inner
    cut = value.find(" #")
    if cut >= 0:
        value = value[:cut].strip()
    if value in {"", "null", "~"}:
        return None
    return value


def parse_manifest_outputs(text: str) -> List[Dict[str, Optional[str]]]:
    """
    The `outputs` entries of configs/data_manifest.yaml, as flat dicts.

    This reads the exact shape the manifest is written and rewritten in (a
    top-level `outputs:` key holding a list of mappings, each with a nested
    `source:` mapping), not YAML in general. Only the entry-level scalars are
    returned; nested blocks are skipped. Keeping the gate free of a YAML
    dependency is worth this much parsing, and `prepare_math.py` still rewrites
    the manifest with a real YAML dumper.
    """
    entries: List[Dict[str, Optional[str]]] = []
    in_outputs = False
    item_indent: Optional[int] = None
    current: Optional[Dict[str, Optional[str]]] = None

    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue

        top = _KEY_VALUE.match(raw)
        if top is not None and not top.group("indent"):
            if top.group("key") == "outputs":
                in_outputs = True
                item_indent = None
                current = None
                continue
            if in_outputs:
                break
            continue

        if not in_outputs:
            continue

        item = _LIST_ITEM.match(raw)
        if item is not None:
            indent = len(item.group("indent"))
            if item_indent is None:
                item_indent = indent
            if indent != item_indent:
                # A nested list such as source.configs, not an output entry.
                continue
            current = {}
            entries.append(current)
            first = _KEY_VALUE.match(item.group("rest"))
            if first is not None and not first.group("indent"):
                current[first.group("key")] = _scalar(first.group("value"))
            continue

        if current is None or item_indent is None:
            continue

        pair = _KEY_VALUE.match(raw)
        if pair is None or len(pair.group("indent")) != item_indent + 2:
            continue
        current[pair.group("key")] = _scalar(pair.group("value"))

    return entries


def resolve_output_path(output_path: str, out_dir: Path) -> Path:
    """Same rule as prepare_math.py: a leading `data/` is replaced by out_dir."""
    p = Path(output_path)
    if len(p.parts) >= 1 and p.parts[0] == "data":
        return out_dir / Path(*p.parts[1:])
    return out_dir / p


def is_train_artifact(name: str) -> bool:
    return str(name).endswith(TRAIN_NAME_SUFFIX)


# -----------------------------
# File reading
# -----------------------------


def load_questions(path: Path) -> Tuple[Set[str], int, int]:
    """Normalized problem statements in one JSONL file, with row counts."""
    keys: Set[str] = set()
    rows = 0
    blank = 0
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            rows += 1
            try:
                row = json.loads(line)
            except Exception as exc:
                raise ValueError(f"{path}: line {lineno} is not valid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}: line {lineno} is not a JSON object")
            key = question_key(row)
            if key:
                keys.add(key)
            else:
                blank += 1
    return keys, rows, blank


def _shorten(text: str, width: int = 100) -> str:
    return text if len(text) <= width else text[: width - 3] + "..."


# -----------------------------
# Main
# -----------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, default="configs/data_manifest.yaml", help="Path to data manifest")
    ap.add_argument("--out_dir", type=str, default="", help="Prepared data root (default: data)")
    ap.add_argument("--data_dir", type=str, default="", help="Alias of --out_dir")
    ap.add_argument("--max_examples", type=int, default=3, help="Overlapping problems printed per failing pair")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    manifest_path = Path(args.manifest).resolve()
    out_dir = Path((args.out_dir or args.data_dir or "data").strip()).resolve()

    if not manifest_path.exists():
        print(f"[FAIL] manifest not found: {manifest_path}")
        return 2

    entries = parse_manifest_outputs(manifest_path.read_text(encoding="utf-8"))
    named = [e for e in entries if e.get("name") and e.get("output_path")]
    if not named:
        print(f"[FAIL] no usable output entries in {manifest_path}")
        return 2

    print(f"[INFO] manifest: {manifest_path}")
    print(f"[INFO] data dir: {out_dir}")

    train: List[Tuple[str, Path, Set[str]]] = []
    evals: List[Tuple[str, Path, Set[str]]] = []
    missing: List[str] = []

    for entry in named:
        name = str(entry["name"])
        path = resolve_output_path(str(entry["output_path"]), out_dir)
        if not path.exists():
            missing.append(f"{name} -> {path}")
            continue
        try:
            keys, rows, blank = load_questions(path)
        except ValueError as exc:
            print(f"[FAIL] {exc}")
            return 2
        if rows and not keys:
            print(
                f"[FAIL] {name}: {rows} rows but no problem statement found in any of them "
                f"({', '.join(QUESTION_KEY_CANDIDATES)}). The gate cannot check this file."
            )
            return 2
        if blank:
            print(f"[WARN] {name}: {blank} of {rows} rows carry no problem statement and are not checked.")
        bucket = train if is_train_artifact(name) else evals
        bucket.append((name, path, keys))
        role = "train" if is_train_artifact(name) else "eval "
        print(f"[INFO] {role} {name}: {len(keys)} distinct problems in {rows} rows ({path})")

    for item in missing:
        print(f"[SKIP] not prepared: {item}")

    if not train or not evals:
        print(
            "[FAIL] nothing to compare: "
            f"{len(train)} training file(s) and {len(evals)} evaluation file(s) were readable. "
            "Prepare the data first; an empty check is not a passing check."
        )
        return 2

    pairs: List[Tuple[str, str, Set[str]]] = []
    for train_name, _train_path, train_keys in train:
        for eval_name, _eval_path, eval_keys in evals:
            pairs.append((train_name, eval_name, eval_keys & train_keys))

    width = max(len(f"{t} x {e}") for t, e, _ in pairs)
    print("")
    print("[INFO] evaluation problems found in each training file:")
    for train_name, eval_name, shared in pairs:
        label = f"{train_name} x {eval_name}"
        print(f"  {label.ljust(width)} : {len(shared)}")

    bad = [(t, e, s) for t, e, s in pairs if s]
    if not bad:
        print("")
        print(f"[OK] no evaluation problem occurs in any training file ({len(pairs)} pairs checked).")
        return 0

    print("")
    print("[FAIL] evaluation problems occur in training files:")
    for train_name, eval_name, shared in bad:
        print(f"  {train_name} x {eval_name}: {len(shared)}")
        for example in sorted(shared)[: max(0, int(args.max_examples))]:
            print(f"    {_shorten(example)}")
    print("")
    print(f"[FAIL] {len(bad)} of {len(pairs)} pairs overlap. Fix the manifest or the preparation, then rerun.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
