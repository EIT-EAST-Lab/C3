#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/audit/no_data_check.py

Fail if the repository contains bundled raw datasets or large artifacts that
should not be distributed (weights, cached datasets, etc.).

Policy (default):
- forbid non-empty generated output directories in the repo root
- forbid common weight/binary extensions
- forbid files over a size threshold

Usage:
  python scripts/audit/no_data_check.py [--root .] [--max_mb 20]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple


DEFAULT_EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "build",
    "dist",
}

NON_DISTRIBUTABLE_DIRS = {
    "data": "Generated datasets must not be committed",
    "artifacts": "Generated reports/plots must not be committed",
    "ckpt": "Generated checkpoints must not be committed",
    "runs": "Generated run directories must not be committed",
    "wandb": "Generated Weights & Biases outputs must not be committed",
    "models": "Downloaded or cached model files must not be committed",
}

FORBIDDEN_EXTS = {
    ".pt",
    ".pth",
    ".ckpt",
    ".bin",
    ".safetensors",
    ".onnx",
    ".so",
    ".dll",
    ".dylib",
    ".parquet",
    ".arrow",
    ".npz",
    ".npy",
    ".pkl",
    ".joblib",
    ".msgpack",
    ".tar",
    ".tgz",
    ".tar.gz",
    ".zip",
    ".7z",
    ".csv",
    ".tsv",
    ".jsonl",
    ".jsonl.gz",
    ".gz",
}

BINARY_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".pdf"}
ALLOWED_FIXTURE_PREFIXES = {
    ("tests", "fixtures"),
    ("examples", "fixtures"),
}


def _is_in_excluded_dir(rel: Path) -> bool:
    parts = set(rel.parts)
    return any(d in parts for d in DEFAULT_EXCLUDE_DIRS)


def _is_allowed_fixture(rel: Path) -> bool:
    parts = rel.parts
    return any(parts[: len(prefix)] == prefix for prefix in ALLOWED_FIXTURE_PREFIXES)


def _scan_files(root: Path, max_bytes: int) -> List[Tuple[Path, str]]:
    hits: List[Tuple[Path, str]] = []
    root = root.resolve()

    # Rule 1: forbid non-empty generated output directories in repo root.
    for dirname, reason in NON_DISTRIBUTABLE_DIRS.items():
        target_dir = root / dirname
        if not target_dir.exists():
            continue
        non_empty = [p for p in target_dir.rglob("*") if p.is_file() and p.stat().st_size > 0]
        if non_empty:
            hits.append((target_dir, reason))

    for p in root.rglob("*"):
        if not p.is_file():
            continue
        try:
            rel = p.relative_to(root)
        except Exception:
            continue

        if _is_in_excluded_dir(rel):
            continue

        if rel.parts and rel.parts[0] in NON_DISTRIBUTABLE_DIRS:
            continue

        if _is_allowed_fixture(rel):
            continue

        suf = p.suffix.lower()
        if suf in BINARY_SUFFIXES:
            continue

        # Special-case double suffix like .tar.gz
        name = p.name.lower()
        if name.endswith(".tar.gz") or name.endswith(".jsonl.gz"):
            suf2 = "." + ".".join(name.split(".")[-2:])
            if suf2 in FORBIDDEN_EXTS:
                hits.append((rel, f"Forbidden archive/artifact extension: {suf2}"))
                continue

        if suf in FORBIDDEN_EXTS:
            hits.append((rel, f"Forbidden artifact extension: {suf}"))
            continue

        try:
            if p.stat().st_size > max_bytes:
                hits.append((rel, f"File too large: {p.stat().st_size} bytes"))
        except OSError:
            continue

    return hits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".", help="Repo root to scan (default: .)")
    ap.add_argument("--max_mb", type=int, default=20, help="Max allowed file size in MB (default: 20)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    max_bytes = int(args.max_mb) * 1024 * 1024

    hits = _scan_files(root, max_bytes=max_bytes)
    if hits:
        print("[FAIL] Found forbidden bundled data / artifacts:")
        for rel, reason in hits[:50]:
            print(f"  - {rel}: {reason}")
        if len(hits) > 50:
            print(f"  ... ({len(hits)-50} more)")
        raise SystemExit(2)

    print("[OK] No bundled datasets / large artifacts found.")


if __name__ == "__main__":
    main()
