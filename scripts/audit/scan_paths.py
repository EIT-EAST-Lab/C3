#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/audit/scan_paths.py

Fail if the repository contains hard-coded private/cluster/local absolute paths.

Notes
-----
We only flag *absolute* paths that appear in file contents (not the file's own
location on disk). To avoid false positives, we require the path to be preceded
by a boundary that typically introduces an absolute path (start-of-line,
whitespace, quote, backtick, '=', ':', '(').

Usage:
  python scripts/audit/scan_paths.py [--root .]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Tuple


DEFAULT_EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "build",
    "dist",
    "ckpt",
    "artifacts",
    "data",
    "runs",
    "wandb",
}

# Absolute Unix paths: require a boundary before the leading slash to avoid
# matching relative paths like "../roles/code/..." or "abc/code/..."
_UNIX_ABS = re.compile(
    r"(?:(?<=^)|(?<=[\s\"'`=:(]))/(code|mnt|home|Users|private|Volumes)/[^\s\"'`]+"
)

# file:// URLs that embed absolute paths
_FILE_URL = re.compile(r"file:///(code|mnt|home|Users|private|Volumes)/[^\s\"'`]+")

# Windows absolute paths (require at least one additional path component)
# This avoids false positives like "e:\n" (escape sequences in strings).
_WINDOWS_ABS = re.compile(r"\b[A-Za-z]:\\(?:[^\\\s\"']+\\)+[^\\\s\"']+")

PATH_PATTERNS = [_UNIX_ABS, _FILE_URL, _WINDOWS_ABS]

BINARY_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".pt", ".bin", ".safetensors", ".so"}


def _iter_text_files(root: Path) -> Iterable[Path]:
    root = root.resolve()
    for p in root.rglob("*"):
        if not p.is_file():
            continue

        try:
            rel = p.relative_to(root)
        except Exception:
            continue

        parts = set(rel.parts)

        # Avoid self-matching on regex literals inside audit scripts.
        if len(rel.parts) >= 2 and rel.parts[0] == "scripts" and rel.parts[1] == "audit":
            continue

        if any(d in parts for d in DEFAULT_EXCLUDE_DIRS):
            continue

        if p.suffix.lower() in BINARY_SUFFIXES:
            continue

        # Skip large files (best-effort)
        try:
            if p.stat().st_size > 5 * 1024 * 1024:
                continue
        except OSError:
            continue

        yield p


def _scan_file(path: Path) -> List[Tuple[int, str, str]]:
    hits: List[Tuple[int, str, str]] = []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return hits

    for i, line in enumerate(text.splitlines(), start=1):
        for pat in PATH_PATTERNS:
            m = pat.search(line)
            if m:
                hits.append((i, m.group(0), line.strip()))
    return hits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".", help="Repo root to scan (default: .)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"Root does not exist: {root}")

    total_hits = 0
    for p in _iter_text_files(root):
        hits = _scan_file(p)
        if not hits:
            continue
        total_hits += len(hits)
        rel = p.relative_to(root)
        print(f"[HIT] {rel}")
        for (lineno, match, line) in hits[:20]:
            print(f"  L{lineno}: {match}")
            print(f"    {line}")
        if len(hits) > 20:
            print(f"  ... ({len(hits)-20} more)")

    if total_hits > 0:
        raise SystemExit(
            f"[FAIL] Found {total_hits} hard-coded absolute path occurrence(s). Please remove them."
        )
    print("[OK] No suspicious absolute paths found.")


if __name__ == "__main__":
    main()
