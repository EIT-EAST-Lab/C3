#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/90_audit/scan_prose.py

Keep the public prose of this repository English and typographically plain.

Two rules:

1. No em dash (U+2014) and no en dash (U+2013) anywhere in the scanned text
   files. They render inconsistently across terminals, LaTeX and Markdown
   viewers, and a colon, a comma, a semicolon or brackets always replace them.

2. No CJK characters, except the ones listed in scripts/90_audit/cjk_allowlist.txt.
   The repository is public and English, but a handful of CJK strings are
   functional: they are the answer cue words and punctuation sets that the CMATH
   benchmark parsers must match byte for byte.

Allowlist format, one entry per line, `#` starts a comment:

    <repo-relative path>:<substring that must occur in the offending line>

An occurrence is allowed when its file matches the entry path and its line
contains the entry substring. Matching on content rather than on a line number
keeps the allowlist valid when the surrounding file is edited.

This file spells the forbidden characters as code points so that it never trips
its own scan.

Usage:
  python scripts/90_audit/scan_prose.py [--root .] [--allowlist <path>]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


SCANNED_SUFFIXES = {".md", ".py", ".sh", ".yaml", ".yml", ".toml", ".cff"}

# Excluded wherever they occur: they nest anywhere and never hold public prose.
ALWAYS_EXCLUDE_DIRS = {".git", "__pycache__", ".mypy_cache", ".pytest_cache"}

# Excluded only at the repository root: these are generated local outputs.
# A nested directory that happens to share one of these names is still scanned.
ROOT_ONLY_EXCLUDE_DIRS = {
    "build",
    "dist",
    ".venv",
    "venv",
    "data",
    "ckpt",
    "runs",
    "wandb",
    "models",
    "artifacts",
}

DEFAULT_ALLOWLIST = "scripts/90_audit/cjk_allowlist.txt"

FORBIDDEN_DASHES = {
    chr(0x2013): "en dash (U+2013)",
    chr(0x2014): "em dash (U+2014)",
}

# CJK symbols and punctuation, the ideograph blocks, compatibility ideographs,
# and the halfwidth/fullwidth forms block that holds the fullwidth colon.
CJK_RANGES = (
    (0x3000, 0x303F),
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
    (0xFF00, 0xFFEF),
)

_CJK = re.compile("[" + "".join(f"{chr(lo)}-{chr(hi)}" for lo, hi in CJK_RANGES) + "]")


def _load_allowlist(path: Path) -> Dict[str, Set[str]]:
    allow: Dict[str, Set[str]] = {}
    if not path.exists():
        return allow
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        rel, _, needle = line.partition(":")
        rel = rel.strip()
        if not rel or not needle:
            continue
        allow.setdefault(rel.replace("\\", "/"), set()).add(needle)
    return allow


def _iter_text_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SCANNED_SUFFIXES:
            continue
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        if any(part in ALWAYS_EXCLUDE_DIRS for part in rel.parts):
            continue
        if rel.parts and rel.parts[0] in ROOT_ONLY_EXCLUDE_DIRS:
            continue
        yield path


def _scan_file(path: Path, rel: str, allow: Dict[str, Set[str]]) -> List[Tuple[int, str, str]]:
    hits: List[Tuple[int, str, str]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return hits

    needles = allow.get(rel, set())
    for lineno, line in enumerate(text.splitlines(), start=1):
        for char, label in FORBIDDEN_DASHES.items():
            if char in line:
                hits.append((lineno, f"forbidden {label}", line.strip()))
                break

        match = _CJK.search(line)
        if match and not any(needle in line for needle in needles):
            code = ord(match.group(0))
            hits.append((lineno, f"CJK character U+{code:04X} is not allowlisted", line.strip()))
    return hits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".", help="Repo root to scan (default: .)")
    ap.add_argument(
        "--allowlist",
        type=str,
        default="",
        help=f"Allowlist file (default: <root>/{DEFAULT_ALLOWLIST})",
    )
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"Root does not exist: {root}")

    allow_path = Path(args.allowlist).resolve() if args.allowlist else root / DEFAULT_ALLOWLIST
    allow = _load_allowlist(allow_path)

    total = 0
    for path in _iter_text_files(root):
        rel = path.relative_to(root).as_posix()
        hits = _scan_file(path, rel, allow)
        if not hits:
            continue
        total += len(hits)
        print(f"[HIT] {rel}")
        for lineno, reason, line in hits[:20]:
            print(f"  L{lineno}: {reason}")
            print(f"    {line}")
        if len(hits) > 20:
            print(f"  ... ({len(hits) - 20} more)")

    if total:
        raise SystemExit(
            f"[FAIL] Found {total} prose violation(s). "
            f"Rewrite them in plain English, or add the functional CJK line to {DEFAULT_ALLOWLIST}."
        )
    print("[OK] Prose is plain English with no em or en dashes.")


if __name__ == "__main__":
    main()
