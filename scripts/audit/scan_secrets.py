#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/audit/scan_secrets.py

Best-effort secret scanner to catch common credential/token leaks before release.

This is NOT a replacement for dedicated secret scanners, but it catches many
high-frequency mistakes (HF tokens, AWS keys, private keys, etc.).

Design notes:
- Avoid patterns that match common git commit hashes or library references.
- Prefer context-aware patterns for services whose keys resemble random hex.

Usage:
  python scripts/audit/scan_secrets.py [--root .]
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

# Intentionally conservative patterns to reduce false positives.
SECRET_PATTERNS: List[Tuple[re.Pattern[str], str]] = [
    # HuggingFace tokens (common formats)
    (re.compile(r"\bhf_[A-Za-z0-9]{24,}\b"), "HuggingFace token"),

    # AWS access key id
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "AWS access key id"),

    # Generic bearer token
    (
        re.compile(r"(?i)\bAuthorization\s*:\s*Bearer\s+[A-Za-z0-9._\-]{20,}\b"),
        "Bearer token",
    ),

    # Private keys
    (re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"), "Private key block"),

    # Common API key assignment patterns (e.g. API_KEY="....")
    (
        re.compile(
            r"(?i)\b(api[_-]?key|secret|token)\b\s*[:=]\s*['\"][A-Za-z0-9_\-]{16,}['\"]"
        ),
        "Key assignment",
    ),

    # W&B keys are 40-hex, which collides with git commit hashes.
    # Only flag if the context explicitly mentions WANDB_API_KEY.
    (
        re.compile(r"(?i)\bWANDB_API_KEY\b\s*[:=]\s*['\"]?[0-9a-f]{40}['\"]?"),
        "W&B API key assignment",
    ),
]

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
        if any(d in parts for d in DEFAULT_EXCLUDE_DIRS):
            continue

        if p.suffix.lower() in BINARY_SUFFIXES:
            continue
        try:
            if p.stat().st_size > 5 * 1024 * 1024:
                continue
        except OSError:
            continue

        yield p


def _scan_file(path: Path) -> List[Tuple[int, str, str, str]]:
    hits: List[Tuple[int, str, str, str]] = []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return hits

    for i, line in enumerate(text.splitlines(), start=1):
        for pat, label in SECRET_PATTERNS:
            m = pat.search(line)
            if m:
                snippet = line.strip()
                hits.append((i, label, m.group(0), snippet))
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
        rel = p.relative_to(root)
        print(f"[HIT] {rel}")
        total_hits += len(hits)
        for (lineno, label, match, snippet) in hits[:20]:
            print(f"  L{lineno}: {label}: {match}")
            print(f"    {snippet}")
        if len(hits) > 20:
            print(f"  ... ({len(hits)-20} more)")

    if total_hits > 0:
        raise SystemExit(
            f"[FAIL] Found {total_hits} potential secret occurrence(s). Please remove/rotate them."
        )
    print("[OK] No obvious secrets found.")


if __name__ == "__main__":
    main()
