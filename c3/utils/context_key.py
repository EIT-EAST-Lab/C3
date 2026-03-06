"""Canonical context-key hashing for C3.

Why this exists
--------------
Several parts of the project need a *stable* and *auditable* mapping from a
textual context (e.g., MAS state_text / replay context string) to an integer key
that fits safely in signed int64 tensors and JSON.

This module defines the single source of truth for context-key hashing.

Definition
----------
We standardize on:

* Hash: BLAKE2b
* Digest size: 8 bytes (64 bits)
* Output key: take the 64-bit unsigned digest as an integer, then mask the
  sign bit to ensure the result fits in signed int64.

This yields a deterministic integer in [0, 2^63).

We also provide a short fingerprint (SHA256 prefix) for collision guarding and
human-friendly error messages.
"""

from __future__ import annotations

import hashlib


_MASK_63 = (1 << 63) - 1


def hash63(text: str) -> int:
    """Return a stable non-negative 63-bit integer hash for `text`.

    The output is guaranteed to be in [0, 2^63) and therefore safe to store in
    signed int64 tensors and typical JSON encodings.
    """

    b = (text or "").encode("utf-8", errors="ignore")
    v = int.from_bytes(hashlib.blake2b(b, digest_size=8).digest(), byteorder="big", signed=False)
    return int(v & _MASK_63)


def fingerprint(text: str, *, n_hex: int = 16) -> str:
    """Return a short SHA256-based fingerprint of `text`.

    Args:
        text: Input text.
        n_hex: Length of returned hex prefix.
    """

    h = hashlib.sha256((text or "").encode("utf-8", errors="ignore")).hexdigest()
    n = max(8, int(n_hex))
    return h[:n]
