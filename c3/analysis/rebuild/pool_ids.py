# c3/analysis/rebuild/pool_ids.py
"""
The stable id of one prepared math row (E1 screening pass, MATHPOOL).

The informative-question pool of the E1 screening pass ships as an id list,
`configs/data/pool_informative_ids.json`, and not as problem statements, because
this repository does not redistribute dataset rows. An id list only works if
both sides mean the same thing by "the id of this row", so both sides call
`row_id`: the screening pass that writes the list, and the builder in
`scripts/10_data/prepare_math.py` that turns the list back into
`data/MATH/pool_informative.jsonl`.

The rule is one line. The id is the `unique_id` the prepared row carries; a row
without one is identified by the first 16 hexadecimal characters of the SHA256
of its `input` field. The fallback hashes the stored string exactly as it
stands, with no whitespace normalization and no case folding, so a row read back
from a prepared file always hashes to the id it was written under. Both prepared
MATH artifacts are written by `_canon_math_row`, which strips the problem
statement once before writing it, so that stored form is what gets hashed.

Standard library only.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

__all__ = ["row_id", "ID_HASH_PREFIX_CHARS"]

# How much of the SHA256 hexadecimal digest a fallback id keeps. Sixteen
# characters are 64 bits, which is far past the collision range of a pool of a
# few thousand problems, and short enough to read in a diff.
ID_HASH_PREFIX_CHARS = 16


def row_id(row: Mapping[str, Any]) -> str:
    """
    The id of one prepared math row: its `unique_id`, or a hash of its `input`.

    Raises ValueError for a row that has neither, because such a row cannot be
    named in the id list at all and a silent empty id would match every other
    row without one.
    """
    unique_id = str(row.get("unique_id") or "").strip()
    if unique_id:
        return unique_id

    problem = str(row.get("input") or "")
    if not problem.strip():
        raise ValueError(
            "row_id needs a unique_id or an input; this row carries neither "
            f"(keys: {sorted(row.keys())})."
        )
    return hashlib.sha256(problem.encode("utf-8")).hexdigest()[:ID_HASH_PREFIX_CHARS]
