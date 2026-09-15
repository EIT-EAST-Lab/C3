# c3/analysis/rebuild/summary.py
"""The single writer of the summary.json defined by the results layout, section 3.4.

Both aggregation families in this package end here: the E1 family goes through
`aggregate_e1.SummaryBuilder` (a thin wrapper that collects keys one at a time)
and the E3a / E5 family calls `build_summary` / `write_summary` directly. The
two paths therefore produce the same document shape and the same three
conventions that used to differ between them (the maintainers' decision of
2026-09-15):

- a git sha that cannot be read is written as null, not as a string,
- the timestamp ends in Z (``2026-09-15T04:05:06Z``),
- an empty note is left out of the entry instead of written as "".

The other package-level value conventions live here for the same reason: the
p-value string of manifest_skeleton.py (`format_p`), reading the manifest
(`load_manifest`), the path shape of a summary `source` (`source_path`) and the
commit stamp (`git_sha`).

One caution about names: `aggregate_e1.source_path` has a different signature
(it joins a results root with the sub-path of a cell). It shapes the joined path
by calling the one here, so both families write the same `source` shape; the
signatures differ, which is why the two are not one function.

Refusals are mechanical and never fatal (results layout revision 2): a key
the manifest does not define, or a key whose unit is "verdict", is dropped with
one line on stderr and the aggregation keeps its exit code of 0, so a partly
collected result tree still produces the keys it can support. Filling a verdict
is the maintainers' action and no script's.

Only the standard library is imported here.
"""

from __future__ import annotations

import datetime
import json
import math
import os
import subprocess
import sys
from typing import Any, Dict, Mapping, Optional

__all__ = [
    "format_p",
    "load_manifest",
    "source_path",
    "git_sha",
    "key_refusal",
    "RESERVED_TOP_LEVEL",
    "build_summary",
    "write_summary",
]


# -----------------------------------------------------------------------------
# value conventions
# -----------------------------------------------------------------------------


def format_p(p: Optional[float]) -> Optional[str]:
    """The manifest's p-value string, or None when the p value is undefined.

    Three tiers (analysis plan revision 13, 2026-09-15): below 0.001 the
    string is ``$<\\!0.001$``; from 0.001 up to but not including 0.01 it is
    ``$<\\!0.01$``; otherwise it is two decimals, ``$= 0.30$``. The middle tier
    exists because ``$= 0.00$`` reads as "p equals zero". The fragment carries
    no letter p: the manifest's own text supplies it.

    None means "not computable"; the caller then leaves the key out of
    summary.json rather than writing null.
    """
    if p is None:
        return None
    try:
        value = float(p)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    if value < 0.001:
        return "$<\\!0.001$"
    if value < 0.01:
        return "$<\\!0.01$"
    return "$= %.2f$" % value


def load_manifest(path: str) -> Dict[str, Dict[str, Any]]:
    """Load `results/manifest.json`: a flat mapping key -> entry."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("manifest is not a JSON object: %s" % path)
    return data


def source_path(path: str) -> str:
    """Path as written into summary.json's `source`.

    The results layout prints repository-relative paths such as
    ``20_data/results/E3a/a3/4b/empty/buckets.jsonl``. The aggregate scripts are
    handed an arbitrary directory, so the rule is mechanical: cut at the last
    ``20_data`` segment when there is one, otherwise keep the path as given.
    Separators are normalised to forward slashes either way.
    """
    parts = os.path.abspath(path).replace("\\", "/").split("/")
    if "20_data" in parts:
        cut = len(parts) - 1 - parts[::-1].index("20_data")
        return "/".join(parts[cut:])
    return path.replace("\\", "/")


def git_sha(cwd: Optional[str] = None) -> Optional[str]:
    """HEAD of the repository holding this file, or None if git is unavailable.

    Read-only (`git rev-parse HEAD`). Any failure returns None instead of
    raising: a missing sha must not stop an aggregation, and null says "not
    recorded" where a string would claim a commit.
    """
    where = cwd or os.path.dirname(os.path.abspath(__file__))
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=where,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    sha = out.stdout.decode("utf-8", "replace").strip()
    return sha or None


# -----------------------------------------------------------------------------
# the document
# -----------------------------------------------------------------------------


def key_refusal(key: str, manifest: Mapping[str, Mapping[str, Any]]) -> Optional[str]:
    """Why this key must not be written, or None when it may be.

    The one place the two aggregation paths share their key-level rule, so they
    cannot drift apart.
    """
    entry = manifest.get(key)
    if entry is None:
        return "not a manifest key"
    if str(entry.get("unit")) == "verdict":
        return "verdict key, the maintainers decide it"
    return None


#: Top-level names of the document that `extra` may not overwrite.
RESERVED_TOP_LEVEL = ("experiment", "generated_by", "keys")


def build_summary(
    experiment: str,
    script: str,
    keys: Mapping[str, Mapping[str, Any]],
    manifest: Mapping[str, Mapping[str, Any]],
    *,
    stream=None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the summary document of the results layout, section 3.4.

    `keys` maps a manifest key to {value, n, source, note}; the unit is taken
    from the manifest so the two cannot drift. Keys the manifest does not define
    and keys whose unit is "verdict" are refused: dropped from the document with
    one line on stderr (see `key_refusal`). The entries are written in sorted
    key order, and an empty note is omitted.

    `extra` adds top-level fields just after `experiment`, which is how the E1
    aggregation records which result tree it read and which key prefix it wrote
    (`results_root`, `key_prefix`). The three names of the document shape are
    reserved: passing one of them raises rather than silently rewriting the
    document the results layout fixes.
    """
    out = stream if stream is not None else sys.stderr
    if extra:
        clash = [name for name in extra if name in RESERVED_TOP_LEVEL]
        if clash:
            raise ValueError("extra may not set the reserved top-level name(s): %s"
                             % ", ".join(sorted(clash)))
    out_keys: Dict[str, Any] = {}
    for key in sorted(keys):
        why = key_refusal(key, manifest)
        if why is not None:
            print("skip %s: %s" % (key, why), file=out)
            continue
        entry = keys[key]
        item: Dict[str, Any] = {
            "value": entry.get("value"),
            "n": entry.get("n"),
            "unit": manifest[key].get("unit"),
            "source": list(entry.get("source") or []),
        }
        note = entry.get("note")
        if note:
            item["note"] = note
        out_keys[key] = item

    doc: Dict[str, Any] = {"experiment": experiment}
    for name in (extra or {}):
        doc[name] = (extra or {})[name]
    doc["generated_by"] = {
        "script": script,
        "git_sha": git_sha(),
        "when": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    doc["keys"] = out_keys
    return doc


def write_summary(
    out_path: str,
    experiment: str,
    script: str,
    keys: Mapping[str, Mapping[str, Any]],
    manifest: Mapping[str, Mapping[str, Any]],
    *,
    stream=None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the summary document and write it as UTF-8 JSON with LF endings."""
    doc = build_summary(experiment, script, keys, manifest, stream=stream, extra=extra)
    parent = os.path.dirname(os.path.abspath(out_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=1, sort_keys=False)
        fh.write("\n")
    return doc
