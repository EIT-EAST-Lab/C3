# Upstream provenance (OpenRLHF)

C3 is built on top of **OpenRLHF** and keeps the upstream package namespace under `openrlhf/` for:
- **traceability / auditability** (a pinned upstream anchor),
- **compatibility** with upstream entrypoints,
- **minimizing diff surface** for future rebases.

> Scope note: this repo vendors the **Python package** `openrlhf/` as the upstream-compatible core.
> The upstream OpenRLHF repository contains additional CLIs/examples/docs that are intentionally not
> vendored in this paper code release.

---

## Upstream repository

- Upstream: `https://github.com/OpenRLHF/OpenRLHF`
- Upstream license: Apache-2.0

---

## Upstream anchor (pinned)

This C3 release is based on OpenRLHF at:

- **UPSTREAM_COMMIT:** `f372a2d41e26c3c47a0f6653fb94c31f5c257942`
- **UPSTREAM_DESCRIBE:** `v0.9.0-3-gf372a2d`
- **UPSTREAM_COMMIT_DATE:** `2025-12-11T12:51:17+00:00`
- **UPSTREAM_VERSION_TXT:** `0.9.1` (from upstream `version.txt`)

This anchor is recorded to make the upstream base **fully reproducible and auditable**.

### How this anchor can be verified

Clone upstream and check the anchor out:

```bash
git clone https://github.com/OpenRLHF/OpenRLHF.git
cd OpenRLHF
git checkout f372a2d41e26c3c47a0f6653fb94c31f5c257942
git rev-parse HEAD
git describe --tags --always
git show -s --format=%cI HEAD
cat version.txt
```

---

## Verifying the vendored `openrlhf/` diff against the pinned upstream

The authoritative file-level change log lives in:

- `docs/41_changes_from_openrlhf.md`

To **reproduce** the “Added / Modified / Removed (not vendored)” lists in that document, do:

### 1) Prepare upstream at the pinned commit

```bash
git clone https://github.com/OpenRLHF/OpenRLHF.git <path-to-OpenRLHF>
cd <path-to-OpenRLHF>
git checkout f372a2d41e26c3c47a0f6653fb94c31f5c257942
```

### 2) Compare the `openrlhf/` package trees

From your C3 repo root:

```bash
diff -rq <path-to-OpenRLHF>/openrlhf ./openrlhf
```

This is a raw byte comparison, so it also reports the 39 vendored `.py` files
that carry the three-line C3 provenance comment at the top of the file. The
lists in `docs/41_changes_from_openrlhf.md` compare non-blank, non-provenance
lines instead; the script in step 3 does that.

### 3) Compute the Added / Modified / Removed lists programmatically

From your C3 repo root:

```bash
python - <<'PY'
from pathlib import Path
import hashlib

c3 = Path("openrlhf")
up = Path("<path-to-OpenRLHF>/openrlhf")

# 39 of the 44 vendored files start with the three-line C3 provenance comment,
# and adding it also moved the blank line next to it in a few files. Hash the
# non-blank, non-provenance lines, or every one of those files reports as
# modified.
PROVENANCE = (
    b"# Derived from OpenRLHF",
    b"# Modified by the C3 authors",
    b"# See docs/40_upstream.md",
)

def files(root: Path):
    return sorted([p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()])

def sha256(p: Path):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for line in f:
            if line.startswith(PROVENANCE) or not line.strip():
                continue
            h.update(line)
    return h.hexdigest()

c3_files = set(files(c3))
up_files = set(files(up))

added = sorted(c3_files - up_files)
removed = sorted(up_files - c3_files)
common = sorted(c3_files & up_files)

modified = []
same = []
for rel in common:
    if sha256(c3 / rel) != sha256(up / rel):
        modified.append(rel)
    else:
        same.append(rel)

print("Added:", len(added))
print("Modified:", len(modified))
print("Removed (not vendored):", len(removed))
print("Same:", len(same))

print("\n# Added")
for x in added:
    print(x)
print("\n# Modified")
for x in modified:
    print(x)
print("\n# Removed (not vendored)")
for x in removed:
    print(x)
PY
```

---

## Where C3 diverges from upstream

All intentional differences are documented in:

- `docs/41_changes_from_openrlhf.md`

In short:

- C3 adds a new top-level module `c3/` and config surface `configs/`.
- Upstream `openrlhf/` is modified only where necessary to integrate:
  - C3 task/role configs,
  - multi-agent rollout / experience fields,
  - reproducibility metadata output.

---

## Rebase policy

When rebasing onto a newer OpenRLHF version:

1. Keep the `openrlhf/` namespace intact.
2. Keep C3 logic under `c3/`.
3. Minimize edits to upstream files; if unavoidable, update:
   - `docs/40_upstream.md` (new pinned commit),
   - `docs/41_changes_from_openrlhf.md` (file-level change log).
4. Re-run the release gate: `bash scripts/90_audit/pre_release.sh`.
