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

If you have the provided `OpenRLHF.zip` (it includes git metadata):

```bash
cd <path-to-OpenRLHF>/OpenRLHF
git rev-parse HEAD
git describe --tags --always
git show -s --format=%cI HEAD
cat version.txt
```

If you use a fresh clone from GitHub, you can verify that the commit exists:

```bash
git clone https://github.com/OpenRLHF/OpenRLHF.git
cd OpenRLHF
git show f372a2d41e26c3c47a0f6653fb94c31f5c257942 --oneline
```

---

## Verifying the vendored `openrlhf/` diff against the pinned upstream

The authoritative file-level change log lives in:

- `docs/CHANGES_FROM_OPENRLHF.md`

To **reproduce** the “Added / Modified / Removed (not vendored)” lists in that document, do:

### Option A (recommended): use the provided `OpenRLHF.zip`

```bash
# 1) Prepare upstream at the pinned commit
cd <path-to-OpenRLHF>/OpenRLHF
git checkout f372a2d41e26c3c47a0f6653fb94c31f5c257942

# 2) Compare openrlhf/ package trees (from your C3 repo root)
cd <path-to-C3>
diff -rq <path-to-OpenRLHF>/OpenRLHF/openrlhf ./openrlhf
```

### Option B: compute Added/Modified/Removed lists programmatically

From your C3 repo root:

```bash
python - <<'PY'
from pathlib import Path
import hashlib

c3 = Path("openrlhf")
up = Path("<path-to-OpenRLHF>/OpenRLHF/openrlhf")

def files(root: Path):
    return sorted([p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()])

def sha256(p: Path):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
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

- `docs/CHANGES_FROM_OPENRLHF.md`

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
   - `docs/UPSTREAM.md` (new pinned commit),
   - `docs/CHANGES_FROM_OPENRLHF.md` (file-level change log).
4. Re-run the release gate: `bash scripts/audit/pre_release.sh`.
