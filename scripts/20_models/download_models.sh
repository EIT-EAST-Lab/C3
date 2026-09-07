#!/usr/bin/env bash
# scripts/20_models/download_models.sh
set -euo pipefail

# Download / cache HuggingFace base models referenced by a registry YAML.
#
# This repo does NOT ship model weights. By default, Transformers will download
# weights on first use. This script is an optional convenience for:
#   - pre-downloading (e.g., on shared clusters),
#   - keeping an explicit local `models/` directory for reproducibility,
#   - avoiding repeated downloads across runs.
#
# It reads `configs/main_results_registry.yaml` and downloads all unique
# `source.type=hf_base` entries.
#
# NOTE: Some models (e.g., Qwen) may require accepting terms on Hugging Face.
#       If the download fails with 401/403, login first:
#         huggingface-cli login
#       or set HF_TOKEN / HUGGINGFACE_HUB_TOKEN in your environment.

_usage() {
  cat <<'USAGE'
Usage:
  bash scripts/20_models/download_models.sh [options]

Options:
  --registry PATH       Registry YAML (default: configs/main_results_registry.yaml)
  --out_dir DIR         Local directory to store snapshots (default: models)
  --only_ids LIST       Comma/space list of run IDs to consider (optional)
  --only_methods LIST   Comma/space list of methods to consider (optional)
  --only_tasks LIST     Comma/space list of tasks to consider (optional)
  --dry_run 0|1         Only print resolved model ids, do not download (default: 0)
  -h, --help            Show this help.

Environment:
  PYTHON    Python executable (default: python)
  HF_TOKEN / HUGGINGFACE_HUB_TOKEN
            Token for gated/private models (optional).
USAGE
}

PYTHON_BIN="${PYTHON:-python}"
REGISTRY="configs/main_results_registry.yaml"
OUT_DIR="models"
ONLY_IDS=""
ONLY_METHODS=""
ONLY_TASKS=""
DRY_RUN="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --registry) REGISTRY="${2:-}"; shift 2 ;;
    --out_dir) OUT_DIR="${2:-}"; shift 2 ;;
    --only_ids) ONLY_IDS="${2:-}"; shift 2 ;;
    --only_methods) ONLY_METHODS="${2:-}"; shift 2 ;;
    --only_tasks) ONLY_TASKS="${2:-}"; shift 2 ;;
    --dry_run) DRY_RUN="${2:-}"; shift 2 ;;
    -h|--help) _usage; exit 0 ;;
    *) echo "[download_models] Unknown arg: $1" >&2; _usage; exit 2 ;;
  esac
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
cd "${REPO_ROOT}"

[[ -f "${REGISTRY}" ]] || { echo "[download_models] ERROR: registry not found: ${REGISTRY}" >&2; exit 1; }

mkdir -p "${OUT_DIR}"

# Extract unique model ids (one per line).
MODELS="$("${PYTHON_BIN}" - <<'PY' "${REGISTRY}" "${ONLY_IDS}" "${ONLY_METHODS}" "${ONLY_TASKS}"
import sys, yaml

reg, only_ids, only_methods, only_tasks = sys.argv[1:5]

def split_list(s: str):
    items = []
    for t in (s or "").replace(",", " ").split():
        t = t.strip()
        if t:
            items.append(t)
    return items

only_id = set(split_list(only_ids))
only_m = set([x.lower() for x in split_list(only_methods)])
only_t = set(split_list(only_tasks))

with open(reg, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

runs = cfg.get("runs") or []
seen = set()
out = []
for r in runs:
    if not isinstance(r, dict):
        continue
    rid = str(r.get("id","")).strip()
    m = str(r.get("method","")).strip()
    t = str(r.get("task","")).strip()
    if only_id and rid not in only_id:
        continue
    if only_m and m.lower() not in only_m:
        continue
    if only_t and t not in only_t:
        continue
    src = r.get("source") or {}
    if not isinstance(src, dict):
        continue
    if str(src.get("type","")).strip() != "hf_base":
        continue
    hf_base = str(src.get("hf_base","")).strip()
    if not hf_base or hf_base in seen:
        continue
    seen.add(hf_base)
    out.append(hf_base)

for x in sorted(out):
    print(x)
PY
)"

if [[ -z "${MODELS}" ]]; then
  echo "[download_models] No hf_base models found in ${REGISTRY}" >&2
  exit 0
fi

echo "[download_models] registry=${REGISTRY}" >&2
echo "[download_models] out_dir=${OUT_DIR}" >&2
echo "[download_models] models:" >&2
echo "${MODELS}" | sed 's/^/  - /' >&2

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[download_models] dry_run=1: done." >&2
  exit 0
fi

# Prefer huggingface_hub.snapshot_download to avoid relying on external CLIs.
"${PYTHON_BIN}" - <<'PY' "${OUT_DIR}" "${MODELS}"
import os, sys
out_dir = sys.argv[1]
models = sys.argv[2].splitlines()

try:
    from huggingface_hub import snapshot_download
except Exception as e:
    raise SystemExit(
        "huggingface_hub is required for download_models.sh. "
        "Install requirements.txt or `pip install huggingface_hub`."
    ) from e

# Token is automatically picked up from env (HF_TOKEN / HUGGINGFACE_HUB_TOKEN).
for mid in models:
    mid = mid.strip()
    if not mid:
        continue
    safe = mid.replace("/", "__")
    local_dir = os.path.join(out_dir, safe)
    os.makedirs(local_dir, exist_ok=True)
    print(f"[download_models] downloading {mid} -> {local_dir}", file=sys.stderr)
    snapshot_download(
        repo_id=mid,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
print("[download_models] OK", file=sys.stderr)
PY
