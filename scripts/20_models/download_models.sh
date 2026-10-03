#!/usr/bin/env bash
# scripts/20_models/download_models.sh
set -euo pipefail

# Download / cache the HuggingFace models the experiments read.
#
# This repo does NOT ship model weights. By default, Transformers will download
# weights on first use. This script is an optional convenience for:
#   - pre-downloading (e.g., on shared clusters),
#   - keeping an explicit local `models/` directory for reproducibility,
#   - avoiding repeated downloads across runs.
#
# Two sets of models, chosen with --models:
#   paper  the models of the paper (arXiv:2603.06859v3): Qwen/Qwen3-4B-Instruct-2507,
#          the base model of every trained workflow and of the measurements;
#          Qwen/Qwen3-8B, the second base model of the measurements; and
#          jhu-clsp/mmBERT-base, the encoder MAPPO's critic and the learned
#          critic of the critic comparison are built on.
#   early  the unique `source.type=hf_base` entries of a registry YAML,
#          `configs/main_results_registry.yaml` by default. These are the base
#          models of the evaluation protocol of the earlier versions of the
#          paper (Qwen2.5-3B-Instruct and Qwen2.5-Coder-3B-Instruct).
# The default, all, is both.
#
# NOTE: Some models (e.g., Qwen) may require accepting terms on Hugging Face.
#       If the download fails with 401/403, login first:
#         hf auth login
#       or set HF_TOKEN / HUGGINGFACE_HUB_TOKEN in your environment.

_usage() {
  cat <<'USAGE'
Usage:
  bash scripts/20_models/download_models.sh [options]

Options:
  --models SET          paper | early | all (default: all)
  --registry PATH       Registry YAML of the early set (default: configs/main_results_registry.yaml)
  --out_dir DIR         Local directory to store snapshots (default: models)
  --only_ids LIST       Comma/space list of registry run IDs to consider (early set only)
  --only_methods LIST   Comma/space list of registry methods to consider (early set only)
  --only_tasks LIST     Comma/space list of registry tasks to consider (early set only)
  --dry_run 0|1         Only print resolved model ids, do not download (default: 0)
  -h, --help            Show this help.

Environment:
  PYTHON    Python executable (default: python)
  HF_TOKEN / HUGGINGFACE_HUB_TOKEN
            Token for gated/private models (optional).
  HF_ENDPOINT
            Hugging Face endpoint. Defaults to the official one; see
            docs/31_network_mirrors.md to route downloads through a mirror.
USAGE
}

PYTHON_BIN="${PYTHON:-python}"
MODEL_SET="all"
REGISTRY="configs/main_results_registry.yaml"
OUT_DIR="models"
ONLY_IDS=""
ONLY_METHODS=""
ONLY_TASKS=""
DRY_RUN="0"

# The models of the paper, as HuggingFace ids.
PAPER_MODELS="Qwen/Qwen3-4B-Instruct-2507
Qwen/Qwen3-8B
jhu-clsp/mmBERT-base"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models) MODEL_SET="${2:-}"; shift 2 ;;
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

# shellcheck source=../_lib/mirrors.sh
source "${REPO_ROOT}/scripts/_lib/mirrors.sh"
c3_mirrors_validate
c3_mirrors_report

case "${MODEL_SET}" in
  paper|early|all) ;;
  *) echo "[download_models] ERROR: --models must be paper, early or all, got: ${MODEL_SET}" >&2; exit 2 ;;
esac

if [[ "${MODEL_SET}" != "paper" ]]; then
  [[ -f "${REGISTRY}" ]] || { echo "[download_models] ERROR: registry not found: ${REGISTRY}" >&2; exit 1; }
fi

mkdir -p "${OUT_DIR}"

EARLY_MODELS=""
if [[ "${MODEL_SET}" != "paper" ]]; then
# Extract unique model ids of the registry (one per line).
EARLY_MODELS="$("${PYTHON_BIN}" - <<'PY' "${REGISTRY}" "${ONLY_IDS}" "${ONLY_METHODS}" "${ONLY_TASKS}"
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
fi

MODELS=""
if [[ "${MODEL_SET}" != "early" ]]; then
  MODELS="${PAPER_MODELS}"
fi
# The early set after the paper set, without repeating an id both carry.
while IFS= read -r mid; do
  mid="${mid%$'\r'}"
  [[ -n "${mid}" ]] || continue
  if ! grep -qxF -- "${mid}" <<< "${MODELS}"; then
    MODELS="${MODELS:+${MODELS}
}${mid}"
  fi
done <<< "${EARLY_MODELS}"

if [[ -z "${MODELS}" ]]; then
  echo "[download_models] No hf_base models found in ${REGISTRY}" >&2
  exit 0
fi

echo "[download_models] models=${MODEL_SET}" >&2
if [[ "${MODEL_SET}" != "paper" ]]; then
  echo "[download_models] registry=${REGISTRY}" >&2
fi
echo "[download_models] out_dir=${OUT_DIR}" >&2
echo "[download_models] models:" >&2
while IFS= read -r mid; do
  if grep -qxF -- "${mid}" <<< "${PAPER_MODELS}" && [[ "${MODEL_SET}" != "early" ]]; then
    echo "  - ${mid}   (paper)" >&2
  else
    echo "  - ${mid}   (early protocol, ${REGISTRY})" >&2
  fi
done <<< "${MODELS}"

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
        "Install the CPU or GPU tier (see the README), or `pip install huggingface_hub`."
    ) from e

# Token is automatically picked up from env (HF_TOKEN / HUGGINGFACE_HUB_TOKEN).
# HF_ENDPOINT is honored by huggingface_hub itself, so a mirror needs no argument here.
# Downloads always resume, and local_dir never uses symlinks: both former keyword
# arguments were removed from the huggingface_hub API.
for mid in models:
    mid = mid.strip()
    if not mid:
        continue
    safe = mid.replace("/", "__")
    local_dir = os.path.join(out_dir, safe)
    os.makedirs(local_dir, exist_ok=True)
    print(f"[download_models] downloading {mid} -> {local_dir}", file=sys.stderr)
    snapshot_download(repo_id=mid, local_dir=local_dir)
print("[download_models] OK", file=sys.stderr)
PY
