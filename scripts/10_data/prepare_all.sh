#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# shellcheck source=../_lib/mirrors.sh
source "${ROOT}/scripts/_lib/mirrors.sh"

# Hugging Face transfer hardening. These are timeouts, not mirrors: the endpoint
# itself stays official unless the caller exported HF_ENDPOINT. HF_HUB_DISABLE_XET
# is set by scripts/_lib/mirrors.sh, which is sourced above.
: "${HF_HUB_ETAG_TIMEOUT:=60}"
: "${HF_HUB_DOWNLOAD_TIMEOUT:=180}"
: "${FALLBACK_TO_OFFICIAL:=1}"     # 1: retry with huggingface.co when a mirror failed

export HF_HUB_ETAG_TIMEOUT HF_HUB_DOWNLOAD_TIMEOUT
if [[ -n "${HF_ENDPOINT:-}" ]]; then
  export HF_ENDPOINT
fi

echo "[INFO] ROOT=${ROOT}"
c3_mirrors_validate
c3_mirrors_report
echo "[INFO] FALLBACK_TO_OFFICIAL=${FALLBACK_TO_OFFICIAL}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/10_data/prepare_all.sh \
    [--data_dir <dir>] [--out_dir <dir>] [--overwrite 0|1] [--strict 0|1] [--update_manifest_sha256 0|1] \
    [--prepare_humaneval 0|1] [--prepare_apps 0|1] [--prepare_mbpp 0|1] [--prepare_mbpp_plus 0|1]

Defaults:
  --prepare_humaneval 0
  --prepare_apps      0
  --prepare_mbpp      1
  --prepare_mbpp_plus 1

Notes:
  - The math datasets are always prepared as pinned by configs/data_manifest.yaml.
  - The code datasets default to MBPP and MBPP+ only (HumanEval and APPS are not downloaded).
  - Downloads go to the official sources. Export HF_ENDPOINT or GITHUB_MIRROR_PREFIX
    to route them through a mirror; see docs/31_network_mirrors.md.
EOF
}

DATA_DIR=""
OUT_DIR=""
OVERWRITE=0
STRICT=0
UPDATE_MANIFEST_SHA256=0

PREPARE_HUMANEVAL=0
PREPARE_APPS=0
PREPARE_MBPP=1
PREPARE_MBPP_PLUS=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_dir) DATA_DIR="$2"; shift 2 ;;
    --out_dir) OUT_DIR="$2"; shift 2 ;;
    --overwrite) OVERWRITE="$2"; shift 2 ;;
    --strict) STRICT="$2"; shift 2 ;;
    --update_manifest_sha256) UPDATE_MANIFEST_SHA256="$2"; shift 2 ;;

    --prepare_humaneval) PREPARE_HUMANEVAL="$2"; shift 2 ;;
    --prepare_apps) PREPARE_APPS="$2"; shift 2 ;;
    --prepare_mbpp) PREPARE_MBPP="$2"; shift 2 ;;
    --prepare_mbpp_plus) PREPARE_MBPP_PLUS="$2"; shift 2 ;;

    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

for v in \
  "$OVERWRITE" "$STRICT" "$UPDATE_MANIFEST_SHA256" \
  "$PREPARE_HUMANEVAL" "$PREPARE_APPS" "$PREPARE_MBPP" "$PREPARE_MBPP_PLUS"; do
  if [[ "$v" != "0" && "$v" != "1" ]]; then
    echo "[FAIL] Flags must be 0 or 1." >&2
    exit 1
  fi
done

# Interpreter override, same knob as the other entrypoints.
PYTHON_BIN="${PYTHON:-python}"

ARGS_COMMON=()
if [[ -n "$DATA_DIR" ]]; then
  ARGS_COMMON+=(--data_dir "$DATA_DIR")
fi
if [[ -n "$OUT_DIR" ]]; then
  ARGS_COMMON+=(--out_dir "$OUT_DIR")
fi
ARGS_COMMON+=(--overwrite "$OVERWRITE")
ARGS_COMMON+=(--strict "$STRICT")
ARGS_COMMON+=(--update_manifest_sha256 "$UPDATE_MANIFEST_SHA256")

run_with_hf_fallback() {
  local script="$1"; shift

  local rc=0
  "$PYTHON_BIN" "$script" "$@" || rc=$?
  if [[ "$rc" -eq 0 ]]; then
    return 0
  fi

  local endpoint="${HF_ENDPOINT:-${C3_HF_OFFICIAL_ENDPOINT}}"
  if [[ "$FALLBACK_TO_OFFICIAL" == "1" && "$endpoint" != "${C3_HF_OFFICIAL_ENDPOINT}" ]]; then
    echo "[WARN] ${script} failed against the Hugging Face mirror ${endpoint}."
    echo "[WARN] Retrying with the official endpoint ${C3_HF_OFFICIAL_ENDPOINT}..."
    export HF_ENDPOINT="${C3_HF_OFFICIAL_ENDPOINT}"
    "$PYTHON_BIN" "$script" "$@"
    return $?
  fi
  return "$rc"
}

# EvalPlus builds its MBPP+ download URL against github.com and caches the
# gunzipped release asset under its own cache directory. When GITHUB_MIRROR_PREFIX
# is set we fetch that asset through the mirror and write it to the exact cache
# path EvalPlus expects, so EvalPlus never contacts github.com itself.
seed_evalplus_mbpp_plus_cache() {
  [[ -n "${GITHUB_MIRROR_PREFIX:-}" ]] || return 0
  [[ "$PREPARE_MBPP_PLUS" == "1" ]] || return 0

  "$PYTHON_BIN" - <<'PY'
import gzip
import os
import shutil
import sys
import tempfile
import urllib.request

prefix = os.environ.get("GITHUB_MIRROR_PREFIX", "").strip()
if not prefix:
    sys.exit(0)

try:
    from evalplus.data.mbpp import MBPP_PLUS_VERSION
    from evalplus.data.utils import get_dataset_metadata
except Exception as exc:
    print(f"[WARN] evalplus is not importable ({exc}); skipping MBPP+ cache seeding.")
    sys.exit(0)

url, cache_path = get_dataset_metadata("MbppPlus", MBPP_PLUS_VERSION, False, False)
if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
    print(f"[INFO] MBPP+ already cached: {cache_path}")
    sys.exit(0)

mirrored = prefix + url
print(f"[INFO] Seeding the EvalPlus MBPP+ cache from {mirrored}")
os.makedirs(os.path.dirname(cache_path), exist_ok=True)
try:
    with tempfile.TemporaryDirectory() as tmpdir:
        gz_path = os.path.join(tmpdir, "MbppPlus.jsonl.gz")
        with urllib.request.urlopen(mirrored, timeout=180) as response:
            with open(gz_path, "wb") as handle:
                shutil.copyfileobj(response, handle)
        tmp_out = cache_path + ".tmp"
        with gzip.open(gz_path, "rb") as src, open(tmp_out, "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.replace(tmp_out, cache_path)
except Exception as exc:
    print(f"[WARN] Could not seed the MBPP+ cache through the mirror ({exc}).")
    print("[WARN] EvalPlus will fall back to downloading from github.com.")
    sys.exit(0)

print(f"[OK] MBPP+ cache seeded: {cache_path}")
PY
}

echo "[INFO] Preparing math datasets..."
run_with_hf_fallback scripts/10_data/prepare_math.py "${ARGS_COMMON[@]}"

seed_evalplus_mbpp_plus_cache

echo "[INFO] Preparing code datasets..."
run_with_hf_fallback scripts/10_data/prepare_code.py \
  "${ARGS_COMMON[@]}" \
  --prepare_humaneval "$PREPARE_HUMANEVAL" \
  --prepare_apps "$PREPARE_APPS" \
  --prepare_mbpp "$PREPARE_MBPP" \
  --prepare_mbpp_plus "$PREPARE_MBPP_PLUS"

echo "[OK] Done."
