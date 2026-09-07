#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# HF network hardening (mirror + timeout)
: "${HF_ENDPOINT:=https://hf-mirror.com}"
: "${HF_HUB_ETAG_TIMEOUT:=60}"
: "${HF_HUB_DOWNLOAD_TIMEOUT:=180}"
: "${HF_HUB_DISABLE_XET:=1}"      # 某些网络环境更稳
: "${FALLBACK_TO_OFFICIAL:=0}"     # 0:不回退官方; 1:失败后回退 huggingface.co

export HF_ENDPOINT HF_HUB_ETAG_TIMEOUT HF_HUB_DOWNLOAD_TIMEOUT HF_HUB_DISABLE_XET

echo "[INFO] ROOT=${ROOT}"
echo "[INFO] HF_ENDPOINT=${HF_ENDPOINT}"
echo "[INFO] HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT}, HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT}"
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

说明：
  - 数学集始终按 manifest 执行。
  - 代码集默认仅准备 MBPP/MBPP+（不下载 HumanEval/APPS）。
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

  if python "$script" "$@"; then
    return 0
  else
    local rc=$?
    if [[ "$FALLBACK_TO_OFFICIAL" == "1" && "$HF_ENDPOINT" == "https://hf-mirror.com" ]]; then
      echo "[WARN] ${script} failed under HF mirror. Retrying with official endpoint..."
      export HF_ENDPOINT="https://huggingface.co"
      echo "[INFO] HF_ENDPOINT=${HF_ENDPOINT}"
      if python "$script" "$@"; then
        return 0
      else
        return $?
      fi
    fi
    return "$rc"
  fi
}

echo "[INFO] Preparing math datasets..."
run_with_hf_fallback scripts/10_data/prepare_math.py "${ARGS_COMMON[@]}"

echo "[INFO] Preparing code datasets..."
run_with_hf_fallback scripts/10_data/prepare_code.py \
  "${ARGS_COMMON[@]}" \
  --prepare_humaneval "$PREPARE_HUMANEVAL" \
  --prepare_apps "$PREPARE_APPS" \
  --prepare_mbpp "$PREPARE_MBPP" \
  --prepare_mbpp_plus "$PREPARE_MBPP_PLUS"

echo "[OK] Done."
