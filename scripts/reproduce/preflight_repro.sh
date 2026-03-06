#!/usr/bin/env bash
set -euo pipefail

# scripts/reproduce/preflight_repro.sh
#
# One-command preflight for public-release readiness.
# This is intentionally "fast-fail + explicit diagnostics", not long-running training.

_usage() {
  cat <<'USAGE'
Usage:
  bash scripts/reproduce/preflight_repro.sh [options]

Options:
  --out_dir DIR            Output report directory (default: artifacts/preflight)
  --task math|code         Smoke task (default: math)
  --skip_data_strict 0|1   Skip strict data verification (default: 0)
  --python PY              Python executable (default: python)
  -h, --help               Show help
USAGE
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
source "${SCRIPT_DIR}/common_env.sh"

OUT_DIR="${REPO_ROOT}/artifacts/preflight"
TASK="math"
SKIP_DATA_STRICT="0"
PYTHON_BIN="${PYTHON:-python}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out_dir) OUT_DIR="${2:-}"; shift 2 ;;
    --task) TASK="${2:-}"; shift 2 ;;
    --skip_data_strict) SKIP_DATA_STRICT="${2:-0}"; shift 2 ;;
    --python) PYTHON_BIN="${2:-}"; shift 2 ;;
    -h|--help) _usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; _usage; exit 2 ;;
  esac
done

mkdir -p "$OUT_DIR"
REPORT_JSON="${OUT_DIR}/preflight_report.json"
LOG_TXT="${OUT_DIR}/preflight.log"

cd "$REPO_ROOT"
c3_repro_export_common_env "${REPO_ROOT}"

echo "[preflight] repo_root=${REPO_ROOT}" | tee "$LOG_TXT"
echo "[preflight] out_dir=${OUT_DIR}" | tee -a "$LOG_TXT"
echo "[preflight] task=${TASK}" | tee -a "$LOG_TXT"

STEP_STATUS=()
HAS_FAIL=0

_run_step() {
  local name="$1"
  shift
  echo "[preflight] RUN ${name}" | tee -a "$LOG_TXT"
  if "$@" >>"$LOG_TXT" 2>&1; then
    echo "[preflight] OK  ${name}" | tee -a "$LOG_TXT"
    STEP_STATUS+=("${name}:ok")
  else
    echo "[preflight] FAIL ${name}" | tee -a "$LOG_TXT"
    STEP_STATUS+=("${name}:fail")
    HAS_FAIL=1
  fi
}

# Ensure preflight is idempotent even if prior Python runs left bytecode caches.
find . -type d -name "__pycache__" -prune -exec rm -rf {} + >/dev/null 2>&1 || true
find . -type f -name "*.py[co]" -delete >/dev/null 2>&1 || true

_run_audit_with_generated_dirs_shelved_if_needed() {
  local generated_dirs=("data" "artifacts" "ckpt" "runs" "wandb" "models")
  local moved_dirs=()

  for rel_dir in "${generated_dirs[@]}"; do
    local abs_dir="${REPO_ROOT}/${rel_dir}"
    local backup_dir="${REPO_ROOT}.${rel_dir}.__preflight_backup__"

    if [[ -d "$abs_dir" ]] && [[ -n "$(ls -A "$abs_dir" 2>/dev/null || true)" ]]; then
      [[ ! -e "$backup_dir" ]] || {
        echo "ERROR: backup dir already exists: $backup_dir" >&2
        return 1
      }
      mv "$abs_dir" "$backup_dir"
      moved_dirs+=("${rel_dir}")
    fi
  done

  local rc=0
  if ! bash scripts/audit/pre_release.sh; then
    rc=1
  fi

  for rel_dir in "${moved_dirs[@]}"; do
    local abs_dir="${REPO_ROOT}/${rel_dir}"
    local backup_dir="${REPO_ROOT}.${rel_dir}.__preflight_backup__"
    mv "$backup_dir" "$abs_dir"
  done

  return $rc
}

_run_step "audit_pre_release" _run_audit_with_generated_dirs_shelved_if_needed

if [[ "${SKIP_DATA_STRICT}" != "1" ]]; then
  _run_step "data_strict" bash scripts/data/prepare_all.sh --out_dir data --strict 1
else
  echo "[preflight] SKIP data_strict (--skip_data_strict=1)" | tee -a "$LOG_TXT"
  STEP_STATUS+=("data_strict:skip")
fi

_run_step "smoke_${TASK}" bash scripts/reproduce/smoke.sh --task "$TASK" --limit 1 --print_example 0

"$PYTHON_BIN" - <<'PY' "$REPORT_JSON" "${STEP_STATUS[@]}"
import json
import sys
from datetime import datetime, timezone

report_path = sys.argv[1]
items = sys.argv[2:]

steps = []
overall_ok = True
for x in items:
    name, status = x.split(":", 1)
    steps.append({"name": name, "status": status})
    if status == "fail":
        overall_ok = False

report = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "overall_status": "ok" if overall_ok else "fail",
    "steps": steps,
}

with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
PY

echo "[preflight] report_json=${REPORT_JSON}" | tee -a "$LOG_TXT"
echo "[preflight] log_txt=${LOG_TXT}" | tee -a "$LOG_TXT"

if [[ "$HAS_FAIL" -eq 1 ]]; then
  echo "[preflight] DONE (with failures)" | tee -a "$LOG_TXT"
  exit 1
fi

echo "[preflight] DONE (all checks passed)" | tee -a "$LOG_TXT"
