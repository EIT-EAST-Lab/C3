#!/usr/bin/env bash
# scripts/90_audit/release_gate.sh
#
# The full local release gate. Everything it runs works on a CPU-only machine
# with the CPU tier installed (see the README): no GPU, no checkpoints.
#
# Environment:
#   PYTHON        Python executable (default: python)
#   DATA_DIR      Prepared dataset directory to verify (default: data).
#                 The manifest check is skipped when the directory is absent.
#   OUT_DIR       Scratch directory for gate artifacts (default: .cache/release_gate)
#   RUN_SFT_EVAL  1 to also run one eval-only pass with a real HF base model
#   HF_BASE       HF model id required when RUN_SFT_EVAL=1
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
source "${REPO_ROOT}/scripts/_lib/common_env.sh"

PYTHON_BIN="${PYTHON:-python}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/.cache/release_gate}"
DATA_DIR="${DATA_DIR:-data}"
RUN_SFT_EVAL="${RUN_SFT_EVAL:-0}"
HF_BASE="${HF_BASE:-${SMOKE_HF_BASE:-}}"

TMP_PYCACHE="$(mktemp -d)"
STATUS_BEFORE="$(mktemp)"
cleanup() {
  rm -rf "${TMP_PYCACHE}" 2>/dev/null || true
  rm -f "${STATUS_BEFORE}" 2>/dev/null || true
}
trap cleanup EXIT

# `git status` is only used to prove the gate leaves no artifacts behind, so it
# must never be able to hang the gate. Bound it, and downgrade to a warning when
# it is unavailable or slow (a network filesystem can stall it for minutes).
_c3_git_status() {
  local out="$1"
  if command -v timeout >/dev/null 2>&1; then
    timeout 120 git -C "${REPO_ROOT}" status --porcelain > "${out}" 2>/dev/null
  else
    git -C "${REPO_ROOT}" status --porcelain > "${out}" 2>/dev/null
  fi
}

IN_GIT_TREE=0
if git -C "${REPO_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  if _c3_git_status "${STATUS_BEFORE}"; then
    IN_GIT_TREE=1
  else
    echo "[release_gate] WARN: git status did not finish; the working-tree check is skipped."
  fi
fi

mkdir -p "${OUT_DIR}"
cd "${REPO_ROOT}"
c3_repro_export_common_env "${REPO_ROOT}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${TMP_PYCACHE}"

echo "[release_gate] 1/8 dependency consistency"
"${PYTHON_BIN}" -m pip check

echo "[release_gate] 2/8 pytest"
"${PYTHON_BIN}" -m pytest -q -p no:cacheprovider tests

echo "[release_gate] 3/8 fixture math smoke (CPU tier import checks included)"
bash scripts/30_smoke/smoke.sh \
  --tier cpu \
  --task tests/fixtures/tasks/mini_math.yaml \
  --limit 1 \
  --print_example 0

echo "[release_gate] 4/8 fixture code smoke"
bash scripts/30_smoke/smoke.sh \
  --tier cpu \
  --task tests/fixtures/tasks/mini_code.yaml \
  --limit 1 \
  --print_example 0 \
  --skip_import_checks 1

echo "[release_gate] 5/8 dummy mechanism figure"
"${PYTHON_BIN}" -m c3.tools.plot_paper_figures mechanism \
  --out_dir "${OUT_DIR}/fig2_dummy" \
  --use_dummy \
  --fmt png

echo "[release_gate] 6/8 dataset manifest verification"
if [[ -d "${DATA_DIR}" ]]; then
  bash scripts/10_data/prepare_all.sh --out_dir "${DATA_DIR}" --strict 1
else
  echo "[release_gate] SKIP: ${DATA_DIR} is absent."
  echo "[release_gate]       Run 'bash scripts/10_data/prepare_all.sh --out_dir ${DATA_DIR}' first"
  echo "[release_gate]       to include the strict manifest check in this gate."
  echo "[release_gate]       Until then this gate is 7 steps, not 8: it did not verify any dataset."
fi

echo "[release_gate] 7/8 model registry resolution (dry run)"
bash scripts/20_models/download_models.sh --dry_run 1

echo "[release_gate] 8/8 pre-release audit"
bash scripts/90_audit/pre_release.sh

if [[ "${RUN_SFT_EVAL}" == "1" ]]; then
  [[ -n "${HF_BASE}" ]] || {
    echo "ERROR: RUN_SFT_EVAL=1 requires HF_BASE or SMOKE_HF_BASE." >&2
    exit 2
  }
  echo "[release_gate] optional paper-facing SFT eval"
  bash scripts/10_data/prepare_all.sh --out_dir data --strict 1
  bash scripts/50_eval/paper_main_results.sh one \
    --id "RELEASE_GATE_SFT" \
    --method "SFT" \
    --task "math" \
    --profile "greedy" \
    --seed "0" \
    --source_type "hf_base" \
    --hf_base "${HF_BASE}" \
    --out_subdir "release_gate_eval"
fi

STATUS_AFTER="$(mktemp)"
if [[ "${IN_GIT_TREE}" -eq 1 ]] && _c3_git_status "${STATUS_AFTER}"; then
  NEW_ENTRIES="$(grep -Fxv -f "${STATUS_BEFORE}" "${STATUS_AFTER}" || true)"
  rm -f "${STATUS_AFTER}"
  if [[ -n "${NEW_ENTRIES}" ]]; then
    echo "[release_gate] FAIL: the gate left new files in the working tree:" >&2
    echo "${NEW_ENTRIES}" >&2
    exit 1
  fi
  echo "[release_gate] the gate left the working tree as it found it"
else
  rm -f "${STATUS_AFTER}"
  echo "[release_gate] WARN: the working-tree check was skipped."
fi

echo "[release_gate] DONE"
