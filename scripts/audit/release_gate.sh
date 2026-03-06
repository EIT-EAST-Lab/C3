#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
source "${REPO_ROOT}/scripts/reproduce/common_env.sh"

PYTHON_BIN="${PYTHON:-python}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/.cache/release_gate}"
RUN_SFT_EVAL="${RUN_SFT_EVAL:-0}"
HF_BASE="${HF_BASE:-${SMOKE_HF_BASE:-}}"

TMP_PYCACHE="$(mktemp -d)"
cleanup() {
  rm -rf "${TMP_PYCACHE}" 2>/dev/null || true
}
trap cleanup EXIT

mkdir -p "${OUT_DIR}"
cd "${REPO_ROOT}"
c3_repro_export_common_env "${REPO_ROOT}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${TMP_PYCACHE}"

echo "[release_gate] 1/5 pytest"
"${PYTHON_BIN}" -m pytest -q -p no:cacheprovider tests

echo "[release_gate] 2/5 fixture math smoke"
bash scripts/reproduce/smoke.sh \
  --task tests/fixtures/tasks/mini_math.yaml \
  --limit 1 \
  --print_example 0 \
  --skip_import_checks 1

echo "[release_gate] 3/5 fixture code smoke"
bash scripts/reproduce/smoke.sh \
  --task tests/fixtures/tasks/mini_code.yaml \
  --limit 1 \
  --print_example 0 \
  --skip_import_checks 1

echo "[release_gate] 4/5 dummy mechanism figure"
"${PYTHON_BIN}" -m c3.tools.plot_paper_figures mechanism \
  --out_dir "${OUT_DIR}/fig2_dummy" \
  --use_dummy \
  --fmt png

echo "[release_gate] 5/5 pre-release audit"
bash scripts/audit/pre_release.sh

if [[ "${RUN_SFT_EVAL}" == "1" ]]; then
  [[ -n "${HF_BASE}" ]] || {
    echo "ERROR: RUN_SFT_EVAL=1 requires HF_BASE or SMOKE_HF_BASE." >&2
    exit 2
  }
  echo "[release_gate] optional paper-facing SFT eval"
  bash scripts/data/prepare_all.sh --out_dir data --strict 1
  bash scripts/reproduce/paper_main_results.sh one \
    --id "RELEASE_GATE_SFT" \
    --method "SFT" \
    --task "math" \
    --profile "greedy" \
    --seed "0" \
    --source_type "hf_base" \
    --hf_base "${HF_BASE}" \
    --out_subdir "release_gate_eval"
fi

echo "[release_gate] DONE"
