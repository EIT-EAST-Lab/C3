#!/usr/bin/env bash
# scripts/audit/pre_release.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON:-python}"

# ---------------------------------------------------------------------------
# Bytecode hygiene
#
# This audit script must NOT dirty the working tree. Running Python entrypoints
# (including compileall) can create __pycache__/ and *.pyc files under the repo
# root. We:
#   1) fail fast if such artifacts already exist in the repo,
#   2) redirect any bytecode output to a temporary directory via
#      PYTHONPYCACHEPREFIX, and
#   3) re-check after the audit to ensure the repo remains clean.
# ---------------------------------------------------------------------------

fail() {
  echo "[FAIL] $*" >&2
  exit 1
}

check_no_bytecode_artifacts() {
  local hits=""
  local hits_arr=()
  mapfile -t hits_arr < <(
    find "${ROOT}" \
      \( -type d -name "__pycache__" -o -type f -name "*.py[co]" \) \
      -not -path "${ROOT}/.git/*" \
      -not -path "${ROOT}/.venv/*" \
      -not -path "${ROOT}/venv/*" \
      -not -path "${ROOT}/env/*" \
      -not -path "${ROOT}/ENV/*" \
      -not -path "${ROOT}/data/*" \
      -not -path "${ROOT}/ckpt/*" \
      -not -path "${ROOT}/runs/*" \
      -not -path "${ROOT}/wandb/*" \
      -not -path "${ROOT}/models/*" \
      -not -path "${ROOT}/artifacts/*" \
      2>/dev/null
  )

  if [[ "${#hits_arr[@]}" -gt 0 ]]; then
    hits="$(printf '%s\n' "${hits_arr[@]:0:50}")"
  fi

  if [[ -n "${hits}" ]]; then
    echo "[FAIL] Found Python bytecode artifacts in the repo (must not be shipped):" >&2
    echo "${hits}" >&2
    echo "[HINT] Clean them with:" >&2
    echo "  find . -type d -name '__pycache__' -prune -exec rm -rf {} +" >&2
    echo "  find . -type f -name '*.py[co]' -delete" >&2
    exit 1
  fi
}

echo "[INFO] Repo root: ${ROOT}"

echo "[0/6] Repo hygiene: ensure no __pycache__/ or *.pyc in the repo..."
check_no_bytecode_artifacts
echo "[OK] repo hygiene passed."

# Redirect bytecode output for any Python execution in this script.
orig_pycacheprefix_set=0
orig_pycacheprefix=""
if [[ -n "${PYTHONPYCACHEPREFIX+x}" ]]; then
  orig_pycacheprefix_set=1
  orig_pycacheprefix="${PYTHONPYCACHEPREFIX}"
fi

orig_dontwrite_set=0
orig_dontwrite=""
if [[ -n "${PYTHONDONTWRITEBYTECODE+x}" ]]; then
  orig_dontwrite_set=1
  orig_dontwrite="${PYTHONDONTWRITEBYTECODE}"
fi

TMP_PYCACHE="$(mktemp -d)"

cleanup() {
  rm -rf "${TMP_PYCACHE}" 2>/dev/null || true

  if [[ "${orig_pycacheprefix_set}" -eq 1 ]]; then
    export PYTHONPYCACHEPREFIX="${orig_pycacheprefix}"
  else
    unset PYTHONPYCACHEPREFIX 2>/dev/null || true
  fi

  if [[ "${orig_dontwrite_set}" -eq 1 ]]; then
    export PYTHONDONTWRITEBYTECODE="${orig_dontwrite}"
  else
    unset PYTHONDONTWRITEBYTECODE 2>/dev/null || true
  fi
}
trap cleanup EXIT

export PYTHONPYCACHEPREFIX="${TMP_PYCACHE}"
# Avoid creating bytecode caches via imports (compileall is still run, but its
# output is redirected by PYTHONPYCACHEPREFIX).
export PYTHONDONTWRITEBYTECODE=1

echo "[1/6] Scan for hard-coded absolute paths..."
"${PY}" "${ROOT}/scripts/audit/scan_paths.py" --root "${ROOT}"

echo "[2/6] Scan for obvious secrets..."
"${PY}" "${ROOT}/scripts/audit/scan_secrets.py" --root "${ROOT}"

echo "[3/6] Check for bundled data/large artifacts..."
"${PY}" "${ROOT}/scripts/audit/no_data_check.py" --root "${ROOT}" --max_mb 20

echo "[4/6] Bash syntax check for *.sh..."
# shellcheck disable=SC2044
for f in $(find "${ROOT}/scripts" -name "*.sh" -type f); do
  bash -n "$f"
done
echo "[OK] bash -n passed."

echo "[5/6] Python syntax compile (best-effort)..."
"${PY}" -m compileall -q "${ROOT}/c3" "${ROOT}/openrlhf" "${ROOT}/scripts" || {
  echo "[FAIL] python compileall failed." >&2
  exit 1
}
echo "[OK] python compileall passed."

echo "[6/6] Repo hygiene: verify audit did not create __pycache__/ or *.pyc..."
check_no_bytecode_artifacts
echo "[OK] repo hygiene post-check passed."

echo "[DONE] pre-release audit passed."
