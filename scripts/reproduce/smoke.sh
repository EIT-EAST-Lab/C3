#!/usr/bin/env bash
# scripts/reproduce/smoke.sh
set -euo pipefail

_usage() {
  cat <<'USAGE'
C3 smoke test.

Default behavior:
  - import-check key entrypoints
  - run a tiny end-to-end env+evaluator pass (NO model inference)

Usage:
  bash scripts/reproduce/smoke.sh [options]

Options:
  --task PATH|math|code     Task YAML path (or shorthand). Default: math.
  --limit N                #examples to run through the env+evaluator. Default: 1.
  --seed S                 Random seed for the smoke test. Default: 0.
  --print_example 0|1      Whether to print one example (prompts/pred/result). Default: 1.

  # Optional (slower): run ONE eval-only call through the OpenRLHF CLI using an HF base model.
  # This does NOT require checkpoints, but it may require GPU/vLLM depending on your setup.
  --eval_sft 0|1           Run one eval-only pass via paper_main_results.sh (default: 0).
  --hf_base ID|PATH        HF model id or local path for --eval_sft (default: from $SMOKE_HF_BASE).
  --profile greedy|n10     Profile for --eval_sft (default: greedy).
  --skip_import_checks 0|1 Skip import-check stage (default: 0).

  -h, --help               Show this help.

Environment knobs:
  PYTHON          Python executable (default: python)
  TRAIN_MOD       Training entry module to import-check (default: openrlhf.cli.train_ppo_ray)
  ANALYSIS_MOD    Analysis entry module to import-check (default: c3.analysis.c3_analysis)
  SMOKE_MOD       Env smoke module (default: c3.tools.c3_env_smoke)
  SMOKE_HF_BASE   Default HF base model for --eval_sft (optional)
USAGE
}

PYTHON_BIN="${PYTHON:-python}"
TRAIN_MOD="${TRAIN_MOD:-openrlhf.cli.train_ppo_ray}"
ANALYSIS_MOD="${ANALYSIS_MOD:-c3.analysis.c3_analysis}"
SMOKE_MOD="${SMOKE_MOD:-c3.tools.c3_env_smoke}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
source "${SCRIPT_DIR}/common_env.sh"

TASK="math"
LIMIT="1"
SEED="0"
PRINT_EXAMPLE="1"
EVAL_SFT="0"
HF_BASE="${SMOKE_HF_BASE:-}"
PROFILE="greedy"
SKIP_IMPORT_CHECKS="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) TASK="${2:-}"; shift 2 ;;
    --limit) LIMIT="${2:-}"; shift 2 ;;
    --seed) SEED="${2:-}"; shift 2 ;;
    --print_example) PRINT_EXAMPLE="${2:-}"; shift 2 ;;
    --eval_sft) EVAL_SFT="${2:-}"; shift 2 ;;
    --hf_base) HF_BASE="${2:-}"; shift 2 ;;
    --profile) PROFILE="${2:-}"; shift 2 ;;
    --skip_import_checks) SKIP_IMPORT_CHECKS="${2:-}"; shift 2 ;;
    -h|--help) _usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; _usage; exit 2 ;;
  esac
done

_task_to_yaml() {
  local t="$1"
  case "$t" in
    math) echo "${REPO_ROOT}/configs/tasks/math.yaml" ;;
    code) echo "${REPO_ROOT}/configs/tasks/code.yaml" ;;
    /*) echo "$t" ;;
    *) echo "${REPO_ROOT}/${t}" ;;
  esac
}

TASK_YAML="$(_task_to_yaml "$TASK")"

cd "${REPO_ROOT}"
c3_repro_export_common_env "${REPO_ROOT}"

_echo() { echo "[smoke] $*" >&2; }

_echo "repo_root=${REPO_ROOT}"
_echo "python=${PYTHON_BIN}"
_echo "task_yaml=${TASK_YAML}"

if [[ "${SKIP_IMPORT_CHECKS}" != "1" ]]; then
  _echo "1) import-check: ${TRAIN_MOD}"
  "${PYTHON_BIN}" -c "import ${TRAIN_MOD}" >/dev/null

  _echo "2) import-check: ${ANALYSIS_MOD}"
  "${PYTHON_BIN}" -c "import ${ANALYSIS_MOD}" >/dev/null

  _echo "3) import-check: ${SMOKE_MOD}"
  "${PYTHON_BIN}" -c "import ${SMOKE_MOD}" >/dev/null
else
  _echo "1-3) skip import-checks (--skip_import_checks=1)"
fi

[[ -f "${TASK_YAML}" ]] || { echo "ERROR: task yaml not found: ${TASK_YAML}" >&2; exit 1; }

_echo "4) preflight: check dataset files referenced by task yaml"
"${PYTHON_BIN}" - <<'PY' "${TASK_YAML}" "${REPO_ROOT}"
import os
import sys
from pathlib import Path
import yaml

task_yaml = sys.argv[1]
repo_root = sys.argv[2]
task_dir = str(Path(task_yaml).resolve().parent)

with open(task_yaml, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

env = cfg.get("environment") or {}
missing = []

for section in ("train_datasets", "eval_suites"):
    for item in env.get(section) or []:
        if not isinstance(item, dict):
            continue
        p = (item.get("path") or "").strip()
        if not p:
            continue
        candidates = []
        if os.path.isabs(p):
            candidates.append(p)
        else:
            candidates.append(os.path.join(repo_root, p))
            candidates.append(os.path.join(task_dir, p))
        if not any(os.path.exists(pp) for pp in candidates):
            missing.append((section, p))

if missing:
    print("ERROR: missing local dataset files required by smoke test:", file=sys.stderr)
    for section, p in missing:
        print(f"  - [{section}] {p}", file=sys.stderr)
    print("", file=sys.stderr)
    print("Prepare datasets first:", file=sys.stderr)
    print("  bash scripts/data/prepare_all.sh --out_dir data", file=sys.stderr)
    raise SystemExit(1)
PY

_echo "5) c3_env_smoke: limit=${LIMIT} seed=${SEED} print_example=${PRINT_EXAMPLE}"
"${PYTHON_BIN}" -m "${SMOKE_MOD}" \
  --task "${TASK_YAML}" \
  --limit "${LIMIT}" \
  --seed "${SEED}" \
  --print_example "${PRINT_EXAMPLE}"

if [[ "${EVAL_SFT}" == "1" ]]; then
  [[ -n "${HF_BASE}" ]] || { echo "ERROR: --hf_base (or SMOKE_HF_BASE) is required when --eval_sft=1" >&2; exit 2; }
  _echo "6) eval-only via paper_main_results.sh (SFT baseline): hf_base=${HF_BASE} profile=${PROFILE}"
  bash scripts/reproduce/paper_main_results.sh one \
    --id "SMOKE_SFT" \
    --method "SFT" \
    --task "${TASK}" \
    --profile "${PROFILE}" \
    --seed "${SEED}" \
    --source_type "hf_base" \
    --hf_base "${HF_BASE}" \
    --out_subdir "smoke_eval"
fi

_echo "OK"