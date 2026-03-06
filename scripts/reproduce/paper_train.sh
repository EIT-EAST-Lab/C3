#!/usr/bin/env bash
set -euo pipefail

# scripts/reproduce/paper_train.sh
#
# One-command launcher for the paper training matrix:
#   METHODS  : MAPPO, MAGRPO, C3
#   TASKS    : math, code
#   SEEDS    : 0..4
#
# Outputs
#   - Run directory:        ckpt/_runs/<run_id>/
#   - Final HF checkpoint:  ckpt/_runs/<run_id>/final_hf/
#
# This script is intentionally non-interactive and non-branchy:
# it always runs the full matrix and fails fast on missing prerequisites.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/common_env.sh"

c3_repro_export_common_env "${REPO_ROOT}"
export WANDB_DISABLED=true

# Ensure Ray is torn down between runs (OpenRLHF reads this env var on exit).
export OPENRLHF_RAY_STOP_ON_EXIT=1

CKPT_ROOT="ckpt"

if [[ -z "${PRETRAIN:-}" ]]; then
  cat >&2 <<'EOF'
ERROR: PRETRAIN is not set.

Set PRETRAIN to a HuggingFace model id or a local HF directory, e.g.:
  export PRETRAIN='Qwen/Qwen2.5-3B-Instruct'
EOF
  exit 1
fi

_require_file() {
  local p="$1"
  [[ -f "$p" ]] || { echo "ERROR: missing required file: $p" >&2; exit 1; }
}

_require_dir() {
  local p="$1"
  [[ -d "$p" ]] || { echo "ERROR: missing required directory: $p" >&2; exit 1; }
}

cd "$REPO_ROOT"

# Basic repo prerequisites.
_require_file "configs/tasks/math.yaml"
_require_file "configs/tasks/code.yaml"
_require_dir "scripts"

# NOTE:
# We do NOT enforce dataset file layout here.
# The training/eval pipeline will error out naturally if required files are missing.
mkdir -p "${CKPT_ROOT}/_runs"

PYTHON_BIN="${PYTHON_BIN:-python}"

run_one() {
  local alg="$1" task="$2" seed="$3"
  local run_id="paper_${alg}_${task}_seed${seed}"
  local run_dir="${CKPT_ROOT}/_runs/${run_id}"
  local final_hf="${run_dir}/final_hf"

  if [[ -e "$run_dir" ]]; then
    echo "ERROR: run_dir already exists: $run_dir" >&2
    echo "       Refusing to overwrite. Remove it (or move it aside) and re-run." >&2
    exit 1
  fi

  echo "[paper_train] RUN alg=${alg} task=${task} seed=${seed}" >&2
  echo "[paper_train]   run_id=${run_id}" >&2
  echo "[paper_train]   run_dir=${run_dir}" >&2
  echo "[paper_train]   final_hf=${final_hf}" >&2

  local task_yaml="configs/tasks/${task}.yaml"

  "$PYTHON_BIN" -m openrlhf.cli.train_ppo_ray \
    --c3_task "$task_yaml" \
    --marl_algorithm "${alg,,}" \
    --policy_sharing_mode shared \
    --pretrain "$PRETRAIN" \
    --seed "$seed" \
    --prompt_max_len 2560 \
    --generate_max_len 512 \
    --temperature 0.7 \
    --top_p 0.8 \
    --top_k 20 \
    --n_samples_per_prompt 8 \
    --ckpt_path "$CKPT_ROOT" \
    --run_id "$run_id" \
    --wandb_run_name "$run_id" \
    --run_dir "$run_dir" \
    --save_path "$final_hf" \
    --use_wandb 0

  if [[ ! -d "$final_hf" ]]; then
    echo "ERROR: training finished but final_hf directory not found: $final_hf" >&2
    exit 1
  fi
}

METHODS=(MAPPO MAGRPO C3)
TASKS=(math code)
SEEDS=(0 1 2 3 4)

for alg in "${METHODS[@]}"; do
  for task in "${TASKS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      run_one "$alg" "$task" "$seed"
    done
  done
done

echo "[paper_train] DONE. Training outputs are under ${CKPT_ROOT}/_runs/paper_*" >&2
