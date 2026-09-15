#!/usr/bin/env bash
set -euo pipefail

# scripts/40_train/paper_train.sh
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
source "${REPO_ROOT}/scripts/_lib/common_env.sh"

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

_paper_train_n_samples() {
  local method="$1"
  "$PYTHON_BIN" - "$method" <<'PY'
import sys

from c3.utils.paper_train_contract import get_paper_train_n_samples

print(get_paper_train_n_samples(sys.argv[1]))
PY
}

_paper_train_recipe_args() {
  "$PYTHON_BIN" - <<'PY'
from c3.utils.paper_train_contract import render_paper_train_args

print(render_paper_train_args())
PY
}

run_one() {
  local alg="$1" task="$2" seed="$3"
  local run_id="paper_${alg}_${task}_seed${seed}"
  local run_dir="${CKPT_ROOT}/_runs/${run_id}"
  local final_hf="${run_dir}/final_hf"
  local n_samples_per_prompt
  local recipe_args_line
  local -a recipe_args=()

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
  n_samples_per_prompt="$(_paper_train_n_samples "$alg")"
  # Assign first, split second: under `set -e` the assignment aborts the run if
  # the helper fails, which a `read` from a command substitution would not. A
  # run that quietly lost the recipe would train on the trainer's defaults.
  recipe_args_line="$(_paper_train_recipe_args)"
  read -r -a recipe_args <<< "$recipe_args_line"
  if [[ ${#recipe_args[@]} -eq 0 ]]; then
    echo "ERROR: the paper training recipe rendered empty" >&2
    echo "       Expected flags from c3/utils/paper_train_contract.py" >&2
    exit 1
  fi

  # The batch sizes and the optimization flags are not written here. They are
  # rendered from c3/utils/paper_train_contract.py, which is this repository's
  # copy of Table 3 of the paper, so that the run follows the table instead of
  # the trainer's argparse defaults. The two deliberate departures from that
  # table, the generation cap and the evaluation interval, are listed there in
  # PAPER_TRAIN_DEPARTURES with their reasons, and are passed below.
  #
  # Decoding and evaluation follow docs/32_evaluation_protocol.md:
  #   - generation cap 2048 for training and evaluation alike, because 512
  #     truncates the harder suites before they reach an answer;
  #   - temperature 0.7, top-p 0.8, top-k 20 for the rollouts, and the same
  #     top-p and top-k for the evaluation, which reads them off these flags and
  #     overrides only the temperature and the sample count;
  #   - one evaluation every ten percent of the run, four samples per problem,
  #     which is the monitoring curve. The numbers of the main table come from
  #     scripts/70_rebuild/final_eval.py after training, not from these.
  "$PYTHON_BIN" -m openrlhf.cli.train_ppo_ray \
    --c3_task "$task_yaml" \
    --marl_algorithm "${alg,,}" \
    --policy_sharing_mode shared \
    --pretrain "$PRETRAIN" \
    --seed "$seed" \
    "${recipe_args[@]}" \
    --generate_max_len 2048 \
    --temperature 0.7 \
    --top_p 0.8 \
    --top_k 20 \
    --n_samples_per_prompt "$n_samples_per_prompt" \
    --eval_every_ratio 0.10 \
    --eval_temperature 0.7 \
    --eval_n_samples_per_prompt 4 \
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
