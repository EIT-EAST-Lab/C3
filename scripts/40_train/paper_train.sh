#!/usr/bin/env bash
set -euo pipefail

# scripts/40_train/paper_train.sh
#
# One-command launcher for the training runs of the paper
# (arXiv:2603.06859v3, "The Trace Is the State: Exact Credit Assignment for
# LLM Agent Teams"). RECIPE picks which trained table to reproduce:
#
#   RECIPE=main  Table 2: the 5-seed run, 2-agent workflow (Reasoner, Actor),
#                512-token generation limit, training seeds 0..4 (default)
#   RECIPE=long  Table 7: the long-generation run, 2-agent workflow,
#                2048-token generation limit, one training seed
#   RECIPE=a3    Table 8: the 5-seed run, 3-agent workflow (Reasoner, Actor,
#                Verifier), training seeds 0..4
#
# Every recipe trains the three methods the paper compares: MAPPO, MAGRPO and
# C3, on the math task of the recipe. The hyperparameters are not written here:
# they are rendered per (recipe, method) from c3/utils/paper_train_contract.py,
# which is this repository's copy of Tables 5 and 6 and of Appendix B.2, so a
# run follows the tables instead of the trainer's argparse defaults.
#
# Environment
#   PRETRAIN    base model, a HuggingFace id or a local HF directory; the paper
#               trains Qwen/Qwen3-4B-Instruct-2507 (required)
#   RECIPE      main | long | a3 (default: main)
#   DRY_RUN     1 prints the final command line of every run and starts none
#   PYTHON_BIN  Python executable (default: python)
#   EXTRA_ARGS  extra trainer flags appended to every run, for the hardware
#               layout of the machine (GPU and vLLM engine counts, colocation,
#               offloading, gradient checkpointing); the hyperparameters of the
#               tables stay in the recipe
#
# Outputs
#   - Run directory:        ckpt/_runs/<run_id>/
#   - Checkpoints:          ckpt/_runs/<run_id>/_actor/ (and _critic/ for MAPPO),
#                           one directory per run, so no run rotates or
#                           overwrites the checkpoints of another
#   - Final HF checkpoint:  ckpt/_runs/<run_id>/final_hf/
#
# The numbers of the tables come from the evaluation that runs after training,
# scripts/70_rebuild/final_eval.py, on each final_hf directory; the evaluation
# that runs during training is the monitoring curve only. See
# scripts/70_rebuild/README.md for the evaluation and aggregation commands.
#
# This script is intentionally non-interactive and non-branchy:
# it runs the full matrix of the recipe and fails fast on missing prerequisites.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/scripts/_lib/common_env.sh"

c3_repro_export_common_env "${REPO_ROOT}"
export WANDB_DISABLED=true

# Ensure Ray is torn down between runs (OpenRLHF reads this env var on exit).
export OPENRLHF_RAY_STOP_ON_EXIT=1

CKPT_ROOT="ckpt"
RECIPE="${RECIPE:-main}"
DRY_RUN="${DRY_RUN:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
read -r -a EXTRA <<< "$EXTRA_ARGS"

if [[ -z "${PRETRAIN:-}" ]]; then
  if [[ "$DRY_RUN" == "1" ]]; then
    PRETRAIN='<PRETRAIN>'
  else
    cat >&2 <<'EOF'
ERROR: PRETRAIN is not set.

Set PRETRAIN to a HuggingFace model id or a local HF directory. The paper trains
  export PRETRAIN='Qwen/Qwen3-4B-Instruct-2507'
EOF
    exit 1
  fi
fi

_require_file() {
  local p="$1"
  [[ -f "$p" ]] || { echo "ERROR: missing required file: $p" >&2; exit 1; }
}

cd "$REPO_ROOT"

# Ask the contract for one piece of the recipe. Assign first, split second:
# under `set -e` the assignment aborts the run if the helper fails, which a
# `read` from a command substitution would not. A run that quietly lost the
# recipe would train on the trainer's defaults.
_contract() {
  "$PYTHON_BIN" -m c3.utils.paper_train_contract "$@"
}

TASK_YAML="$(_contract task --recipe "$RECIPE")"
SEEDS_LINE="$(_contract seeds --recipe "$RECIPE")"
read -r -a SEEDS <<< "$SEEDS_LINE"
_require_file "$TASK_YAML"
if [[ ${#SEEDS[@]} -eq 0 ]]; then
  echo "ERROR: recipe ${RECIPE} lists no training seed" >&2
  exit 1
fi

# NOTE:
# We do NOT enforce dataset file layout here.
# The training/eval pipeline will error out naturally if required files are missing.
if [[ "$DRY_RUN" != "1" ]]; then
  mkdir -p "${CKPT_ROOT}/_runs"
fi

run_one() {
  local alg="$1" seed="$2"
  local run_id="paper_${RECIPE}_${alg}_math_seed${seed}"
  local run_dir="${CKPT_ROOT}/_runs/${run_id}"
  local final_hf="${run_dir}/final_hf"
  local recipe_args_line
  local -a recipe_args=()
  local -a cmd=()

  recipe_args_line="$(_contract args --recipe "$RECIPE" --method "$alg")"
  read -r -a recipe_args <<< "$recipe_args_line"
  if [[ ${#recipe_args[@]} -eq 0 ]]; then
    echo "ERROR: the paper training recipe ${RECIPE} rendered empty for ${alg}" >&2
    echo "       Expected flags from c3/utils/paper_train_contract.py" >&2
    exit 1
  fi

  cmd=(
    "$PYTHON_BIN" -m openrlhf.cli.train_ppo_ray
    --c3_task "$TASK_YAML"
    --marl_algorithm "${alg,,}"
    --pretrain "$PRETRAIN"
    --seed "$seed"
    "${recipe_args[@]}"
    --ckpt_path "$run_dir"
    --run_id "$run_id"
    --wandb_run_name "$run_id"
    --run_dir "$run_dir"
    --save_path "$final_hf"
    --use_wandb 0
    ${EXTRA[@]+"${EXTRA[@]}"}
  )

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "# recipe=${RECIPE} alg=${alg} seed=${seed}"
    printf '%q ' "${cmd[@]}"
    printf '\n'
    return 0
  fi

  if [[ -e "$run_dir" ]]; then
    echo "ERROR: run_dir already exists: $run_dir" >&2
    echo "       Refusing to overwrite. Remove it (or move it aside) and re-run." >&2
    exit 1
  fi

  echo "[paper_train] RUN recipe=${RECIPE} alg=${alg} seed=${seed}" >&2
  echo "[paper_train]   run_id=${run_id}" >&2
  echo "[paper_train]   run_dir=${run_dir}" >&2
  echo "[paper_train]   final_hf=${final_hf}" >&2

  "${cmd[@]}"

  if [[ ! -d "$final_hf" ]]; then
    echo "ERROR: training finished but final_hf directory not found: $final_hf" >&2
    exit 1
  fi
}

METHODS=(MAPPO MAGRPO C3)

for alg in "${METHODS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_one "$alg" "$seed"
  done
done

if [[ "$DRY_RUN" == "1" ]]; then
  echo "# recipe=${RECIPE}: $(( ${#METHODS[@]} * ${#SEEDS[@]} )) runs, none started (DRY_RUN=1)"
else
  echo "[paper_train] DONE. Training outputs are under ${CKPT_ROOT}/_runs/paper_${RECIPE}_*" >&2
fi
