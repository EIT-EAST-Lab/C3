#!/usr/bin/env bash
set -euo pipefail

# scripts/reproduce/paper_main_results.sh
#
# Run evaluation-only sweeps for paper main results, and aggregate tables.
#
# Usage:
#   bash scripts/reproduce/paper_main_results.sh sweep \
#     --registry configs/main_results_registry.yaml \
#     --ckpt_root ckpt \
#     --out_dir ckpt/paper_main_results \
#     [--only_methods SFT|MAPPO|MAGRPO|C3] \
#     [--resume 1]
#
#   bash scripts/reproduce/paper_main_results.sh one \
#     --id <run_id> \
#     --profile greedy|n10 \
#     --registry configs/main_results_registry.yaml
#
#   # Direct one-shot mode (no registry lookup):
#   bash scripts/reproduce/paper_main_results.sh one \
#     --id <run_id> \
#     --profile greedy|n10 \
#     --method C3 \
#     --task math|code|configs/tasks/*.yaml \
#     --source_type train_run_dir|hf_base \
#     --train_run_dir ckpt/_runs/<run_id> \
#     --hf_base Qwen/Qwen2.5-3B-Instruct \
#     --seed 0 \
#     --out_subdir main_results
#
# Notes:
# - This script never trains. It runs openrlhf.cli.train_ppo_ray in --eval_only mode.
# - For hf_base entries, artifacts are written under ckpt/_runs/_sft_main_results/<id>/...
# - For train_run_dir entries, artifacts are written under <train_run_dir>/<out_subdir>/...

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/common_env.sh"
PYTHON_BIN="${PYTHON_BIN:-python}"

CKPT_ROOT="ckpt"
REGISTRY="configs/main_results_registry.yaml"
OUT_DIR=""
ONLY_METHODS=""
ONLY_TASKS=""
ONLY_IDS=""
PROFILE=""
RUN_ID=""
RESUME="0"
VLLM_NUM_ENGINES="${VLLM_NUM_ENGINES:-1}"
VLLM_TP="${VLLM_TP:-1}"
VLLM_MEM_UTIL="${VLLM_MEM_UTIL:-0.85}"
PROMPT_MAX_LEN="${PROMPT_MAX_LEN:-2560}"
GENERATE_MAX_LEN="${GENERATE_MAX_LEN:-512}"
DIRECT_METHOD=""
DIRECT_TASK=""
DIRECT_SOURCE_TYPE=""
DIRECT_HF_BASE=""
DIRECT_TRAIN_RUN_DIR=""
DIRECT_SEED=""
DIRECT_OUT_SUBDIR=""
DIRECT_EXTRA_EVAL_ARGS=""
DIRECT_ALG_FOR_EVAL=""

usage() {
  cat <<'EOF'
Usage:
  bash scripts/reproduce/paper_main_results.sh sweep [options]
  bash scripts/reproduce/paper_main_results.sh one   [options]

Commands:
  sweep   Run all runs x profiles defined in registry (with optional filters).
  one     Run one specified (id, profile).

Options:
  --registry FILE     Registry YAML (default: configs/main_results_registry.yaml)
  --ckpt_root DIR     Root for ckpt/_runs and sft container dirs (default: ckpt)
  --out_dir DIR       Aggregation output dir (default: <ckpt_root>/paper_main_results)
  --only_methods M    Comma-separated whitelist of methods (SFT,MAPPO,MAGRPO,C3)
  --only_tasks T      Comma-separated whitelist of tasks (math,code)
  --only_ids IDS      Comma-separated whitelist of run IDs
  --profile P         Profile for `one` command (greedy|n10)
  --id RUN_ID         Run ID for `one` command
  --resume 0|1        Skip eval if artifacts exist (default: 0)
  --method M          Direct `one` mode: method label (e.g., C3/SFT)
  --task T            Direct `one` mode: math|code|path/to/task.yaml
  --source_type S     Direct `one` mode: train_run_dir|hf_base
  --train_run_dir D   Direct `one` mode source path
  --hf_base ID|PATH   Direct `one` mode source model
  --seed N            Direct `one` mode seed (default: 0)
  --out_subdir NAME   Direct `one` mode eval output subdir
  --extra_eval_args   Direct `one` mode extra eval args
  --alg_for_eval A    Direct `one` mode algorithm override (default: auto)

Environment:
  PYTHON_BIN              Python interpreter (default: python)
  VLLM_NUM_ENGINES        vLLM engines for eval-only (default: 1)
  VLLM_TP                 vLLM tensor parallel size (default: 1)
  VLLM_MEM_UTIL           vLLM memory util (default: 0.85)
  PROMPT_MAX_LEN          Prompt max len (default: 2560)
  GENERATE_MAX_LEN        Generation max len (default: 512)

EOF
}

_task_to_yaml() {
  local raw="${1:-}"
  local lower="${raw,,}"
  case "$lower" in
    math) echo "configs/tasks/math.yaml" ;;
    code) echo "configs/tasks/code.yaml" ;;
    *)
      if [[ "$raw" == /* ]]; then
        echo "$raw"
      elif [[ "$raw" == *.yaml ]]; then
        echo "${REPO_ROOT}/$raw"
      else
        echo "ERROR: unknown task: $raw (expected math|code|path/to/*.yaml)" >&2
        exit 2
      fi
      ;;
  esac
}

_eval_profile_params() {
  local profile="${1,,}"
  case "$profile" in
    greedy) echo "1|0" ;;
    n10)    echo "10|0.7" ;;
    *) echo "ERROR: unknown profile: $profile (expected greedy|n10)" >&2; exit 2 ;;
  esac
}

_sft_run_root() {
  local ckpt_root="$1" run_id="$2"
  local container="${ckpt_root}/_runs/_sft_main_results/${run_id}"
  mkdir -p "$container"
  echo "$container"
}

_detect_policy_source() {
  local run_root="$1"

  if [[ -d "$run_root/final_hf" ]]; then
    echo "shared|$run_root/final_hf|"
    return 0
  fi

  if ls "$run_root"/actor_* >/dev/null 2>&1; then
    echo "per_role||$run_root/actor_{role}"
    return 0
  fi

  echo "shared|$run_root|"
}

_is_sane_jsonl() {
  local p="$1"
  [[ -f "$p" ]] || return 1
  [[ -s "$p" ]] || return 1
  head -n 1 "$p" >/dev/null 2>&1 || return 1
  return 0
}

_infer_alg_from_method() {
  local m="${1,,}"
  case "$m" in
    sft)   echo "none" ;;
    mappo) echo "mappo" ;;
    magrpo) echo "magrpo" ;;
    c3)    echo "c3" ;;
    *)     echo "auto" ;;
  esac
}

_ray_stop_if_needed() {
  if command -v ray >/dev/null 2>&1; then
    ray stop --force >/dev/null 2>&1 || true
  fi
}

_run_eval_one() {
  local run_id="$1" method="$2" task="$3" profile="$4" seed="$5"
  local source_type="$6" source_val="$7" out_subdir="$8" extra_eval_args="$9" alg_for_eval="${10}"

  local task_yaml
  task_yaml="$(_task_to_yaml "$task")"
  [[ -f "$task_yaml" ]] || { echo "ERROR: task yaml not found: $task_yaml" >&2; exit 1; }

  local task_tag="$task"
  if [[ "$task_tag" == /* ]]; then
    task_tag="$(basename "$task_tag")"
  fi
  if [[ "$task_tag" == *.yaml ]]; then
    task_tag="${task_tag%.yaml}"
  fi
  task_tag="${task_tag,,}"

  local alg
  alg="$alg_for_eval"
  [[ -n "$alg" && "$alg" != "auto" ]] || alg="$(_infer_alg_from_method "$method")"

  local run_root
  local hf_pretrain=""
  if [[ "$source_type" == "hf_base" ]]; then
    run_root="$(_sft_run_root "$CKPT_ROOT" "$run_id")"
    hf_pretrain="$source_val"
  else
    run_root="$source_val"
  fi

  if [[ "$run_root" != /* ]]; then
    run_root="${REPO_ROOT}/${run_root}"
  fi

  if [[ -d "$run_root" && "$(basename "$run_root")" == "final_hf" ]]; then
    run_root="$(dirname "$run_root")"
  fi

  if [[ "$source_type" == "train_run_dir" ]] && [[ ! -d "$run_root" ]]; then
    echo "[main_results] WARN: train_run_dir not found; skipping: id=$run_id task=$task profile=$profile" >&2
    echo "[main_results]       train_run_dir=$run_root" >&2
    echo "[main_results]       (Tip) Run SFT-only sweeps via: --only_methods SFT" >&2
    return 0
  fi

  local eval_dir="${run_root%/}/${out_subdir}/${task_tag}/${profile}"
  local eval_jsonl="${eval_dir}/eval_only.jsonl"
  local eval_metrics="${eval_jsonl}.metrics.jsonl"
  mkdir -p "$eval_dir"

  if [[ "$RESUME" == "1" ]] && _is_sane_jsonl "$eval_jsonl" && _is_sane_jsonl "$eval_metrics"; then
    echo "[main_results] SKIP (resume): id=$run_id task=$task profile=$profile -> $eval_dir" >&2
    return 0
  fi

  rm -f "$eval_jsonl" "$eval_metrics" || true

  local mode pretrain pattern
  if [[ "$source_type" == "hf_base" ]]; then
    mode="shared"
    pretrain="$hf_pretrain"
    pattern=""
  else
    IFS='|' read -r mode pretrain pattern <<< "$(_detect_policy_source "$run_root")"
  fi

  local n_samples temp
  IFS='|' read -r n_samples temp <<< "$(_eval_profile_params "$profile")"
  local ray_tag
  ray_tag="$(printf '%s' "${run_id}_${task_tag}_${profile}" | sha1sum | cut -c1-10)"
  local ray_tmp_short="/tmp/c3r_${ray_tag}"

  echo "[main_results] RUN id=$run_id method=$method task=$task profile=$profile seed=$seed" >&2
  echo "[main_results]   run_root=$run_root" >&2
  echo "[main_results]   eval_dir=$eval_dir" >&2
  echo "[main_results]   policy_mode=$mode" >&2
  echo "[main_results]   alg_for_eval=$alg" >&2

  c3_repro_export_common_env "${REPO_ROOT}"
  export WANDB_DISABLED=true

  local cmd=(
    "$PYTHON_BIN" -m openrlhf.cli.train_ppo_ray
    --c3_task "$task_yaml"
    --marl_algorithm "$alg"

    --eval_only
    --eval_global_step 0

    --seed "$seed"

    --actor_learning_rate 0
    --critic_learning_rate 0
    --max_norm 0

    --adam_betas 0.9 0.999
    --l2 0

    --init_kl_coef 0
    --train_batch_size 1
    --micro_train_batch_size 1
    --rollout_batch_size 1
    --micro_rollout_batch_size 1

    --vllm_num_engines "$VLLM_NUM_ENGINES"
    --vllm_tensor_parallel_size "$VLLM_TP"
    --vllm_gpu_memory_utilization "$VLLM_MEM_UTIL"
    --enable_prefix_caching

    --prompt_max_len "$PROMPT_MAX_LEN"
    --generate_max_len "$GENERATE_MAX_LEN"

    --eval_steps 1
    --eval_n_samples_per_prompt "$n_samples"
    --eval_temperature "$temp"

    --eval_dump_path "$eval_jsonl"
    --eval_dump_mode overwrite
    --run_dir "${eval_dir}/driver_run"
    --log_dir "${eval_dir}/logs"
    --ray_tmpdir "${ray_tmp_short}"

    --use_wandb 0
  )

  if [[ "$mode" == "shared" ]]; then
    cmd+=(--policy_sharing_mode shared --pretrain "$pretrain")
  else
    cmd+=(--policy_sharing_mode per_role --pretrain_by_role_pattern "$pattern")
  fi

  if [[ -n "$extra_eval_args" ]]; then
    # shellcheck disable=SC2206
    extra=( $extra_eval_args )
    cmd+=("${extra[@]}")
  fi

  (
    cd "$REPO_ROOT"
    "${cmd[@]}"
  )

  if ! _is_sane_jsonl "$eval_jsonl" || ! _is_sane_jsonl "$eval_metrics"; then
    echo "ERROR: missing or invalid eval artifacts in $eval_dir" >&2
    exit 1
  fi

  _ray_stop_if_needed
}

# -------------------------- registry parsing (sweep) --------------------------

_parse_registry_tsv() {
  local registry="$1"

  "$PYTHON_BIN" - <<'PY' "$registry" "$REPO_ROOT"
import os, sys, yaml
reg = sys.argv[1]
repo = sys.argv[2]

def norm_path(p: str) -> str:
    p = (p or "").strip()
    if not p:
        return p
    if os.path.isabs(p) or p.startswith("~"):
        return os.path.expanduser(p)
    return os.path.normpath(os.path.join(repo, p))

with open(reg, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

defaults = cfg.get("defaults") or {}
D_out_subdir = str(defaults.get("out_subdir", "main_results"))
D_profiles = defaults.get("profiles", ["greedy", "n10"]) or ["greedy", "n10"]
D_extra = str(defaults.get("extra_eval_args", "") or "")

runs = cfg.get("runs") or []
if not isinstance(runs, list):
    raise ValueError("registry.runs must be a list")

for r in runs:
    if not isinstance(r, dict):
        continue
    run_id = str(r.get("id", "")).strip()
    method = str(r.get("method", "")).strip()
    task = str(r.get("task", "")).strip()
    seed = r.get("seed", 0)
    try:
        seed = int(seed)
    except Exception:
        seed = 0

    alg_for_eval = str(r.get("alg_for_eval", "") or "").strip() or "auto"

    out_subdir = str(r.get("out_subdir", "") or "").strip() or D_out_subdir
    extra = str(r.get("extra_eval_args", "") or "").strip() or D_extra

    src = r.get("source") or {}
    stype = str(src.get("type", "")).strip()
    if stype == "hf_base":
        sval = str(src.get("hf_base", "")).strip()
    else:
        sval = str(src.get("train_run_dir", "")).strip()
        sval = norm_path(sval)

    profiles = r.get("profiles", None)
    if not profiles:
        profiles = D_profiles
    for prof in profiles:
        prof = str(prof).strip()
        print("\\t".join([
            run_id, method, task, str(seed), alg_for_eval,
            stype, sval, out_subdir, extra, prof
        ]))
PY
}

# -------------------------- command handlers --------------------------

_do_sweep() {
  local out_dir="$1"
  mkdir -p "$out_dir"

  local tsv
  tsv="$(_parse_registry_tsv "$REGISTRY")"

  while IFS=$'\t' read -r run_id method task seed alg_for_eval stype sval out_subdir extra_eval_args profile; do
    [[ -n "$run_id" ]] || continue

    if [[ -n "$ONLY_METHODS" ]]; then
      case ",${ONLY_METHODS}," in
        *",${method},"*) : ;;
        *) continue ;;
      esac
    fi

    if [[ -n "$ONLY_TASKS" ]]; then
      case ",${ONLY_TASKS}," in
        *",${task},"*) : ;;
        *) continue ;;
      esac
    fi

    if [[ -n "$ONLY_IDS" ]]; then
      case ",${ONLY_IDS}," in
        *",${run_id},"*) : ;;
        *) continue ;;
      esac
    fi

    _run_eval_one "$run_id" "$method" "$task" "$profile" "$seed" "$stype" "$sval" "$out_subdir" "$extra_eval_args" "$alg_for_eval"
  done <<< "$tsv"

  echo "[main_results] Aggregating tables to: $out_dir" >&2
  "$PYTHON_BIN" -m c3.tools.main_results aggregate \
    --registry "$REGISTRY" \
    --ckpt_root "$CKPT_ROOT" \
    --out_dir "$out_dir" \
    --strict 0
}

_do_one() {
  [[ -n "$RUN_ID" ]] || { echo "ERROR: --id required for one" >&2; exit 2; }
  [[ -n "$PROFILE" ]] || { echo "ERROR: --profile required for one" >&2; exit 2; }

  # Direct mode: run one eval without registry lookup.
  if [[ -n "$DIRECT_SOURCE_TYPE" ]]; then
    [[ -n "$DIRECT_METHOD" ]] || { echo "ERROR: --method required for direct one mode" >&2; exit 2; }
    [[ -n "$DIRECT_TASK" ]] || { echo "ERROR: --task required for direct one mode" >&2; exit 2; }
    [[ -n "$DIRECT_SEED" ]] || DIRECT_SEED="0"
    [[ -n "$DIRECT_OUT_SUBDIR" ]] || DIRECT_OUT_SUBDIR="main_results"

    local source_val=""
    case "$DIRECT_SOURCE_TYPE" in
      hf_base)
        [[ -n "$DIRECT_HF_BASE" ]] || { echo "ERROR: --hf_base required when --source_type=hf_base" >&2; exit 2; }
        source_val="$DIRECT_HF_BASE"
        ;;
      train_run_dir)
        [[ -n "$DIRECT_TRAIN_RUN_DIR" ]] || { echo "ERROR: --train_run_dir required when --source_type=train_run_dir" >&2; exit 2; }
        if [[ "$DIRECT_TRAIN_RUN_DIR" == /* ]]; then
          source_val="$DIRECT_TRAIN_RUN_DIR"
        else
          source_val="${REPO_ROOT}/${DIRECT_TRAIN_RUN_DIR}"
        fi
        ;;
      *)
        echo "ERROR: unknown --source_type: $DIRECT_SOURCE_TYPE (expected hf_base|train_run_dir)" >&2
        exit 2
        ;;
    esac

    _run_eval_one \
      "$RUN_ID" \
      "$DIRECT_METHOD" \
      "$DIRECT_TASK" \
      "$PROFILE" \
      "$DIRECT_SEED" \
      "$DIRECT_SOURCE_TYPE" \
      "$source_val" \
      "$DIRECT_OUT_SUBDIR" \
      "$DIRECT_EXTRA_EVAL_ARGS" \
      "${DIRECT_ALG_FOR_EVAL:-auto}"
    return 0
  fi

  local tsv
  tsv="$(_parse_registry_tsv "$REGISTRY")"

  while IFS=$'\t' read -r run_id method task seed alg_for_eval stype sval out_subdir extra_eval_args profile; do
    [[ "$run_id" == "$RUN_ID" ]] || continue
    [[ "$profile" == "$PROFILE" ]] || continue
    _run_eval_one "$run_id" "$method" "$task" "$profile" "$seed" "$stype" "$sval" "$out_subdir" "$extra_eval_args" "$alg_for_eval"
    exit 0
  done <<< "$tsv"

  echo "ERROR: run_id/profile not found in registry: id=$RUN_ID profile=$PROFILE" >&2
  exit 1
}

CMD="${1:-}"
shift || true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --registry) REGISTRY="${2:-}"; shift 2 ;;
    --ckpt_root) CKPT_ROOT="${2:-}"; shift 2 ;;
    --out_dir) OUT_DIR="${2:-}"; shift 2 ;;
    --only_methods) ONLY_METHODS="${2:-}"; shift 2 ;;
    --only_tasks) ONLY_TASKS="${2:-}"; shift 2 ;;
    --only_ids) ONLY_IDS="${2:-}"; shift 2 ;;
    --profile) PROFILE="${2:-}"; shift 2 ;;
    --id) RUN_ID="${2:-}"; shift 2 ;;
    --resume) RESUME="${2:-0}"; shift 2 ;;
    --method) DIRECT_METHOD="${2:-}"; shift 2 ;;
    --task) DIRECT_TASK="${2:-}"; shift 2 ;;
    --source_type) DIRECT_SOURCE_TYPE="${2:-}"; shift 2 ;;
    --hf_base) DIRECT_HF_BASE="${2:-}"; shift 2 ;;
    --train_run_dir) DIRECT_TRAIN_RUN_DIR="${2:-}"; shift 2 ;;
    --seed) DIRECT_SEED="${2:-}"; shift 2 ;;
    --out_subdir) DIRECT_OUT_SUBDIR="${2:-}"; shift 2 ;;
    --extra_eval_args) DIRECT_EXTRA_EVAL_ARGS="${2:-}"; shift 2 ;;
    --alg_for_eval) DIRECT_ALG_FOR_EVAL="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="${CKPT_ROOT}/paper_main_results"
fi

case "$CMD" in
  sweep) _do_sweep "$OUT_DIR" ;;
  one)   _do_one ;;
  *) usage; exit 2 ;;
esac
