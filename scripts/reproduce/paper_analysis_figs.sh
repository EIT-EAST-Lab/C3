#!/usr/bin/env bash
# scripts/reproduce/paper_analysis_figs.sh
set -euo pipefail

# Paper analysis/diagnostics figure pipeline.
#
# This script is the "single entry" replacement for the old examples/c3/analysis/* bash files.
# It runs sequentially by default (predictable + minimal), but you can parallelize by
# running separate processes with different splits/variants.

_usage() {
  cat <<'USAGE'
Usage:
  # Fig2 (paper): mechanism diagnostics (C3 / MAPPO / MAGRPO / SFT)
  bash scripts/reproduce/paper_analysis_figs.sh fig2 \
    --suite math \
    --out_dir artifacts/fig2_mechanism \
    --run_c3 ckpt/_runs/<C3_run_dir> \
    --run_mappo ckpt/_runs/<MAPPO_run_dir> \
    --run_magrpo ckpt/_runs/<MAGRPO_run_dir> \
    --run_sft ckpt/_runs/_sft_main_results/<SFT_dir> \
    --mappo_critic_ckpt <PATH_TO_MAPPO_CRITIC>

Subcommands:
  fig2                    Run + aggregate + plot mechanism diagnostics (paper Fig2).

Common options:
  --suite math|code       Which suite/task YAML to use (default: math)
  --task_yaml PATH        Override task YAML (default: configs/tasks/<suite>.yaml)
  --analysis_yaml PATH    Optional analysis config (default: configs/analysis.yaml if exists)
  --analysis_seed INT     Seed used for bucket generation (default: 0)
  --splits LIST           Comma/space split list (default: suite defaults)
  --engine auto|hf|vllm   Policy engine for build-buckets (default: vllm)
  --tp INT                vLLM tensor parallel size for build-buckets (default: 1)
  --python PY             Python executable (default: python)
  --resume 0|1            Skip if metrics JSON already exist (default: 1)

Fig2-specific (paper mechanism):
  --run_c3 DIR
  --run_mappo DIR
  --run_magrpo DIR
  --run_sft DIR
  --mappo_critic_ckpt PATH        Required for MAPPO credit (mode=mappo_v)
  --mappo_critic_base_ckpt PATH   Optional, recommended if critic arch != policy arch
  --calibration_csv PATH          Optional Fig2 calibration bins csv for inset
  --calibration_methods LIST      Optional method list for inset (default: C3,MAGRPO)

Outputs:
  fig2:
    <out_dir>/analysis_results.summary.json
    <out_dir>/fig2_mechanism*.{pdf,png}
USAGE
}

# -------------------------- repo + defaults --------------------------

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
source "${SCRIPT_DIR}/common_env.sh"

PYTHON_BIN="python"
RESUME=1

ANALYSIS_MOD="${ANALYSIS_MOD:-c3.analysis.c3_analysis}"
ANALYSIS_RESULTS_MOD="${ANALYSIS_RESULTS_MOD:-c3.tools.analysis_results}"
PLOT_MOD="${PLOT_MOD:-c3.tools.plot_paper_figures}"

SUITE="math"
TASK_YAML=""
ANALYSIS_YAML=""
ANALYSIS_SEED=0
ENGINE="vllm"
TP=1
SPLITS=""

# fig2 runs
RUN_C3=""; RUN_MAPPO=""; RUN_MAGRPO=""; RUN_SFT=""
MAPPO_CRITIC_CKPT=""; MAPPO_CRITIC_BASE_CKPT=""

CALIBRATION_CSV=""
CALIBRATION_METHODS="C3,MAGRPO"

_split_list() {
  local s="${1:-}"
  s="${s//,/ }"
  echo "$s" | xargs || true
}

_task_to_yaml() {
  local suite="$1"
  echo "${REPO_ROOT}/configs/tasks/${suite}.yaml"
}

_default_splits_for_suite() {
  local suite="$1"
  case "${suite,,}" in
    math) echo "MATH500 CMATH-test GSM8K-test" ;;
    code) echo "MBPP+ MBPP-test" ;;
    *) echo "" ;;
  esac
}

_maybe_set_default_analysis_yaml() {
  local p="${REPO_ROOT}/configs/analysis.yaml"
  if [[ -z "$ANALYSIS_YAML" && -f "$p" ]]; then
    ANALYSIS_YAML="$p"
  fi
}

_policy_ckpt_for_run() {
  local run_root="$1"
  if [[ -d "$run_root/final_hf" ]]; then
    echo "$run_root/final_hf"
  elif [[ "$(basename "$run_root")" == "final_hf" ]]; then
    echo "$run_root"
  else
    echo "$run_root"
  fi
}

_analysis_seed_dir() {
  local run_root="$1"
  echo "${run_root%/}/analysis/seed${ANALYSIS_SEED}"
}

_is_sane_json() {
  local f="$1"
  [[ -s "$f" ]] || return 1
  if grep -qE '(^|[^A-Za-z0-9_])NaN([^A-Za-z0-9_]|$)' "$f"; then
    return 1
  fi
  return 0
}

_run_py() {
  (
    cd "$REPO_ROOT"
    c3_repro_export_common_env "${REPO_ROOT}"
    export PYTHONUNBUFFERED=1
    "$PYTHON_BIN" "$@"
  )
}

# -------------------------- core: credit + influence --------------------------

_compute_credit_influence() {
  local run_root="$1" label="$2" split="$3" method="$4"
  # method: c3 | magrpo | mappo | sft | noreplay | noloo

  local seed_dir buckets_dir metrics_dir
  seed_dir="$(_analysis_seed_dir "$run_root")"
  buckets_dir="${seed_dir}/buckets"
  metrics_dir="${seed_dir}/metrics"
  mkdir -p "$buckets_dir" "$metrics_dir"

  local credit_out="${metrics_dir}/credit_${label}_${split}.json"
  local infl_out="${metrics_dir}/influence_${label}_${split}.json"

  if [[ "$RESUME" == "1" ]] && _is_sane_json "$credit_out" && _is_sane_json "$infl_out"; then
    echo "[analysis] SKIP (resume): ${label} split=${split} -> ${seed_dir}" >&2
    return 0
  fi

  rm -f "$credit_out" "$infl_out" || true

  local policy_ckpt
  policy_ckpt="$(_policy_ckpt_for_run "$run_root")"

  local actor_buckets="${buckets_dir}/buckets_${label}_${split}.actor.jsonl"
  local reasoner_buckets="${buckets_dir}/buckets_${label}_${split}.reasoner.jsonl"

  # Build actor buckets (for credit)
  cmd=( -m "$ANALYSIS_MOD" build-buckets
        --task "$TASK_YAML"
        --split "$split"
        --policy_ckpt "$policy_ckpt"
        --method "$label"
        --target_role actor
        --seed "$ANALYSIS_SEED"
        --engine "$ENGINE"
        --tp "$TP"
        --out "$actor_buckets" )
  if [[ -n "$ANALYSIS_YAML" ]]; then cmd+=( --analysis_yaml "$ANALYSIS_YAML" ); fi
  if [[ "$RESUME" != "1" ]]; then cmd+=( --overwrite ); fi
  echo "[analysis] build-buckets(actor): ${label} ${split}" >&2
  _run_py "${cmd[@]}"

  # Credit
  local credit_mode
  case "${method,,}" in
    magrpo) credit_mode="magrpo_mean" ;;
    mappo)  credit_mode="mappo_v" ;;
    *)      credit_mode="c3_loo" ;;
  esac

  cmd=( -m "$ANALYSIS_MOD" credit
        --bucket "$actor_buckets"
        --mode "$credit_mode"
        --out "$credit_out" )

  if [[ "$credit_mode" == "mappo_v" ]]; then
    [[ -n "$MAPPO_CRITIC_CKPT" ]] || { echo "ERROR: --mappo_critic_ckpt required for MAPPO credit" >&2; exit 1; }
    cmd+=( --mappo_critic_ckpt "$MAPPO_CRITIC_CKPT" )
    # When critic ckpt is DeepSpeed, c3_analysis may need a policy base HF dir for tokenizer/model skeleton.
    cmd+=( --policy_ckpt "$policy_ckpt" )
    if [[ -n "$MAPPO_CRITIC_BASE_CKPT" ]]; then
      cmd+=( --mappo_critic_base_ckpt "$MAPPO_CRITIC_BASE_CKPT" )
    fi
  fi

  echo "[analysis] credit(${credit_mode}): ${label} ${split}" >&2
  _run_py "${cmd[@]}"

  # Build reasoner buckets (for influence): record next teammate action
  cmd=( -m "$ANALYSIS_MOD" build-buckets
        --task "$TASK_YAML"
        --split "$split"
        --policy_ckpt "$policy_ckpt"
        --method "$label"
        --target_role reasoner
        --next_role actor
        --record_next_teammate
        --seed "$ANALYSIS_SEED"
        --engine "$ENGINE"
        --tp "$TP"
        --out "$reasoner_buckets" )
  if [[ -n "$ANALYSIS_YAML" ]]; then cmd+=( --analysis_yaml "$ANALYSIS_YAML" ); fi
  if [[ "$RESUME" != "1" ]]; then cmd+=( --overwrite ); fi

  echo "[analysis] build-buckets(reasoner): ${label} ${split}" >&2
  _run_py "${cmd[@]}"

  # Influence
  cmd=( -m "$ANALYSIS_MOD" influence
        --bucket "$reasoner_buckets"
        --out "$infl_out" )

  echo "[analysis] influence: ${label} ${split}" >&2
  _run_py "${cmd[@]}"

  if ! _is_sane_json "$credit_out" || ! _is_sane_json "$infl_out"; then
    echo "ERROR: missing or invalid analysis metrics for ${label} ${split} under ${seed_dir}" >&2
    exit 1
  fi
}

# -------------------------- fig2 (paper mechanism) pipeline --------------------------

_run_fig2_mechanism() {
  local out_dir="$1"
  mkdir -p "$out_dir"

  for split in $(_split_list "$SPLITS"); do
    _compute_credit_influence "$RUN_C3"     "C3"     "$split" "c3"
    _compute_credit_influence "$RUN_MAPPO"  "MAPPO"  "$split" "mappo"
    _compute_credit_influence "$RUN_MAGRPO" "MAGRPO" "$split" "magrpo"
    _compute_credit_influence "$RUN_SFT"    "SFT"    "$split" "sft"
  done

  echo "[fig2] aggregate -> $out_dir" >&2
  _run_py -m "$ANALYSIS_RESULTS_MOD" aggregate \
    --analysis_root "${RUN_C3%/}/analysis" \
    --analysis_root "${RUN_MAPPO%/}/analysis" \
    --analysis_root "${RUN_MAGRPO%/}/analysis" \
    --analysis_root "${RUN_SFT%/}/analysis" \
    --out_dir "$out_dir" \
    --suite "$SUITE"

  local summary_json="$out_dir/analysis_results.summary.json"
  [[ -f "$summary_json" ]] || { echo "ERROR: missing summary json: $summary_json" >&2; exit 1; }

  echo "[fig2] plot -> $out_dir" >&2
  local mech_cmd=( -m "$PLOT_MOD" mechanism
    --out_dir "$out_dir"
    --summary_json "$summary_json"
    --calibration_methods "$CALIBRATION_METHODS" )
  if [[ -n "$CALIBRATION_CSV" ]]; then
    mech_cmd+=( --calibration_csv "$CALIBRATION_CSV" )
  fi
  _run_py "${mech_cmd[@]}"

  echo "[fig2] OK: $out_dir" >&2
}

# -------------------------- parse args --------------------------

if [[ $# -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  _usage
  exit 0
fi
SUB="$1"; shift

OUT_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) PYTHON_BIN="${2:-}"; shift 2 ;;
    --resume) RESUME="${2:-}"; shift 2 ;;

    --suite) SUITE="${2:-}"; shift 2 ;;
    --task_yaml) TASK_YAML="${2:-}"; shift 2 ;;
    --analysis_yaml) ANALYSIS_YAML="${2:-}"; shift 2 ;;
    --analysis_seed) ANALYSIS_SEED="${2:-}"; shift 2 ;;
    --splits) SPLITS="${2:-}"; shift 2 ;;
    --engine) ENGINE="${2:-}"; shift 2 ;;
    --tp) TP="${2:-}"; shift 2 ;;

    --out_dir) OUT_DIR="${2:-}"; shift 2 ;;

    --run_c3) RUN_C3="${2:-}"; shift 2 ;;
    --run_mappo) RUN_MAPPO="${2:-}"; shift 2 ;;
    --run_magrpo) RUN_MAGRPO="${2:-}"; shift 2 ;;
    --run_sft) RUN_SFT="${2:-}"; shift 2 ;;
    --mappo_critic_ckpt) MAPPO_CRITIC_CKPT="${2:-}"; shift 2 ;;
    --mappo_critic_base_ckpt) MAPPO_CRITIC_BASE_CKPT="${2:-}"; shift 2 ;;

    --calibration_csv) CALIBRATION_CSV="${2:-}"; shift 2 ;;
    --calibration_methods) CALIBRATION_METHODS="${2:-}"; shift 2 ;;

    -h|--help) _usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; _usage; exit 2 ;;
  esac
done

OUT_DIR="${OUT_DIR:-${REPO_ROOT}/artifacts/${SUB}}"

if [[ -z "$TASK_YAML" ]]; then
  TASK_YAML="$(_task_to_yaml "$SUITE")"
fi
if [[ ! -f "$TASK_YAML" ]]; then
  echo "ERROR: task_yaml not found: $TASK_YAML" >&2
  exit 1
fi

_maybe_set_default_analysis_yaml

if [[ -z "$SPLITS" ]]; then
  SPLITS="$(_default_splits_for_suite "$SUITE")"
fi

case "$SUB" in
  fig2)
    [[ -n "$RUN_C3" && -n "$RUN_MAPPO" && -n "$RUN_MAGRPO" && -n "$RUN_SFT" ]] || {
      echo "ERROR: fig2 requires --run_c3 --run_mappo --run_magrpo --run_sft" >&2
      exit 2
    }
    _run_fig2_mechanism "$OUT_DIR"
    ;;

  *)
    echo "Unknown subcommand: $SUB" >&2
    _usage
    exit 2
    ;;
esac
