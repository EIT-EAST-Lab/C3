#!/usr/bin/env bash
# Common reproducibility environment settings for reproduce scripts.

c3_repro_export_common_env() {
  local repo_root="$1"
  export PYTHONPATH="${repo_root}:${PYTHONPATH:-}"
  export TOKENIZERS_PARALLELISM=false

  # Keep reproduce scripts independent from local TF/Keras integration variants.
  export TRANSFORMERS_NO_TF=1
  export USE_TF=0
  export USE_FLAX=0
}
