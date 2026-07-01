#!/bin/bash
# Generate fine-grained relevance supervision with NATIVE vLLM offline batched
# inference. No server and no OPENAI_BASE_URL needed -- this loads the model
# weights in-process, so it must run on a machine with a GPU.
#
# Uses constrained (guided-JSON) decoding + greedy sampling, skips the
# span-not-found regeneration loop, and loops over the models below.

# Usage:
#   ./run_generate_vllm.sh            # full run
#   ./run_generate_vllm.sh --dry-run  # only the first 10 samples, overwrite existing
set -euo pipefail

# --- Argument parsing ---
DRY_RUN=0
for arg in "$@"; do
  case $arg in
    --dry-run) DRY_RUN=1 ;;
    *) echo "Unknown argument: $arg" >&2; exit 1 ;;
  esac
done

# In dry-run mode only generate a handful of samples and overwrite prior output,
# so a crash surfaces fast without a full pass over the dataset.
DRY_RUN_ARGS=()
if [ "$DRY_RUN" -eq 1 ]; then
  echo "DRY RUN: limiting to 10 samples per model."
  DRY_RUN_ARGS=(--to_sample 10 --force_rewrite)
fi

# long-embed base -> loads templates/long-embed-system.template + long-embed-user.template
TEMPLATE_FILE="templates/long-embed.template"

# HuggingFace model ids to loop over.
MODELS=(
    "google/gemma-4-31B-it"
    "google/gemma-4-26B-A4B-it"
    "google/gemma-4-12B-it"
    "google/gemma-4-E4B-it"
    "mistralai/Ministral-3-14B-Instruct-2512"
    "Qwen/Qwen3.6-27B"
)

for MODEL_NAME in "${MODELS[@]}"; do
    echo "Running llm_extraction.py (vLLM offline) with model: $MODEL_NAME"
    python llm_extraction.py \
        --input_data_name 'dwzhu/LongEmbed' \
        --template_file "$TEMPLATE_FILE" \
        --psg_key passage \
        --model_name "$MODEL_NAME" \
        --generation_client vllm \
        --batch_size 128 \
        --vllm_max_model_len 32768 \
        --max_gen_tokens 8192 \
        --vllm_gpu_memory_utilization 0.9 \
        --vllm_guided_json \
        --skip-regeneration \
        "${DRY_RUN_ARGS[@]}"
done
