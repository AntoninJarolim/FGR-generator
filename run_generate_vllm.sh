#!/bin/bash
# Generate fine-grained relevance supervision with NATIVE vLLM offline batched
# inference. No server and no OPENAI_BASE_URL needed -- this loads the model
# weights in-process, so it must run on a machine with a GPU.
#
# Uses constrained (guided-JSON) decoding + greedy sampling, skips the
# span-not-found regeneration loop, and loops over the models below.
set -euo pipefail

# long-embed base -> loads templates/long-embed-system.template + long-embed-user.template
TEMPLATE_FILE="templates/long-embed.template"

# HuggingFace model ids to loop over.
# NOTE: verify these ids resolve to real HF repos before running.
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
        --model_name "$MODEL_NAME" \
        --generation_client vllm \
        --batch_size 128 \
        --vllm_gpu_memory_utilization 0.9 \
        --vllm_guided_json \
        --skip-regeneration
done
