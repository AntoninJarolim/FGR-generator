#!/bin/zsh
# Generate fine-grained relevance supervision with NATIVE vLLM offline batched
# inference. No server and no OPENAI_BASE_URL needed -- this loads the model
# weights in-process, so it must run on a machine with a GPU.
#
# Uses constrained (guided-JSON) decoding + greedy sampling, skips the
# span-not-found regeneration loop, and loops over the models below.

# Usage:
#   ./run_generate_vllm.sh                  # full run
#   ./run_generate_vllm.sh --dry-run        # only the first 10 samples, overwrite existing
#   ./run_generate_vllm.sh --force_rewrite  # any other arg is passed through to llm_extraction.py
set -euo pipefail

# --- Argument parsing ---
# --dry-run is consumed here; everything else is forwarded to llm_extraction.py.
DRY_RUN=0
EXTRA_ARGS=()
for arg in "$@"; do
  case $arg in
    --dry-run) DRY_RUN=1 ;;
    *) EXTRA_ARGS+=("$arg") ;;
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

# HuggingFace model ids to loop over. Only models that fit in the local
# 2x RTX 3090 (48 GB total) budget are enabled here.
MODELS=(
    "google/gemma-4-12B-it"
    "google/gemma-4-E4B-it"
    "mistralai/Ministral-3-14B-Instruct-2512"
)

# Larger models that do NOT fit in 48 GB of VRAM (need >48 GB in bf16). Kept
# here for a future run on bigger hardware; not looped over below.
LARGE_MODELS=(
    "google/gemma-4-31B-it"
    "google/gemma-4-26B-A4B-it"
    "Qwen/Qwen3.6-27B"
)

# On big-GPU machines (e.g. MetaCentrum DGX via dgx_run_generate_vllm.sh) run
# the large models instead of the local-VRAM-sized ones.
if [ "${RUN_LARGE_MODELS:-0}" -eq 1 ]; then
  echo "RUN_LARGE_MODELS=1: using the LARGE_MODELS list."
  MODELS=("${LARGE_MODELS[@]}")
fi

# This script runs non-interactively, so ~/.zshrc is NOT sourced and the `conda`
# shell function is never defined -- bare `conda activate` would fail with a
# "run conda init" error. Source conda's profile script to load the function,
# deriving the base from the on-PATH conda binary so no path is hardcoded.
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate fgr-generator

# `conda activate` here sets CONDA_PREFIX correctly but does NOT always prepend
# the env's bin to PATH, so a bare `python` can resolve to the base env's
# interpreter (which lacks this project's deps -> "ModuleNotFoundError: jsonlines").
# Whether it broke depended on the inherited PATH, hence the intermittent crash.
# Prepend the env bin, and below invoke the interpreter by absolute path so the
# right Python is used regardless of PATH ordering, aliases, or command hashing.
export PATH="$CONDA_PREFIX/bin:$PATH"
PYTHON="$CONDA_PREFIX/bin/python"
echo $(which pip)

# vLLM's FlashInfer sampler JIT-compiles a CUDA kernel at first use and needs
# nvcc. There is no system CUDA toolkit (/usr/local/cuda is absent), but the
# pip `nvidia-cuda-nvcc` package ships a full toolkit under site-packages. Point
# CUDA_HOME at it (and add its bin to PATH) so FlashInfer can find nvcc.
CUDA_HOME="$(echo "$CONDA_PREFIX"/lib/python*/site-packages/nvidia/cu13)"
if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
  echo "ERROR: nvcc not found under $CUDA_HOME/bin (nvidia-cuda-nvcc missing?)" >&2
  exit 1
fi
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
echo "Using CUDA_HOME=$CUDA_HOME"

# The pip nvcc (13.2) is newer than FlashInfer 0.6.12's bundled CCCL headers, so
# JIT-compiling FlashInfer's sampling kernel fails with "CUDA compiler and CUDA
# toolkit headers are incompatible". We decode greedily (guided JSON), so the
# FlashInfer sampler adds nothing -- disable it and use vLLM's native Torch
# sampler, which needs no runtime compilation.
export VLLM_USE_FLASHINFER_SAMPLER=0

for MODEL_NAME in "${MODELS[@]}"; do
    echo "Running llm_extraction.py (vLLM offline) with model: $MODEL_NAME"
    "$PYTHON" llm_extraction.py \
        --input_data_name 'dwzhu/LongEmbed' \
        --template_file "$TEMPLATE_FILE" \
        --psg_key passage \
        --model_name "$MODEL_NAME" \
        --generation_client vllm \
        --batch_size 128 \
        --vllm_max_model_len 65536 \
        --max_gen_tokens 8192 \
        --vllm_gpu_memory_utilization 0.9 \
        --vllm_guided_json \
        --skip-regeneration \
        "${DRY_RUN_ARGS[@]}" \
        "${EXTRA_ARGS[@]}"
done
