#!/bin/zsh
# Generate fine-grained relevance supervision with NATIVE vLLM offline batched
# inference. No server and no OPENAI_BASE_URL needed -- this loads the model
# weights in-process, so it must run on a machine with a GPU.
#
# Emits the XML/delimiter span format (templates/long-embed-xml). By default it
# runs UNCONSTRAINED (free decoding, delimiter parsing only); pass --constrained
# to add per-document span-grammar decoding, which GUARANTEES every generated
# span is a verbatim contiguous substring of its own document. Greedy sampling,
# skips the span-not-found regeneration loop, loops over the models below.

# Usage:
#   ./run_generate_vllm.sh                  # full run, unconstrained
#   ./run_generate_vllm.sh --constrained    # span-grammar constrained decoding
#   ./run_generate_vllm.sh --constrained --reasoning
#                                           # reasoning models think first, grammar
#                                           # engages after </think>; writes to a
#                                           # separate "-reasoning" dir
#   ./run_generate_vllm.sh --dry-run        # only the first 10 samples, overwrite existing
#   ./run_generate_vllm.sh --json           # JSON format instead of XML (unconstrained only)
#   ./run_generate_vllm.sh --model google/gemma-4-31B-it   # just this model, not a whole list
#   ./run_generate_vllm.sh --model A --model B             # repeatable
#   ./run_generate_vllm.sh --force_rewrite  # any other arg is passed through to llm_extraction.py
set -euo pipefail

# --- Argument parsing ---
# --dry-run, --json and --constrained are consumed here; everything else is
# forwarded to llm_extraction.py.
DRY_RUN=0
CONSTRAINED=0
REASONING=0
JSON_FORMAT=0
PICKED_MODELS=()
EXTRA_ARGS=()
while [ $# -gt 0 ]; do
  case $1 in
    --dry-run) DRY_RUN=1 ;;
    --constrained) CONSTRAINED=1 ;;
    --reasoning) REASONING=1 ;;
    --json) JSON_FORMAT=1 ;;
    --model) shift; PICKED_MODELS+=("$1") ;;
    --model=*) PICKED_MODELS+=("${1#--model=}") ;;
    *) EXTRA_ARGS+=("$1") ;;
  esac
  shift
done

# In dry-run mode only generate a handful of samples and overwrite prior output,
# so a crash surfaces fast without a full pass over the dataset.
DRY_RUN_ARGS=()
if [ "$DRY_RUN" -eq 1 ]; then
  echo "DRY RUN: limiting to 10 samples per model."
  DRY_RUN_ARGS=(--to_sample 10 --force_rewrite)
fi

# The template fixes only the OUTPUT FORMAT (long-embed-xml -> delimiter-wrapped
# <spans>/<s> output); it never changes with the decoding mode. The CONSTRAINT
# is orthogonal and set by the single --constrained flag, passed straight to
# llm_extraction.py, which both triggers the span-grammar machinery and routes
# output to the "-constrained" dir:
#   * default (unconstrained): free decoding -> data/extracted_relevancy/long-embed-xml/
#   * --constrained: per-document substring grammar, every span guaranteed
#     verbatim -> data/extracted_relevancy/long-embed-xml-constrained/
TEMPLATE_FILE="templates/long-embed-xml.template"
if [ "$JSON_FORMAT" -eq 1 ]; then
  # The span grammar is a substring grammar over the XML span format; there is
  # deliberately no JSON-constrained mode (see llm_extraction.py's mode_dir
  # comment), so refuse the combination rather than silently creating a
  # long-embed-json-constrained dir nothing else in eval/ knows about.
  if [ "$CONSTRAINED" -eq 1 ]; then
    echo "ERROR: --json cannot be combined with --constrained; the span grammar" >&2
    echo "       applies to the XML span format only." >&2
    exit 1
  fi
  TEMPLATE_FILE="templates/long-embed-json.template"
  echo "JSON format: writing to data/extracted_relevancy/long-embed-json/."
fi

MODE_ARGS=()
if [ "$CONSTRAINED" -eq 1 ]; then
  echo "CONSTRAINED: per-document span-grammar decoding (verbatim spans)."
  MODE_ARGS=(--constrained)
else
  echo "UNCONSTRAINED (default): free decoding."
fi

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
    "Qwen/Qwen3.6-27B"
)

# Frontier-scale MoE models; need a multi-GPU node (e.g. several 80 GB+ GPUs
# with tensor parallelism) even at bf16/fp8.
EXTREME_MODELS=(
    "Qwen/Qwen3-235B-A22B-Thinking-2507"
)

# model -> vLLM reasoning parser. Needed per-model because vLLM auto-detects
# none, and a wrong name disables the span grammar. Models absent from the map
# run without a parser -- correct for non-reasoning ones (Ministral-3).
typeset -A REASONING_PARSERS
REASONING_PARSERS=(
    "Qwen/Qwen3.6-27B"                  qwen3
    "Qwen/Qwen3-235B-A22B-Thinking-2507" qwen3
    "google/gemma-4-12B-it"             gemma4
    "google/gemma-4-31B-it"             gemma4
    "google/gemma-4-E4B-it"             gemma4
    "google/gemma-4-26B-A4B-it"         gemma4
)
# gemma-4 only: its template pre-closes an empty thought channel, so the parser
# alone would leave it unable to reason.
typeset -A NEEDS_ENABLE_THINKING
NEEDS_ENABLE_THINKING=(
    "google/gemma-4-12B-it"             1
    "google/gemma-4-31B-it"             1
    "google/gemma-4-E4B-it"             1
    "google/gemma-4-26B-A4B-it"         1
)

# CoT/answer token split when --reasoning is on. Sized from the unconstrained
# Qwen3.6-27B run (p90 CoT ~4K tokens, max ~8.7K); answers are ~100 tokens.
MAX_GEN_TOKENS=8192
THINKING_TOKEN_BUDGET=12288
if [ "$REASONING" -eq 1 ]; then
  MAX_GEN_TOKENS=16384
else
  THINKING_TOKEN_BUDGET=""
fi

# On big-GPU machines (e.g. MetaCentrum DGX via dgx_run_generate_vllm.sh) run
# the large or extreme models instead of the local-VRAM-sized ones.
if [ ${#PICKED_MODELS[@]} -gt 0 ]; then
  # One or more --model flags: run exactly those, ignoring the lists above. This
  # is how a Slurm job takes a single model, so the models run concurrently in
  # separate allocations instead of sequentially in one.
  echo "--model given: running ${#PICKED_MODELS[@]} explicitly named model(s)."
  MODELS=("${PICKED_MODELS[@]}")
elif [ "${RUN_EXTREME_MODELS:-0}" -eq 1 ]; then
  echo "RUN_EXTREME_MODELS=1: using the EXTREME_MODELS list."
  MODELS=("${EXTREME_MODELS[@]}")
elif [ "${RUN_LARGE_MODELS:-0}" -eq 1 ]; then
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
# toolkit headers are incompatible". We decode greedily (temperature=0), so the
# FlashInfer sampler adds nothing -- disable it and use vLLM's native Torch
# sampler, which needs no runtime compilation.
export VLLM_USE_FLASHINFER_SAMPLER=0

for MODEL_NAME in "${MODELS[@]}"; do
    REASONING_ARGS=()
    if [ "$REASONING" -eq 1 ]; then
      PARSER="${REASONING_PARSERS[$MODEL_NAME]:-}"
      if [ -n "$PARSER" ]; then
        REASONING_ARGS=(--reasoning_parser "$PARSER"
                        --thinking_token_budget "$THINKING_TOKEN_BUDGET")
        if [ -n "${NEEDS_ENABLE_THINKING[$MODEL_NAME]:-}" ]; then
          REASONING_ARGS+=(--enable_thinking)
        fi
        echo "REASONING: $MODEL_NAME -> parser=$PARSER, budget=$THINKING_TOKEN_BUDGET."
      else
        echo "REASONING: $MODEL_NAME has no parser mapping -- running without reasoning."
      fi
    fi

    echo "Running llm_extraction.py (vLLM offline) with model: $MODEL_NAME"
    "$PYTHON" llm_extraction.py \
        --input_data_name 'dwzhu/LongEmbed' \
        --template_file "$TEMPLATE_FILE" \
        --psg_key passage \
        --model_name "$MODEL_NAME" \
        --generation_client vllm \
        --batch_size 128 \
        --vllm_max_model_len 65536 \
        --max_gen_tokens "$MAX_GEN_TOKENS" \
        --vllm_gpu_memory_utilization 0.9 \
        "${MODE_ARGS[@]}" \
        "${REASONING_ARGS[@]}" \
        --skip-regeneration \
        "${DRY_RUN_ARGS[@]}" \
        "${EXTRA_ARGS[@]}"
done
