#!/bin/bash
# Generate fine-grained relevance supervision through the e-INFRA CZ
# OpenAI-compatible API. No GPU needed -- this only sends API requests, but
# the fgr-generator conda env is still activated so python has the repo deps.
#
# Mirrors run_generate_vllm.sh (same dataset, template, psg key, batch size,
# --skip-regeneration) but drops the vLLM engine params, which are meaningless
# for an API client. The span-not-found regeneration loop is skipped: samples
# with unmatched spans are only annotated with extraction_error, not re-queried.
#
# Usage:
#   ./run_generate_einfra.sh                  # full run
#   ./run_generate_einfra.sh --dry-run        # only the first 10 samples, overwrite existing
#   ./run_generate_einfra.sh --force_rewrite  # any other arg is passed through to llm_extraction.py
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

DRY_RUN_ARGS=()
if [ "$DRY_RUN" -eq 1 ]; then
  echo "DRY RUN: limiting to 10 samples per model."
  DRY_RUN_ARGS=(--to_sample 10 --force_rewrite)
fi

# long-embed-json base -> loads templates/long-embed-json-system.template +
# long-embed-json-user.template (JSON-object span output). This API path decodes
# freely (no guided JSON), so outputs land in data/extracted_relevancy/long-embed-json/.
TEMPLATE_FILE="templates/long-embed-json.template"

# Models served by the e-INFRA CZ API.
EINFRA_MODELS=(
    "glm-5.2"
)

# This script runs non-interactively, so the shell rc is NOT sourced and the
# `conda` shell function is never defined -- bare `conda activate` would fail
# with a "run conda init" error. Source conda's profile script to load the
# function, deriving the base from the on-PATH conda binary so no path is
# hardcoded.
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate fgr-generator

# `conda activate` here sets CONDA_PREFIX correctly but does NOT always prepend
# the env's bin to PATH, so a bare `python` can resolve to the base env's
# interpreter (which lacks this project's deps -> "ModuleNotFoundError").
# Prepend the env bin, and below invoke the interpreter by absolute path so the
# right Python is used regardless of PATH ordering, aliases, or command hashing.
export PATH="$CONDA_PREFIX/bin:$PATH"
PYTHON="$CONDA_PREFIX/bin/python"

# --- e-INFRA environment ---
# llm_extraction.py builds its client from env vars: OPENAI_BASE_URL (the
# API endpoint, '/v1' is appended in code) and E_INFRA_API_TOKEN (the key),
# so both must be exported before python starts. `set -a` marks everything
# sourced from the env file for export.
ENV_FILE=".env_einfra"
if [ ! -f "$ENV_FILE" ]; then
  echo "ERROR: $ENV_FILE not found!" >&2
  exit 1
fi
set -a
source "$ENV_FILE"
set +a

if [ -z "${OPENAI_BASE_URL:-}" ]; then
  echo "ERROR: OPENAI_BASE_URL not set in $ENV_FILE" >&2
  exit 1
fi
if [ -z "${E_INFRA_API_TOKEN:-}" ]; then
  echo "WARNING: E_INFRA_API_TOKEN not set ($ENV_FILE or shell); requests may be unauthorized." >&2
fi
echo "Using e-INFRA endpoint: $OPENAI_BASE_URL"

for MODEL_NAME in "${EINFRA_MODELS[@]}"; do
    echo "Running llm_extraction.py (e-INFRA API) with model: $MODEL_NAME"
    "$PYTHON" llm_extraction.py \
        --input_data_name 'dwzhu/LongEmbed' \
        --template_file "$TEMPLATE_FILE" \
        --psg_key passage \
        --model_name "$MODEL_NAME" \
        --generation_client ollama \
        --api_token_env_var E_INFRA_API_TOKEN \
        --batch_size 128 \
        --skip-regeneration \
        "${DRY_RUN_ARGS[@]}" \
        "${EXTRA_ARGS[@]}"
done
