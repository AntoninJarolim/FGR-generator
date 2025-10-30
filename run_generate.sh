#!/bin/bash

# Load .env and export OPENAI_BASE_URL → OLLAMA_HOST
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo ".env file not found!"
    exit 1
fi

if [ -z "$OPENAI_BASE_URL" ]; then
    echo "OPENAI_BASE_URL not set in .env"
    exit 1
fi

export OLLAMA_HOST="$OPENAI_BASE_URL"
echo "Using remote Ollama server: $OLLAMA_HOST"

# --- Model Definitions ---
DEFAULT_LLM_MODELS=(
  "llama3.1:8b"
  "llama3.3:70b"
  "gemma3:4b"
  "mixtral:8x7b"
  "gemma3:12b"
  "qwen3:14b"
  "phi4:14b"
  "gpt-oss:20b"
  "gemma3:27b"
  "gemma2:27b"
  "deepseek-r1:32b"
  "qwen3:32b"
  "qwen2.5:32b"
  "qwen2.5:72b"
  "gpt-oss:120b"
)

EINFRA_LLM_MODELS=(
    "gpt-oss-120b"
    "deepseek-r1"
)

# --- Argument Parsing and Configuration ---
LLM_MODELS=("${DEFAULT_LLM_MODELS[@]}")
PULL_ONLY=0

# Parse arguments
for arg in "$@"
do
    case $arg in
        --einfra)
        echo "Using e-infra models"
        LLM_MODELS=("${EINFRA_LLM_MODELS[@]}")
        ;;
        pull)
        PULL_ONLY=1
        ;;
    esac
done

# --- Main Logic ---
if [[ "$PULL_ONLY" -eq 1 ]]; then
  for model in "${LLM_MODELS[@]}"; do
    echo "Pulling model: $model"
    if ollama pull "$model"; then
      echo "✅ Successfully pulled: $model"
    else
      echo "❌ Failed to pull: $model"
    fi
    echo "---------------------------------"
  done
  exit 0
fi


# List of template files
TEMPLATE_FILES=(
    "templates/FDM-evidence-pos.template"
    # "templates/FDM-evidence-neg.template"
    # Add more template files here
)

# Loop through each model and then each template file
for MODEL_NAME in "${LLM_MODELS[@]}"; do
    for TEMPLATE_FILE in "${TEMPLATE_FILES[@]}"; do
        echo "Running llm_extraction.py with model: $MODEL_NAME and template: $TEMPLATE_FILE"
        python llm_extraction.py \
            --input_data_name data/to-generate/for_llm_annotation.json \
            --template_file "$TEMPLATE_FILE" \
            --psg_key clean_text \
            --model_name "$MODEL_NAME" \
            --generation_client ollama \
            --batch_size 50 \
	          --max_fixes 10
        echo "Finished running for model: $MODEL_NAME and template: $TEMPLATE_FILE"
        echo "----------------------------------------------------"
    done
done
