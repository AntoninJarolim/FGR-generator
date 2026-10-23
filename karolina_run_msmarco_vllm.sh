#!/bin/bash
# Karolina (IT4Innovations) Slurm wrapper for MS MARCO span supervision in the
# reason-then-constrain mode: the model thinks freely, then a per-document
# llguidance substring grammar engages so every span is verbatim.
#
# Unlike run_generate_vllm.sh (which hardcodes LongEmbed and a 64K context) this
# calls llm_extraction.py directly with settings sized for MS MARCO's ~80-token
# passages, and shards the 532,751-pair train split across concurrent jobs.
#
# Prereqs, both on the LOGIN node (compute nodes are air-gapped):
#     bash karolina_setup_env.sh
#     python prefetch_hf_assets.py --models google/gemma-4-31B-it --datasets msmarco
# The second command writes data/msmarco/pairs-{train,validation}.jsonl, which is
# what this script slices. It must exist before submitting.
#
# Usage:
#   # 1k pilot on the 1h experimental queue -- run this FIRST, it sizes the rest:
#   sbatch -p qgpu_exp -t 01:00:00 -J fgr-msmarco-pilot karolina_run_msmarco_vllm.sh --pilot
#   # one shard of the full train run (submit all 32 to run concurrently):
#   sbatch -J fgr-msmarco-s0 karolina_run_msmarco_vllm.sh --shard 0 --num-shards 32
#   # the small validation split in a single job:
#   sbatch karolina_run_msmarco_vllm.sh --split validation
#   # after all shards finish, build the dataset (CPU only, no GPU needed):
#   ./karolina_run_msmarco_vllm.sh --aggregate
#   # anything else is passed straight through to llm_extraction.py
#
# Sharding needs NO merge step. Every shard generates into one directory named by
# --run_tag (the split), writing batch files stamped with their own row range, so
# they never collide and each shard resumes over what the others already did.
# --aggregate then makes a single pass over the full range: get_all_responses
# reads every batch file and write_output pairs each row id with its input record.
# Shards pass --skip_output so they never race to write that one dataset file.
#
# After the pilot, set the token budgets from the measured p99 CoT length:
#   FGR_MAX_MODEL_LEN=4096 FGR_MAX_GEN_TOKENS=2048 FGR_THINKING_BUDGET=1536 \
#     FGR_BATCH_SIZE=2048 sbatch karolina_run_msmarco_vllm.sh --shard 0 --num-shards 32
# The defaults below are the deliberately generous PILOT values; llm_extraction.py
# refuses a context that leaves no room for the passage, so a bad combination
# fails fast instead of silently truncating every passage to "".

#SBATCH --job-name=fgr-msmarco-vllm
#SBATCH --account=fta-26-64
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "$REPO_DIR"
echo "Repo dir: $REPO_DIR"
echo "Host: $(hostname); Job: ${SLURM_JOB_ID:-none}; GPUs: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

FGR_SCRATCH="/scratch/project/fta-26-64/ajarolim"
export HF_HOME="${HF_HOME:-$FGR_SCRATCH/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export CONDA_ENVS_PATH="$FGR_SCRATCH/.conda/envs"
export CONDA_PKGS_DIRS="$FGR_SCRATCH/.conda/pkgs"
export TMPDIR="${SCRATCH:-$FGR_SCRATCH}/tmp"
mkdir -p "$HF_HOME" "$TMPDIR" logs

# We decode greedily, so FlashInfer's sampler adds nothing and its JIT build
# against the pip nvcc is a known failure mode. Use the Torch sampler.
export VLLM_USE_FLASHINFER_SAMPLER=0

if ! command -v conda >/dev/null 2>&1; then
  module load Anaconda3/2024.02-1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fgr-generator
export PATH="$CONDA_PREFIX/bin:$PATH"
PYTHON="$CONDA_PREFIX/bin/python"
echo "PYTHON=$PYTHON"

# --- Argument parsing -------------------------------------------------------
MODEL_NAME="${FGR_MODEL:-google/gemma-4-31B-it}"
SPLIT="train"
SHARD=""
NUM_SHARDS=""
PILOT=0
AGGREGATE=0
PASSTHRU_ARGS=()
while [ $# -gt 0 ]; do
  case $1 in
    --pilot)       PILOT=1 ;;
    --aggregate)   AGGREGATE=1 ;;
    --split)       shift; SPLIT="$1" ;;
    --shard)       shift; SHARD="$1" ;;
    --num-shards)  shift; NUM_SHARDS="$1" ;;
    --model)       shift; MODEL_NAME="$1" ;;
    *)             PASSTHRU_ARGS+=("$1") ;;
  esac
  shift
done

PAIRS_FILE="data/msmarco/pairs-${SPLIT}.jsonl"
if [ ! -f "$PAIRS_FILE" ]; then
  echo "ERROR: $PAIRS_FILE missing. Materialize it on a LOGIN node first:" >&2
  echo "  python prefetch_hf_assets.py --models --datasets msmarco" >&2
  exit 1
fi
TOTAL=$(wc -l < "$PAIRS_FILE")
echo "Pairs file: $PAIRS_FILE ($TOTAL pairs)"

# All shards of a split share this generation dir, so there is nothing to merge.
#
# The pilot gets its OWN tag rather than seeding the real run's dir. Decoding is
# greedy, so a pilot row and a full-run row differ only when one of them hits its
# token budget -- and the whole point of the pilot is to run generous budgets and
# then lower them. Sharing the dir would leave the first 1000 rows of the dataset
# generated under different caps than the other 531,751, and --force_rewrite
# resume would keep them instead of regenerating.
RUN_TAG="$SPLIT"
if [ "$PILOT" -eq 1 ]; then
  RUN_TAG="${SPLIT}-pilot"
fi

# --- Model-specific reasoning wiring ----------------------------------------
# vLLM auto-detects no parser, and a wrong name silently disables the span
# grammar (llm_extraction.py aborts the run if that happens). Resolved before the
# aggregate branch too, because --constrained and --reasoning_parser are what
# select the output directory (ms-marco-xml-constrained-reasoning) -- the
# aggregate pass has to look in the same place the shards wrote to.
case "$MODEL_NAME" in
  google/gemma-4*)  REASONING_ARGS=(--reasoning_parser gemma4 --enable_thinking) ;;
  Qwen/Qwen3*)      REASONING_ARGS=(--reasoning_parser qwen3) ;;
  *)
    echo "ERROR: no reasoning parser known for $MODEL_NAME. Add one rather than" >&2
    echo "       running without it -- unconstrained output is not the goal here." >&2
    exit 1 ;;
esac

# --- Aggregate: turn every shard's batch files into the dataset --------------
# No model is loaded (--skip_generation), so this needs no GPU and can run on a
# login node. It must cover the FULL range: write_output pairs a row id with
# input_data[row_id - from_sample] and skips anything outside the loaded slice.
if [ "$AGGREGATE" -eq 1 ]; then
  echo "AGGREGATE: pairing all generated rows for split=$SPLIT over [0, $TOTAL)."
  "$PYTHON" llm_extraction.py \
      --input_data_name "$PAIRS_FILE" \
      --from_sample 0 \
      --to_sample "$TOTAL" \
      --template_file templates/ms-marco-xml.template \
      --psg_key passage \
      --model_name "$MODEL_NAME" \
      --run_tag "$RUN_TAG" \
      --constrained \
      "${REASONING_ARGS[@]}" \
      --skip_generation \
      --skip-regeneration \
      "${PASSTHRU_ARGS[@]}"
  echo "ALL DONE."
  exit 0
fi

# --- Shard range ------------------------------------------------------------
# Contiguous equal chunks over line indices. The pairs file is written in qrels
# order and never rewritten, so a given (shard, num-shards) always covers the
# same rows -- safe to requeue.
if [ "$PILOT" -eq 1 ]; then
  FROM=0
  TO=1000
  echo "PILOT: first $TO pairs only."
elif [ -n "$SHARD" ] || [ -n "$NUM_SHARDS" ]; then
  if [ -z "$SHARD" ] || [ -z "$NUM_SHARDS" ]; then
    echo "ERROR: --shard and --num-shards must be given together." >&2
    exit 1
  fi
  CHUNK=$(( (TOTAL + NUM_SHARDS - 1) / NUM_SHARDS ))
  FROM=$(( SHARD * CHUNK ))
  TO=$(( FROM + CHUNK ))
  [ "$TO" -gt "$TOTAL" ] && TO=$TOTAL
  if [ "$FROM" -ge "$TOTAL" ]; then
    echo "Shard $SHARD/$NUM_SHARDS starts past the end ($FROM >= $TOTAL); nothing to do."
    exit 0
  fi
  echo "SHARD $SHARD/$NUM_SHARDS: rows [$FROM, $TO) of $TOTAL."
else
  FROM=0
  TO=$TOTAL
  echo "FULL SPLIT: rows [$FROM, $TO)."
fi

# --- Token budgets ----------------------------------------------------------
# Pilot-sized defaults; see the header on resetting these from the measured p99.
OUTPUT_ARGS=(--skip_output)
if [ "$PILOT" -eq 1 ]; then
  OUTPUT_ARGS=()
fi

MAX_MODEL_LEN="${FGR_MAX_MODEL_LEN:-8192}"
MAX_GEN_TOKENS="${FGR_MAX_GEN_TOKENS:-4096}"
THINKING_BUDGET="${FGR_THINKING_BUDGET:-3072}"
BATCH_SIZE="${FGR_BATCH_SIZE:-512}"

# Qwen3.6-27B has 4 KV heads, not divisible by 8, so TP follows the 4 GPUs we
# request rather than defaulting to every visible card.
FGR_TP="${FGR_TP:-$(echo "${CUDA_VISIBLE_DEVICES:-0,1,2,3}" | tr ',' '\n' | grep -c .)}"

echo "==================================================================="
echo "model=$MODEL_NAME  TP=$FGR_TP  split=$SPLIT  rows=[$FROM,$TO)"
echo "max_model_len=$MAX_MODEL_LEN  max_gen=$MAX_GEN_TOKENS"
echo "thinking_budget=$THINKING_BUDGET  batch_size=$BATCH_SIZE"
echo "==================================================================="

# --force_rewrite is RESUME, not overwrite: prepare_out_dir reads the batch files
# already in the output dir and skips those row ids. 532k pairs cannot finish in
# one 24h allocation, so every job needs it, including the first. With a shared
# --run_tag dir it also means a shard skips rows other shards already produced.
#
# --skip_output: concurrent shards generate only, because writing the extracted
# dataset is the --aggregate pass's job. The pilot is a single job over its own
# tag with nothing to race against, so it writes its own output and is
# immediately checkable.
#
# --no_large_doc_grammar: the raised llguidance limits (and the in-process engine
# they force) exist for LongEmbed's 231KB documents. MS MARCO passages compile
# inside the default limits, so use the faster multiprocess engine.
"$PYTHON" llm_extraction.py \
    --input_data_name "$PAIRS_FILE" \
    --from_sample "$FROM" \
    --to_sample "$TO" \
    --template_file templates/ms-marco-xml.template \
    --psg_key passage \
    --model_name "$MODEL_NAME" \
    --run_tag "$RUN_TAG" \
    --generation_client vllm \
    --constrained \
    --no_large_doc_grammar \
    "${OUTPUT_ARGS[@]}" \
    "${REASONING_ARGS[@]}" \
    --thinking_token_budget "$THINKING_BUDGET" \
    --vllm_max_model_len "$MAX_MODEL_LEN" \
    --max_gen_tokens "$MAX_GEN_TOKENS" \
    --vllm_tensor_parallel_size "$FGR_TP" \
    --vllm_gpu_memory_utilization 0.9 \
    --batch_size "$BATCH_SIZE" \
    --skip-regeneration \
    --force_rewrite \
    "${PASSTHRU_ARGS[@]}"

echo "ALL DONE."
