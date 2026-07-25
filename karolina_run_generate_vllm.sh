#!/bin/bash
# Karolina (IT4Innovations) Slurm wrapper around run_generate_vllm.sh for the
# LARGE_MODELS list (gemma-4-31B-it, gemma-4-26B-A4B-it, Qwen3.6-27B).
#
# Karolina uses Slurm (NOT PBS/qsub). GPU nodes are 8x A100-40GB and qgpu
# allows partial-node allocation billed per GPU, so we request only the GPUs we
# need. Weights and the conda env live on project scratch (home is a 25 GB
# quota); build them first on the LOGIN node with:
#     bash karolina_setup_env.sh
#     python prefetch_hf_assets.py       # weights + dataset + aux tokenizer (offline prep)
#
# Usage:
#   # Dry run (10 samples/model, both modes) on the 1h experimental queue:
#   sbatch -p qgpu_exp -t 01:00:00 karolina_run_generate_vllm.sh --dry-run
#   # Full run (both modes, all 3 large models) on the standard GPU queue:
#   sbatch karolina_run_generate_vllm.sh
#
# tensor_parallel_size is forced to 4 (FGR_TP): Qwen3.6-27B has 4 KV heads,
# which are NOT divisible by 8, so the default TP=(#GPUs)=8 would crash. TP=4
# works for all three large models and fits comfortably (max 62.5 GB / 4 =
# ~16 GB per 40 GB card).

#SBATCH --job-name=fgr-large-vllm
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

# --- Locate the repo -------------------------------------------------------
# Under Slurm the batch script runs from a spool copy; SLURM_SUBMIT_DIR holds
# the submission directory. Fall back to the script's dir when run by hand.
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "$REPO_DIR"
echo "Repo dir: $REPO_DIR"
echo "Host: $(hostname); Job: ${SLURM_JOB_ID:-none}; GPUs: ${CUDA_VISIBLE_DEVICES:-unset}"

# --- Scratch caches / conda env location -----------------------------------
FGR_SCRATCH="/scratch/project/fta-26-64/ajarolim"
export HF_HOME="${HF_HOME:-$FGR_SCRATCH/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"   # weights are pre-downloaded; don't hit the network
export CONDA_ENVS_PATH="$FGR_SCRATCH/.conda/envs"   # so `conda activate fgr-generator` finds the scratch env
export CONDA_PKGS_DIRS="$FGR_SCRATCH/.conda/pkgs"
export TMPDIR="${SCRATCH:-$FGR_SCRATCH}/tmp"
mkdir -p "$HF_HOME" "$TMPDIR"
echo "HF_HOME=$HF_HOME  CONDA_ENVS_PATH=$CONDA_ENVS_PATH"

# Compute nodes should inherit the login PATH (--export=ALL). If conda is
# somehow absent, load the module used to build the env.
if ! command -v conda >/dev/null 2>&1; then
  module load Anaconda3/2024.02-1
fi

# --- Run both decoding modes over the LARGE_MODELS list ---------------------
export RUN_LARGE_MODELS=1
FGR_TP="${FGR_TP:-4}"

RUNNER=bash
command -v zsh >/dev/null 2>&1 && RUNNER=zsh

# FGR_MODES selects which decoding passes to run (space-separated): "1 0" for
# both (default), "1" constrained only, "0" unconstrained only. Splitting the
# two modes into separate concurrent jobs (each its own 4-GPU allocation) halves
# wall-clock at identical per-GPU billing.
#
# The mode maps to run_generate_vllm.sh's --constrained flag: constrained adds
# span-grammar decoding, unconstrained is the bare default. Both use the same
# long-embed-xml template; outputs land in long-embed-xml{-constrained} dirs.
for MODE in ${FGR_MODES:-1 0}; do   # 1 = constrained (span grammar), 0 = unconstrained
  MODE_FLAG=()
  [ "$MODE" -eq 1 ] && MODE_FLAG=(--constrained)
  echo "==================================================================="
  echo "=== Decoding mode $([ "$MODE" -eq 1 ] && echo CONSTRAINED || echo UNCONSTRAINED)  (TP=$FGR_TP) ==="
  echo "==================================================================="
  "$RUNNER" ./run_generate_vllm.sh \
      --vllm_tensor_parallel_size "$FGR_TP" \
      "${MODE_FLAG[@]}" \
      "$@"
done

echo "ALL DONE."
