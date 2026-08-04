#!/bin/bash
# Karolina (IT4Innovations) Slurm wrapper for the gpt-oss second judge.
#
# The judge model is fixed to openai/gpt-oss-120b on purpose: three of the six
# systems under comparison are gemma-4-31B arms and one is Qwen3.6-27B, so a
# gemma or Qwen judge would be scoring its own family. Do NOT swap the model to
# work around a runtime problem -- report the problem instead.
#
# Prereqs, both on the LOGIN node (compute nodes are air-gapped):
#     bash karolina_setup_env.sh
#     HF_HOME=/scratch/project/fta-26-64/ajarolim/.cache/huggingface \
#       python -c "from huggingface_hub import snapshot_download; \
#                  snapshot_download('openai/gpt-oss-120b', \
#                    ignore_patterns=['*.pth','*.gguf','*.bin','original/*','metal/*'])"
#
# Usage:
#   # Smoke test, 20 comparisons, 1h experimental queue:
#   sbatch -p qgpu_exp -t 01:00:00 -J fgr-judge-smoke karolina_run_judge_vllm.sh \
#       --limit 20 --tag smoke
#   # Full run, 16,246 comparisons:
#   sbatch karolina_run_judge_vllm.sh

#SBATCH --job-name=fgr-gptoss-judge
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

if ! command -v conda >/dev/null 2>&1; then
  module load Anaconda3/2024.02-1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fgr-generator
PYTHON="$CONDA_PREFIX/bin/python"
echo "PYTHON=$PYTHON"

# --gpus is what we were actually allocated, so TP follows it. gpt-oss-120b has
# 8 KV heads, so TP in {1,2,4,8} all divide cleanly.
FGR_TP="${FGR_TP:-$(echo "${CUDA_VISIBLE_DEVICES:-0,1,2,3}" | tr ',' '\n' | grep -c .)}"

# max_model_len 16384: the longest judge prompt measured over the 16,246
# comparisons is ~7.8K tokens, leaving generous room for the analysis channel
# plus the JSON within --max-tokens.
#
# --reasoning-effort low: the API judge ran with thinking DISABLED and used the
# JSON `reasoning` field as its chain of thought. 'low' is the closest analogue,
# and keeps the analysis channel from eating the token budget.
echo "=== judge: openai/gpt-oss-120b, TP=$FGR_TP ==="
"$PYTHON" eval/judge_vllm.py \
    --model openai/gpt-oss-120b \
    --items data/eval/judge_items_full.json \
    --tensor-parallel-size "$FGR_TP" \
    --max-model-len 16384 \
    --max-tokens 2000 \
    --reasoning-parser openai_gptoss \
    --reasoning-effort low \
    "$@"

echo "ALL DONE."
