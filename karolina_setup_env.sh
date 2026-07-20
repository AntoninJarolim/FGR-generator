#!/bin/bash
# One-time environment setup for Karolina (IT4Innovations), run on the LOGIN
# node so it consumes NO GPU node-hours. Creates the `fgr-generator` conda env
# on the project scratch (home has only a 25 GB quota) and installs the pinned
# deps. Model weights / dataset are pre-downloaded separately, on the login
# node, with the generic `prefetch_hf_assets.py` (offline-prep, not Karolina-only).
set -euo pipefail

# Project scratch: no quota, unlike the 25 GB home. Keep env + caches here.
FGR_SCRATCH="/scratch/project/fta-26-64/ajarolim"
export CONDA_ENVS_PATH="$FGR_SCRATCH/.conda/envs"
export CONDA_PKGS_DIRS="$FGR_SCRATCH/.conda/pkgs"
mkdir -p "$CONDA_ENVS_PATH" "$CONDA_PKGS_DIRS"

# The login node exposes conda via the Anaconda3 module (already on PATH here).
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

REPO_DIR="$FGR_SCRATCH/FGR-generator"
cd "$REPO_DIR"

if ! conda env list | grep -qE "/\.conda/envs/fgr-generator\s"; then
  # python 3.13: requirements-lock.txt was frozen on 3.13, where numpy==2.4.4
  # is compatible. On 3.12 mistral-common caps numpy<2.4 -> ResolutionImpossible.
  echo "Creating conda env fgr-generator (python 3.13) at $CONDA_ENVS_PATH ..."
  conda create -y -n fgr-generator python=3.13
fi

conda activate fgr-generator
echo "Activated env prefix: $CONDA_PREFIX"

REQ_FILE="requirements-lock.txt"
[ -f "$REQ_FILE" ] || REQ_FILE="requirements.txt"
echo "Installing python deps from $REQ_FILE ..."
"$CONDA_PREFIX/bin/pip" install --no-input -r "$REQ_FILE"

echo "DONE: fgr-generator env ready at $CONDA_PREFIX"
"$CONDA_PREFIX/bin/python" -c "import vllm, torch, transformers; print('vllm', vllm.__version__, '| torch', torch.__version__, '| transformers', transformers.__version__)"
