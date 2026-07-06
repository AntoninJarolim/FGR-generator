#!/bin/bash
# MetaCentrum (DGX) wrapper around run_generate_vllm.sh.
#
# Submits as a PBS job that (1) initializes the MetaCentrum environment
# (modules, scratch, caches), (2) creates/updates the `fgr-generator` conda
# env from requirements-lock.txt (a pip freeze of the known-working local
# env; falls back to the unpinned requirements.txt), and (3) runs
# run_generate_vllm.sh with RUN_EXTREME_MODELS=1 so the EXTREME_MODELS list
# (Qwen3-235B-A22B-Thinking-2507) is used. llm_extraction.py auto-sets
# vLLM tensor_parallel_size to the number of allocated GPUs.
#
# Usage (from the repo directory on a MetaCentrum frontend, repo on /storage):
#   qsub dgx_run_generate_vllm.sh
#   qsub -q gpu@pbs-m1.metacentrum.cz dgx_run_generate_vllm.sh   # non-DGX GPU queue
#   ./dgx_run_generate_vllm.sh --dry-run     # directly on an interactive GPU node;
#                                            # extra args are forwarded to run_generate_vllm.sh
#
# Tutorial this follows: https://gist.github.com/vlccek/892c5d49fe0ce99a4f62238b15bc1e9d
# If storage access fails with permission errors on the frontend, run `kinit`.

#PBS -N fgr-large-vllm
#PBS -q gpu_dgx@pbs-m1.metacentrum.cz
#PBS -l select=1:ncpus=64:ngpus=4:mem=64gb:scratch_ssd=1000gb
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o logs/
#PBS -e logs/

set -euo pipefail

# --- Locate the repo -------------------------------------------------------
# Under PBS the script runs from a spool copy, so $0 is useless; qsub records
# the submission directory in PBS_O_WORKDIR. When run by hand, fall back to
# the script's own directory.
REPO_DIR="${PBS_O_WORKDIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "$REPO_DIR"
echo "Repo dir: $REPO_DIR"
echo "Host: $(hostname); GPUs: ${CUDA_VISIBLE_DEVICES:-unset}"

# --- MetaCentrum init ------------------------------------------------------
# Frontends/compute nodes provide conda via the mambaforge module. Skip ifo logs/
# PBS -e logs/ conda is already on PATH (e.g. running outside MetaCentrum for testing).
if ! command -v conda >/dev/null 2>&1; then
  module add mambaforge
fi

## Init TMPDIRi, scratchdir should be created by PBS automatically 
if [ -n "${SCRATCHDIR:-}" ]; then
  export TMPDIR="$SCRATCHDIR"
  export HF_HOME="${HF_HOME:-$SCRATCHDIR/hf-cache}"
  mkdir -p "$HF_HOME"
  echo "Scratch: $SCRATCHDIR (TMPDIR, HF_HOME=$HF_HOME)"
  # MetaCentrum-provided cleanup helper; wipe scratch when the job ends.
  trap 'command -v clean_scratch >/dev/null 2>&1 && clean_scratch' TERM EXIT
else
  echo "WARNING: SCRATCHDIR not set (interactive run?); using default tmp/HF cache."
fi

# --- Conda env: create or sync `fgr-generator` -----------------------------
# Named env (not --prefix) so run_generate_vllm.sh's `conda activate
# fgr-generator` works unchanged. NOTE: the mambaforge module's install lives
# on read-only AFS, so a named env is NOT created under $CONDA_BASE/envs --
# conda/mamba falls back to the first WRITABLE entry in envs_dirs (a /storage
# path, e.g. ~/projects/mamba). Never hardcode the env path; after
# `conda activate` use $CONDA_PREFIX, which points at the real location.
# Set FGR_SKIP_ENV_SETUP=1 to skip when the env is known to be ready.
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

# The package cache next to the read-only AFS install is unusable ("Could not
# open lockfile .../pkgs/cache/cache.lock"); download packages into scratch
# (or home when run interactively) instead.
export CONDA_PKGS_DIRS="${SCRATCHDIR:-$HOME}/conda-pkgs"
mkdir -p "$CONDA_PKGS_DIRS"

if [ "${FGR_SKIP_ENV_SETUP:-0}" -ne 1 ]; then
  if ! conda env list | grep -qE '^fgr-generator\s'; then
    echo "Creating conda env fgr-generator (python 3.13)..."
    # mamba where available (the module ships it), plain conda otherwise.
    if command -v mamba >/dev/null 2>&1; then
      mamba create -y -n fgr-generator python=3.13
    else
      conda create -y -n fgr-generator python=3.13
    fi
  fi
  echo "Lest activate fgr-generator"
  conda activate fgr-generator
  # $CONDA_PREFIX is the activated env's real location, wherever envs_dirs put
  # it ($CONDA_BASE/envs is read-only AFS here, so it is NOT there -- see the
  # note above). The previous hardcoded $CONDA_BASE/envs/fgr-generator/bin/pip
  # path killed the job with "No such file or directory".
  echo "Activated env prefix: $CONDA_PREFIX"
  if [ ! -x "$CONDA_PREFIX/bin/pip" ]; then
    echo "ERROR: pip not found in activated env ($CONDA_PREFIX/bin/pip)." >&2
    exit 1
  fi
  # requirements-lock.txt is the pip freeze of the working local env
  # (requirements.txt is unpinned and may lag behind it).
  REQ_FILE="requirements-lock.txt"
  [ -f "$REQ_FILE" ] || REQ_FILE="requirements.txt"
  echo "Installing python deps from $REQ_FILE ..."
  "$CONDA_PREFIX/bin/pip" install -r "$REQ_FILE"
  conda deactivate
fi

# --- Run the generation over the extreme-size models -----------------------
export RUN_EXTREME_MODELS=1

# run_generate_vllm.sh has a zsh shebang but is bash-compatible; MetaCentrum
# nodes are not guaranteed to have zsh, so pick whichever exists.
RUNNER=bash
command -v zsh >/dev/null 2>&1 && RUNNER=zsh

"$RUNNER" ./run_generate_vllm.sh "$@"
