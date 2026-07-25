#!/bin/bash
# Activate the conda env the retrieval-scoring metrics need, and expose its
# interpreter as $PYTHON. SOURCE this -- do not execute it:
#
#     source eval/eval_env.sh
#     $PYTHON eval/retrieval_score.py --stage aggregate
#
# The metrics need pylate + GTE-ModernColBERT, which live in a DIFFERENT env
# than generation (fgr-generator): pylate pins its own torch/transformers, so
# the two cannot share one env. Hence this indirection instead of a bare
# `python`, which would resolve to whatever env the caller happens to be in.
#
# Nothing here is machine-specific: the env is referenced BY NAME and conda's
# base is derived from the on-PATH conda binary. Override the env name with
# $FGR_EVAL_ENV. Machine-local settings (e.g. HF_HOME pointing at a shared
# model cache off the home quota) belong in .env_eval, which is gitignored --
# never hardcode absolute paths in this file.

# No `set -euo pipefail` here: this file is sourced, and those options would
# leak into the calling script (turning a benign non-zero into an exit).

FGR_EVAL_ENV="${FGR_EVAL_ENV:-modernCoLBERT-benchmark}"
_EVAL_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Machine-local overrides (HF_HOME, FGR_EVAL_ENV, ...). Gitignored; optional.
if [ -f "$_EVAL_ENV_DIR/.env_eval" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$_EVAL_ENV_DIR/.env_eval"
  set +a
fi

# These scripts run non-interactively, so the shell rc is NOT sourced and the
# `conda` shell function is never defined -- bare `conda activate` would fail
# with a "run conda init" error. Source conda's profile script to load the
# function, deriving the base from the on-PATH conda binary so no path is
# hardcoded (same approach as run_generate_einfra.sh).
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not on PATH; cannot activate '$FGR_EVAL_ENV'." >&2
  return 1 2>/dev/null || exit 1
fi
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$FGR_EVAL_ENV"

# `conda activate` sets CONDA_PREFIX but does not always prepend the env's bin
# to PATH, so a bare `python` can still resolve to the base env (which lacks
# pylate). Prepend it, and invoke the interpreter via $CONDA_PREFIX/bin/python
# so the right Python is used regardless of PATH ordering or command hashing.
# Use $CONDA_PREFIX rather than $CONDA_BASE/envs/<name>: on clusters with a
# read-only base, named envs live in the user's own envs_dirs instead.
export PATH="$CONDA_PREFIX/bin:$PATH"
PYTHON="$CONDA_PREFIX/bin/python"

echo "eval env: $FGR_EVAL_ENV ($CONDA_PREFIX)${HF_HOME:+  HF_HOME=$HF_HOME}"
