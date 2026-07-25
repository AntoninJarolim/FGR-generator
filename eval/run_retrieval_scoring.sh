#!/usr/bin/env bash
# Full retrieval plausibility/comprehensiveness scoring: every system and
# baseline, both directions, context rungs 0 and 2048, split across two GPUs.
# Resumable — rerunning skips pairs already scored. Run from the repo root:
#
#     bash eval/run_retrieval_scoring.sh
#
# Add a new model later by appending its extraction file to GPU0/GPU1 below, or
# score it ad hoc:
#     source eval/eval_env.sh
#     $PYTHON eval/retrieval_score.py --stage score --system <file>
#     $PYTHON eval/retrieval_score.py --stage aggregate
set -euo pipefail
cd "$(dirname "$0")/.."

# Activates the pylate env and sets $PYTHON (see eval/eval_env.sh); no absolute
# paths here -- machine-local bits like HF_HOME go in the gitignored .env_eval.
source "$(dirname "$0")/eval_env.sh"
PY="$PYTHON"
CON=data/extracted_relevancy/long-embed-xml-constrained
UNC=data/extracted_relevancy/long-embed-xml

# Constrained systems + the three trivial floors. The unconstrained arms are
# omitted from the default run (they serve the hallucination/format-tax story,
# not the retrieval ranking) — add them here when there is GPU time to spare.
GPU0_SYSTEMS=(
  "$CON/google~gemma-4-12B-it_from0-to12612.jsonl"
  "$CON/mistralai~Ministral-3-14B-Instruct-2512_from0-to12612.jsonl"
  "baseline:lexical"
)
GPU1_SYSTEMS=(
  "$CON/google~gemma-4-E4B-it_from0-to12612.jsonl"
  "baseline:random"
  "baseline:lead_k"
)

# narrativeqa is subsampled 20% (deterministically, same pairs for every system)
# because each ablated 32K-token re-encode costs ~8.5s; comprehensiveness runs
# only at the ctx=2048 rung (bare-span removal is predicted flat on these doc
# lengths — add "--directions comprehensiveness --contexts 0" runs later if
# needed; everything is resumable).
NQA_FRAC=0.2

run_queue () {
  local device=$1; shift
  for system in "$@"; do
    echo "=== [$device] $system plausibility ==="
    $PY eval/retrieval_score.py --stage score --device "$device" \
        --system "$system" --directions plausibility --contexts 0 2048 \
        --narrativeqa-frac $NQA_FRAC
    echo "=== [$device] $system comprehensiveness ==="
    $PY eval/retrieval_score.py --stage score --device "$device" \
        --system "$system" --directions comprehensiveness --contexts 2048 \
        --narrativeqa-frac $NQA_FRAC
  done
}

$PY eval/retrieval_score.py --stage corpus --device cuda:0   # no-op when cached

run_queue cuda:0 "${GPU0_SYSTEMS[@]}" &
PID0=$!
run_queue cuda:1 "${GPU1_SYSTEMS[@]}" &
PID1=$!
wait $PID0 $PID1

$PY eval/retrieval_score.py --stage aggregate
echo "done — view with: python eval/summary_table.py"
