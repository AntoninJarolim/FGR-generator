#!/usr/bin/env bash
# Retrieval plausibility/comprehensiveness for the new Karolina-cluster models
# (gemma-4-31B, gemma-4-26B-A4B, Qwen3.6-27B), constrained arm, both GPUs.
# Same protocol as eval/run_retrieval_scoring.sh; resumable. From repo root:
#     bash eval/run_retrieval_new_models.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY=/mnt/data/ijarolim/conda/envs/modernCoLBERT-benchmark/bin/python
export HF_HOME=/mnt/data/ijarolim/.hfcache
LE=data/extracted_relevancy/long-embed-json-constrained
NQA_FRAC=0.2

GPU0_SYSTEMS=(
  "$LE/google~gemma-4-31B-it_from0-to12612.jsonl"
  "$LE/google~gemma-4-26B-A4B-it_from0-to12612.jsonl"
)
GPU1_SYSTEMS=(
  "$LE/Qwen~Qwen3.6-27B_from0-to12612.jsonl"
)

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

run_queue cuda:0 "${GPU0_SYSTEMS[@]}" &
PID0=$!
run_queue cuda:1 "${GPU1_SYSTEMS[@]}" &
PID1=$!
wait $PID0 $PID1

$PY eval/retrieval_score.py --stage aggregate
echo "done — view with: python eval/summary_table.py"
