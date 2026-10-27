# Fine-Grained Relevancy generator


## What goes in `data/`

Only produced span data and what is derived from it: raw model responses,
extracted spans, human spans and `eval/`. Input datasets are never stored there.
LongEmbed is read from the HF cache; the materialized MS MARCO pairs go to
`$FGR_PAIRS_DIR/msmarco/` (default `~/.cache/fgr-generator/pairs`, written by
`prefetch_hf_assets.py --datasets msmarco`, read by `karolina_run_msmarco_vllm.sh`).

## Human span annotations (MS MARCO dev-small)

`data/human_spans/msmarco-spans.jsonl` is a copy of the merged human dataset from the
`explaining-ms-marco` repo (`data/spans/merged/spans.jsonl`): 319 query-passage
pairs, one row per pair, every annotator's highlighted spans as character
offsets. Refresh it by copying that file; do not edit it here.

`custom_utils/human_spans.py` flattens it into the same
`{query, passage, split, qid, doc_id, selected_spans}` records the extraction
outputs use, one record per (pair, annotator), so a human slot can be scored
like a model (`python custom_utils/human_spans.py` runs its self-test). All 319
pairs are in the BEIR `validation` split and are covered by
`data/extracted_spans/ms-marco-*/google~gemma-4-31B-it_validation.jsonl`.
